"""Score the medical adapter against its unadapted base on held-out medical QA.

Both models are served by the same deployment, so the only difference is the
LoRA adapter. Each question is asked once per model at temperature 0 with
thinking disabled. By default the reply must start with the answer (a letter, or
yes/no/maybe). Results land in serving/benchmarks/<timestamp>-medeval-<scoring>/.

    python -m serving.medical_eval --api-version v1 --budget-usd 1.5
    python -m serving.medical_eval --scoring logprob --datasets medqa,medmcqa

Datasets (fetched once from the Hugging Face datasets server, then cached):
  medqa     GBaker/MedQA-USMLE-4-options, test split (USMLE-style, 4 options)
  medmcqa   openlifescienceai/medmcqa, validation split, seeded sample
  pubmedqa  qiaojin/PubMedQA pqa_labeled (expert-labelled, with abstract context)
"""

import argparse
import asyncio
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import re
import time

import httpx

from serving.benchmark import DEPLOYMENTS, estimated_cost
from serving.chat import resolve_endpoint
from serving.config import load_credentials

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "serving/benchmarks/eval-data"
ROWS_URL = "https://datasets-server.huggingface.co/rows"
MODELS = ["qwen-medical", "Qwen/Qwen3-8B"]
SOURCES = {
    "medqa": ("GBaker/MedQA-USMLE-4-options", "default", "test"),
    "medmcqa": ("openlifescienceai/medmcqa", "default", "validation"),
    "pubmedqa": ("qiaojin/PubMedQA", "pqa_labeled", "train"),
}
LETTERS = "ABCD"


def fetch_rows(dataset, config, split):
    rows, offset = [], 0
    with httpx.Client(timeout=60) as client:
        while True:
            for attempt in range(5):
                response = client.get(ROWS_URL, params={"dataset": dataset, "config": config, "split": split,
                                                        "offset": offset, "length": 100})
                if response.status_code == 200:
                    break
                time.sleep(2 ** attempt)
            response.raise_for_status()
            page = response.json()
            rows += [item["row"] for item in page["rows"]]
            offset += 100
            if offset >= page["num_rows_total"]:
                return rows


def to_item(name, row):
    """Normalise a dataset row to {id, prompt, answer, kind, subject}."""
    if name == "medqa":
        options = "\n".join(f"{k}. {row['options'][k]}" for k in LETTERS)
        return {"prompt": f"{row['question']}\n\n{options}", "answer": row["answer_idx"], "kind": "letter",
                "subject": row.get("meta_info")}
    if name == "medmcqa":
        options = "\n".join(f"{k}. {row['op' + k.lower()]}" for k in LETTERS)
        return {"id": row["id"], "prompt": f"{row['question']}\n\n{options}", "answer": LETTERS[row["cop"]],
                "kind": "letter", "subject": row.get("subject_name")}
    context = "\n".join(row["context"]["contexts"])
    return {"id": str(row["pubid"]), "prompt": f"Abstract:\n{context}\n\nQuestion: {row['question']}",
            "answer": row["final_decision"], "kind": "ynm", "subject": None}


def load_dataset(name, limit, seed):
    path = DATA_DIR / f"{name}.jsonl"
    if not path.is_file():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        rows = fetch_rows(*SOURCES[name])
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    items = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        item = to_item(name, json.loads(line))
        item.setdefault("id", f"{name}-{index}")
        items.append({**item, "dataset": name})
    if limit and len(items) > limit:
        items = random.Random(seed).sample(items, limit)
    return items


def instruction(kind):
    if kind == "letter":
        return "Answer with the letter of the single best option only (A, B, C, or D)."
    return "Answer with exactly one word: yes, no, or maybe."


def parse_answer(text, kind):
    """The first answer token in the reply, or None when there is none."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    if kind == "letter":
        match = re.search(r"\b([A-D])\b", text)
        return match.group(1) if match else None
    match = re.search(r"\b(yes|no|maybe)\b", text, flags=re.I)
    return match.group(1).lower() if match else None


STATED_PATTERNS = {
    "letter": [r"answer(?: is|:)[:\s*]*\(?([A-D])\b", r"\*\*\(?([A-D])[.):*]", r"(?m)^\s*\(?([A-D])[.)]\s"],
    "ynm": [r"answer(?: is|:)[:\s*]*(yes|no|maybe)\b", r"\*\*(yes|no|maybe)\b", r"(?m)^\s*(yes|no|maybe)\b"],
}


def parse_stated_answer(text, kind):
    """The answer an explained reply commits to ("The answer is B", "**B.**"), or None.

    Used on the longer retry, where a bare first-letter match would also catch
    words such as "Vitamin A" inside the explanation.
    """
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    for pattern in STATED_PATTERNS[kind]:
        match = re.search(pattern, text, flags=re.I)
        if match:
            return match.group(1).upper() if kind == "letter" else match.group(1).lower()
    return None


def wilson(correct, total, z=1.96):
    if not total:
        return None, None
    p = correct / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return centre - half, centre + half


def mcnemar_exact(only_a, only_b):
    """Two-sided exact McNemar p-value from the discordant pair counts."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def summarise(records, mode="strict"):
    """Accuracy per model and dataset, plus the paired adapter-vs-base comparison.

    strict: the short first reply must start with the answer.
    lenient: replies with no answer get a longer retry, scored on the answer they state.
    """
    parsed_key, correct_key = ("parsed", "correct") if mode == "strict" else ("parsed_lenient", "correct_lenient")
    records = [{**r, "parsed": r.get(parsed_key), "correct": r.get(correct_key, False)} for r in records]
    by_key = defaultdict(dict)
    for record in records:
        by_key[(record["dataset"], record["id"])][record["model"]] = record
    out = {}
    for dataset in sorted({d for d, _ in by_key}):
        pairs = [v for (d, _), v in by_key.items() if d == dataset and all(m in v for m in MODELS)]
        entry = {"questions": len(pairs)}
        for model in MODELS:
            correct = sum(p[model]["correct"] for p in pairs)
            low, high = wilson(correct, len(pairs))
            entry[model] = {"accuracy": correct / len(pairs) if pairs else None, "ci95": [low, high],
                            "unparsed": sum(p[model]["parsed"] is None for p in pairs),
                            "errors": sum(p[model].get("error") is not None for p in pairs)}
        adapter_only = sum(p[MODELS[0]]["correct"] and not p[MODELS[1]]["correct"] for p in pairs)
        base_only = sum(p[MODELS[1]]["correct"] and not p[MODELS[0]]["correct"] for p in pairs)
        entry["paired"] = {"adapter_only_correct": adapter_only, "base_only_correct": base_only,
                           "agreement": sum(p[MODELS[0]]["parsed"] == p[MODELS[1]]["parsed"] for p in pairs)
                           / len(pairs) if pairs else None,
                           "mcnemar_p": mcnemar_exact(adapter_only, base_only)}
        if dataset == "pubmedqa":
            entry["answer_distribution"] = {m: {a: sum(p[m]["parsed"] == a for p in pairs)
                                                for a in ("yes", "no", "maybe", None)} for m in MODELS}
        out[dataset] = entry
    return out


LOGPROB_PREFILL = "The answer is ("


def letter_choice(top_logprobs):
    """Most likely option letter among the next-token candidates, with each letter's probability."""
    probs = {}
    for candidate in top_logprobs:
        letter = candidate["token"].strip()
        if len(letter) == 1 and letter in LETTERS:
            probs[letter] = probs.get(letter, 0.0) + math.exp(candidate["logprob"])
    return (max(probs, key=probs.get) if probs else None), probs


async def ask(client, endpoint, headers, model, item, max_tokens, prefill=None):
    """Generated reply text, or with `prefill` the next token's top-20 log-probabilities after it."""
    messages = [{"role": "user", "content": f"{item['prompt']}\n\n{instruction(item['kind'])}"}]
    body = {"model": model, "max_tokens": max_tokens, "temperature": 0, "seed": 0,
            "messages": messages, "chat_template_kwargs": {"enable_thinking": False}}
    if prefill:
        messages.append({"role": "assistant", "content": prefill})
        body.update(max_tokens=1, logprobs=True, top_logprobs=20,
                    add_generation_prompt=False, continue_final_message=True)
    error = None
    for attempt in range(4):
        try:
            response = await client.post(endpoint + "/v1/chat/completions", headers=headers, json=body)
            if response.status_code < 500:
                response.raise_for_status()
                choice = response.json()["choices"][0]
                if prefill:
                    return choice["logprobs"]["content"][0]["top_logprobs"], None
                return choice["message"]["content"], None
            error = f"HTTP {response.status_code}"
        except httpx.HTTPStatusError as exc:
            return "", f"HTTP {exc.response.status_code}: {exc.response.text[:200]}"
        except httpx.HTTPError as exc:
            error = repr(exc)[:200]
        await asyncio.sleep(2 ** attempt)
    return "", error


async def run(args):
    credentials = load_credentials(ROOT)
    headers = {"Modal-Key": credentials["MODAL_KEY"], "Modal-Secret": credentials["MODAL_SECRET"]}
    endpoint = resolve_endpoint(credentials, args.api_version)
    deployment = DEPLOYMENTS[args.api_version]
    limits = {"medqa": args.medqa_limit, "medmcqa": args.medmcqa_limit, "pubmedqa": args.pubmedqa_limit}
    items = [item for name in args.datasets.split(",") for item in load_dataset(name, limits[name], args.seed)]
    if args.scoring == "logprob":
        items = [item for item in items if item["kind"] == "letter"]  # yes/no/maybe stays generative
    jobs = [(model, item) for item in items for model in MODELS]
    print(f"{len(items)} questions x {len(MODELS)} models on {args.api_version}", flush=True)

    out_dir = ROOT / "serving/benchmarks" / f"{time.strftime('%Y%m%d-%H%M%S')}-medeval-{args.scoring}"
    out_dir.mkdir(parents=True, exist_ok=True)
    records, started, stopped = [], time.monotonic(), False
    queue = iter(jobs)
    # Billing assumes every replica the deployment may run is up.
    spent = lambda: estimated_cost(time.monotonic() - started, deployment)

    async def worker(client):
        nonlocal stopped
        for model, item in queue:
            if spent() >= args.budget_usd:
                stopped = True
                return
            if args.scoring == "logprob":
                top, error = await ask(client, endpoint, headers, model, item, 1, prefill=LOGPROB_PREFILL)
                parsed, probs = letter_choice(top or [])
                records.append({"dataset": item["dataset"], "id": item["id"], "model": model,
                                "subject": item["subject"], "answer": item["answer"], "letter_probs": probs,
                                "parsed": parsed, "correct": parsed == item["answer"], "error": error,
                                "parsed_lenient": parsed, "correct_lenient": parsed == item["answer"]})
                continue
            text, error = await ask(client, endpoint, headers, model, item, args.max_tokens)
            parsed = parse_answer(text, item["kind"])
            record = {"dataset": item["dataset"], "id": item["id"], "model": model,
                      "subject": item["subject"], "answer": item["answer"], "reply": text,
                      "parsed": parsed, "correct": parsed == item["answer"], "error": error,
                      "parsed_lenient": parsed, "retry_reply": None}
            if parsed is None and args.retry_tokens and error is None:
                retry, error = await ask(client, endpoint, headers, model, item, args.retry_tokens)
                record.update(retry_reply=retry, parsed_lenient=parse_stated_answer(retry, item["kind"]),
                              error=error)
            record["correct_lenient"] = record["parsed_lenient"] == item["answer"]
            records.append(record)
            if len(records) % 500 == 0:
                print(f"  {len(records)}/{len(jobs)} answered, est. ${spent():.2f}", flush=True)

    async with httpx.AsyncClient(timeout=httpx.Timeout(args.request_timeout, connect=30)) as client:
        await asyncio.gather(*(worker(client) for _ in range(args.concurrency)))

    (out_dir / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    summary = {"args": vars(args), "api_version": args.api_version, "models": MODELS,
               "estimated_cost_usd": round(spent(), 3), "budget_stopped": stopped,
               "wall_s": round(time.monotonic() - started, 1), "datasets": summarise(records),
               "datasets_lenient": summarise(records, "lenient")}
    if args.scoring == "logprob":
        # How much next-token probability fell on a letter at all; near 1 means the prefill worked.
        summary["mean_letter_mass"] = {
            f"{dataset}/{model}": sum(sum(r["letter_probs"].values()) for r in subset) / len(subset)
            for dataset in sorted({r["dataset"] for r in records}) for model in MODELS
            if (subset := [r for r in records if r["dataset"] == dataset and r["model"] == model])}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for mode in ("datasets", "datasets_lenient"):
        for name, entry in summary[mode].items():
            line = ", ".join(f"{m} {entry[m]['accuracy']:.1%} ({entry[m]['unparsed']} unparsed)" for m in MODELS)
            print(f"{mode} {name} (n={entry['questions']}): {line}; "
                  f"McNemar p={entry['paired']['mcnemar_p']:.3g}", flush=True)
    print(f"Estimated cost ${summary['estimated_cost_usd']:.2f}; results in {out_dir}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api-version", default="v1", choices=sorted(DEPLOYMENTS))
    parser.add_argument("--datasets", default="medqa,medmcqa,pubmedqa")
    parser.add_argument("--medqa-limit", type=int, default=0, help="0 = all 1,273 test questions")
    parser.add_argument("--medmcqa-limit", type=int, default=1000)
    parser.add_argument("--pubmedqa-limit", type=int, default=0, help="0 = all 1,000 labelled questions")
    parser.add_argument("--scoring", choices=["generate", "logprob"], default="generate",
                        help="generate: parse a short reply, plus a longer retry for the lenient score. "
                             "logprob: prefill 'The answer is (' and take the likeliest letter; "
                             "multiple-choice sets only, unaffected by answer formatting")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--retry-tokens", type=int, default=384,
                        help="Re-ask unparsed questions with this many tokens for the lenient score; 0 disables")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--budget-usd", type=float, default=1.5)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    asyncio.run(run(main()))
