"""Check that serving stacks give the same greedy answers to the same prompts.

Data parallelism copies the model; it should not change what it says. Each
prompt is sent alone (no batching neighbours) at temperature 0. The reference
API is asked twice, so run-to-run noise on one stack is measured before
blaming another. Results land in serving/benchmarks/<timestamp>-parity/.
"""

import argparse
import json
from pathlib import Path
import time

import httpx

from serving.benchmark import DEPLOYMENTS, PROMPTS, estimated_cost
from serving.chat import resolve_endpoint
from serving.config import load_credentials

ROOT = Path(__file__).resolve().parents[1]
PARITY_PROMPTS = PROMPTS + [
    "What does an HbA1c of 8.5% mean for a patient with diabetes?",
    "Give three red-flag symptoms of a headache that need same-day assessment.",
    "Explain the mechanism of action of ACE inhibitors in plain language.",
    "How is iron-deficiency anaemia usually investigated in an adult?",
    "What is the difference between type 1 and type 2 diabetes?",
    "List the main contraindications to giving a live vaccine.",
    "Describe the stages of wound healing in one paragraph.",
    "When should a patient with chest pain call emergency services?",
]


def first_difference(a, b):
    """Index of the first differing character, or None when the texts are identical."""
    for index, (left, right) in enumerate(zip(a, b)):
        if left != right:
            return index
    return None if len(a) == len(b) else min(len(a), len(b))


def compare(reference, candidate):
    """Exact-match rate and how far answers agree before diverging."""
    diffs = [first_difference(a, b) for a, b in zip(reference, candidate)]
    diverged = [(d, max(len(a), 1)) for d, a in zip(diffs, reference) if d is not None]
    return {
        "prompts": len(diffs),
        "exact_match_rate": sum(d is None for d in diffs) / len(diffs) if diffs else None,
        "mean_first_diff_char": (sum(d for d, _ in diverged) / len(diverged)) if diverged else None,
        "mean_shared_prefix_fraction": (sum(min(d / n, 1.0) for d, n in diverged) + (len(diffs) - len(diverged)))
        / len(diffs) if diffs else None,
    }


def ask(client, endpoint, headers, prompt, max_tokens):
    response = client.post(endpoint + "/v1/chat/completions", headers=headers, json={
        "model": "qwen-medical", "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0, "seed": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apis", default="v1,v3", help="Comma-separated; the first is the reference")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--budget-usd", type=float, default=0.5)
    args = parser.parse_args()
    apis = args.apis.split(",")
    credentials = load_credentials(ROOT)
    headers = {"Modal-Key": credentials["MODAL_KEY"], "Modal-Secret": credentials["MODAL_SECRET"]}
    runs = [(apis[0], 0), (apis[0], 1)] + [(api, 0) for api in apis[1:]]
    answers, busy, spent = {}, {}, 0.0
    with httpx.Client(timeout=httpx.Timeout(1500, connect=30)) as client:
        for api, attempt in runs:
            if spent >= args.budget_usd:
                print(f"Budget reached; skipping {api}", flush=True)
                continue
            endpoint = resolve_endpoint(credentials, api)
            started = time.monotonic()
            answers[f"{api}#{attempt}"] = [ask(client, endpoint, headers, p, args.max_tokens)
                                           for p in PARITY_PROMPTS]
            # One replica serves sequential requests; the idle tail is charged once per API.
            elapsed = time.monotonic() - started
            busy[api] = busy.get(api, 0.0) + elapsed
            spent = sum(estimated_cost(t, DEPLOYMENTS[a], replicas=1) for a, t in busy.items())
            print(f"{api} run {attempt}: {elapsed:.0f}s, est. ${spent:.2f} so far", flush=True)

    reference = answers.get(f"{apis[0]}#0", [])
    report = {"reference": apis[0], "max_tokens": args.max_tokens, "estimated_cost_usd": round(spent, 3),
              "comparisons": {key: compare(reference, value) for key, value in answers.items()
                              if key != f"{apis[0]}#0"},
              "answers": answers, "prompts": PARITY_PROMPTS}
    out_dir = ROOT / "serving/benchmarks" / f"{time.strftime('%Y%m%d-%H%M%S')}-parity"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "parity.json").write_text(json.dumps(report, indent=2) + "\n")
    for key, result in report["comparisons"].items():
        print(f"{key} vs {apis[0]}#0: exact {result['exact_match_rate']:.0%}, "
              f"shared prefix {result['mean_shared_prefix_fraction']:.0%}", flush=True)
    print(f"Results in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
