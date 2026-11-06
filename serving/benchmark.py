"""Simulate concurrent users against a serving API version under a spending cap.

Each simulated user sends requests back to back. Results land in
serving/benchmarks/<timestamp>-<api>/ as requests.jsonl and summary.json.
"""

import argparse
import asyncio
import json
from pathlib import Path
import random
import time

import httpx

from serving.chat import resolve_endpoint
from serving.config import load_credentials

ROOT = Path(__file__).resolve().parents[1]
RAD_CONFIG_DICT = "qwen-peft-rad-config"
SCHEDULER_SYNC_S = 3  # The scheduler publishes its state every 2 seconds.
DEFAULT_V2_CONFIGS = [
    {"policy": "vllm", "max_active": 64, "chunk": 2048},
    {"policy": "rad", "max_active": 64, "chunk": 256},
]
# Modal list prices per GPU-hour; CPU and memory add a few percent.
GPU_USD_PER_HOUR = {"L4": 0.80}
DEPLOYMENTS = {
    "v1": {"gpus": 1, "gpu": "L4", "idle_tail_s": 300},
    "v2": {"gpus": 2, "gpu": "L4", "idle_tail_s": 120},
}
PROMPTS = [
    "Explain what an antibody is in two sentences.",
    "What is the difference between a virus and a bacterium?",
    "Summarize the first-line management of uncomplicated hypertension in adults.",
    "A 58-year-old with type 2 diabetes reports numb feet and burning pain at night. "
    "Outline the likely diagnosis, the examination you would perform, and initial management.",
    "List common side effects of metformin and how to counsel a patient about them.",
    "Describe how the kidneys regulate blood pressure through the renin-angiotensin-aldosterone system.",
    "A 24-year-old presents with fever, neck stiffness, and photophobia for one day. "
    "Walk through the differential diagnosis, urgent investigations, and empirical treatment.",
    "What are the warning signs of sepsis that should prompt urgent care?",
]


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))]


def estimated_cost(elapsed_s, deployment):
    rate = GPU_USD_PER_HOUR[deployment["gpu"]] * deployment["gpus"]
    return (elapsed_s + deployment["idle_tail_s"]) * rate / 3600


def parse_config(text):
    config = {}
    for item in text.split(","):
        key, _, value = item.partition("=")
        config[key.strip()] = value.strip() if key.strip() == "policy" else int(value)
    return config


def summarize(records, wall_s):
    ok = [r for r in records if r["ok"]]
    gaps = [gap for r in ok for gap in r["token_gaps_s"]]
    tokens = sum(r["completion_tokens"] or 0 for r in ok)
    return {
        "requests": len(records), "failed": len(records) - len(ok),
        "ttft_p50_s": percentile([r["ttft_s"] for r in ok], 0.5),
        "ttft_p99_s": percentile([r["ttft_s"] for r in ok], 0.99),
        "tbt_p50_s": percentile(gaps, 0.5), "tbt_p99_s": percentile(gaps, 0.99),
        "e2e_p50_s": percentile([r["e2e_s"] for r in ok], 0.5),
        "e2e_p99_s": percentile([r["e2e_s"] for r in ok], 0.99),
        "output_tokens_per_s": tokens / wall_s if wall_s else None,
        "requests_per_s": len(ok) / wall_s if wall_s else None,
        "wall_s": wall_s,
    }


def build_prompt(request_id, args):
    """Unique text per request so prefix caching cannot skip prefill."""
    prompt = f"[case {request_id}] {random.choice(PROMPTS)}"
    if args.prompt_chars <= 0:
        return prompt
    sentence = f"Record {request_id}: symptoms are unchanged from the prior note. "
    copies = args.prompt_chars // len(sentence) + 1
    return f"{prompt}\n{(sentence * copies)[:args.prompt_chars]}"


async def stream_request(client, endpoint, headers, request_id, args):
    prompt = build_prompt(request_id, args)
    payload = {
        "model": "qwen-medical", "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens, "temperature": 0, "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if args.fixed_output:
        payload.update(ignore_eos=True, min_tokens=args.max_tokens)
    record = {"id": request_id, "ok": False, "ttft_s": None, "e2e_s": None,
              "token_gaps_s": [], "completion_tokens": None, "prompt_tokens": None, "error": None}
    started = last = time.monotonic()
    try:
        async with client.stream("POST", endpoint + "/v1/chat/completions",
                                 headers=headers, json=payload) as response:
            if response.status_code != 200:
                record["error"] = f"HTTP {response.status_code}: {(await response.aread())[:300]!r}"
                return record
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                if line[6:] == "[DONE]":
                    record["ok"] = record["ttft_s"] is not None
                    break
                chunk = json.loads(line[6:])
                if usage := chunk.get("usage"):
                    record["completion_tokens"] = usage["completion_tokens"]
                    record["prompt_tokens"] = usage["prompt_tokens"]
                if any(c.get("delta", {}).get("content") for c in chunk.get("choices", [])):
                    now = time.monotonic()
                    if record["ttft_s"] is None:
                        record["ttft_s"] = now - started
                    else:
                        record["token_gaps_s"].append(now - last)
                    last = now
    except (httpx.HTTPError, ValueError) as error:
        record["error"] = f"{type(error).__name__}: {error}"
    record["e2e_s"] = time.monotonic() - started
    if not record["ok"] and not record["error"]:
        record["error"] = "Stream ended before [DONE]"
    return record


def config_store():
    import modal

    return modal.Dict.from_name(RAD_CONFIG_DICT, environment_name="main", create_if_missing=True)


async def scheduler_steps():
    await asyncio.sleep(SCHEDULER_SYNC_S)
    return (await config_store().get.aio("applied"))["steps"]


async def apply_config(config, version, timeout_s=60):
    store = config_store()
    await store.put.aio("requested", {**config, "version": version})
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        applied = await store.get.aio("applied")
        if applied and applied.get("version") == version:
            return applied
        await asyncio.sleep(1)
    raise RuntimeError(f"The v2 scheduler did not apply {config} within {timeout_s}s")


async def run(args):
    deployment = DEPLOYMENTS[args.api_version]
    credentials = load_credentials(ROOT)
    endpoint = resolve_endpoint(credentials, args.api_version)
    headers = {"Modal-Key": credentials["MODAL_KEY"], "Modal-Secret": credentials["MODAL_SECRET"]}
    if args.api_version == "v1" and args.config:
        raise SystemExit("--config only applies to v2, whose scheduler reads runtime knobs")
    configs = ([parse_config(text) for text in args.config] or DEFAULT_V2_CONFIGS
               if args.api_version == "v2" else [None])
    out_dir = ROOT / "serving/benchmarks" / f"{time.strftime('%Y%m%d-%H%M%S')}-{args.api_version}"
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    run_started = time.monotonic()
    spent = lambda: estimated_cost(time.monotonic() - run_started, deployment)
    summaries, next_id, stopped = [], 0, False
    print(f"API {args.api_version}: {deployment['gpus']}x {deployment['gpu']}, budget ${args.budget_usd:.2f} "
          f"(includes a {deployment['idle_tail_s']}s idle tail after the run)", flush=True)

    async with httpx.AsyncClient(timeout=httpx.Timeout(args.request_timeout, connect=30)) as client:
        print("Warming up; a cold start can take several minutes...", flush=True)
        warm_client = httpx.AsyncClient(timeout=httpx.Timeout(1500, connect=30))
        async with warm_client:
            warm = await stream_request(warm_client, endpoint, headers, "warmup", args)
        if not warm["ok"]:
            raise SystemExit(f"Warm-up failed: {warm['error']}")
        print(f"Warm after {time.monotonic() - run_started:.0f}s, estimated ${spent():.2f} so far", flush=True)

        with (out_dir / "requests.jsonl").open("w") as log:
            for config_index, config in enumerate(configs):
                steps = None
                if config is not None:
                    steps = (await apply_config(config, f"{out_dir.name}-{config_index}"))["steps"]
                    print(f"\nScheduler config: {config}", flush=True)
                for users in args.users:
                    if spent() >= args.budget_usd:
                        stopped = True
                        break
                    records = []

                    async def user_loop(user):
                        nonlocal next_id
                        for _ in range(args.rounds):
                            if spent() >= args.budget_usd:
                                return
                            next_id += 1
                            record = await stream_request(client, endpoint, headers, next_id, args)
                            record.update(user=user, users=users, config=config)
                            records.append(record)
                            log.write(json.dumps(record) + "\n")

                    level_started = time.monotonic()
                    await asyncio.gather(*(user_loop(user) for user in range(users)))
                    summary = {"api_version": args.api_version, "config": config, "users": users,
                               **summarize(records, time.monotonic() - level_started)}
                    if steps is not None:
                        after = await scheduler_steps()
                        summary["scheduler_steps"] = {k: after[k] - steps.get(k, 0) for k in after}
                        steps = after
                    summaries.append(summary)
                    print(f"  {users:>3} users: {summary['requests']} req, {summary['failed']} failed, "
                          f"TTFT p50 {fmt(summary['ttft_p50_s'])} p99 {fmt(summary['ttft_p99_s'])}, "
                          f"TBT p99 {fmt(summary['tbt_p99_s'])}, "
                          f"{fmt(summary['output_tokens_per_s'], 'tok/s')} | est. ${spent():.2f}", flush=True)
                if stopped:
                    break

    result = {"args": vars(args), "deployment": deployment, "budget_stopped": stopped,
              "estimated_cost_usd": round(spent(), 3), "summaries": summaries}
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    if stopped:
        print("Stopped launching requests at the budget cap.", flush=True)
    print(f"\nEstimated cost ${result['estimated_cost_usd']:.2f}; results in {out_dir}", flush=True)


def fmt(value, unit="s"):
    return "n/a" if value is None else f"{value:.2f}{unit}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-version", default="v2", choices=sorted(DEPLOYMENTS))
    parser.add_argument("--users", default="1,4,16",
                        type=lambda text: [int(n) for n in text.split(",")],
                        help="Comma-separated concurrent user counts")
    parser.add_argument("--rounds", type=int, default=2, help="Requests per user at each level")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--fixed-output", action=argparse.BooleanOptionalAction, default=True,
                        help="Generate exactly --max-tokens per request for comparable runs")
    parser.add_argument("--config", action="append", default=[],
                        help="v2 scheduler knobs, e.g. policy=rad,max_active=64,chunk=256; repeatable. "
                             "Defaults to stock vLLM vs RAD at 64 active requests")
    parser.add_argument("--budget-usd", type=float, default=2.0)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenario", choices=("light", "capacity"), default="light",
                        help="capacity uses 8, 16, and 32 users with long prompts, above v1's 8-sequence cap")
    parser.add_argument("--prompt-chars", type=int, default=None,
                        help="Extra characters of unique chart text appended to each prompt")
    args = parser.parse_args()
    if args.scenario == "capacity":
        if args.users == [1, 4, 16]:
            args.users = [8, 16, 32]
        if args.prompt_chars is None:
            args.prompt_chars = 3000
        if args.rounds == 2:
            args.rounds = 1
    if args.prompt_chars is None:
        args.prompt_chars = 0
    if args.max_tokens < 1 or args.rounds < 1 or min(args.users) < 1 or max(args.users) > 128:
        parser.error("--max-tokens and --rounds must be positive; --users must be between 1 and 128")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
