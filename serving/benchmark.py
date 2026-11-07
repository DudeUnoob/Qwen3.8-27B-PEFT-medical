"""Simulate concurrent users against a serving API version under a spending cap.

Each simulated user sends requests back to back. Results land in
serving/benchmarks/<timestamp>-<label>/ as requests.jsonl and summary.json.
"""

import argparse
import asyncio
import json
from pathlib import Path
import random
import re
import time

import httpx

from serving.chat import resolve_endpoint
from serving.config import load_credentials

ROOT = Path(__file__).resolve().parents[1]
RAD_CONFIG_DICT = "qwen-peft-rad-config"
SCHEDULER_SYNC_S = 3  # The scheduler publishes its state every 2 seconds.
STATS_POLL_S = 2
MAX_MODEL_LEN = 2048
DEFAULT_V2_CONFIGS = [
    {"policy": "vllm", "max_active": 64, "chunk": 2048},
    {"policy": "rad", "max_active": 64, "chunk": 256},
]
# Workspace list prices per GPU-hour (`modal billing rates`, Nov 2026); CPU and memory add a few percent.
GPU_USD_PER_HOUR = {"L4": 0.80, "A10G": 1.10}
DEPLOYMENTS = {
    "v1": {"app": "qwen-peft-medical-vllm", "gpus": 1, "max_replicas": 1,
           "gpu_choices": ["L4", "A10G"], "idle_tail_s": 300},
    "v2": {"app": "qwen-peft-medical-vllm-v2", "gpus": 2, "max_replicas": 1,
           "gpu_choices": ["L4", "A10G"], "idle_tail_s": 120},
    "v3": {"app": "qwen-peft-medical-vllm-v3", "gpus": 1, "max_replicas": 2,
           "gpu_choices": ["L4", "A10G"], "idle_tail_s": 120},
}
# Presets fill in only the options left unset on the command line.
SCENARIOS = {
    "light": {"users": [1, 4, 8], "rounds": 2, "prompt_chars": 0, "max_tokens": 128},
    "capacity": {"users": [8, 16, 32, 64], "rounds": 1, "prompt_chars": 3000, "max_tokens": 128},
    "long-context": {"users": [16, 32, 64], "rounds": 1, "prompt_chars": 6000, "max_tokens": 256},
    "long-decode": {"users": [1, 16], "rounds": 1, "prompt_chars": 0, "max_tokens": 512},
    "burst": {"users": [24], "rounds": 1, "prompt_chars": 3000, "max_tokens": 128},
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


def gpu_price(deployment):
    """The dearest GPU the deployment may land on, so the budget guard never underestimates."""
    return max(GPU_USD_PER_HOUR[gpu] for gpu in deployment["gpu_choices"])


def estimated_cost(elapsed_s, deployment, replicas=None):
    """Cost of `replicas` containers (default: the maximum) for elapsed time plus their idle tail."""
    replicas = deployment["max_replicas"] if replicas is None else replicas
    return (elapsed_s + deployment["idle_tail_s"]) * replicas * deployment["gpus"] * gpu_price(deployment) / 3600


class GpuMeter:
    """Integrates billed GPU-seconds from sampled container counts.

    Between samples the larger neighbouring count is used, and before the first
    sample or after a failed poll the maximum replica count is assumed, so the
    estimate errs high. The idle tail is charged for every container still up.
    """

    def __init__(self, deployment, start):
        self.deployment = deployment
        self.start = start
        self.samples = []  # (monotonic time, containers or None)

    def record(self, at, containers):
        self.samples.append((at, containers))

    def _count(self, containers):
        return self.deployment["max_replicas"] if containers is None else containers

    def container_seconds(self, until, since=None):
        since = self.start if since is None else since
        points = [(t, self._count(c)) for t, c in self.samples if since < t < until]
        before = [self._count(c) for t, c in self.samples if t <= since]
        level = before[-1] if before else self.deployment["max_replicas"]
        total, last_t = 0.0, since
        for t, count in points:
            total += (t - last_t) * max(level, count)
            level, last_t = count, t
        return total + (until - last_t) * level

    def gpu_seconds(self, until, since=None):
        return self.container_seconds(until, since) * self.deployment["gpus"]

    def current_containers(self):
        return self._count(self.samples[-1][1]) if self.samples else self.deployment["max_replicas"]

    def cost(self, now):
        tail = self.deployment["idle_tail_s"] * self.current_containers()
        return (self.container_seconds(now) + tail) * self.deployment["gpus"] * gpu_price(self.deployment) / 3600

    def replica_range(self, since, until):
        counts = [self._count(c) for t, c in self.samples if since <= t <= until]
        return (min(counts), max(counts)) if counts else (None, None)


def parse_config(text):
    config = {}
    for item in text.split(","):
        key, _, value = item.partition("=")
        config[key.strip()] = value.strip() if key.strip() == "policy" else int(value)
    return config


METRIC_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eE]+|NaN|[+-]Inf)")


def parse_metrics(text):
    """Sum Prometheus samples by metric name, dropping labels."""
    values = {}
    for line in text.splitlines():
        match = METRIC_LINE.match(line)
        if match:
            try:
                values[match[1]] = values.get(match[1], 0.0) + float(match[3])
            except ValueError:
                continue
    return values


def replica_metrics(values):
    """The vLLM signals the benchmark keeps, keyed so replicas can be told apart."""
    pick = lambda *names: next((values[n] for n in names if n in values), None)
    return {
        "replica": pick("process_start_time_seconds"),
        "preemptions": pick("vllm:num_preemptions_total", "vllm:num_preemptions"),
        "kv_cache_usage": pick("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
    }


def time_per_output_token(record):
    """Mean decode time per token after the first. Unlike single gaps, it is not
    distorted when the proxy delivers several tokens in one network read."""
    tokens = record.get("completion_tokens") or 0
    if tokens < 2 or record.get("ttft_s") is None or record.get("e2e_s") is None:
        return None
    return (record["e2e_s"] - record["ttft_s"]) / (tokens - 1)


def summarize(records, wall_s):
    ok = [r for r in records if r["ok"]]
    gaps = [gap for r in ok for gap in r["token_gaps_s"]]
    tpots = [t for t in map(time_per_output_token, ok) if t is not None]
    tokens = sum(r["completion_tokens"] or 0 for r in ok)
    return {
        "requests": len(records), "failed": len(records) - len(ok),
        "ttft_p50_s": percentile([r["ttft_s"] for r in ok], 0.5),
        "ttft_p99_s": percentile([r["ttft_s"] for r in ok], 0.99),
        "tbt_p50_s": percentile(gaps, 0.5), "tbt_p99_s": percentile(gaps, 0.99),
        "tpot_p50_s": percentile(tpots, 0.5), "tpot_p99_s": percentile(tpots, 0.99),
        "e2e_p50_s": percentile([r["e2e_s"] for r in ok], 0.5),
        "e2e_p99_s": percentile([r["e2e_s"] for r in ok], 0.99),
        "output_tokens": tokens,
        "prompt_tokens_max": max((r.get("prompt_tokens") or 0 for r in ok), default=None),
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


def fits_context(prompt_chars, max_tokens):
    """Conservative check: about 4 characters per token plus the chat template and question."""
    return prompt_chars // 4 + 96 + max_tokens <= MAX_MODEL_LEN


async def stream_request(client, endpoint, headers, request_id, args, max_tokens=None, clock_zero=None):
    max_tokens = args.max_tokens if max_tokens is None else max_tokens
    prompt = build_prompt(request_id, args)
    payload = {
        "model": "qwen-medical", "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0, "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if args.fixed_output:
        payload.update(ignore_eos=True, min_tokens=max_tokens)
    record = {"id": request_id, "ok": False, "ttft_s": None, "e2e_s": None,
              "token_gaps_s": [], "completion_tokens": None, "prompt_tokens": None, "error": None}
    started = last = time.monotonic()
    if clock_zero is not None:
        record["start_s"] = started - clock_zero
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


def serve_function(deployment):
    import modal

    return modal.Function.from_name(deployment["app"], "serve", environment_name="main")


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


async def poll_containers(function, meter, timeline, stop):
    """Sample the live container count; a failed poll is recorded as unknown (charged at the maximum)."""
    while not stop.is_set():
        now = time.monotonic()
        try:
            stats = await function.get_current_stats.aio()
            meter.record(now, stats.num_total_runners)
            timeline.append({"t_s": round(now - meter.start, 1), "containers": stats.num_total_runners,
                             "running_inputs": stats.num_running_inputs, "backlog": stats.backlog})
        except Exception as error:  # noqa: BLE001 - any failure must fall back to the safe estimate
            meter.record(now, None)
            timeline.append({"t_s": round(now - meter.start, 1), "containers": None, "error": str(error)[:200]})
        try:
            await asyncio.wait_for(stop.wait(), STATS_POLL_S)
        except asyncio.TimeoutError:
            pass


async def sample_metrics(client, endpoint, headers, samples):
    """Fetch /metrics a few times. Each GET may reach a different replica; keep one reading per replica."""
    by_replica = {}
    for _ in range(samples):
        try:
            response = await client.get(endpoint + "/metrics", headers=headers, timeout=30)
            if response.status_code == 200:
                reading = replica_metrics(parse_metrics(response.text))
                by_replica[reading["replica"]] = reading
        except httpx.HTTPError:
            continue
    return by_replica


async def wait_for_replicas(client, endpoint, headers, function, args, count, timeout_s=900):
    """Hold the warm count at `count` and wait until that many replicas answer.

    Waves of requests that overflow one replica force Modal to route to the
    others, and /metrics probes in the same wave reveal each replica's process.
    """
    deadline = time.monotonic() + timeout_s
    seen = set()
    while time.monotonic() < deadline:
        stats = await function.get_current_stats.aio()
        if stats.num_total_runners >= count:
            wave = [stream_request(client, endpoint, headers, f"warm-{i}", args, max_tokens=32)
                    for i in range(args.replica_inputs * count)]
            probes = [sample_metrics(client, endpoint, headers, 1) for _ in range(count * 2)]
            results = await asyncio.gather(*wave, *probes)
            readings = [reading for found in results[len(wave):] for reading in found.values()]
            if readings and all(reading["replica"] is None for reading in readings):
                # vLLM did not expose a process start time; wait out one cold start instead.
                await asyncio.sleep(90)
                return count
            seen.update(reading["replica"] for reading in readings if reading["replica"] is not None)
            if len(seen) >= count:
                return len(seen)
        await asyncio.sleep(5)
    raise RuntimeError(f"Fewer than {count} replicas were serving after {timeout_s}s")


async def run(args):
    deployment = DEPLOYMENTS[args.api_version]
    credentials = load_credentials(ROOT)
    endpoint = resolve_endpoint(credentials, args.api_version)
    headers = {"Modal-Key": credentials["MODAL_KEY"], "Modal-Secret": credentials["MODAL_SECRET"]}
    if args.api_version != "v2" and args.config:
        raise SystemExit("--config only applies to v2, whose scheduler reads runtime knobs")
    if args.warm_replicas and deployment["max_replicas"] < 2:
        raise SystemExit("--warm-replicas only applies to v3, which has more than one replica")
    configs = ([parse_config(text) for text in args.config] or DEFAULT_V2_CONFIGS
               if args.api_version == "v2" else [None])
    label = args.label or args.api_version
    out_dir = ROOT / "serving/benchmarks" / f"{time.strftime('%Y%m%d-%H%M%S')}-{label}-{args.scenario}"
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    run_started = time.monotonic()
    meter = GpuMeter(deployment, run_started)
    spent = lambda: meter.cost(time.monotonic())
    summaries, timeline, next_id, stopped = [], [], 0, False
    function = serve_function(deployment) if deployment["max_replicas"] > 1 else None
    stop_polling = asyncio.Event()
    poller = (asyncio.create_task(poll_containers(function, meter, timeline, stop_polling))
              if function else None)
    print(f"API {label}: up to {deployment['max_replicas']}x {deployment['gpus']} GPU, "
          f"budget ${args.budget_usd:.2f} at ${gpu_price(deployment):.2f}/GPU-hour "
          f"(includes a {deployment['idle_tail_s']}s idle tail per container)", flush=True)

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(args.request_timeout, connect=30)) as client:
            if args.warm_replicas:
                await function.update_autoscaler.aio(min_containers=args.warm_replicas)
                print(f"Holding {args.warm_replicas} replica(s) warm until the run ends", flush=True)
            print("Warming up; a cold start can take several minutes...", flush=True)
            async with httpx.AsyncClient(timeout=httpx.Timeout(1500, connect=30)) as warm_client:
                warm = await stream_request(warm_client, endpoint, headers, "warmup", args)
                if not warm["ok"]:
                    raise SystemExit(f"Warm-up failed: {warm['error']}")
                if args.warm_replicas and args.warm_replicas > 1:
                    found = await wait_for_replicas(warm_client, endpoint, headers, function, args,
                                                    args.warm_replicas)
                    print(f"{found} replicas serving", flush=True)
            print(f"Warm after {time.monotonic() - run_started:.0f}s, estimated ${spent():.2f} so far", flush=True)
            metrics_before = await sample_metrics(client, endpoint, headers, args.metrics_samples)

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
                        level_started = time.monotonic()

                        async def user_loop(user):
                            nonlocal next_id
                            for _ in range(args.rounds):
                                if spent() >= args.budget_usd:
                                    return
                                next_id += 1
                                record = await stream_request(client, endpoint, headers, next_id, args,
                                                              clock_zero=level_started)
                                record.update(user=user, users=users, config=config)
                                records.append(record)
                                log.write(json.dumps(record) + "\n")

                        await asyncio.gather(*(user_loop(user) for user in range(users)))
                        level_ended = time.monotonic()
                        low, high = meter.replica_range(level_started, level_ended)
                        summary = {"api_version": args.api_version, "label": label, "scenario": args.scenario,
                                   "config": config, "users": users,
                                   **summarize(records, level_ended - level_started),
                                   "replicas_min": low if function else 1,
                                   "replicas_max": high if function else 1,
                                   "gpu_seconds": meter.gpu_seconds(level_ended, level_started),
                                   "usd_per_gpu_hour": gpu_price(deployment)}
                        metrics_after = await sample_metrics(client, endpoint, headers, args.metrics_samples)
                        summary["preemptions"] = sum(
                            (reading["preemptions"] or 0) - (metrics_before.get(key, {}).get("preemptions") or 0)
                            for key, reading in metrics_after.items())
                        summary["metrics_replicas_seen"] = len(metrics_after)
                        metrics_before = {**metrics_before, **metrics_after}
                        if steps is not None:
                            after = await scheduler_steps()
                            summary["scheduler_steps"] = {k: after[k] - steps.get(k, 0) for k in after}
                            steps = after
                        summaries.append(summary)
                        print(f"  {users:>3} users: {summary['requests']} req, {summary['failed']} failed, "
                              f"TTFT p50 {fmt(summary['ttft_p50_s'])} p99 {fmt(summary['ttft_p99_s'])}, "
                              f"TPOT p50 {fmt(summary['tpot_p50_s'], 's', 3)} p99 {fmt(summary['tpot_p99_s'], 's', 3)}, "
                              f"{fmt(summary['output_tokens_per_s'], 'tok/s')}, "
                              f"replicas {summary['replicas_min']}-{summary['replicas_max']}, "
                              f"preemptions {summary['preemptions']:.0f} | est. ${spent():.2f}", flush=True)
                    if stopped:
                        break
    finally:
        if args.warm_replicas:
            await function.update_autoscaler.aio(min_containers=0)
            print("Released warm replicas (min_containers=0)", flush=True)
        if poller:
            stop_polling.set()
            await poller

    result = {"args": vars(args), "label": label, "deployment": deployment, "budget_stopped": stopped,
              "estimated_cost_usd": round(spent(), 3),
              "run_gpu_seconds": round(meter.gpu_seconds(time.monotonic()), 1),
              "summaries": summaries, "container_timeline": timeline}
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    if stopped:
        print("Stopped launching requests at the budget cap.", flush=True)
    print(f"\nEstimated cost ${result['estimated_cost_usd']:.2f}; results in {out_dir}", flush=True)


def fmt(value, unit="s", digits=2):
    return "n/a" if value is None else f"{value:.{digits}f}{unit}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-version", default="v2", choices=sorted(DEPLOYMENTS))
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="light",
                        help="Preset users, rounds, prompt size, and output length; explicit options win")
    parser.add_argument("--users", default=None,
                        type=lambda text: [int(n) for n in text.split(",")],
                        help="Comma-separated concurrent user counts")
    parser.add_argument("--rounds", type=int, default=None, help="Requests per user at each level")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--prompt-chars", type=int, default=None,
                        help="Extra characters of unique chart text appended to each prompt")
    parser.add_argument("--fixed-output", action=argparse.BooleanOptionalAction, default=True,
                        help="Generate exactly --max-tokens per request for comparable runs")
    parser.add_argument("--config", action="append", default=[],
                        help="v2 scheduler knobs, e.g. policy=rad,max_active=64,chunk=256; repeatable. "
                             "Defaults to stock vLLM vs RAD at 64 active requests")
    parser.add_argument("--warm-replicas", type=int, default=0,
                        help="v3: hold this many replicas warm for the run, then release them")
    parser.add_argument("--replica-inputs", type=int, default=8,
                        help="v3: per-replica concurrency (V3_MAX_NUM_SEQS of the deploy), used to overflow "
                             "one replica while waiting for warm replicas")
    parser.add_argument("--label", default=None,
                        help="Name for this stack in results, e.g. v3-wide; defaults to the API version")
    parser.add_argument("--metrics-samples", type=int, default=4,
                        help="GETs of vLLM /metrics after each level, to reach each replica")
    parser.add_argument("--budget-usd", type=float, default=2.0)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    for key, value in SCENARIOS[args.scenario].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.max_tokens < 1 or args.rounds < 1 or min(args.users) < 1 or max(args.users) > 128:
        parser.error("--max-tokens and --rounds must be positive; --users must be between 1 and 128")
    if not fits_context(args.prompt_chars, args.max_tokens):
        parser.error(f"--prompt-chars {args.prompt_chars} plus --max-tokens {args.max_tokens} "
                     f"may exceed the {MAX_MODEL_LEN}-token context")
    if args.warm_replicas < 0 or args.warm_replicas > DEPLOYMENTS[args.api_version]["max_replicas"]:
        parser.error("--warm-replicas must be between 0 and the deployment's replica count")
    return args


if __name__ == "__main__":
    asyncio.run(run(main()))
