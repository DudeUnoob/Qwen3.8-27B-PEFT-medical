import pytest

from serving import benchmark


def test_cost_estimate_includes_both_gpus_idle_tail_and_dearest_fallback():
    v2 = benchmark.DEPLOYMENTS["v2"]
    # Two GPUs at the A10G price, because the run may land on the fallback.
    assert benchmark.estimated_cost(3600 - v2["idle_tail_s"], v2) == pytest.approx(2.20)


def test_v3_cost_defaults_to_every_replica_until_counts_are_known():
    v3 = benchmark.DEPLOYMENTS["v3"]
    assert benchmark.estimated_cost(3600 - v3["idle_tail_s"], v3) == pytest.approx(2.20)
    assert benchmark.estimated_cost(3600 - v3["idle_tail_s"], v3, replicas=1) == pytest.approx(1.10)


def test_meter_integrates_sampled_container_counts_conservatively():
    meter = benchmark.GpuMeter(benchmark.DEPLOYMENTS["v3"], start=0.0)
    meter.record(0.0, 1)
    meter.record(10.0, 2)   # 0-10 s charged at 2: the larger neighbouring count
    meter.record(20.0, 2)
    meter.record(30.0, 1)   # 20-30 s charged at 2
    assert meter.container_seconds(40.0) == pytest.approx(10 * 2 + 10 * 2 + 10 * 2 + 10 * 1)
    assert meter.gpu_seconds(40.0, since=30.0) == pytest.approx(10)
    assert meter.replica_range(5.0, 25.0) == (2, 2)
    # One container still up, so one idle tail.
    assert meter.cost(40.0) == pytest.approx((70 + 120) * 1.10 / 3600)


def test_meter_charges_unknown_counts_at_the_maximum():
    meter = benchmark.GpuMeter(benchmark.DEPLOYMENTS["v3"], start=0.0)
    assert meter.container_seconds(10.0) == pytest.approx(20)  # no samples yet
    meter.record(10.0, None)
    assert meter.current_containers() == 2
    assert meter.cost(20.0) == pytest.approx((40 + 2 * 120) * 1.10 / 3600)


def test_summary_reports_latency_percentiles_and_throughput():
    records = [
        {"ok": True, "ttft_s": 1.0, "e2e_s": 3.0, "token_gaps_s": [0.1, 0.2], "completion_tokens": 3},
        {"ok": True, "ttft_s": 2.0, "e2e_s": 4.0, "token_gaps_s": [0.3], "completion_tokens": 2},
        {"ok": False, "ttft_s": None, "e2e_s": 1.0, "token_gaps_s": [], "completion_tokens": None},
    ]
    summary = benchmark.summarize(records, wall_s=5.0)
    assert (summary["requests"], summary["failed"]) == (3, 1)
    assert summary["ttft_p99_s"] == 2.0
    assert summary["tbt_p50_s"] == 0.2
    assert summary["output_tokens_per_s"] == 1.0
    assert summary["tpot_p50_s"] == 1.0 and summary["tpot_p99_s"] == 2.0


def test_tpot_ignores_bursty_delivery():
    # Tokens arriving two per network read give zero gaps, but TPOT stays the true average.
    record = {"ttft_s": 1.0, "e2e_s": 11.0, "completion_tokens": 101, "token_gaps_s": [0.0, 0.2] * 50}
    assert benchmark.time_per_output_token(record) == pytest.approx(0.1)
    assert benchmark.time_per_output_token({"ttft_s": 1.0, "e2e_s": 2.0, "completion_tokens": 1}) is None


def test_config_parsing_keeps_policy_as_text():
    assert benchmark.parse_config("policy=rad,max_active=32,chunk=256") == {
        "policy": "rad", "max_active": 32, "chunk": 256}


def test_metrics_parsing_sums_labelled_series_and_names_the_replica():
    text = "\n".join([
        "# HELP vllm:num_preemptions_total Preemptions",
        'vllm:num_preemptions_total{engine="0",model_name="Qwen/Qwen3-8B"} 3.0',
        'vllm:num_preemptions_total{engine="1",model_name="Qwen/Qwen3-8B"} 2.0',
        'vllm:kv_cache_usage_perc{engine="0"} 0.25',
        "process_start_time_seconds 1.7e9",
    ])
    reading = benchmark.replica_metrics(benchmark.parse_metrics(text))
    assert reading == {"replica": 1.7e9, "preemptions": 5.0, "kv_cache_usage": 0.25}
    assert benchmark.replica_metrics({}) == {"replica": None, "preemptions": None, "kv_cache_usage": None}


def test_scenarios_fill_only_unset_options():
    args = benchmark.main(["--api-version", "v3", "--scenario", "long-context", "--users", "8"])
    assert (args.users, args.prompt_chars, args.max_tokens, args.rounds) == ([8], 6000, 256, 1)
    assert benchmark.main(["--scenario", "capacity"]).users == [8, 16, 32, 64]


def test_cli_rejects_prompts_that_overflow_context_and_bad_warm_counts():
    with pytest.raises(SystemExit):
        benchmark.main(["--prompt-chars", "6000", "--max-tokens", "512"])
    with pytest.raises(SystemExit):
        benchmark.main(["--api-version", "v3", "--warm-replicas", "3"])
    assert benchmark.main(["--api-version", "v3", "--warm-replicas", "2"]).warm_replicas == 2
