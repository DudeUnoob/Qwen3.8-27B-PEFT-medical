import pytest

from serving import benchmark


def test_cost_estimate_includes_both_gpus_and_idle_tail():
    v2 = benchmark.DEPLOYMENTS["v2"]
    assert benchmark.estimated_cost(3600 - v2["idle_tail_s"], v2) == pytest.approx(1.60)


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


def test_config_parsing_keeps_policy_as_text():
    assert benchmark.parse_config("policy=rad,max_active=32,chunk=256") == {
        "policy": "rad", "max_active": 32, "chunk": 256}
