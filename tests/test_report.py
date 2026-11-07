import json

import pytest

from serving import report


def summary(stack, users, tok_s, gpu_seconds, wall_s=10.0, config=None):
    return {"api_version": stack.split("-")[0], "label": stack, "config": config, "users": users,
            "output_tokens_per_s": tok_s, "output_tokens": tok_s * wall_s, "gpu_seconds": gpu_seconds,
            "wall_s": wall_s, "usd_per_gpu_hour": 1.10, "ttft_p50_s": 1.0, "ttft_p99_s": 2.0,
            "tpot_p50_s": 0.07, "tpot_p99_s": 0.08, "failed": 0}


def write_run(tmp_path, name, scenario, summaries):
    run = tmp_path / name
    run.mkdir()
    (run / "summary.json").write_text(json.dumps({"args": {"scenario": scenario}, "summaries": summaries}))
    return str(run / "summary.json")


def test_cost_per_million_tokens_uses_billed_gpu_seconds():
    row = summary("v1", 8, 100.0, gpu_seconds=10.0)
    assert report.usd_per_million_tokens(row) == pytest.approx(10 * 1.10 / 3600 / 1000 * 1e6)


def test_stack_names_split_v2_by_policy():
    assert report.stack_name({"api_version": "v2", "config": {"policy": "rad"}}) == "v2-rad"
    assert report.stack_name({"api_version": "v3", "label": "v3-wide", "config": None}) == "v3-wide"


def test_efficiency_crossover_and_repeats(tmp_path):
    paths = [
        write_run(tmp_path, "a", "capacity", [summary("v1", 8, 100.0, 10.0), summary("v1", 16, 100.0, 10.0)]),
        write_run(tmp_path, "b", "capacity", [summary("v3-wide", 8, 90.0, 20.0),
                                              summary("v3-wide", 16, 190.0, 20.0)]),
        write_run(tmp_path, "c", "capacity", [summary("v3-wide", 16, 210.0, 20.0)]),
    ]
    tables = report.table(report.load(paths))
    rows = tables["capacity"]
    assert rows["v3-wide"][16]["runs"] == 2
    assert rows["v3-wide"][16]["output_tokens_per_s"] == pytest.approx(200.0)
    assert rows["v3-wide"][16]["tok_s_range"] == (190.0, 210.0)
    assert report.efficiency(rows, "v3-wide", 16) == pytest.approx(1.0)  # 200 tok/s on 2 GPUs vs 100 on 1
    assert report.crossover(rows, "v3-wide", "v1") == 16
    text = report.markdown(tables)
    assert "v3-wide > v1 from 16 users" in text
    assert "| v3-wide | 16 | 200.0 (190–210)" in text


def test_older_runs_get_tpot_from_their_request_log(tmp_path):
    old = summary("v1", 4, 50.0, 10.0)
    del old["tpot_p50_s"], old["tpot_p99_s"]
    path = write_run(tmp_path, "old", "light", [old])
    (tmp_path / "old" / "requests.jsonl").write_text(json.dumps(
        {"ok": True, "users": 4, "config": None, "ttft_s": 1.0, "e2e_s": 2.27, "completion_tokens": 128}) + "\n")
    rows = report.table(report.load([path]))["light"]
    assert rows["v1"][4]["tpot_p50_s"] == pytest.approx(0.01)
