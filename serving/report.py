"""Compare benchmark runs across serving stacks. Reads local summary.json files only.

    python -m serving.report serving/benchmarks/2026*/summary.json --out docs/benchmark-tables.md --charts docs/img
"""

import argparse
from collections import defaultdict
import glob
import json
import sys
from pathlib import Path
from statistics import mean

from serving.benchmark import percentile, time_per_output_token

# Fixed slot per stack so a stack keeps its color in every chart (validated default palette, light mode).
STACK_COLORS = {"v1": "#2a78d6", "v2-rad": "#eb6834", "v3-narrow": "#1baf7a",
                "v3-wide": "#eda100", "v2-vllm": "#e87ba4"}
STACK_MARKERS = {"v1": "o", "v2-rad": "s", "v3-narrow": "^", "v3-wide": "D", "v2-vllm": "v"}
STACK_ORDER = list(STACK_COLORS)
METRICS = ["output_tokens_per_s", "ttft_p50_s", "ttft_p99_s", "tpot_p50_s", "tpot_p99_s", "tbt_p99_s"]


def stack_name(summary):
    """A v2 run splits by scheduler policy; other stacks use their benchmark label."""
    label = summary.get("label") or summary["api_version"]
    config = summary.get("config")
    return f"{label}-{config['policy']}" if config and label == "v2" else label


def usd_per_million_tokens(summary):
    tokens = summary.get("output_tokens")
    if not tokens or summary.get("gpu_seconds") is None:
        return None
    return summary["gpu_seconds"] * summary["usd_per_gpu_hour"] / 3600 / tokens * 1e6


def load(paths):
    """Group level summaries by (scenario, stack, users); repeats become a list."""
    groups = defaultdict(list)
    for path in paths:
        data = json.loads(Path(path).read_text())
        scenario = data.get("args", {}).get("scenario", "light")
        backfill = tpot_by_level(Path(path).with_name("requests.jsonl"))
        for summary in data["summaries"]:
            summary = {**summary, "usd_per_mtok": usd_per_million_tokens(summary)}
            if summary.get("tpot_p50_s") is None:
                tpots = backfill.get((json.dumps(summary.get("config"), sort_keys=True), summary["users"]), [])
                summary.update(tpot_p50_s=percentile(tpots, 0.5), tpot_p99_s=percentile(tpots, 0.99))
            groups[(summary.get("scenario", scenario), stack_name(summary), summary["users"])].append(summary)
    return groups


def tpot_by_level(requests_path):
    """Per-request TPOT for runs recorded before the benchmark reported it."""
    levels = defaultdict(list)
    if requests_path.is_file():
        for line in requests_path.read_text().splitlines():
            record = json.loads(line)
            if record.get("ok") and (tpot := time_per_output_token(record)) is not None:
                levels[(json.dumps(record.get("config"), sort_keys=True), record["users"])].append(tpot)
    return levels


def combine(runs):
    """Mean of each metric across repeats, plus the tok/s range when there is more than one run."""
    pick = lambda key: [r[key] for r in runs if r.get(key) is not None]
    row = {key: mean(pick(key)) if pick(key) else None for key in METRICS + ["usd_per_mtok", "gpu_seconds"]}
    rates = pick("output_tokens_per_s")
    row.update(runs=len(runs), failed=sum(r.get("failed", 0) for r in runs),
               tok_s_range=(min(rates), max(rates)) if len(rates) > 1 else None,
               replicas_max=max((r.get("replicas_max") or 1 for r in runs), default=1),
               preemptions=sum(r.get("preemptions") or 0 for r in runs),
               gpus=max(((r.get("gpu_seconds") or 0) / r["wall_s"] for r in runs if r.get("wall_s")),
                        default=None) or None)
    return row


def table(groups):
    """{scenario: {stack: {users: row}}}"""
    out = defaultdict(lambda: defaultdict(dict))
    for (scenario, stack, users), runs in groups.items():
        out[scenario][stack][users] = combine(runs)
    return out


def ordered(stacks):
    return sorted(stacks, key=lambda s: (STACK_ORDER.index(s) if s in STACK_ORDER else len(STACK_ORDER), s))


def efficiency(rows, stack, users, baseline="v1"):
    """Throughput per GPU actually billed, relative to the single-GPU baseline at the same load."""
    row, base = rows.get(stack, {}).get(users), rows.get(baseline, {}).get(users)
    if not row or not base or not row["output_tokens_per_s"] or not base["output_tokens_per_s"] or not row["gpus"]:
        return None
    return row["output_tokens_per_s"] / row["gpus"] / base["output_tokens_per_s"]


def crossover(rows, faster, slower):
    """Smallest user count at which `faster` beats `slower` on tok/s, or None."""
    shared = sorted(set(rows.get(faster, {})) & set(rows.get(slower, {})))
    for users in shared:
        a, b = rows[faster][users]["output_tokens_per_s"], rows[slower][users]["output_tokens_per_s"]
        if a and b and a > b:
            return users
    return None


def fmt(value, digits=2, unit=""):
    return "—" if value is None else f"{value:.{digits}f}{unit}"


def markdown(tables):
    lines = []
    for scenario in sorted(tables):
        rows = tables[scenario]
        lines += [f"### Scenario `{scenario}`", "",
                  "| Stack | Users | tok/s | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | GPUs billed | "
                  "Eff. vs v1 | $/1M tok | Preempt. | Failed | Runs |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for stack in ordered(rows):
            for users in sorted(rows[stack]):
                row = rows[stack][users]
                rate = fmt(row["output_tokens_per_s"], 1)
                if row["tok_s_range"]:
                    rate += f" ({row['tok_s_range'][0]:.0f}–{row['tok_s_range'][1]:.0f})"
                eff = efficiency(rows, stack, users)
                lines.append(
                    f"| {stack} | {users} | {rate} | {fmt(row['ttft_p50_s'])} s | {fmt(row['ttft_p99_s'])} s | "
                    f"{fmt(row['tpot_p50_s'], 3)} s | {fmt(row['tpot_p99_s'], 3)} s | {fmt(row['gpus'], 1)} | "
                    f"{'—' if eff is None else f'{eff:.0%}'} | {fmt(row['usd_per_mtok'])} | "
                    f"{row['preemptions']:.0f} | {row['failed']} | {row['runs']} |")
        pairs = [(a, b) for a in ordered(rows) for b in ordered(rows) if a != b]
        found = [(a, b, crossover(rows, a, b)) for a, b in pairs]
        found = [f"{a} > {b} from {u} users" for a, b, u in found
                 if u is not None and u != min(set(rows[a]) & set(rows[b]))]
        if found:
            lines += ["", "Crossovers (tok/s): " + "; ".join(found) + "."]
        lines.append("")
    return "\n".join(lines)


def charts(tables, out_dir):
    """Small multiples: one panel per scenario, one line per stack, for tok/s and for $ per 1M tokens."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for metric, title, filename in [("output_tokens_per_s", "Output tokens per second", "throughput.png"),
                                    ("usd_per_mtok", "GPU $ per 1M output tokens", "cost.png")]:
        scenarios = sorted(tables)
        fig, axes = plt.subplots(1, len(scenarios), figsize=(4.2 * len(scenarios), 3.6), squeeze=False)
        fig.patch.set_facecolor("#fcfcfb")
        for axis, scenario in zip(axes[0], scenarios):
            axis.set_facecolor("#fcfcfb")
            ends = []
            for stack in ordered(tables[scenario]):
                points = [(u, r[metric]) for u, r in sorted(tables[scenario][stack].items()) if r[metric] is not None]
                if not points:
                    continue
                xs, ys = zip(*points)
                color = STACK_COLORS.get(stack, "#52514e")
                axis.plot(xs, ys, color=color, linewidth=2, solid_capstyle="round", solid_joinstyle="round",
                          marker=STACK_MARKERS.get(stack, "o"), markersize=7,
                          markeredgecolor="#fcfcfb", markeredgewidth=2, label=stack)
                ends.append((stack, xs[-1], ys[-1]))
            # Direct-label line ends only where they cannot collide; the legend names the rest.
            span = max((y for _, _, y in ends), default=0) or 1
            for stack, x, y in ends:
                if all(abs(y - other_y) > 0.06 * span for other, other_x, other_y in ends
                       if other != stack and other_x == x):
                    axis.annotate(stack, (x, y), xytext=(6, 0), textcoords="offset points",
                                  va="center", fontsize=8, color="#52514e")
            axis.set_xscale("log", base=2)
            users = sorted({u for s in tables[scenario].values() for u in s})
            axis.set_xticks(users, [str(u) for u in users])
            axis.set_title(scenario, fontsize=10, color="#0b0b0b", loc="left")
            axis.set_xlabel("Concurrent users", fontsize=8, color="#52514e")
            axis.grid(True, color="#e1e0d9", linewidth=1)
            axis.set_axisbelow(True)
            for side in ("top", "right"):
                axis.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                axis.spines[side].set_color("#e1e0d9")
            axis.tick_params(colors="#52514e", labelsize=8)
            axis.set_ylim(bottom=0)
        handles, labels = {}, {}
        for axis in axes[0]:
            for handle, label in zip(*axis.get_legend_handles_labels()):
                handles.setdefault(label, handle)
        if not handles:  # e.g. cost for runs recorded before GPU-seconds were metered
            plt.close(fig)
            continue
        fig.legend([handles[k] for k in ordered(handles)], ordered(handles), loc="upper right",
                   ncol=len(handles), frameon=False, fontsize=8)
        fig.suptitle(title, x=0.01, ha="left", fontsize=11, color="#0b0b0b")
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        path = out_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="summary.json files or globs")
    parser.add_argument("--out", type=Path, help="Write the markdown tables here instead of printing")
    parser.add_argument("--charts", type=Path, help="Directory for throughput.png and cost.png")
    args = parser.parse_args()
    paths = sorted({p for pattern in args.paths for p in (glob.glob(pattern) or [pattern])})
    tables = table(load(paths))
    text = markdown(tables)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        print(text)
    if args.charts:
        for path in charts(tables, args.charts):
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
