#!/usr/bin/env bash
# RAD vs stock vLLM on v2 (two pinned A10Gs); each run measures both configurations.
set -u
cd "$(dirname "$0")/.."
export PYTHONUTF8=1 PYTHONWARNINGS=ignore
log=serving/benchmarks/followups.log
for args in "v2-ab-r1 capacity" "v2-ab-r2 capacity" "v2-ab-r3 capacity" "v2-ab-long-r1 long-context"; do
  set -- $args
  echo "=== $1 (rerun with warm-up retry)" | tee -a "$log"
  .venv/Scripts/python.exe -m serving.benchmark --api-version v2 --label "$1" --scenario "$2" --budget-usd 1.0 2>&1 \
    | grep -vE "Deprecation|set_event_loop" | tail -12 | tee -a "$log"
done
echo "V2 DONE" | tee -a "$log"
