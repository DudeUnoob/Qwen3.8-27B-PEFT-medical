#!/usr/bin/env bash
# Follow-up experiments for the inference write-up, all on pinned A10Gs.
# Each line: label | per-run budget | benchmark arguments. Stops before the running estimate passes CAP.
set -u
cd "$(dirname "$0")/.."
export PYTHONUTF8=1 PYTHONWARNINGS=ignore
CAP=7.50
spent=0.08  # exp-v1-32 capacity run 1, already done
log=serving/benchmarks/followups.log
PY=.venv/Scripts/python.exe

run() {
  local label=$1 budget=$2; shift 2
  if awk -v s="$spent" -v b="$budget" -v c="$CAP" 'BEGIN { exit !(s + b > c) }'; then
    echo "SKIP $label: \$$spent spent, a \$$budget run could pass the \$$CAP cap" | tee -a "$log"; return
  fi
  echo "=== $label (spent so far \$$spent)" | tee -a "$log"
  out=$($PY -m serving.benchmark --label "$label" --budget-usd "$budget" "$@" 2>&1 | grep -vE "Deprecation|set_event_loop")
  echo "$out" | tail -8 | tee -a "$log"
  cost=$(echo "$out" | grep -oE "Estimated cost \\\$[0-9.]+" | grep -oE "[0-9.]+$" | tail -1)
  spent=$(awk -v s="$spent" -v c="${cost:-$budget}" 'BEGIN { printf "%.2f", s + c }')
}

# 1. Is the multi-GPU gain the second GPU or the cap? One GPU at 8 vs 32 in flight.
run exp-v1-32-r2 0.6 --api-version exp-v1-32 --scenario capacity
run exp-v1-32-r3 0.6 --api-version exp-v1-32 --scenario capacity
run exp-v1-8-r1  0.6 --api-version exp-v1-8  --scenario capacity
run exp-v1-8-r2  0.6 --api-version exp-v1-8  --scenario capacity
run exp-v1-8-r3  0.6 --api-version exp-v1-8  --scenario capacity
run exp-v1-32-long-r1 0.6 --api-version exp-v1-32 --scenario long-context
run exp-v1-32-long-r2 0.6 --api-version exp-v1-32 --scenario long-context
run exp-v1-32-long-r3 0.6 --api-version exp-v1-32 --scenario long-context
run exp-v1-8-long-r1  0.8 --api-version exp-v1-8  --scenario long-context

# 2. Did RAD help? Stock vLLM and RAD on the same two pinned A10Gs (each run measures both).
run v2-ab-r1 1.0 --api-version v2 --scenario capacity
run v2-ab-r2 1.0 --api-version v2 --scenario capacity
run v2-ab-r3 1.0 --api-version v2 --scenario capacity
run v2-ab-long-r1 1.0 --api-version v2 --scenario long-context

# 3. What does keeping the adapter separate cost? Merged weights vs LoRA, both at 32 in flight.
run exp-merged-32-r1 0.6 --api-version exp-merged-32 --scenario capacity
run exp-merged-32-r2 0.6 --api-version exp-merged-32 --scenario capacity
run exp-merged-32-r3 0.6 --api-version exp-merged-32 --scenario capacity

# 4. Two-GPU reference on the same pinned hardware: v3-wide, both replicas warm.
run v3-wide-r1 0.8 --api-version v3 --scenario capacity --warm-replicas 2 --replica-inputs 32
run v3-wide-r2 0.8 --api-version v3 --scenario capacity --warm-replicas 2 --replica-inputs 32
run v3-wide-r3 0.8 --api-version v3 --scenario capacity --warm-replicas 2 --replica-inputs 32

# Same answers? Merged vs LoRA (parity asks the first API twice for a noise floor).
echo "=== parity (spent so far \$$spent)" | tee -a "$log"
$PY -m serving.parity --apis exp-v1-32,exp-merged-32 --budget-usd 0.3 2>&1 | grep -vE "Deprecation|set_event_loop" | tail -12 | tee -a "$log"
echo "DONE, estimated total \$$spent" | tee -a "$log"
