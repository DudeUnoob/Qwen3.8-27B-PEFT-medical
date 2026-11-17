"""Experiment server: one 24 GB GPU, like v1, with the knobs the follow-up questions need.

Read at deploy time and baked into the image, so the container and the decorator agree:
    EXP_LABEL        part of the app name, so experiments never replace the live v1 endpoint
    EXP_MAX_NUM_SEQS sequences in flight (v1 uses 8)
    EXP_WEIGHTS      "lora": base model + adapter, as v1 serves it
                     "merged": the adapter folded into the weights (serving/merge_adapter.py)
    EXP_GPU          one GPU type, pinned, so runs are comparable (no fallback)

    EXP_LABEL=v1-32 EXP_MAX_NUM_SEQS=32 modal deploy --env main serving/modal_app_exp.py
    EXP_LABEL=merged-32 EXP_MAX_NUM_SEQS=32 EXP_WEIGHTS=merged modal deploy --env main serving/modal_app_exp.py

Either way the served model is called "qwen-medical", so the benchmark and the evals run unchanged.
"""

import json
import os
from pathlib import Path
import subprocess

import modal

BASE_MODEL = "Qwen/Qwen3-8B"
ADAPTER_PATH = Path("/adapters/medical")
MERGED_PATH = Path("/merged/medical")
MODEL_NAME = "qwen-medical"
PORT = 8000

LABEL = os.environ.get("EXP_LABEL", "v1-32")
MAX_NUM_SEQS = int(os.environ.get("EXP_MAX_NUM_SEQS", "32"))
WEIGHTS = os.environ.get("EXP_WEIGHTS", "lora")
GPU = os.environ.get("EXP_GPU", "A10G")
if WEIGHTS not in ("lora", "merged"):
    raise ValueError("EXP_WEIGHTS must be lora or merged")

app = modal.App(f"qwen-peft-medical-exp-{LABEL}")
image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.11.0")
    .entrypoint([])
    .run_commands("ln -sf /usr/bin/python3 /usr/bin/python")
    .env({"EXP_LABEL": LABEL, "EXP_MAX_NUM_SEQS": str(MAX_NUM_SEQS), "EXP_WEIGHTS": WEIGHTS, "EXP_GPU": GPU})
)
adapters = modal.Volume.from_name("qwen-peft-adapters")
merged = modal.Volume.from_name("qwen-peft-merged", create_if_missing=True)
hf_cache = modal.Volume.from_name("qwen-peft-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("qwen-peft-vllm-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU,
    min_containers=0,
    max_containers=1,
    scaledown_window=120,  # an experiment endpoint; no need for v1's five-minute idle window
    timeout=1800,
    volumes={
        "/adapters": adapters,
        "/merged": merged,
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
)
@modal.concurrent(max_inputs=MAX_NUM_SEQS)
@modal.web_server(PORT, startup_timeout=1200, requires_proxy_auth=True)
def serve():
    print("GPU:", subprocess.check_output([
        "nvidia-smi", "--query-gpu=name", "--format=csv,noheader"
    ], text=True).strip(), f"| {LABEL} | max-num-seqs {MAX_NUM_SEQS} | {WEIGHTS}", flush=True)

    common = [
        "--host", "0.0.0.0", "--port", str(PORT),
        "--dtype", "bfloat16",
        "--tensor-parallel-size", "1",
        "--max-model-len", "2048",
        "--max-num-seqs", str(MAX_NUM_SEQS),
        "--gpu-memory-utilization", "0.90",
        "--enforce-eager",
    ]
    if WEIGHTS == "merged":
        if not (MERGED_PATH / "config.json").is_file():
            raise FileNotFoundError("Run serving/merge_adapter.py first to write /merged/medical")
        subprocess.Popen(["vllm", "serve", str(MERGED_PATH), "--served-model-name", MODEL_NAME, *common])
        return

    config = json.loads((ADAPTER_PATH / "adapter_config.json").read_text())
    if config.get("base_model_name_or_path") != BASE_MODEL or config.get("r") != 64:
        raise ValueError("Expected the medical notebook's rank-64 LoRA adapter on Qwen3-8B")
    subprocess.Popen([
        "vllm", "serve", BASE_MODEL, *common,
        "--enable-lora", "--max-loras", "1", "--max-lora-rank", "64",
        "--lora-modules", f"{MODEL_NAME}={ADAPTER_PATH}",
    ])
