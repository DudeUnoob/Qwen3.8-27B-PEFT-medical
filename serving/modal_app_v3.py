"""API v3: data-parallel replicas of the single-GPU Qwen3-8B PEFT server.

Each container holds one 24 GB GPU and a full copy of the model, like v1.
Modal's load balancer spreads requests across up to MAX_REPLICAS containers
and starts the second one only when in-flight requests pass the target.

The per-replica sequence cap is read at deploy time:
    V3_MAX_NUM_SEQS=32 modal deploy --env main serving/modal_app_v3.py
It is baked into the image environment so the container uses the same value.
Shares the adapter and cache volumes with v1 and v2.
"""

import json
import os
from pathlib import Path
import subprocess

import modal

BASE_MODEL = "Qwen/Qwen3-8B"
ADAPTER_PATH = Path("/adapters/medical")
MODEL_NAME = "qwen-medical"
PORT = 8000
MAX_REPLICAS = 2
MAX_NUM_SEQS = int(os.environ.get("V3_MAX_NUM_SEQS", "8"))
# Scale out before a replica is full; it may burst up to MAX_NUM_SEQS meanwhile.
TARGET_INPUTS = max(1, MAX_NUM_SEQS * 3 // 4)

app = modal.App("qwen-peft-medical-vllm-v3")
image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.11.0")
    .entrypoint([])
    .run_commands("ln -sf /usr/bin/python3 /usr/bin/python")
    .env({"V3_MAX_NUM_SEQS": str(MAX_NUM_SEQS)})
)
adapters = modal.Volume.from_name("qwen-peft-adapters")
hf_cache = modal.Volume.from_name("qwen-peft-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("qwen-peft-vllm-cache", create_if_missing=True)


@app.function(
    image=image,
    # One 24 GB GPU per replica. Replicas never talk to each other, so each
    # can fall back to A10G on its own without breaking the others.
    gpu=["L4", "A10G"],
    min_containers=0,
    max_containers=MAX_REPLICAS,
    scaledown_window=120,  # Each idle replica bills its own GPU; scale down sooner than v1.
    timeout=1800,
    volumes={
        "/adapters": adapters,
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
)
@modal.concurrent(max_inputs=MAX_NUM_SEQS, target_inputs=TARGET_INPUTS)
@modal.web_server(PORT, startup_timeout=1200, requires_proxy_auth=True)
def serve():
    print("GPU:", subprocess.check_output([
        "nvidia-smi", "--query-gpu=name", "--format=csv,noheader"
    ], text=True).strip(), f"| max-num-seqs {MAX_NUM_SEQS}", flush=True)
    config = json.loads((ADAPTER_PATH / "adapter_config.json").read_text())
    if config.get("base_model_name_or_path") != BASE_MODEL:
        raise ValueError(f"Expected adapter trained on {BASE_MODEL}")
    if config.get("peft_type") != "LORA" or config.get("r") != 64:
        raise ValueError("Expected the medical notebook's rank-64 LoRA adapter")
    if not any((ADAPTER_PATH / name).is_file() for name in (
        "adapter_model.safetensors", "adapter_model.bin"
    )):
        raise FileNotFoundError("Upload adapter weights to /adapters/medical first")

    subprocess.Popen([
        "vllm", "serve", BASE_MODEL,
        "--host", "0.0.0.0", "--port", str(PORT),
        "--dtype", "bfloat16",
        "--tensor-parallel-size", "1",
        "--max-model-len", "2048",
        "--max-num-seqs", str(MAX_NUM_SEQS),
        "--gpu-memory-utilization", "0.90",
        "--enforce-eager",
        "--enable-lora", "--max-loras", "1", "--max-lora-rank", "64",
        "--lora-modules", f"{MODEL_NAME}={ADAPTER_PATH}",
    ])
