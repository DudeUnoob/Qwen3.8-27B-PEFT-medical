"""Single-GPU Qwen3-8B PEFT inference. Upload the adapter before deploying."""

import json
from pathlib import Path
import subprocess

import modal

BASE_MODEL = "Qwen/Qwen3-8B"
ADAPTER_PATH = Path("/adapters/medical")
MODEL_NAME = "qwen-medical"
PORT = 8000

app = modal.App("qwen-peft-medical-vllm")
image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.11.0")
    .entrypoint([])
    .run_commands("ln -sf /usr/bin/python3 /usr/bin/python")
)
adapters = modal.Volume.from_name("qwen-peft-adapters")
hf_cache = modal.Volume.from_name("qwen-peft-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("qwen-peft-vllm-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu=["L4", "A10G"],  # One 24 GB GPU; allow a fallback when L4 is unavailable.
    min_containers=0,
    max_containers=1,
    scaledown_window=300,
    timeout=1800,
    volumes={
        "/adapters": adapters,
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
)
@modal.concurrent(max_inputs=8)
@modal.web_server(PORT, startup_timeout=1200, requires_proxy_auth=True)
def serve():
    print("GPU:", subprocess.check_output([
        "nvidia-smi", "--query-gpu=name", "--format=csv,noheader"
    ], text=True).strip(), flush=True)
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
        "--max-num-seqs", "8",
        "--gpu-memory-utilization", "0.90",
        "--enforce-eager",
        "--enable-lora", "--max-loras", "1", "--max-lora-rank", "64",
        "--lora-modules", f"{MODEL_NAME}={ADAPTER_PATH}",
    ])
