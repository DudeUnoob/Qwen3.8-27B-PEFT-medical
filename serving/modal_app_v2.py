"""API v2: Qwen3-8B PEFT inference on two tensor-parallel 24 GB GPUs with the RAD scheduler.

Shares the adapter and cache volumes with v1 (serving/modal_app.py).
"""

import json
from pathlib import Path
import subprocess

import modal

BASE_MODEL = "Qwen/Qwen3-8B"
ADAPTER_PATH = Path("/adapters/medical")
MODEL_NAME = "qwen-medical"
PORT = 8000
GPU_COUNT = 2
RAD_CONFIG_DICT = "qwen-peft-rad-config"
MAX_NUM_SEQS = 128
MAX_NUM_BATCHED_TOKENS = 2048
HERE = Path(__file__).parent

app = modal.App("qwen-peft-medical-vllm-v2")
image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.11.0")
    .entrypoint([])
    .run_commands("ln -sf /usr/bin/python3 /usr/bin/python")
    .pip_install("modal==1.6.3")  # The engine-core subprocess reads modal.Dict.
    .run_commands("python -c \"import pathlib, site; "
                  "pathlib.Path(site.getsitepackages()[0], 'rad.pth').write_text('/opt/rad\\n')\"")
    .env({"RAD_CONFIG_DICT": RAD_CONFIG_DICT})
    .add_local_file(HERE / "rad_policy.py", "/opt/rad/rad_policy.py")
    .add_local_file(HERE / "rad_scheduler.py", "/opt/rad/rad_scheduler.py")
)
adapters = modal.Volume.from_name("qwen-peft-adapters")
hf_cache = modal.Volume.from_name("qwen-peft-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("qwen-peft-vllm-cache", create_if_missing=True)


@app.function(
    image=image,
    # Two 24 GB GPUs. A10G is the fallback when a pair of L4s is unavailable.
    gpu=[f"L4:{GPU_COUNT}", f"A10G:{GPU_COUNT}"],
    min_containers=0,
    max_containers=1,
    scaledown_window=120,  # Two GPUs bill while idle; scale down sooner than v1.
    timeout=1800,
    volumes={
        "/adapters": adapters,
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
)
@modal.concurrent(max_inputs=MAX_NUM_SEQS)
@modal.web_server(PORT, startup_timeout=1200, requires_proxy_auth=True)
def serve():
    gpus = subprocess.check_output([
        "nvidia-smi", "--query-gpu=name", "--format=csv,noheader"
    ], text=True).strip().splitlines()
    print("GPUs:", gpus, flush=True)
    if len(gpus) != GPU_COUNT:
        raise RuntimeError(f"Expected {GPU_COUNT} GPUs for tensor parallelism, got {gpus}")
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
        "--tensor-parallel-size", str(GPU_COUNT),
        "--max-model-len", "2048",
        # Upper bounds; RAD's runtime knobs are clamped to these.
        "--max-num-seqs", str(MAX_NUM_SEQS),
        "--max-num-batched-tokens", str(MAX_NUM_BATCHED_TOKENS),
        "--enable-chunked-prefill",
        "--scheduler-cls", "rad_scheduler.RADScheduler",
        "--gpu-memory-utilization", "0.90",
        "--enforce-eager",
        "--enable-lora", "--max-loras", "1", "--max-lora-rank", "64",
        "--lora-modules", f"{MODEL_NAME}={ADAPTER_PATH}",
    ])
