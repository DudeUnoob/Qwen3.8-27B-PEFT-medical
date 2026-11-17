"""Fold the rank-64 medical adapter into Qwen3-8B and save the result for vLLM.

The merged model answers like base + adapter, but vLLM serves it as a plain model, with no
LoRA kernels on the hot path. serving/modal_app_exp.py serves it with EXP_WEIGHTS=merged.

    modal run --env main serving/merge_adapter.py

Writes bfloat16 safetensors plus the tokenizer to the qwen-peft-merged volume at /merged/medical.
"""

from pathlib import Path

import modal

BASE_MODEL = "Qwen/Qwen3-8B"
ADAPTER_PATH = Path("/adapters/medical")
OUT_PATH = Path("/merged/medical")

app = modal.App("qwen-peft-medical-merge")
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.8.0", "transformers==4.57.1", "peft==0.17.1", "accelerate==1.10.1", "safetensors"
)
adapters = modal.Volume.from_name("qwen-peft-adapters")
merged = modal.Volume.from_name("qwen-peft-merged", create_if_missing=True)
hf_cache = modal.Volume.from_name("qwen-peft-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",  # the bf16 model (16.4 GB) fits; merging on the GPU takes minutes, not tens of minutes
    timeout=3600,
    volumes={"/adapters": adapters, "/merged": merged, "/root/.cache/huggingface": hf_cache},
)
def merge():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(base, str(ADAPTER_PATH)).merge_and_unload()
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT_PATH, safe_serialization=True, max_shard_size="5GB")
    AutoTokenizer.from_pretrained(BASE_MODEL).save_pretrained(OUT_PATH)
    merged.commit()
    print("Merged model written:", sorted(p.name for p in OUT_PATH.iterdir()))


@app.local_entrypoint()
def main():
    merge.remote()
