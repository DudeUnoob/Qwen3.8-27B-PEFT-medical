"""Verify endpoint authentication, adapter registration, and generation."""

import json
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import modal

from serving.config import load_credentials

ROOT = Path(__file__).resolve().parents[1]


def main():
    credentials = load_credentials(ROOT)
    endpoint = modal.Function.from_name(
        "qwen-peft-medical-vllm", "serve", environment_name="main"
    ).get_web_url().rstrip("/")
    headers = {
        "Modal-Key": credentials["MODAL_KEY"],
        "Modal-Secret": credentials["MODAL_SECRET"],
        "Content-Type": "application/json",
    }
    try:
        with urlopen(endpoint + "/v1/models", timeout=30):
            raise AssertionError("Unauthenticated request unexpectedly succeeded")
    except HTTPError as error:
        assert error.code in (401, 403), error.code
    print("Unauthenticated access rejected", flush=True)

    started = time.monotonic()
    with urlopen(Request(endpoint + "/v1/models", headers=headers), timeout=1500) as response:
        models = json.load(response)
    model_ids = [entry["id"] for entry in models["data"]]
    assert "qwen-medical" in model_ids, model_ids
    print("Medical adapter registered:", model_ids, flush=True)

    payload = {
        "model": "qwen-medical",
        "messages": [{"role": "user", "content": "Reply with a short greeting."}],
        "max_tokens": 64,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    generation_start = time.monotonic()
    request = Request(endpoint + "/v1/chat/completions", headers=headers,
                      data=json.dumps(payload).encode())
    with urlopen(request, timeout=180) as response:
        result = json.load(response)
    answer = result["choices"][0]["message"]["content"]
    assert answer and answer.strip(), result
    assert result["model"] == "qwen-medical", result["model"]
    report = {
        "endpoint": endpoint,
        "base_model": "Qwen/Qwen3-8B",
        "adapter": "qwen-medical",
        "gpu_preference": ["L4", "A10G"], "gpu_count": 1,
        "auth_rejection_verified": True,
        "models": model_ids,
        "response": answer,
        "usage": result.get("usage"),
        "generation_seconds": round(time.monotonic() - generation_start, 2),
        "total_seconds_including_startup": round(time.monotonic() - started, 2),
    }
    (ROOT / "serving/deployment.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
