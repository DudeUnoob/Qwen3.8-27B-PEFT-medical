"""Stream a chat from the deployed medical adapter."""

import argparse
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from serving.config import load_credentials

ROOT = Path(__file__).resolve().parents[1]
ENDPOINT_KEYS = {"v1": "MODEL_ENDPOINT", "v2": "MODEL_ENDPOINT_V2"}


def resolve_endpoint(credentials, api_version):
    key = ENDPOINT_KEYS[api_version]
    endpoint = credentials.get(key, "").rstrip("/")
    if not endpoint.startswith("https://"):
        raise RuntimeError(f"Set {key} in the local .env file to the deployed {api_version} HTTPS URL")
    return endpoint


def respond(messages, model, api_version="v1"):
    credentials = load_credentials(ROOT)
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    endpoint = resolve_endpoint(credentials, api_version)
    request = Request(endpoint + "/v1/chat/completions",
                      data=json.dumps(payload).encode(), headers={
                          "Content-Type": "application/json",
                          "Modal-Key": credentials["MODAL_KEY"],
                          "Modal-Secret": credentials["MODAL_SECRET"],
                      })
    started = time.monotonic()
    first_token = None
    content = []
    usage = None
    completed = False
    with urlopen(request, timeout=1500) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                completed = True
                break
            chunk = json.loads(data)
            if "error" in chunk:
                raise RuntimeError(str(chunk["error"]))
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices", []):
                token = choice.get("delta", {}).get("content")
                if token:
                    if first_token is None:
                        first_token = time.monotonic() - started
                    print(token, end="", flush=True)
                    content.append(token)
    if not completed:
        raise RuntimeError("Stream ended before completion; the partial response was not saved")
    elapsed = time.monotonic() - started
    stats = [f"total {elapsed:.2f}s"]
    if first_token is not None:
        stats.insert(0, f"first token {first_token:.2f}s")
    if usage:
        stats.append(f"{usage['completion_tokens']} output tokens")
    print("\n[" + " | ".join(stats) + "]", flush=True)
    return "".join(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", help="One prompt; omit for interactive chat")
    parser.add_argument("--model", default="qwen-medical",
                        choices=["qwen-medical", "Qwen/Qwen3-8B"])
    parser.add_argument("--api-version", default="v1", choices=sorted(ENDPOINT_KEYS),
                        help="v1: one GPU, stock vLLM scheduler; v2: two tensor-parallel GPUs with RAD")
    args = parser.parse_args()
    print(f"Model: {args.model} | API {args.api_version}", flush=True)
    print("A cold start can take several minutes. /reset clears history; /quit exits.", flush=True)
    messages = []
    while True:
        try:
            prompt = args.prompt if args.prompt is not None else input("\nYou: ").strip()
            if args.prompt is None and prompt == "/quit":
                break
            if args.prompt is None and prompt == "/reset":
                messages = []
                print("History cleared.")
                continue
            if not prompt:
                if args.prompt is not None:
                    parser.error("Prompt must not be empty")
                continue
            messages.append({"role": "user", "content": prompt})
            print("\nAssistant: ", end="", flush=True)
            try:
                answer = respond(messages, args.model, args.api_version)
            except (HTTPError, URLError, TimeoutError, RuntimeError) as error:
                messages.pop()
                if isinstance(error, HTTPError):
                    detail = error.read().decode(errors="replace")[:1500]
                    print(f"\nHTTP {error.code}: {detail}")
                else:
                    print(f"\nRequest failed: {error}")
                print("If the conversation exceeds 2048 tokens, use /reset.")
                if args.prompt is not None:
                    raise SystemExit(1)
                continue
            messages.append({"role": "assistant", "content": answer})
            if args.prompt is not None:
                break
        except (EOFError, KeyboardInterrupt):
            print("\nChat closed.")
            break


if __name__ == "__main__":
    main()
