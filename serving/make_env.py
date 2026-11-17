"""Create a Modal proxy token for the benchmark CLI and write the local .env, without printing secrets.

    python -m serving.make_env

Writes MODAL_KEY / MODAL_SECRET plus the endpoint URL of every deployed stack. An existing .env is
kept, with only those keys replaced. The token's secret is shown nowhere: Modal shows it only once,
at creation, and this script hands it straight to the file.
"""

from pathlib import Path

import modal
from dotenv import dotenv_values

from serving.chat import ENDPOINT_KEYS

ROOT = Path(__file__).resolve().parents[1]
APPS = {
    "v1": "qwen-peft-medical-vllm",
    "v2": "qwen-peft-medical-vllm-v2",
    "v3": "qwen-peft-medical-vllm-v3",
    "exp-v1-8": "qwen-peft-medical-exp-v1-8",
    "exp-v1-32": "qwen-peft-medical-exp-v1-32",
    "exp-merged-32": "qwen-peft-medical-exp-merged-32",
}


def main():
    path = ROOT / ".env"
    values = {k: v for k, v in dotenv_values(path).items() if v is not None} if path.is_file() else {}

    token = modal.Workspace.from_context().proxy_tokens.create(name="benchmark-cli")
    values["MODAL_KEY"] = token.token_id
    values["MODAL_SECRET"] = token.token_secret

    for version, app in APPS.items():
        try:
            values[ENDPOINT_KEYS[version]] = modal.Function.from_name(app, "serve", environment_name="main").get_web_url()
        except Exception as error:  # not deployed: leave the key unset
            print(f"skipped {version}: {type(error).__name__}")

    path.write_text("".join(f"{k}={v}\n" for k, v in values.items()), encoding="utf-8")
    print("Wrote .env:", ", ".join(k for k in values))


if __name__ == "__main__":
    main()
