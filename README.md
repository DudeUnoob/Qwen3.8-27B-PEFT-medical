# Qwen PEFT medical

Fine-tuning `Qwen/Qwen3-8B` using rank-64 QLoRA on medical instructions,
with single-GPU inference through vLLM on Modal.

The medical training notebook is `qwen_peft_medical.ipynb`. The repository
name contains an older model label; the notebook and exported adapter both
identify `Qwen/Qwen3-8B` as the actual base model.

See [serving instructions](serving/README.md) for deployment, authentication,
and API examples. Use the project's isolated environment:

```sh
source .venv/bin/activate
modal deploy --env main serving/modal_app.py
python serving/smoke_test.py
```

To recreate the environment, run `python3 -m venv .venv`, then
`.venv/bin/python -m pip install -e .`.
vLLM and CUDA dependencies run inside Modal, not on this Mac.

The downloaded weights are stored in the ignored `final/` folder. API
credentials live in an ignored `.env` with owner-only permissions.

The [private chat webpage](webapp/README.md) runs in a separate CPU Modal app
and adds GitHub login restricted to the owner's verified account ID.
See its setup instructions to configure GitHub OAuth credentials.

Use model name `qwen-medical` with the OpenAI-compatible `/v1` API and the
Modal proxy headers shown in the serving instructions. An end-to-end smoke
test verified authentication, adapter registration, and generation on one
NVIDIA A10 GPU. L4 is preferred, with A10G as a capacity fallback; the app
scales to zero after five idle minutes. The adapter is uploaded to
`qwen-peft-adapters:/medical`.

## Configuration and shutdown

Copy `.env.example` to `.env` and fill in your own credentials, endpoint URLs,
and numeric GitHub user ID locally. The published source contains placeholders;
deployment URLs, credentials, model weights, and verification reports stay local.

Both Modal apps were stopped after verification. To restart, deploy the inference
app, update the named web secret from `.env`, then deploy the web app using the
linked setup instructions. To stop both apps again:

```sh
modal app stop qwen-peft-medical-web --env main
modal app stop qwen-peft-medical-vllm --env main
```

Stopping the apps terminates CPU/GPU containers. Saved adapter and cache volumes
remain independently of the apps and can incur storage charges.
