# Single-GPU serving

## Try it now

From the project directory:

```sh
.venv/bin/python serving/chat.py
```

Type a prompt to see the response stream into the terminal. `/reset` clears
conversation history and `/quit` exits. The client reads `.env` automatically.
Cold starts can take several minutes; timing includes that wait.
Both clients use python-dotenv, so quoted values, comments, whitespace, and
`export` declarations are parsed consistently with Modal's dotenv loader.
If a stream ends before `[DONE]`, the terminal client reports a failure and
does not add the partial turn to conversation history.

For one prompt, or a comparison with the unadapted base:

```sh
.venv/bin/python serving/chat.py 'Explain what an antibody is in two sentences.'
.venv/bin/python serving/chat.py --model Qwen/Qwen3-8B 'Explain what an antibody is in two sentences.'
```

Both requests run through the same deployed server. Only `qwen-medical`
applies the trained adapter. A side-by-side response comparison can reveal
behavior differences; it does not measure medical accuracy.

To verify authentication, model registration, and generation automatically:

```sh
.venv/bin/python serving/smoke_test.py
```

This serves the adapter from `qwen_peft_medical.ipynb` on its actual base,
`Qwen/Qwen3-8B`. The notebook uses rank-64 QLoRA and saves `results/final`.
Training quantization is not needed at inference: vLLM loads the base in BF16
and applies the trained LoRA adapter. The steering notebook is unrelated.

## Export from Colab

After the medical notebook's training and save cell completes, run:

```python
import shutil
from google.colab import files

shutil.make_archive('/content/qwen-medical-adapter', 'zip', save_path)
files.download('/content/qwen-medical-adapter.zip')
```

Keep the whole archive, including `adapter_config.json`,
`adapter_model.safetensors` (or `.bin`), and tokenizer files.
Download it before the Colab runtime expires. Do not commit weights to Git.

## Upload and deploy

Use Modal SDK 1.6.3. Authenticate with `modal setup` if needed.
Unzip the exported adapter to a local `final` directory, then run from repo root:

```sh
modal volume create qwen-peft-adapters --env main
modal volume put qwen-peft-adapters ./final /medical --env main
modal deploy --env main serving/modal_app.py
```

The app uses at most one GPU, preferring L4 and falling back to A10G when
L4 capacity is unavailable. Both have 24 GB of VRAM. It scales to zero
after five idle minutes.
It caches base weights and compilation artifacts in persistent volumes.
The initial baseline limits context to 2048 tokens and eight sequences.
The initial live smoke test verified memory fit, authentication, adapter
registration, and generation on an NVIDIA A10. It is a serving check,
not a benchmark or an evaluation of medical answer quality.
The container uses the official `vllm/vllm-openai:v0.11.0` image so its
CUDA and Python dependencies stay together.

## Verify the adapter

Save your proxy credentials and the inference URL as `MODEL_ENDPOINT` in the
ignored `.env`, with owner-only permissions. Load them for curl requests:

```sh
set -a
source .env
set +a
export ENDPOINT="$MODEL_ENDPOINT"
```

Keep credentials out of source control. These are workspace proxy credentials;
they can authenticate other protected endpoints in the same workspace.

```sh
curl --fail-with-body "$ENDPOINT/v1/models" \
  -H "Modal-Key: $MODAL_KEY" -H "Modal-Secret: $MODAL_SECRET"

curl --fail-with-body "$ENDPOINT/v1/chat/completions" \
  -H "Modal-Key: $MODAL_KEY" -H "Modal-Secret: $MODAL_SECRET" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen-medical","messages":[{"role":"user","content":"What model are you?"}],"max_tokens":128,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
```

Always request `qwen-medical` to select the adapter. `/v1/models` also
lists the unadapted base. The endpoint requires Modal proxy authentication.
Run `python serving/smoke_test.py` to verify rejected unauthenticated access,
adapter registration, and generation. It writes a credential-free report to
`serving/deployment.json`. The first authenticated request starts the GPU and
downloads base weights; subsequent cold starts can use the persistent cache.

To stop serving, run `modal app stop qwen-peft-medical-vllm --env main`.
The adapter and cache volumes persist independently of the app.
