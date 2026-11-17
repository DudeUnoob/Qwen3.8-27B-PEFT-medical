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

## API v2: two GPUs with the RAD scheduler

Diagrams and the decision log are in
[docs/serving-v2-architecture.md](../docs/serving-v2-architecture.md).

| | v1 (`modal_app.py`) | v2 (`modal_app_v2.py`) |
| --- | --- | --- |
| GPUs | 1× L4 (A10G fallback) | 2× L4, tensor parallel |
| Scheduler | stock vLLM | RAD ([arXiv:2508.01002](https://arxiv.org/abs/2508.01002)) |
| Concurrent sequences | 8 | up to 128 |
| Idle scale-down | 5 min | 2 min |
| Approximate GPU cost | $0.80/h | $1.60/h |

Both serve the same base model and adapter from the same volumes, through the
same OpenAI-compatible `/v1/chat/completions` path. "API version" names the
serving stack, not the OpenAI path.

```sh
modal deploy --env main serving/modal_app_v2.py
```

Save the printed URL as `MODEL_ENDPOINT_V2` in `.env`, then chat with
`python -m serving.chat --api-version v2`. The web app shows a v1/v2 selector
once `MODEL_ENDPOINT_V2` is in the `qwen-medical-web-config` secret.

RAD (`rad_scheduler.py`, policy in `rad_policy.py`) alternates two modes. In
prefill mode it prefills one request at a time in `chunk`-token pieces until
`max_active` requests are decoding. In decode mode it batches one decode
iteration for every active request until one finishes. After
`cycle_requests` admissions it drains all active requests before starting a
new cycle. It subclasses vLLM's V1 scheduler and restricts which requests the
stock scheduler sees each step, so vLLM still owns KV allocation, preemption,
and LoRA. A prefill that cannot get KV blocks forces a decode step so memory
can drain. vLLM marks custom scheduler classes as a non-public interface; pin
the image tag when upgrading.

The knobs change at runtime through the `qwen-peft-rad-config` Modal Dict, so
a batching sweep needs only one cold start. `policy=vllm` runs vLLM's stock
prefill-first scheduler under the same caps, as a baseline on the same GPUs.

## API v3: data-parallel replicas

`modal_app_v3.py` runs v1's single-GPU server as up to two replicas. Modal's
load balancer routes each request to one replica and starts the second only
when in-flight requests pass 75% of a replica's cap. Replicas share nothing,
so each can fall back to A10G on its own. Set the per-replica cap at deploy
time with `V3_MAX_NUM_SEQS` (default 8). Design, diagrams, and the decision log
are in [docs/serving-v3-data-parallel.md](../docs/serving-v3-data-parallel.md).

```sh
modal deploy --env main serving/modal_app_v3.py
python -m serving.chat --api-version v3
python -m serving.smoke_test --api-version v3
```

Save the printed URL as `MODEL_ENDPOINT_V3` in `.env`.

## Load testing under a budget

```sh
python -m pip install -e '.[bench]'
python -m serving.benchmark --api-version v2 --users 1,4,16 --rounds 2 --budget-usd 2
python -m serving.benchmark --api-version v2 \
  --config policy=rad,max_active=16,chunk=256 \
  --config policy=rad,max_active=64,chunk=256 \
  --config policy=rad,max_active=64,chunk=512
python -m serving.benchmark --api-version v1 --users 1,4,8
python -m serving.benchmark --api-version v3 --label v3-narrow --warm-replicas 2 --scenario capacity
python -m serving.parity --apis v1,v3,v2
python -m serving.report "serving/benchmarks/*/summary.json" --charts docs/img
```

Scenarios: `light`, `capacity`, `long-context`, `long-decode`, and `burst`.
Each one presets users, prompt size, and reply length; explicit flags win.
Per-user speed is reported as TPOT, the mean time per output token after the
first, because single inter-token gaps are distorted when several tokens
arrive in one network read. For v3, the cost guard polls the live container
count, so only replicas that are actually up are charged.

Each user sends `--rounds` streaming requests back to back. Each level reports
time to first token (TTFT), time between tokens (TBT), end-to-end latency, and
output tokens per second. On v2 it also reports the scheduler's decode and
prefill step counts. By default each request generates exactly `--max-tokens`
(128) tokens, so runs are comparable. Each prompt gets a unique prefix so
prefix caching does not hide prefill work. Results go to the ignored
`serving/benchmarks/` directory.

The cost guard counts elapsed GPU time from the first request (including any
cold start) plus the idle scale-down tail. It stops launching requests once
the estimate reaches `--budget-usd`; requests already in flight finish. The
estimate uses the workspace list price of the dearest GPU the stack may fall
back to (A10G, $1.10/h) and excludes CPU and memory, which add a
few percent. Check actual spend in the Modal dashboard.

## Medical accuracy eval

```sh
python -m serving.medical_eval --api-version v1
python -m serving.medical_eval --scoring logprob --datasets medqa,medmcqa
```

Asks `qwen-medical` and the unadapted base the same MedQA, MedMCQA, and PubMedQA
questions through one deployment, so only the adapter differs. Prefer
`--scoring logprob` for multiple choice: it reads the likeliest letter after
"The answer is (", so a model that explains before answering is not marked wrong.
Results from 7 Nov 2026, with the throughput comparison, are in
[docs/serving-benchmark-results.md](../docs/serving-benchmark-results.md).

An 8B model fits on one 24 GB GPU. Over PCIe, without NVLink, tensor
parallelism can make a single request slower; v2 should gain throughput at
high concurrency from extra KV cache space. Compare v1 and v2 at the same
user counts before deciding which to keep.
