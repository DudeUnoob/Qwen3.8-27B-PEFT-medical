# Open questions: cap, scheduler, merged weights (17 Nov 2026)

Follows [serving-benchmark-results.md](serving-benchmark-results.md). Every run here is on **one GPU type, NVIDIA A10G, no fallback**, so nothing differs by hardware. Normal-prompt results are the **median of three runs**, with the min-max range where it matters. The runs agreed within a few percent. Estimated GPU cost of the whole set: about $2.70.

Raw request logs are in the ignored `serving/benchmarks/` folder; `serving/run_followups.sh` and `serving/run_v2_ab.sh` reproduce everything.

## 1. How much of the multi-GPU gain was v1's cap?

`serving/modal_app_exp.py` serves the v1 setup (one GPU, LoRA) with the sequence cap set at deploy time. Capacity scenario (~750-token prompts, 128-token replies):

| Users | 1 GPU, cap 8: tok/s | 1 GPU, cap 32: tok/s | 2 copies (v3-wide, 32 each): tok/s |
| ---: | ---: | ---: | ---: |
| 8 | 118 | 115 | 126 |
| 16 | 117 | 193 | 233 |
| 32 | 117 | 292 | 388 |
| 64 | 117 (116-117) | 296 (293-296) | 590 (587-593) |

At 64 users: median TTFT 35.4 s (cap 8), 13.9 s (cap 32), 3.7 s (v3-wide). TPOT 0.058, 0.078, 0.078 s.

Raising the cap alone gives 2.5x on the same GPU. Two copies give almost exactly 2x one GPU at cap 32. Together, 5.0x the cap-8 baseline.

Long-context scenario (~1,500-token prompts, 256-token replies), one GPU:

| Users | cap 8: tok/s | cap 8: TTFT p50 | cap 32: tok/s | cap 32: TTFT p99 | cap 32: preemptions |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 119 | 17.7 s | 173 | 6.6 s | 0 |
| 32 | 118 | 35.1 s | 171 | 30.9 s | 3 |
| 64 | 118 | 69.5 s | 170 | 79.1 s | 8 |

About 20 long conversations fit in one GPU's KV cache, so cap 32 overshoots and vLLM evicts.

## 2. Did RAD help?

v2 (two A10Gs, tensor parallel), each run measuring vLLM's stock scheduler (`policy=vllm, max_active=64, chunk=2048`) and RAD (`policy=rad, max_active=64, chunk=256`) back to back:

| Users | stock tok/s | RAD tok/s | stock TTFT p50 | RAD TTFT p50 | stock TPOT | RAD TPOT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 92 | 86 | 2.1 s | 2.2 s | 0.072 | 0.080 |
| 16 | 154 | 138 | 2.8 s | 3.6 s | 0.081 | 0.091 |
| 32 | 232 | 191 | 5.0 s | 6.7 s | 0.096 | 0.118 |
| 64 | 315 | 252 | 8.7 s | 12.1 s | 0.129 | 0.163 |

Long-context, one run: stock 313 tok/s, RAD 281 at 64 users, no preemptions either way.

RAD trailed at every load, about 20% at 64 users. Likely cause (untested): RAD prefills in 256-token chunks one prompt at a time, while the stock config packs up to 2,048 prefill tokens per step.

On these GPUs, v2 with normal prompts lands near one GPU at cap 32 (315 vs 296) and about half of v3-wide (590). With long prompts its pooled cache clearly beats one GPU (313 vs 170). The 7 Nov v2 run (401 tok/s with RAD) did not record its GPU type.

## 3. LoRA vs merged weights

`serving/merge_adapter.py` folds the adapter into the bf16 base (`merge_and_unload`) and writes it to the `qwen-peft-merged` volume. One A10G, cap 32, capacity scenario:

| Users | LoRA tok/s | merged tok/s | LoRA TPOT | merged TPOT |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 115 | 140 | 0.058 | 0.046 |
| 64 | 296 | 319 | 0.078 | 0.070 |

Greedy parity (`serving/parity.py`, 16 prompts): LoRA vs itself 50% exact / 76% shared prefix, merged vs LoRA 19% / 40%. Log-probability scoring (`serving/medical_eval.py --scoring logprob --models qwen-medical`), same day:

| | MedQA (1,273) | MedMCQA (1,000) |
| --- | ---: | ---: |
| LoRA | 63.5% | 57.7% |
| merged | 63.1% | 58.8% |

Same answer letter on 98.8% (MedQA) and 96.8% (MedMCQA) of questions.

## Reproduce

```sh
python -m serving.make_env                       # proxy token + endpoint URLs into .env
EXP_LABEL=v1-8 EXP_MAX_NUM_SEQS=8 modal deploy --env main serving/modal_app_exp.py
EXP_LABEL=v1-32 EXP_MAX_NUM_SEQS=32 modal deploy --env main serving/modal_app_exp.py
modal run --env main serving/merge_adapter.py
EXP_LABEL=merged-32 EXP_MAX_NUM_SEQS=32 EXP_WEIGHTS=merged modal deploy --env main serving/modal_app_exp.py
V2_GPU=A10G modal deploy --strategy recreate --env main serving/modal_app_v2.py
V3_GPU=A10G V3_MAX_NUM_SEQS=32 modal deploy --strategy recreate --env main serving/modal_app_v3.py
bash serving/run_followups.sh && bash serving/run_v2_ab.sh
```

On Windows set `PYTHONUTF8=1`. A two-GPU cold start can pass Modal's ~150 s request window; the benchmark now retries its warm-up on the resulting HTTP 303.
