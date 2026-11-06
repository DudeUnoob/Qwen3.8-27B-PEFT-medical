# v1 and v2: what the live tests showed

Both apps serve the same model. The web page selector only chooses which Modal app handles the request. The browser talks to the CPU web app, which checks the GitHub session and proxies the stream with server-side Modal credentials. Both apps load `Qwen/Qwen3-8B` in BF16, attach the same rank-64 medical LoRA as `qwen-medical`, and answer on `/v1/chat/completions`. They share the adapter volume and the weight caches, and each scales to zero when idle.

## What is different

| | v1 | v2 |
| --- | --- | --- |
| GPUs | 1× L4, or 1× A10G when L4 is unavailable | 2× L4, or 2× A10G when a pair of L4s is unavailable |
| Model placement | The whole model sits on one GPU | Tensor parallelism splits each layer across both GPUs |
| Scheduler | vLLM's normal prefill-first scheduler | RAD alternates prompt prefill and token decode |
| Requests in flight | At most 8 | Up to 128, with the live RAD cap at 64 by default |
| Context | 2048 tokens | 2048 tokens |
| Idle before shutdown | 5 minutes | 2 minutes |
| GPU cost while running | About one 24 GB GPU | About two 24 GB GPUs |

v1 is the single-user server. One 24 GB GPU holds the full 8B model, and vLLM mixes the few active requests itself.

v2 is the throughput server. Splitting the weights leaves more memory for the key/value cache, so more conversations can stay active. RAD prefills one waiting prompt in chunks until enough requests are decoding, then decodes one token for all of them together. That favors tokens per second across many users. It does not guarantee a smooth gap between tokens for one user. The cross-GPU exchange after each layer adds delay to every token, because these GPUs are paired over PCIe.

## Light load, 6 Nov 2026

`serving/benchmark.py` sent real streaming requests. Every success returned exactly 128 new tokens and 127 inter-token gaps. Prompts were 26–53 tokens. v1 ran on one A10-class GPU that was already warm. v2 cold-started on two A10Gs. Raw files: `serving/benchmarks/20261106-154134-v1` and `serving/benchmarks/20261106-154227-v2`.

| Users at once | v1 tokens/sec | v2, stock vLLM | v2, RAD |
| --- | ---: | ---: | ---: |
| 1 | 17.9 | 13.3 | 13.3 |
| 4 | 68.7 | 48.2 | 50.1 |
| 8 | 134.0 | 95.5 | 97.0 |

No request failed. v1 time to first token was 0.26–0.45 seconds. v2 was 0.75–1.9 seconds. The gap between tokens was about 0.07 seconds on v1 and 0.19 seconds on v2.

v2 was slower because this load never used the extra capacity. Eight short requests fit in one GPU, and v1's cap is also 8, so nothing waited in line. The second GPU only added the per-token exchange. RAD and stock vLLM on v2 matched each other: RAD logged about 260 decode steps and 4–18 prefill steps, so the scheduler was not the bottleneck.

The script's cost estimate was about $0.24 using the L4 list price, including idle time before shutdown. The GPUs that ran were A10-class, so the Modal dashboard charge is higher than that estimate.

## When v2 can be a gain

v2 is worth its cost only when more requests are in flight than one GPU is allowed to run. v1 stops at 8 sequences, so user 9 and beyond wait. v2 can decode up to 64 at once. The capacity scenario in `serving/benchmark.py` is that comparison: 8, 16, and 32 concurrent users, one round each, with about 3,000 extra characters of unique chart text on every prompt so the runs cannot share a cached prefix.

```sh
python -m serving.benchmark --api-version v1 --scenario capacity --budget-usd 4
python -m serving.benchmark --api-version v2 --scenario capacity --budget-usd 4 \
  --config policy=rad,max_active=64,chunk=256
```

A gain means v2's output tokens per second at 16 and 32 users exceed v1's, which should flatten once eight sequences fill the single GPU. One user, or eight short prompts, is the wrong test: v1 remains the better server for this private chat.

## Capacity load, 6 Nov 2026

The same client, with `--scenario capacity`. Each request again returned 128 tokens. Prompts were 694–764 tokens, unique per request. 8, then 16, then 32 users, one round each. v1 cold-started in 78 seconds on one A10G. v2 cold-started in 65 seconds on two A10Gs and ran RAD with `max_active=64` and `chunk=256`. Raw files: `serving/benchmarks/20261106-155411-v1` and `serving/benchmarks/20261106-155640-v2`.

| Users at once | v1 tokens/sec | v1 TTFT p50 | v2 RAD tokens/sec | v2 TTFT p50 |
| --- | ---: | ---: | ---: | ---: |
| 8 | 96.6 | 1.94 s | 102.6 | 1.22 s |
| 16 | 101.4 | 10.38 s | 180.6 | 1.85 s |
| 32 | 102.9 | 20.23 s | 308.9 | 3.16 s |

No request failed. v1's throughput stayed near 100 tokens per second once eight sequences filled the GPU, and the median wait for the first token grew from 2 seconds to 20 seconds because the extra users queued. v2 kept admitting them: at 32 users it produced about three times as many tokens per second, and the median first token arrived in 3.2 seconds instead of 20. RAD's prefill steps rose with the crowd (25, then 49, then 97) while each level still finished in about 127 decode steps, which is one shared decode pass per generated token rather than a separate pass per user.

At eight long requests the two servers were tied. Past that cap, v2 is the gain: more tokens per second and a much shorter wait to start streaming. It still costs two GPUs. The script priced both runs at about $0.10 each on the L4 list price, including idle time before shutdown. Check the Modal dashboard for the A10-class charge.

A repeat about five minutes later matched. v1 was still warm. v2 cold-started again in 68 seconds. Same prompt size, 694–764 tokens, and the same 128-token replies. Files: `serving/benchmarks/20261106-155933-v1` and `serving/benchmarks/20261106-160052-v2`.

| Users at once | v1 tokens/sec | v1 TTFT p50 | v2 RAD tokens/sec | v2 TTFT p50 |
| --- | ---: | ---: | ---: | ---: |
| 8 | 97.6 | 1.69 s | 100.7 | 1.35 s |
| 16 | 102.4 | 10.30 s | 177.5 | 1.96 s |
| 32 | 103.6 | 20.05 s | 304.4 | 3.49 s |
