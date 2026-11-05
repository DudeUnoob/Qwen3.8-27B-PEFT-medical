"""Resource-Aware Dynamic (RAD) step selection, after Bari et al. (arXiv:2508.01002).

Kept free of vLLM imports so the policy can be unit tested on any machine.
"""

from dataclasses import asdict, dataclass

POLICIES = ("rad", "vllm")


@dataclass(frozen=True)
class RadParams:
    """Runtime scheduler knobs.

    policy: "rad" alternates prefill and decode modes; "vllm" defers to vLLM's
        stock prefill-first scheduler under the same caps, as a baseline.
    max_active: RAD's b_col, the decode batch size and active-request cap.
        For "vllm" it caps concurrently running sequences.
    chunk: RAD's prefill chunk, LCM(b_row, b_col, b_red). For "vllm" it is the
        per-step token budget.
    cycle_requests: RAD's N, requests started before a cycle must drain.
    """

    policy: str = "rad"
    max_active: int = 64
    chunk: int = 256
    cycle_requests: int = 256

    def as_dict(self):
        return asdict(self)


def parse_params(raw: dict, *, max_seqs: int, max_tokens: int) -> RadParams:
    """Validate requested knobs and clamp them to the limits vLLM started with."""
    defaults = RadParams()
    policy = raw.get("policy", defaults.policy)
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
    values = {}
    for name in ("max_active", "chunk", "cycle_requests"):
        value = int(raw.get(name, getattr(defaults, name)))
        if value < 1:
            raise ValueError(f"{name} must be positive, got {value}")
        values[name] = value
    return RadParams(policy=policy,
                     max_active=min(values["max_active"], max_seqs),
                     chunk=min(values["chunk"], max_tokens),
                     cycle_requests=values["cycle_requests"])


def choose_mode(*, decoding: int, prefilling: int, waiting: int, started: int,
                params: RadParams, prefill_blocked: bool = False) -> str:
    """Return "idle", "decode", or "prefill" for the next batch (Algorithm 1).

    decoding: active requests whose prefill is complete (the set D).
    prefilling: active requests with an unfinished prefill.
    waiting: queued requests that hold no KV cache yet.
    started: requests admitted during the current cycle.
    prefill_blocked: the previous prefill step could not allocate KV blocks,
        so decodes must run and free memory before another prefill is tried.
    """
    if not (decoding or prefilling or waiting):
        return "idle"
    if decoding and (decoding >= params.max_active or prefill_blocked
                     or not (prefilling or waiting)
                     or (not prefilling and started >= params.cycle_requests)):
        return "decode"
    if not (prefilling or waiting) or (not prefilling and started >= params.cycle_requests):
        return "idle"
    return "prefill"
