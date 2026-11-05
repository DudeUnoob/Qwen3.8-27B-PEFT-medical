from collections import deque
from dataclasses import dataclass, field
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "serving"))

from rad_policy import RadParams, choose_mode, parse_params  # noqa: E402


class FakeQueue(deque):
    def peek_request(self):
        return self[0]

    def pop_request(self):
        return self.popleft()

    def prepend_requests(self, requests):
        self.extendleft(reversed(list(requests)))


class FakeScheduler:
    """Mimics the parts of vLLM's V1 Scheduler.schedule() that RAD relies on."""

    def __init__(self, max_seqs=8, max_tokens=512):
        self.max_num_running_reqs = max_seqs
        self.max_num_scheduled_tokens = max_tokens
        self.scheduler_config = SimpleNamespace(long_prefill_token_threshold=0)
        self.running, self.waiting, self.policy = [], FakeQueue(), "fcfs"
        self.preempt_next = False

    def _new_tokens(self, request, budget):
        tokens = request.num_tokens - request.num_computed_tokens
        threshold = self.scheduler_config.long_prefill_token_threshold
        if 0 < threshold < tokens:
            tokens = threshold
        return min(tokens, budget)

    def schedule(self):
        budget, scheduled, preempted = self.max_num_scheduled_tokens, {}, False
        if self.preempt_next and self.running:
            self.preempt_next, preempted = False, True
            victim = self.running.pop()
            victim.num_computed_tokens = 0
            self.waiting.appendleft(victim)
        for request in self.running:
            tokens = self._new_tokens(request, budget)
            if tokens:
                scheduled[request.name] = tokens
                budget -= tokens
        while (not preempted and self.waiting and budget > 0
               and len(self.running) < self.max_num_running_reqs):
            request = self.waiting.pop_request()
            self.running.append(request)
            scheduled[request.name] = self._new_tokens(request, budget)
            budget -= scheduled[request.name]
        for request in self.running:
            request.num_computed_tokens += scheduled.get(request.name, 0)
        return SimpleNamespace(num_scheduled_tokens=scheduled,
                               total_num_scheduled_tokens=sum(scheduled.values()))


@dataclass(eq=False)
class FakeRequest:
    name: str
    prompt: int
    output_limit: int
    outputs: int = 0
    num_computed_tokens: int = 0
    finished: bool = field(default=False)

    @property
    def num_tokens(self):
        return self.prompt + self.outputs


@pytest.fixture
def rad(monkeypatch):
    for name in ("vllm", "vllm.v1", "vllm.v1.core", "vllm.v1.core.sched"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    scheduler_module = ModuleType("vllm.v1.core.sched.scheduler")
    scheduler_module.Scheduler = FakeScheduler
    queue_module = ModuleType("vllm.v1.core.sched.request_queue")
    queue_module.create_request_queue = lambda policy: FakeQueue()
    monkeypatch.setitem(sys.modules, scheduler_module.__name__, scheduler_module)
    monkeypatch.setitem(sys.modules, queue_module.__name__, queue_module)
    monkeypatch.delenv("RAD_CONFIG_DICT", raising=False)
    sys.modules.pop("rad_scheduler", None)
    return importlib.import_module("rad_scheduler")


def step(scheduler):
    """Schedule once, then emulate vLLM sampling and finishing requests."""
    output = scheduler.schedule()
    for request in list(scheduler.running):
        if request.name in output.num_scheduled_tokens and request.num_computed_tokens == request.num_tokens:
            request.outputs += 1
            if request.outputs >= request.output_limit:
                scheduler.running.remove(request)
                request.finished = True
    return output.num_scheduled_tokens


def test_policy_alternates_prefill_and_decode_modes():
    params = RadParams(max_active=2, chunk=256, cycle_requests=8)
    assert choose_mode(decoding=0, prefilling=0, waiting=0, started=0, params=params) == "idle"
    assert choose_mode(decoding=1, prefilling=0, waiting=3, started=1, params=params) == "prefill"
    assert choose_mode(decoding=2, prefilling=0, waiting=3, started=2, params=params) == "decode"
    assert choose_mode(decoding=1, prefilling=0, waiting=0, started=1, params=params) == "decode"
    assert choose_mode(decoding=1, prefilling=0, waiting=3, started=8, params=params) == "decode"
    assert choose_mode(decoding=1, prefilling=0, waiting=3, started=1, params=params,
                       prefill_blocked=True) == "decode"


def test_params_are_validated_and_clamped_to_startup_limits():
    params = parse_params({"policy": "rad", "max_active": 500, "chunk": 4096},
                          max_seqs=128, max_tokens=2048)
    assert (params.max_active, params.chunk) == (128, 2048)
    with pytest.raises(ValueError):
        parse_params({"policy": "sjf"}, max_seqs=128, max_tokens=2048)
    with pytest.raises(ValueError):
        parse_params({"chunk": 0}, max_seqs=128, max_tokens=2048)


def test_rad_prefills_in_chunks_one_request_at_a_time_then_decodes(rad, monkeypatch):
    monkeypatch.setenv("RAD_MAX_ACTIVE", "2")
    monkeypatch.setenv("RAD_CHUNK", "256")
    scheduler = rad.RADScheduler()
    a, b, c = (FakeRequest(name, prompt=300, output_limit=3) for name in "abc")
    scheduler.waiting.extend([a, b, c])

    assert step(scheduler) == {"a": 256}
    assert step(scheduler) == {"a": 44}
    assert step(scheduler) == {"b": 256}
    assert step(scheduler) == {"b": 44}
    # Two decoding requests fill b_col, so c waits while both decode together.
    assert step(scheduler) == {"a": 1, "b": 1}
    assert list(scheduler.waiting) == [c]
    assert step(scheduler) == {"a": 1, "b": 1}
    assert a.finished and b.finished
    assert step(scheduler) == {"c": 256}
    assert scheduler.step_counts["decode"] == 2


def test_rad_restores_queues_and_returns_preempted_requests(rad, monkeypatch):
    monkeypatch.setenv("RAD_MAX_ACTIVE", "2")
    scheduler = rad.RADScheduler()
    a, b, c = (FakeRequest(name, prompt=10, output_limit=50) for name in "abc")
    scheduler.waiting.extend([a, b, c])
    step(scheduler)
    step(scheduler)
    real_waiting = scheduler.waiting
    scheduler.preempt_next = True
    assert step(scheduler) == {"a": 1}
    assert scheduler.waiting is real_waiting
    assert list(scheduler.waiting) == [b, c]
    assert scheduler.running == [a]
    assert (scheduler.max_num_running_reqs, scheduler.max_num_scheduled_tokens) == (8, 512)
    assert scheduler.scheduler_config.long_prefill_token_threshold == 0


def test_vllm_baseline_policy_uses_stock_prefill_first_scheduling(rad, monkeypatch):
    monkeypatch.setenv("RAD_POLICY", "vllm")
    monkeypatch.setenv("RAD_CHUNK", "512")
    scheduler = rad.RADScheduler()
    scheduler.waiting.extend(FakeRequest(name, prompt=100, output_limit=5) for name in "abc")
    assert step(scheduler) == {"a": 100, "b": 100, "c": 100}
