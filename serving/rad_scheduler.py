"""vLLM v0.11 V1 scheduler that applies the RAD policy from serving/rad_policy.py.

Loaded inside vLLM's engine-core process with
``--scheduler-cls rad_scheduler.RADScheduler``. Each step restricts which
requests the stock scheduler can see, so KV allocation, preemption, LoRA, and
output bookkeeping remain vLLM's own code.
"""

import logging
import os
import threading
import time

from vllm.v1.core.sched.request_queue import create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler

from rad_policy import RadParams, choose_mode, parse_params

logger = logging.getLogger(__name__)
CONFIG_POLL_SECONDS = 2.0


def is_decoding(request) -> bool:
    return request.num_tokens - request.num_computed_tokens <= 1


class RADScheduler(Scheduler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_seqs = self.max_num_running_reqs
        self.max_tokens = self.max_num_scheduled_tokens
        self.base_long_prefill = self.scheduler_config.long_prefill_token_threshold
        self.rad_params = parse_params({
            "policy": os.environ.get("RAD_POLICY", "rad"),
            "max_active": os.environ.get("RAD_MAX_ACTIVE", 64),
            "chunk": os.environ.get("RAD_CHUNK", 256),
            "cycle_requests": os.environ.get("RAD_CYCLE_REQUESTS", 256),
        }, max_seqs=self.max_seqs, max_tokens=self.max_tokens)
        self.started_in_cycle = 0
        self.prefill_blocked = False
        self.step_counts = {"decode": 0, "prefill": 0, "blocked_prefill": 0, "vllm": 0}
        logger.info("RAD scheduler active with %s", self.rad_params)
        dict_name = os.environ.get("RAD_CONFIG_DICT")
        if dict_name:
            threading.Thread(target=self._watch_config, args=(dict_name,),
                             daemon=True, name="rad-config").start()

    def _watch_config(self, dict_name):
        import modal

        store = modal.Dict.from_name(dict_name, create_if_missing=True)
        applied_version = None
        while True:
            try:
                requested = store.get("requested")
                if requested and requested.get("version") != applied_version:
                    params = parse_params(requested, max_seqs=self.max_seqs,
                                          max_tokens=self.max_tokens)
                    self.rad_params = params
                    applied_version = requested.get("version")
                    logger.info("RAD parameters updated to %s", params)
                store["applied"] = {**self.rad_params.as_dict(), "version": applied_version,
                                    "steps": dict(self.step_counts),
                                    "running": len(self.running), "waiting": len(self.waiting),
                                    "updated_at": time.time()}
            except Exception:
                logger.exception("Could not sync RAD parameters from %s", dict_name)
            time.sleep(CONFIG_POLL_SECONDS)

    def schedule(self):
        params: RadParams = self.rad_params
        if params.policy == "vllm":
            self.step_counts["vllm"] += 1
            return self._schedule_with(
                running=self.running, show_waiting=True,
                max_running=max(params.max_active, len(self.running)),
                token_budget=params.chunk, long_prefill=self.base_long_prefill)

        decoding = [r for r in self.running if is_decoding(r)]
        prefilling = [r for r in self.running if not is_decoding(r)]
        if not decoding and not prefilling:
            self.started_in_cycle = 0
        mode = choose_mode(decoding=len(decoding), prefilling=len(prefilling),
                           waiting=len(self.waiting), started=self.started_in_cycle,
                           params=params, prefill_blocked=self.prefill_blocked)
        if mode == "decode":
            self.prefill_blocked = False
            self.step_counts["decode"] += 1
            return self._schedule_with(running=decoding, show_waiting=False,
                                       max_running=self.max_seqs,
                                       token_budget=self.max_tokens, long_prefill=0)
        if mode == "prefill" and prefilling:
            self.step_counts["prefill"] += 1
            output = self._schedule_with(running=prefilling[:1], show_waiting=False,
                                         max_running=self.max_seqs,
                                         token_budget=params.chunk, long_prefill=params.chunk)
        else:
            # Idle and new-prefill steps both admit at most one request so an
            # empty step still returns finished request IDs to the workers.
            self.step_counts["prefill"] += 1
            admitted_before = len(self.running)
            output = self._schedule_with(running=[], show_waiting=mode == "prefill",
                                         max_running=1,
                                         token_budget=params.chunk, long_prefill=params.chunk)
            self.started_in_cycle += len(self.running) - admitted_before
        self.prefill_blocked = output.total_num_scheduled_tokens == 0 and bool(decoding)
        if self.prefill_blocked:
            self.step_counts["blocked_prefill"] += 1
        return output

    def _schedule_with(self, *, running, show_waiting, max_running, token_budget, long_prefill):
        """Run the stock scheduler over a subset of requests, then restore state."""
        all_running = self.running
        visible = set(map(id, running))
        real_waiting = self.waiting
        saved = (self.max_num_running_reqs, self.max_num_scheduled_tokens,
                 self.scheduler_config.long_prefill_token_threshold)
        self.running = list(running)
        if not show_waiting:
            self.waiting = create_request_queue(self.policy)
        self.max_num_running_reqs = max_running
        self.max_num_scheduled_tokens = token_budget
        self.scheduler_config.long_prefill_token_threshold = long_prefill
        try:
            output = super().schedule()
        finally:
            survivors = set(map(id, self.running))
            original = set(map(id, all_running))
            admitted = [r for r in self.running if id(r) not in original]
            self.running = [r for r in all_running
                            if id(r) not in visible or id(r) in survivors] + admitted
            if not show_waiting:
                # Requests preempted during this step were prepended here.
                real_waiting.prepend_requests(self.waiting)
                self.waiting = real_waiting
            (self.max_num_running_reqs, self.max_num_scheduled_tokens,
             self.scheduler_config.long_prefill_token_threshold) = saved
        return output
