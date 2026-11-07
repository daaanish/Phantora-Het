"""Shared training-region counts and native control transport."""

import ctypes
import json
import os


def expected_events(workload, pp_rank, iterations):
    layers = sum(kind == "gdn" for _, kind in workload.pipeline_layers(pp_rank))
    calls = layers * workload.gradient_accumulation * iterations
    return {f"{op}/{direction}": calls * factor for op, factor in (
        ("causal_conv1d", 1), ("l2norm", 2), ("chunk_gated_delta_rule", 1),
    ) for direction in ("forward", "backward")}


class TrainingRegion:
    def __init__(self, rank, world_size, expected, sink=None):
        self.message = dict(version=1, run_id=os.environ["PHANTORA_RUN_ID"], rank=rank,
                            world_size=world_size, expected=expected)
        self.sink = sink
        if sink is None:
            self.library = ctypes.CDLL("libcuda.so.1")
            self.send = self.library.cuda_coverage_control
            self.send.argtypes = (ctypes.c_char_p, ctypes.c_size_t)
            self.send.restype = ctypes.c_int
        self.emit("register")

    def start_observer(self):
        from qwen_hybrid_observer import DeviceOnlyObserver
        self.observer = DeviceOnlyObserver(lambda observation: self.emit("device_only", observation))
        self.observer.__enter__()
        self.emit("observer_start")

    def stop_observer(self):
        self.observer.__exit__(None, None, None)
        self.emit("observer_finish")

    def emit(self, phase, observation=None):
        message = dict(self.message, phase=phase)
        if phase != "register":
            message["expected"] = {}
        if observation is not None:
            message["observation"] = observation
        if self.sink is not None:
            self.sink(message)
        else:
            payload = json.dumps(message, sort_keys=True, separators=(",", ":")).encode()
            if self.send(payload, len(payload)) != 0:
                raise RuntimeError(f"native coverage control failed: {phase}")
