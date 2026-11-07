import os
import unittest
from unittest.mock import patch

import torch
from qwen_hybrid_control import TrainingRegion
from qwen_hybrid_observer import DeviceOnlyObserver


class DeviceOnlyObserverTests(unittest.TestCase):
    def test_actual_cpu_dispatch_classifies_factories_and_transfer(self):
        events = []
        source = torch.tensor([1., 2.])
        with DeviceOnlyObserver(events.append, target_device="cpu"):
            torch.empty(3, device="cpu")
            torch.empty_strided((2, 3), (1, 2), device="cpu")
            torch.zeros(3, device="cpu")
            torch.ones_like(source, device="cpu")
            source.to(device="cpu", copy=True)
            source + source  # no device argument, hence no duplicate observation
        kinds = [(event["op"], event["kind"]) for event in events]
        self.assertIn(("aten::empty", "allocation_only"), kinds)
        self.assertIn(("aten::empty_strided", "allocation_only"), kinds)
        self.assertIn(("aten::zeros", "unsupported"), kinds)
        self.assertIn(("aten::ones_like", "unsupported"), kinds)
        self.assertIn(("aten::_to_copy", "native_transfer"), kinds)
        self.assertFalse(any(name == "aten::add" for name, _ in kinds))

    def test_default_observer_does_not_treat_cpu_as_cuda(self):
        events = []
        with DeviceOnlyObserver(events.append):
            torch.zeros(2, device="cpu")
        self.assertEqual(events, [])

    def test_region_emits_activation_and_deactivation(self):
        messages = []
        with patch.dict(os.environ, PHANTORA_RUN_ID="test"):
            region = TrainingRegion(0, 1, {}, sink=messages.append)
            region.start_observer()
            region.emit("start")
            torch.ones(2, device="cpu")
            region.stop_observer()
            region.emit("finish")
        self.assertEqual([message["phase"] for message in messages],
            ["register", "observer_start", "start", "observer_finish", "finish"])


if __name__ == "__main__": unittest.main()
