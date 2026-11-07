from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from qwen_hybrid_config import HybridConfig
from qwen_hybrid_control import TrainingRegion, expected_events

spec = importlib.util.spec_from_file_location("hybrid_run", Path(__file__).parent / "docker/megatron/hybrid_run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class HybridRunnerTests(unittest.TestCase):
    def worker(self, code):
        return subprocess.Popen([sys.executable, "-c", code], start_new_session=True,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def test_failed_worker_stops_other_hosts(self):
        processes = [self.worker("import time; time.sleep(30)"), self.worker("raise SystemExit(7)")]
        with self.assertRaisesRegex(RuntimeError, "worker failed"):
            runner.wait_all(processes, 3)
        self.assertTrue(all(process.poll() is not None for process in processes))
        self.assertEqual(processes[1].returncode, 7)

    def test_timeout_cleans_up_workers(self):
        processes = [self.worker("import time; time.sleep(30)")]
        start = time.monotonic()
        with self.assertRaises(TimeoutError):
            runner.wait_all(processes, 0.1)
        self.assertLess(time.monotonic() - start, 3)
        self.assertIsNotNone(processes[0].poll())

    def test_status_wait_rejects_stale_run_id(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            path.write_text(json.dumps(dict(version=1, run_id="old", phase="complete")))
            with self.assertRaisesRegex(RuntimeError, "stale"):
                runner.wait_status(path, "new", "complete", 0.1)

    def status(self):
        return dict(version=1, run_id="run", phase="complete", valid=True, expected_world=2,
            ranks={str(rank): dict(started=True, finished=True, exited=True, observer_started=True, observer_finished=True,
                host=dict(hostname="host-1", pid=rank + 100)) for rank in range(2)})

    def test_final_status_requires_every_unique_completed_rank(self):
        self.assertTrue(runner.validate_final_status(self.status(), "run", 2))
        for mutate in (
            lambda status: status["ranks"].pop("1"),
            lambda status: status["ranks"]["1"].update(host=status["ranks"]["0"]["host"]),
            lambda status: status["ranks"]["1"].update(finished=False),
            lambda status: status["ranks"]["1"].update(observer_started=False),
        ):
            status = deepcopy(self.status())
            mutate(status)
            with self.assertRaises(RuntimeError):
                runner.validate_final_status(status, "run", 2)
        status = self.status()
        status["valid"] = False
        self.assertFalse(runner.validate_final_status(status, "run", 2))

    def test_region_manifest_counts_all_iterations_and_excludes_initialization(self):
        config = HybridConfig(pipeline_parallel_size=2, world_size=2)
        self.assertEqual(expected_events(config, 0, 4)["l2norm/backward"], 128)
        self.assertEqual(expected_events(config, 1, 4)["causal_conv1d/forward"], 32)
        events = []
        with patch.dict(os.environ, PHANTORA_RUN_ID="run"):
            region = TrainingRegion(0, 2, expected_events(config, 0, 4), sink=events.append)
            region.emit("start")
            region.emit("finish")
        self.assertEqual([event["phase"] for event in events], ["register", "start", "finish"])
        self.assertEqual(events[0]["world_size"], 2)
        self.assertEqual(events[1]["expected"], {})


if __name__ == "__main__":
    unittest.main()
