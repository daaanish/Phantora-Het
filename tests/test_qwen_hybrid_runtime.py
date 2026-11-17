import builtins
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import qwen_hybrid_runtime_check as runtime
from test_qwen_hybrid_runner import runner


class RuntimeMetadataTests(unittest.TestCase):
    def test_metadata_build_gate_imports_no_gpu_packages(self):
        versions = dict(runtime.EXPECTED_STACK, torch="2.9.1")
        original = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in ("torch", "triton", "fla", "causal_conv1d", "megatron"):
                raise AssertionError(f"runtime package imported in metadata gate: {name}")
            return original(name, *args, **kwargs)

        with patch.object(runtime.metadata, "version", side_effect=versions.__getitem__), \
                patch.object(runtime.metadata, "requires", return_value=[]), patch("builtins.__import__", guarded):
            self.assertEqual(runtime.check_metadata(), versions)

    def test_metadata_requires_fla_core_with_matching_wrapper_installed(self):
        versions = dict(runtime.EXPECTED_STACK, torch="2.9.1")
        versions.pop("fla-core")

        def installed_version(name):
            if name not in versions:
                raise runtime.metadata.PackageNotFoundError(name)
            return versions[name]

        with patch.object(runtime.metadata, "version", side_effect=installed_version), \
                patch.object(runtime.metadata, "requires", return_value=[]):
            with self.assertRaisesRegex(runtime.metadata.PackageNotFoundError, "fla-core"):
                runtime.check_metadata()
            versions["fla-core"] = "0.4.1"
            with self.assertRaisesRegex(RuntimeError, "fla-core: expected 0.4.0, found 0.4.1"):
                runtime.check_metadata()

    def test_runtime_rejects_incorrect_patched_torch_revision(self):
        for revision in (None, "wrong-revision"):
            with self.subTest(revision=revision), patch.object(runtime, "check_metadata", return_value={}), \
                    patch.object(torch, "__version__", "2.9.1"), \
                    patch.object(torch.version, "git_version", revision):
                with self.assertRaisesRegex(RuntimeError, "pinned patched torch revision"):
                    runtime.check_runtime(replay=False)

    def test_runtime_rejects_wrong_or_absent_cuda_build(self):
        for cuda in (None, "12.7"):
            with patch.object(runtime, "check_metadata", return_value={}), patch.object(torch, "__version__", "2.9.1"), patch.object(torch.version, "git_version", "bbb19f8c856fcf8d7c60267fecb976b4749e2cff"), patch.object(torch.version, "cuda", cuda):
                with self.assertRaisesRegex(RuntimeError, "CUDA 12.8"):
                    runtime.check_runtime(replay=False)

    def test_replay_gpu_identity_is_required_and_unambiguous(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with self.assertRaises(ValueError):
                runner.gpu_name_from_db(path)
            manifest = path / "manifest.md"
            manifest.write_text("- **GPU:** NVIDIA L40S\n")
            self.assertEqual(runner.gpu_name_from_db(path), "NVIDIA L40S")
            manifest.write_text("- **GPU:** NVIDIA L40S\n- **GPU:** other GPU\n")
            with self.assertRaises(ValueError):
                runner.gpu_name_from_db(path)


if __name__ == "__main__":
    unittest.main()
