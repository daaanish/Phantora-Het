"""Profile missing keys with bench.py CUPTI timing; each aliased exp.out call gets fresh logits.

Run inside the container by profile_inside.sh. Env: QH_SRC (build source, default /qh-src),
QH_KEYS, QH_OUT, QH_WARMUP, QH_ITERS. Unchanged from the QH-07n..QH-07v profile-all.py
except that the source root comes from QH_SRC."""
from pathlib import Path
import runpy
import sys

import torch

import os
root = Path(os.environ.get("QH_SRC", "/qh-src"))
sys.path.insert(0, str(root / "tests/perfdb"))
from gdn_profile import assert_real_execution
import bench

assert_real_execution()
assert torch.cuda.device_count() == 1
assert torch.cuda.get_device_capability(0) == (12, 0)
torch.manual_seed(42)
shared_build = runpy.run_path(str(root / "phantora/phantora/src/qwen_ops.py"))["build"]
build = bench.build
import os
warmup, iters = int(os.environ.get("QH_WARMUP", 3)), int(os.environ.get("QH_ITERS", 5))


def fresh_exp_out(op, payload):
    if op != "QwenOp" or payload["op"] != "exp_out":
        return build(op, payload)
    first, second = payload["inputs"]
    assert first == second and first["dtype"] == "Float"
    thunks = []
    for _ in range(warmup + iters):
        logits = torch.randn(first["shape"], dtype=torch.float32, device="cuda")
        logits.sub_(logits.amax(dim=-1, keepdim=True))
        thunks.append(shared_build(payload, allocator=lambda info: logits))
    thunks = iter(thunks)
    return lambda: next(thunks)()


bench.build = fresh_exp_out
sys.argv = ["bench.py", "--ref", os.environ.get("QH_KEYS", "/qhrun/keys"), "--out", os.environ.get("QH_OUT", "/qhrun/measured"),
            "--warmup", str(warmup), "--iters", str(iters)]
bench.main()
