"""Check that the torch in use is the build's torch (run inside the container).

Usage: python3 verify_torch.py            verify this process
       python3 verify_torch.py <pid>      verify another process's mapped libtorch
Env:   QH_TORCH_PREFIX (default /qh-torch), QH_TORCH_REVISION (required git_version).
Adapted from the QH-02 exact-gate verify_torch.py; only the prefix and revision moved to env.
"""
import os
from pathlib import Path
import sys

PREFIX = Path(os.environ.get("QH_TORCH_PREFIX", "/qh-torch")) / "torch"
REVISION = os.environ["QH_TORCH_REVISION"]


def verify_maps(pid):
    paths = set()
    for line in Path(f"/proc/{pid}/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6:
            path = Path(fields[5])
            if path.name.startswith(("libtorch", "libc10")) and ".so" in path.name:
                paths.add(path)
    assert any(path.name.startswith("libtorch_cpu.so") for path in paths), paths
    assert all(path.is_relative_to(PREFIX / "lib") for path in paths), paths
    print("verified mapped libtorch/libc10:", ", ".join(map(str, sorted(paths))), flush=True)


def verify_torch():
    import torch

    assert Path(torch.__file__).is_relative_to(PREFIX), torch.__file__
    assert torch.__version__ == "2.9.1", torch.__version__
    assert torch.version.git_version == REVISION, torch.version.git_version
    assert torch.version.cuda == "12.8", torch.version.cuda
    assert "sm_120" in torch._C._cuda_getArchFlags(), torch._C._cuda_getArchFlags()
    assert callable(torch.profiler.enable_function_tracer)
    print("verified torch:", torch.__file__, torch.__version__, torch.version.git_version,
          torch.version.cuda, torch._C._cuda_getArchFlags(), flush=True)
    verify_maps(os.getpid())


if __name__ == "__main__":
    if len(sys.argv) > 1:
        verify_maps(int(sys.argv[1]))
    else:
        verify_torch()
