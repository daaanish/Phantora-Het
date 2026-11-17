#!/usr/bin/env python3
"""One GPU-free Phantora simulation on gpu01 (discovery or replay), run by run.sh.

Derived from the QH-07a..QH-07v launcher; the workload is now given on the command line
instead of being edited in with sed. --perf-db is copied, never modified; keys it lacks
are written to <output>/discover-*/perfdb.missing/."""

import argparse
import __future__
import types
from dataclasses import asdict
import hashlib
import json
import re
import os
from pathlib import Path
import socket
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

parser = argparse.ArgumentParser(description=__doc__)
for name in ("tests", "image", "native-dist", "torch-prefix", "causal-prefix",
             "upgraded-prefix", "perf-db", "calibrated-netconfig", "output-dir"):
    parser.add_argument("--" + name, type=lambda path: Path(path).resolve(), required=True)
parser.add_argument("--preset", required=True, help="hybrid preset, e.g. 0.8b or 0.8b-l4")
parser.add_argument("--sequence-length", type=int, required=True)
parser.add_argument("--micro-batch-size", type=int, default=1)
parser.add_argument("--gradient-accumulation", type=int, default=8)
parser.add_argument("--tensor-parallel-size", type=int, default=2)
parser.add_argument("--iterations", type=int, default=4)
parser.add_argument("--mtp-layers", type=int, default=0, help="trained MTP layers (0 = off)")
parser.add_argument("--vram-mib", type=int, default=97887)
parser.add_argument("--sim-log-level", default="info", help="simulator log level; debug enables per-event aggregation")
parser.add_argument("--timeout-s", type=int, default=1500)
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

sys.path[:0] = [str(args.tests / "docker" / "megatron"), str(args.tests)]
# Rocky's host Python is 3.9. Defer the config's type annotations outside the SIF.
if sys.version_info < (3, 10):
    name = "qwen_hybrid_config"
    source = args.tests / f"{name}.py"
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(compile(source.read_text(), str(source), "exec", __future__.annotations.compiler_flag), module.__dict__)
from hybrid_run import stop_processes, validate_final_status, wait_all
from qwen_hybrid_config import HYBRID_PRESETS, HybridConfig
from qwen_hybrid_control import expected_events

TP = args.tensor_parallel_size
WORKLOAD = HybridConfig(model=HYBRID_PRESETS[args.preset], tensor_parallel_size=TP, world_size=TP,
                        sequence_length=args.sequence_length, micro_batch_size=args.micro_batch_size,
                        gradient_accumulation=args.gradient_accumulation,
                        global_batch_size=args.micro_batch_size * args.gradient_accumulation,
                        mtp_num_layers=args.mtp_layers)
ITERATIONS = args.iterations
HOST = "gpu01.pronet.rcac.purdue.edu"
GPU = "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def sha256(path):
    with path.open("rb") as source:
        digest = hashlib.sha256()
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


def prepare(output, shared):
    run_id = uuid.uuid4().hex
    events = expected_events(WORKLOAD, 0, ITERATIONS)
    plan = dict(run_id=run_id, expected_world=TP, expected={str(rank): events for rank in range(TP)})
    write_json(output / "plan.json", plan)
    (output / "netconfig.toml").write_bytes(args.calibrated_netconfig.read_bytes())
    for name in ("home", "tmp", "triton"):
        (shared / name).mkdir()
    binds = [
        (args.tests, "/phantora/tests", "ro"),
        (args.native_dist, "/phantora/dist", "ro"),
        (args.torch_prefix, "/qh-torch", "ro"),
        (args.causal_prefix, "/qh-causal", "ro"),
        (args.upgraded_prefix, "/qh-upgraded", "ro"),
        (output, "/qh-run", "rw"),
        (shared, "/qh-ipc", "rw"),
    ]
    env = dict(NVIDIA_VISIBLE_DEVICES="none", PYTHONDONTWRITEBYTECODE="1",
               PHANTORA_LOG="info",
               LD_LIBRARY_PATH="/qh-torch/torch/lib:/usr/local/python3.11.9/lib:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib",
               PYTHONPATH="/qh-torch:/qh-causal:/qh-upgraded",
               PHANTORA_SOCKET_PREFIX="/qh-ipc/p", PHANTORA_GPU_NAME=GPU,
               PHANTORA_IGNORE_CPU_TIME="1", PHANTORA_USE_CUPTI="1")
    runner_env = dict(PHANTORA_COVERAGE="1", PHANTORA_RUN_ID=run_id, PHANTORA_NGPU=str(TP),
                      PHANTORA_VRAM_MIB=str(args.vram_mib), CUDA_DEVICE_MAX_CONNECTIONS="1")
    # Apptainer ranks and the server share the host PID namespace and the same socket directory.
    apptainer = ["apptainer", "exec", "--cleanenv", "--contain", "--no-mount", "home,cwd", "--pwd", "/qh-run"]
    for path, target, mode in binds:
        apptainer += ["--bind", f"{path}:{target}:{mode}"]

    def command(argv, extra):
        return apptainer + ["--env", ",".join(f"{key}={value}" for key, value in (env | extra).items()),
                            str(args.image), "env", "HOME=/qh-ipc/home", "TMPDIR=/qh-ipc/tmp",
                            "TRITON_CACHE_DIR=/qh-ipc/triton", *argv]

    commands = dict(
        runtime=command(["python3", "-c",
                         "import glob, runpy, sys; assert not glob.glob('/dev/nvidia*'); "
                         "sys.path.insert(0, '/phantora/tests'); "
                         "runpy.run_path('/phantora/tests/qwen_hybrid_runtime_check.py', run_name='__main__')"],
                        {"PHANTORA": "1", "LD_LIBRARY_PATH": "/phantora/dist:" + env["LD_LIBRARY_PATH"]}),
        simulator=command(["/phantora/dist/phantora_server", "--netconfig", "/qh-run/netconfig.toml",
                           "--perf-db", "/qh-run/perfdb", "--run-id", run_id, "--expected-ranks", str(TP),
                           "--coverage-plan", "/qh-run/plan.json", "--coverage-status", "/qh-run/status.json",
                           "--coverage-mode", "discovery"], {"PHANTORA_LOG": args.sim_log_level}),
        worker=command(["/phantora/dist/phantora_run", "torchrun", "--standalone", f"--nproc-per-node={TP}",
                        "/phantora/tests/test_megatron.py", "--hybrid-preset", args.preset,
                        "--tensor_parallel_size", str(TP), "--data_parallel_size", "1",
                        "--pipeline_model_parallel_size", "1", "--sequence_length", str(args.sequence_length),
                        "--micro_batch_size", str(args.micro_batch_size),
                        "--global_batch_size", str(args.micro_batch_size * args.gradient_accumulation),
                        "--gradient_accumulation", str(args.gradient_accumulation),
                        "--iterations", str(ITERATIONS),
                        *(["--mtp-layers", str(args.mtp_layers)] if args.mtp_layers else [])], runner_env),
    )
    write_json(output / "commands.json", commands)
    write_json(output / "workload.json", dict(
        workload=asdict(WORKLOAD), iterations=ITERATIONS, cpu_time_policy="ignored in replay",
        gpu=GPU, gpu_per_host=[TP], host=HOST, vram_mib=args.vram_mib, dtype="bfloat16", device_visibility="none",
        perf_db=str(args.perf_db), perf_db_manifest=(args.perf_db / "manifest.md").read_text(),
        network_policy="same-host loopback rate from prior 8 MiB NCCL all-reduce; one fixed rate at all sizes",
        calibrated_netconfig=str(args.calibrated_netconfig), timing_result_accepted=False))
    return run_id, commands


def wait_phase(path, run_id, phase, deadline, server):
    while time.monotonic() < deadline:
        if path.exists():
            status = json.loads(path.read_text())
            if status.get("version") != 1 or status.get("run_id") != run_id:
                raise RuntimeError("unexpected simulator status")
            if status.get("phase") == phase:
                return status
        if server.poll() is not None:
            raise RuntimeError(f"simulator exited {server.returncode} before reporting {phase}")
        time.sleep(0.1)
    raise TimeoutError(f"simulator did not report {phase}")


def main():
    netconfig = args.calibrated_netconfig.read_text()
    if (not re.search(r'^host_mapping = \["gpu01\.pronet\.rcac\.purdue\.edu"\]$', netconfig, re.M)
            or not re.search(r'^loopback_speed = 271\.91597719$', netconfig, re.M)):
        raise ValueError("expected saved gpu01 netconfig with the measured 271.91597719 Gbit/s loopback rate")
    if not args.dry_run:
        if socket.getfqdn() != HOST and socket.gethostname() != HOST:
            raise RuntimeError(f"execute on {HOST}; observed {socket.gethostname()}")
        required = [args.image, args.native_dist / "phantora_server", args.native_dist / "phantora_run",
                    args.native_dist / "libcuda.so.1", args.torch_prefix / "torch" / "version.py",
                    args.tests / "test_megatron.py", args.perf_db / "compute.csv", args.perf_db / "memcpy.csv"]
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(path)
        for path in (args.causal_prefix, args.upgraded_prefix):
            if not path.is_dir():
                raise FileNotFoundError(path)
    args.output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="dry-run-" if args.dry_run else "discover-", dir=args.output_dir))
    print(output, flush=True)
    with tempfile.TemporaryDirectory(prefix="qh07a-") as scratch:
        run_id, commands = prepare(output, Path(scratch))
        if args.dry_run:
            return 0
        shutil.copytree(args.perf_db, output / "perfdb")
        paths = [*sorted(args.native_dist.glob("*")), *sorted((output / "perfdb").glob("*")),
                 *sorted(args.tests.glob("qwen_hybrid_*.py")), args.tests / "test_megatron.py",
                 args.torch_prefix / "torch" / "version.py", args.calibrated_netconfig]
        write_json(output / "input-hashes.json", {str(path): sha256(path) for path in paths if path.is_file()})
        result = dict(run_id=run_id, exit_code=1, valid=False, timing_result_accepted=False, artifact_directory=str(output))
        processes = []
        deadline = time.monotonic() + args.timeout_s
        clean_env = {k: v for k, v in os.environ.items()
                     if k != "CUDA_VISIBLE_DEVICES" and not k.startswith(("APPTAINERENV_", "SINGULARITYENV_", "APPTAINER_BIND", "SINGULARITY_BIND"))}
        try:
            for name in ("runtime", "simulator", "worker"):
                with (output / f"{name}.log").open("w") as log:
                    process = subprocess.Popen(commands[name], stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True, env=clean_env)
                processes.append(process)
                if name == "runtime" and process.wait(timeout=60) != 0:
                    raise RuntimeError(f"runtime probe exited {process.returncode}")
                if name == "simulator":
                    wait_phase(output / "status.json", run_id, "ready", min(deadline, time.monotonic() + 30), process)
            result["worker_exit_codes"] = wait_all(processes[-1:], max(0, deadline - time.monotonic()))
            status = wait_phase(output / "status.json", run_id, "complete", deadline, processes[1])
            result["valid"] = validate_final_status(status, run_id, TP)
            result["exit_code"] = 0
        except (OSError, ValueError, RuntimeError, TimeoutError, subprocess.SubprocessError) as error:
            result["error"] = str(error)
        finally:
            stop_processes(processes)
            write_json(output / "runner.json", result)
        return result["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
