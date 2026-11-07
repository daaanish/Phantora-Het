#!/usr/bin/env python3
"""Bounded, isolated multi-host hybrid replay; leaves legacy run.sh untouched."""

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import time
import uuid

TESTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TESTS))
from qwen_hybrid_config import DEBUG_MODEL, QWEN35_9B, HybridConfig
from qwen_hybrid_control import expected_events


def stop_processes(processes):
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for process in processes:
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=2)


def wait_all(processes, timeout):
    deadline = time.monotonic() + timeout
    try:
        while True:
            codes = [process.poll() for process in processes]
            if any(code is not None and code != 0 for code in codes):
                raise RuntimeError(f"worker failed: exit codes {codes}")
            if all(code == 0 for code in codes):
                return codes
            if time.monotonic() >= deadline:
                raise TimeoutError("workers timed out")
            time.sleep(0.05)
    except BaseException:
        stop_processes(processes)
        raise


def wait_status(path, run_id, phase, timeout):
    deadline = time.monotonic() + timeout
    while True:
        if path.exists():
            value = json.loads(path.read_text())
            if value.get("version") != 1 or value.get("run_id") != run_id:
                raise RuntimeError("stale or incompatible simulator status")
            if value.get("phase") == phase:
                return value
        if time.monotonic() >= deadline:
            raise TimeoutError(f"simulator did not report {phase}")
        time.sleep(0.05)


def validate_final_status(status, run_id, world):
    if status.get("run_id") != run_id or status.get("version") != 1 or status.get("phase") != "complete":
        raise RuntimeError("stale/incomplete final status")
    ranks = status.get("ranks", {})
    if status.get("expected_world") != world or set(ranks) != {str(rank) for rank in range(world)}:
        raise RuntimeError("final status is missing expected ranks")
    hosts = set()
    for rank in ranks.values():
        if not all(rank.get(field) is True for field in ("started", "finished", "exited", "observer_started", "observer_finished")):
            raise RuntimeError("rank did not complete its lifecycle")
        host = (rank["host"]["hostname"], rank["host"]["pid"])
        if host in hosts:
            raise RuntimeError("duplicate rank host identity")
        hosts.add(host)
    return status.get("valid") is True


def gpu_name_from_db(directory):
    manifest = directory / "manifest.md"
    names = [line.removeprefix("- **GPU:** ").strip() for line in manifest.read_text().splitlines()
             if line.startswith("- **GPU:** ")] if manifest.exists() else []
    if len(names) != 1 or not names[0]:
        raise ValueError("hybrid replay requires one nonempty GPU identity in the DB manifest")
    return names[0]


def run(args):
    world = args.tensor_parallel_size * args.data_parallel_size * args.pipeline_parallel_size
    if args.ngpu <= 0 or world % args.ngpu:
        raise ValueError("TP*DP*PP must be divisible by positive --ngpu")
    if args.iterations < 2 or args.timeout <= 0 or not math.isfinite(args.timeout) or args.vram_mib <= 0:
        raise ValueError("iterations must be at least 2, timeout finite/positive and VRAM positive")
    denominator = args.micro_batch_size * args.data_parallel_size
    if denominator <= 0 or args.global_batch_size % denominator:
        raise ValueError("global batch must be divisible by microbatch * DP")
    ga = args.gradient_accumulation if args.gradient_accumulation is not None else args.global_batch_size // denominator
    workload = HybridConfig(model=DEBUG_MODEL if args.preset == "debug" else QWEN35_9B,
        tensor_parallel_size=args.tensor_parallel_size, data_parallel_size=args.data_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size, world_size=world,
        sequence_length=args.sequence_length, micro_batch_size=args.micro_batch_size,
        gradient_accumulation=ga, global_batch_size=args.global_batch_size)
    db = Path(args.perf_db).resolve()
    if not db.is_dir():
        raise ValueError("--perf-db must name an existing database directory")
    gpu_name = gpu_name_from_db(db)
    run_id = uuid.uuid4().hex
    directory = Path(args.output_dir).resolve() / run_id
    directory.mkdir(parents=True, exist_ok=False)
    nhost = world // args.ngpu
    # Core's default rank order is TP-DP-PP; PP is the slowest-varying coordinate.
    plan = dict(run_id=run_id, expected_world=world, expected={str(rank): expected_events(
        workload, rank // (args.tensor_parallel_size * args.data_parallel_size), args.iterations,
    ) for rank in range(world)})
    (directory / "plan.json").write_text(json.dumps(plan, indent=2))
    template = runpy.run_path(str(Path(__file__).with_name("config_gen.py")))["NETCONFIG_TEMPLATE"]
    (directory / "netconfig.toml").write_text(template.format(
        host_list=repr([f"host-{i}" for i in range(1, nhost + 1)]), nracks=(nhost + 1) // 2))
    common_env = dict(PHANTORA_SOCKET_PREFIX="/run/phantora/p", PHANTORA_IGNORE_CPU_TIME="1", PHANTORA_GPU_NAME=gpu_name)
    assumptions = dict(gpu_name=gpu_name, image=args.image, ngpu_per_host=args.ngpu,
        nhost=nhost, perf_db=str(db), ignore_cpu_time=True, vram_mib=args.vram_mib,
        netconfig=(directory / "netconfig.toml").read_text(), full_attention_backend="local_unfused",
        memory_validated=False, software_gate="qwen_hybrid_runtime_check.py; actual HYBRID_RUNTIME is in each host log",
        expected_software=dict(torch="2.9.1", megatron_core="0.16.0", fla="0.4.0",
            causal_conv1d="1.5.3.post1", triton="3.7.0"))
    (directory / "workload.json").write_text(json.dumps(dict(workload=asdict(workload), assumptions=assumptions), indent=2))
    volumes = [f"{directory}:/run/phantora", f"{TESTS}:/phantora/tests:ro"]
    simulator_command = ["/phantora/dist/phantora_server", "--netconfig", "/run/phantora/netconfig.toml",
        "--perf-db", "/run/phantora/perfdb", "--run-id", run_id, "--expected-ranks", str(world),
        "--coverage-plan", "/run/phantora/plan.json", "--coverage-status", "/run/phantora/status.json",
        "--coverage-mode", args.mode]
    services = {"simulator": dict(image=args.image, volumes=[*volumes, f"{db}:/run/phantora/perfdb:ro"],
        environment=common_env, command=simulator_command, ipc="host", pid="host")}
    for index in range(1, nhost + 1):
        services[f"host-{index}"] = dict(image=args.image, volumes=volumes,
            hostname=f"host-{index}", command=["sleep", "infinity"], ipc="host", pid="host",
            environment=dict(common_env, PHANTORA_NGPU=str(args.ngpu), PHANTORA_COVERAGE="1",
                PHANTORA_RUN_ID=run_id, PHANTORA_VRAM_MIB=str(args.vram_mib), CUDA_DEVICE_MAX_CONNECTIONS="1"))
    compose_file = directory / "compose.json"
    compose_file.write_text(json.dumps(dict(services=services), indent=2))
    compose = ["docker", "compose", "-p", f"phantora-qh-{run_id}", "-f", str(compose_file)]
    deadline = time.monotonic() + args.timeout
    processes, logs = [], []
    result = dict(run_id=run_id, valid=False, exit_code=1, artifact_directory=str(directory), assumptions=assumptions)
    try:
        with (directory / "setup.log").open("w") as log:
            subprocess.run([*compose, "up", "-d"], stdout=log, stderr=subprocess.STDOUT, check=True,
                           timeout=min(60, args.timeout))
        wait_status(directory / "status.json", run_id, "ready", max(0, deadline - time.monotonic()))
        model_args = ["--hybrid-preset", args.preset, "--tensor_parallel_size", str(args.tensor_parallel_size),
            "--pipeline_model_parallel_size", str(args.pipeline_parallel_size), "--data_parallel_size", str(args.data_parallel_size),
            "--sequence_length", str(args.sequence_length), "--micro_batch_size", str(args.micro_batch_size),
            "--global_batch_size", str(args.global_batch_size), "--gradient_accumulation", str(ga), "--iterations", str(args.iterations)]
        for index in range(1, nhost + 1):
            log = (directory / f"host-{index}.log").open("w")
            logs.append(log)
            command = [*compose, "exec", "-T", f"host-{index}", "/phantora/dist/phantora_run", "torchrun",
                "--nproc_per_node", str(args.ngpu), "--nnodes", str(nhost), "--node_rank", str(index - 1),
                "--master_addr", "host-1", "--master_port", "12345", "/phantora/tests/test_megatron.py", *model_args]
            processes.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        result["worker_exit_codes"] = wait_all(processes, max(0, deadline - time.monotonic()))
        status = wait_status(directory / "status.json", run_id, "complete", max(0, deadline - time.monotonic()))
        valid = validate_final_status(status, run_id, world)
        result.update(valid=valid, exit_code=0 if valid or args.mode == "discovery" else 1)
    except (OSError, ValueError, RuntimeError, TimeoutError, subprocess.SubprocessError) as error:
        result["error"] = str(error)
    finally:
        stop_processes(processes)
        for log in logs:
            log.close()
        for filename, command, timeout in (
            ("simulator.log", [*compose, "logs", "--no-color", "simulator"], 15),
            ("cleanup.log", [*compose, "down", "--timeout", "1", "--remove-orphans"], 30),
        ):
            try:
                with (directory / filename).open("w") as log:
                    completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
                if completed.returncode != 0:
                    result.setdefault("cleanup_errors", []).append(f"{filename}: exit {completed.returncode}")
            except (OSError, subprocess.SubprocessError) as error:
                result.setdefault("cleanup_errors", []).append(f"{filename}: {error}")
        if result.get("cleanup_errors"):
            result["exit_code"] = 1
        (directory / "runner.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, sort_keys=True))
    return result["exit_code"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="phantora:hybrid")
    parser.add_argument("--perf-db", required=True)
    parser.add_argument("--preset", choices=["debug", "9b"], default="debug")
    parser.add_argument("--mode", choices=["strict", "discovery"], default="strict")
    parser.add_argument("--output-dir", default=".cache/qwen-hybrid/runs")
    for name, default in (("tensor_parallel_size", 1), ("data_parallel_size", 1), ("pipeline_parallel_size", 1),
                          ("ngpu", 1), ("sequence_length", 128), ("micro_batch_size", 1),
                          ("global_batch_size", 8), ("iterations", 4), ("vram_mib", 143771)):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=default)
    parser.add_argument("--gradient-accumulation", type=int)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    try:
        return run(args)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
