#!/usr/bin/env python3
"""Prepare or execute debug then full-depth 9B replay cases with explicit host GPUs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

CASES = ((1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2), (2, 2, 2))


def commands(args):
    hosts = [int(value) for value in args.gpus_per_host.split(",")]
    if len(hosts) != len(CASES):
        raise ValueError("give five explicit GPUs-per-host values, one per case")
    result = []
    for preset in (["debug", "9b"] if args.preset == "both" else [args.preset]):
        for (tp, dp, pp), ngpu in zip(CASES, hosts):
            if ngpu <= 0 or tp * dp * pp % ngpu:
                raise ValueError("each case world must divide its explicit GPUs-per-host")
            case = f"{preset}-tp{tp}-dp{dp}-pp{pp}"
            command = [sys.executable, str(Path(__file__).with_name("hybrid_run.py")),
                "--image", args.image, "--perf-db", args.perf_db, "--preset", preset,
                "--mode", args.mode, "--tensor-parallel-size", str(tp),
                "--data-parallel-size", str(dp), "--pipeline-parallel-size", str(pp),
                "--ngpu", str(ngpu), "--iterations", str(args.iterations),
                "--output-dir", str(args.out / case)]
            result.append(dict(case=case, gpus_per_host=ngpu, hosts=tp * dp * pp // ngpu, command=command))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perf-db", required=True)
    parser.add_argument("--image", default="phantora:hybrid")
    parser.add_argument("--preset", choices=["debug", "9b", "both"], default="debug")
    parser.add_argument("--mode", choices=["discovery", "strict"], default="strict")
    parser.add_argument("--gpus-per-host", required=True, help="five comma-separated explicit case values, e.g. 1,2,1,1,4")
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.iterations < 2: parser.error("at least two iterations required")
    try: matrix = commands(args)
    except ValueError as error: parser.error(str(error))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "matrix.json").write_text(json.dumps(dict(cases=matrix, memory_validated=False,
        topology="virtual replay topology; match actual hardware explicitly for calibration"), indent=2) + "\n")
    for case in matrix:
        print(json.dumps(case), flush=True)
        if args.execute:
            with (args.out / (case["case"] + ".log")).open("x") as log:
                process = subprocess.run(case["command"], stdout=log, stderr=subprocess.STDOUT)
            if process.returncode: return process.returncode
    return 0


if __name__ == "__main__": raise SystemExit(main())
