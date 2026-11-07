#!/usr/bin/env python3
"""Aggregate complete unrounded rank series; compare only matched calibration cases."""
import argparse
import json
import math
from pathlib import Path
from qwen_hybrid_config import HybridConfig, ModelSpec
from qwen_hybrid_control import expected_events


def aggregate(records, run_id, world):
    ranks = {}
    for record in records:
        rank = record.get("rank")
        if type(rank) is not int or not 0 <= rank < world or rank in ranks:
            raise ValueError("duplicate/invalid rank metric")
        if record.get("run_id") != run_id or record.get("world_size") != world or record.get("version") != 1:
            raise ValueError("stale/incompatible rank metric")
        values = record.get("durations")
        if not isinstance(values, list) or len(values) < 2 or record.get("iterations") != len(values) or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("complete positive unrounded iteration series required")
        if type(record.get("tokens_per_iteration")) is not int or record["tokens_per_iteration"] <= 0:
            raise ValueError("positive global token count required")
        ranks[rank] = record
    if set(ranks) != set(range(world)):
        raise ValueError("missing rank metrics")
    first = ranks[0]
    if any((r["iterations"], r["tokens_per_iteration"]) != (first["iterations"], first["tokens_per_iteration"]) for r in ranks.values()):
        raise ValueError("rank workload mismatch")
    maxima = [max(r["durations"][index] for r in ranks.values()) for index in range(1, first["iterations"])]
    seconds = sum(maxima) / len(maxima)
    return dict(steady_max_rank_seconds=maxima, mean_seconds=seconds,
                tokens_per_second=first["tokens_per_iteration"] / seconds,
                iterations=first["iterations"], tokens_per_iteration=first["tokens_per_iteration"],
                discarded_iterations=[0], world_size=world, memory_validated=False)


def read_records(paths):
    result = []
    for path in paths:
        for line in path.read_text().splitlines():
            if "HYBRID_METRICS " in line:
                result.append(json.loads(line.split("HYBRID_METRICS ", 1)[1]))
    return result


def report(bundle):
    """Bundle explicitly archives matched workload/software/hardware/topology policies.

    Replay status is native status.json. Hardware status is an explicit completed
    actual-run record, with rank metrics from actual execution, never discovery.
    """
    required = {"run_id", "world_size", "kind", "status", "records", "match"}
    if set(bundle) not in (required, required | {"runner"}) or bundle["kind"] not in ("replay", "hardware"):
        raise ValueError("invalid timing bundle")
    match = bundle["match"]
    if set(match) != {"workload", "software", "gpu", "gpus_per_host", "network", "cpu_time_policy"} or any(not value for value in match.values()):
        raise ValueError("complete workload/software/hardware/topology assumptions required")
    policy = {"replay": "ignored", "hardware": "wall_clock_elapsed"}
    if match["cpu_time_policy"] != policy:
        raise ValueError("CPU policy must explicitly separate ignored replay CPU from actual wall clock")
    try:
        workload_values = dict(match["workload"])
        workload_values["model"] = ModelSpec(**workload_values["model"])
        workload = HybridConfig(**workload_values)
    except (TypeError, KeyError) as error:
        raise ValueError("archived full HybridConfig workload required") from error
    if workload.world_size != bundle["world_size"]:
        raise ValueError("workload TP*DP*PP/world mismatch")
    placement = match["gpus_per_host"]
    if not isinstance(placement, list) or any(type(n) is not int or n <= 0 for n in placement) or sum(placement) != workload.world_size:
        raise ValueError("explicit per-host GPU placement must cover world")
    if type(bundle["world_size"]) is not int or bundle["world_size"] <= 0:
        raise ValueError("positive integer report world required")
    status = bundle["status"]
    if status.get("version") != 1 or status.get("run_id") != bundle["run_id"] or status.get("phase") != "complete" or status.get("valid") is not True or status.get("expected_world") != bundle["world_size"]:
        raise ValueError("complete valid matching all-rank status required")
    if bundle["kind"] == "replay":
        if status.get("mode") != "strict":
            raise ValueError("discovery is not an accepted timing result")
        runner = bundle.get("runner", {})
        codes = runner.get("worker_exit_codes")
        if runner.get("run_id") != bundle["run_id"] or runner.get("valid") is not True or runner.get("exit_code") != 0 or runner.get("error") or runner.get("cleanup_errors") or not isinstance(codes, list) or len(codes) != len(placement) or any(type(code) is not int or code != 0 for code in codes):
            raise ValueError("successful matching all-host runner evidence required")
        misses = status.get("timing_misses", {})
        if status.get("errors") != [] or status.get("malformed_messages") != 0 or set(misses) != {"compute", "memcpy", "flash_attn", "gdn", "malformed_gdn"} or any(type(n) is not int or n != 0 for n in misses.values()):
            raise ValueError("native errors/missing timing evidence")
        ranks = status.get("ranks", {})
        if set(ranks) != {str(rank) for rank in range(workload.world_size)}:
            raise ValueError("all native ranks required")
        categories = status.get("categories")
        if not isinstance(categories, dict) or any(categories.get(name) for name in ("unmapped_compute", "malformed_compute", "unmapped_device_only")):
            raise ValueError("unsupported/malformed operator categories invalidate report")
        hosts = set()
        placement_counts = {}
        iterations = bundle["records"][0]["iterations"] if bundle["records"] else 0
        for index, rank in ranks.items():
            if any(rank.get(field) is not True for field in ("started", "finished", "exited", "observer_started", "observer_finished")):
                raise ValueError("native rank lifecycle/observer incomplete")
            host = rank.get("host", {})
            identity = (host.get("hostname"), host.get("pid"))
            if not isinstance(identity[0], str) or not identity[0] or type(identity[1]) is not int or identity[1] <= 0 or identity in hosts:
                raise ValueError("invalid/duplicate native rank identity")
            hosts.add(identity)
            placement_counts[identity[0]] = placement_counts.get(identity[0], 0) + 1
            pp_rank = int(index) // (workload.tensor_parallel_size * workload.data_parallel_size)
            expected = expected_events(workload, pp_rank, iterations)
            actual = rank.get("actual", {})
            if rank.get("expected") != expected or any(type(n) is not int for n in rank.get("expected", {}).values()) or set(actual) - set(expected) or any(type(actual.get(name, 0)) is not int or actual.get(name, 0) != count for name, count in expected.items()):
                raise ValueError("native fused count/workload mismatch")
        if sorted(placement_counts.values()) != sorted(placement):
            raise ValueError("native host placement contradicts archived topology")
    result = aggregate(bundle["records"], bundle["run_id"], bundle["world_size"])
    if result["tokens_per_iteration"] != workload.tokens_per_iteration:
        raise ValueError("rank token count contradicts archived workload")
    return dict(result, effective_cpu_time_policy=policy[bundle["kind"]], run_id=bundle["run_id"], kind=bundle["kind"], match=match)


def compare(replay, hardware):
    if replay["kind"] != "replay" or hardware["kind"] != "hardware" or replay["match"] != hardware["match"] or any(replay[name] != hardware[name] for name in ("world_size", "iterations", "tokens_per_iteration")):
        raise ValueError("calibration requires identical workload/software/hardware/topology and iterations")
    error = abs(replay["mean_seconds"] / hardware["mean_seconds"] - 1)
    return dict(relative_time_error=error, tolerance=0.10, passed=error <= 0.10,
                replay=replay, hardware=hardware, memory_validated=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--status", type=Path)
    parser.add_argument("--runner", type=Path, help="required successful runner.json for replay")
    parser.add_argument("--logs", nargs="+", type=Path)
    parser.add_argument("--match", type=Path, help="archived observed workload/software/hardware/topology metadata")
    parser.add_argument("--kind", choices=["replay", "hardware"])
    parser.add_argument("--save-bundle", type=Path)
    parser.add_argument("--hardware-bundle", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.bundle:
        bundle = json.loads(args.bundle.read_text())
    else:
        if not all((args.status, args.logs, args.match, args.kind)):
            parser.error("supply --bundle or --status --logs --match --kind")
        status = json.loads(args.status.read_text())
        bundle = dict(run_id=status["run_id"], world_size=status["expected_world"],
            kind=args.kind, status=status, records=read_records(args.logs),
            match=json.loads(args.match.read_text()))
        if args.runner:
            bundle["runner"] = json.loads(args.runner.read_text())
    result = report(bundle)
    if args.save_bundle:
        args.save_bundle.parent.mkdir(parents=True, exist_ok=True)
        args.save_bundle.write_text(json.dumps(bundle, indent=2) + "\n")
    if args.hardware_bundle:
        result = compare(result, report(json.loads(args.hardware_bundle.read_text())))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
