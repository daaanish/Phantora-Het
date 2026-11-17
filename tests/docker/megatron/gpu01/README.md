# Running Phantora experiments on gpu01

Every gpu01 run starts from a pushed commit of this repository. A run records the
commit it used, so any result traces back to exact code, and code that is not
committed and pushed cannot be run by accident.

## The rule

1. Edit, commit and push on your own machine.
2. On gpu01: `git -C ~/Phantora/Phantora-Het pull`.
3. Run `run.sh`. It refuses if the clone has local changes or is not at its upstream commit.

Never edit files in `~/Phantora/Phantora-Het` or in `~/Phantora/builds` by hand.

## Layout on gpu01

| Path | What | Written by |
|---|---|---|
| `~/Phantora/Phantora-Het/` | clone of `daaanish/Phantora-Het`, branch `qwen-hybrid-layout` | `git pull` only |
| `~/Phantora/env/` | copies of the external dependencies (`causal-deps`, `upgraded-deps`, `cargo-home`) | `setup_env.sh`, once |
| `~/Phantora/builds/<commit>/` | `src/` (git archive of the commit), `dist/` (native simulator), `build.json`, logs | `build.sh`, once per commit |
| `~/Phantora/builds/torch-<pytorch commit>/` | torch built from the commit the `pytorch` submodule pins | `build.sh`, once per pin |
| `~/Phantora/runs/<date>-<name>-<commit8>/` | one directory per run | `run.sh` |
| `~/Phantora/Phantora/phantora-gpu01.sif` | base image: CUDA 12.8, Python 3.11, Megatron, Rust | unchanged |
| `~/Phantora/qh*/` | runs QH-02 to QH-07v, made before this workflow | kept as records; not used by it |

Builds and runs are never modified after they finish. `src/` and `dist/` are read-only.

## One-time setup

```bash
cd ~/Phantora
git clone -b qwen-hybrid-layout https://github.com/daaanish/Phantora-Het.git
Phantora-Het/tests/docker/megatron/gpu01/setup_env.sh
```

The clone does not need the `pytorch` submodule checked out; `build.sh` fetches only
what it needs. `setup_env.sh` copies the dependencies that used to live under `/tmp`
and never overwrites an existing copy.

## Running

```bash
cd ~/Phantora/Phantora-Het && git pull
tests/docker/megatron/gpu01/run.sh <name> --preset 0.8b --seq 8192 --base-db <db-dir>
```

- `<name>`: short label, e.g. `seq8192-layout`. The directory adds the date and commit.
- `--preset`: `0.8b` (24 layers) or `0.8b-l4` (4 layers); `debug` and `9b` also exist.
- `--base-db`: the timing database to start from. It is copied into the run, never changed.
- `--stages`: any of `discovery,profile,replay,real` (default `discovery,profile,replay`).
- `--mtp N`: train N multi-token-prediction layers (Qwen3.5 has 1). Default 0, as in every
  run before QH-07w; applies to the simulated and the real stages alike.
- Other options (`--mb`, `--ga`, `--tp`, `--iterations`, `--vram-mib`, `--sim-log`,
  `--expandable`) are listed at the top of `run.sh`. Defaults match the QH-07 sweeps:
  TP 2, micro-batch 1, accumulation 8, 4 iterations, 97,887 MiB simulated memory.

The first run of a new commit builds it first: about 2 minutes for the native
simulator, and about 5 minutes more if the `pytorch` pin changed. Later runs reuse it.

Run long jobs detached so they survive a dropped SSH session:

```bash
setsid nohup tests/docker/megatron/gpu01/run.sh <name> ... > ~/Phantora/runs/<name>.out 2>&1 < /dev/null &
```

`run.sh` prints the run directory first; `progress.log` inside it shows each stage.

### Stages

| Stage | Uses | Does |
|---|---|---|
| `discovery` | no GPU | simulates with `--base-db`; keys the DB lacks are charged 0 and listed in `discovery/perfdb.missing/` |
| `profile` | GPU 0 | times the missing keys (fused GDN keys with `gdn_profile.py`, `exp_out` separately, the rest with `bench.py`), copies the base DB to `db/` and appends every positive new timing |
| `replay` | no GPU | simulates again with `db/` (or `--base-db` if there was no profile stage); this is the Phantora step time |
| `real` | GPUs 0 and 1 | one real training run with the same configuration |

GPU stages wait until the GPUs are idle (no process, ≤16 MiB, 0% utilization),
checking once a minute for up to 8 hours.

### What a run leaves

```
commit.txt        the exact commit
config.json       every option, including the base DB path
build.json        build and torch provenance, test results of the build
progress.log      one line per stage with exit code and step time
discovery/        worker.log, status.json, runner.json, input-hashes.json, perfdb.missing/
profile/          keys/, measured*/, profile.log, bench*.log, gdn.log, merge.json
db/               base DB + newly profiled rows (feed it to later runs as --base-db)
replay/           as discovery, plus events.jsonl.gz (per-event charges) with --sim-log debug
real/             ranks/ (per-rank logs), trainer.log, mem.csv (device memory every 2 s)
summary.json      step times (mean of iterations 2..N, slower rank), validity, missing keys, % vs real
```

## Typical uses

Rerun a point after a simulator change, from that point's existing DB:

```bash
run.sh l4-24576 --preset 0.8b-l4 --seq 24576 --base-db ~/Phantora/qh07q-l4-sweep-3CPzh4/seq-24576/db
```

Check only that the current code reproduces a replay, with no profiling:

```bash
run.sh repro --preset 0.8b --seq 8192 --stages discovery,replay --base-db <db that already has every key>
```

A full point including the real run:

```bash
run.sh seq4096 --preset 0.8b --seq 4096 --base-db <db> --stages discovery,profile,replay,real
```

## Recording results

Copy the numbers you keep into `results/<study>/` with the run directory path
(`gpu01:/home/dathar/Phantora/runs/...`) and the commit, and add a row to
`QWEN_HYBRID_TRACKER.md`. Follow the rules in the tracker: never overwrite a number,
add a column or row when the method changes.

## Things to know

- **Torch version string.** The built torch reports `torch.version.git_version`
  `bbb19f8c…` (the base revision the code checks for provenance in `gdn_common.py`,
  `qwen_hybrid_runtime_check.py` and `gdn.rs`), not the fork commit. The fork commit
  (`daaanish/pytorch`, adds strides to the tracer) is in `builds/torch-*/torch.json`
  under `source_commit`, with the files taken from it. The QH-07u runs used a torch
  built the same way.
- **Torch builds are incremental.** `build.sh` copies the compiled PyTorch tree from
  the base image, replaces only the files that differ from the pinned commit, and stops
  if that would recompile CUDA code or touch more than 20 files. A larger PyTorch
  change needs a new base image.
- **One code path for all stages.** Simulation, profiling and real runs all use the
  build's torch and the commit's `tests/`. The tracer change only acts when Phantora
  enables the tracer, so it does not affect profiling or real runs.
- **Disk.** A native build is about 15 MB; a torch build about 1.5 GB; a 24-layer
  replay with `--sim-log debug` about 30 MB (the raw simulator log is aggregated, then
  deleted). Delete old builds with `chmod -R u+w <dir> && rm -rf <dir>`; runs keep their
  own `build.json`, so a deleted build can be rebuilt from the commit.
- **`--allow-unpushed`** lets you run a commit that is not on the upstream branch. Use
  it only for throwaway checks; the result is not reproducible from GitHub.

## Files here

| File | Role |
|---|---|
| `paths.sh` | every location above, in one place |
| `setup_env.sh` | one-time dependency copies |
| `build.sh` | builds a commit (native simulator + pinned torch) |
| `run.sh` | runs one configuration through the chosen stages |
| `simulate.py` | one GPU-free simulation (discovery or replay) |
| `profile_inside.sh`, `profile_keys.py` | profiling inside the container |
| `real_inside.sh` | real run inside the container |
| `merge.py` | appends new positive timings to the run's DB |
| `metrics.py` | step time from `HYBRID_METRICS` lines |
| `events_agg.py` | debug simulator log to per-event table |
| `verify_torch.py` | checks the container uses the build's torch |
| `netconfig.toml` | gpu01 network calibration (271.916 Gbit/s loopback) |
