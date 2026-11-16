# SPECTRA: Sensor Provenance and Execution Through Recursive Arguments

This repository contains the implementation of the SPECTRA-CAM circuits and the harness used for
the paper's performance evaluation. The prover is written in Rust on arecibo's SuperNova
implementation. Its four step circuits prove, in order, dead pixel correction, digital gain,
demosaicing and white balance.



## Requirements

- Docker with Compose v2. The image provides CUDA 12.8.2 on Ubuntu 24.04, Rust 1.97.1 and the
  pinned Python packages, so nothing else is needed on the host.
- For GPU runs, an NVIDIA GPU and the NVIDIA Container Toolkit.
- Bash on the host for the sweep scripts.

The arecibo and pasta-msm forks are included under `src/supernova/third-party/`, so there is
nothing else to fetch.

## Quick start

From the repository root:

```sh
./scripts/dev.sh build                                  # build the image once
./scripts/dev.sh pipeline --width 32 --height 32        # input frames for a 32x32 image
./scripts/dev.sh check                                  # check every step, no folding
./scripts/dev.sh benchmark --batch-size 2               # setup, prove, verify, one CSV row
```

On CPU, the 32x32 benchmark takes about 15 s and 150 MiB of RAM.

Add `--gpu` right after `dev.sh` to run any command with GPU access, for example
`./scripts/dev.sh --gpu benchmark`. Both modes use the same image and the same compiled binary,
which has CUDA support built in. In the default mode the GPU is hidden from the container, so
pasta-msm computes every MSM on the CPU. In GPU mode pasta-msm runs MSMs above its internal size
threshold on the GPU, and everything else stays on the CPU.

## Commands

| Command | What it does |
|---|---|
| `build` | Builds the Docker image. |
| `pipeline [--width W] [--height H]` | Writes the synthetic input frame and every stage output to `scripts/python/out/`. The default size is 1920x1080, and both dimensions must be even. |
| `check` | Runs every step in a test constraint system without folding. It reports the first unsatisfied constraint, or that all steps and the verifier's evidence checks hold. |
| `run` | Proves and verifies one frame. Public parameters are cached in `src/supernova/pp/`. |
| `benchmark [--batch-size B] [--benchmark-output PATH]` | Like `run`, but with a fresh setup and timings. It appends one row to `PATH` (default `benchmarks/supernova.csv`) if the proof verifies. |

`run` and `benchmark` prove the frame most recently written by `pipeline`. The batch size `B` is
the number of rows each step handles; the frame height must be divisible by `2B`.

## Benchmark output

Each `benchmark` row contains:

- the frame size, the number of steps, `B`, and the constraints and R1CS variables per step for
  each stage;
- wall-clock times for preparation, setup, proving (total and per stage), recursive
  verification, the verifier's evidence check, and the whole run;
- peak resident memory (`VmHWM`, in MiB) of the whole process; GPU memory is not included;
- run metadata: UTC time, git commit, branch, clean or dirty worktree, CPU model and available
  threads, OS, architecture, GPU model and GPU memory.

The proving time in the paper is the `proving_time_s` column. It covers the whole proving loop
and excludes preparation, setup and verification.

A CSV whose header differs from the current one is rejected, so use a new output path after
changing the columns.

## Reproducing the evaluation

The paper's results come from `scripts/sweep_sizes.sh`. For each frame size it generates the input
frame once, then benchmarks every batch size on CPU and on GPU in random order:

```sh
BENCH_CPUSET=0-127 scripts/sweep_sizes.sh ServerC              # every size of tab:sizes
BENCH_CPUSET=0-5   scripts/sweep_sizes.sh Jetson 1280x720      # one size
scripts/sweep_sizes.sh ServerA --cpu-only 64x64 128x128        # CPU only
```

- Without sizes, the script runs the classifier sizes 16x16 to 224x224 in steps of 16, then
  640x480, 1280x720, 1920x1080, 2048x1080 (2K) and 3840x2160 (4K).
- `BENCH_CPUSET` sets the CPUs the container may use. The default is `0-7`. The paper uses every
  hardware thread of each device, so set it to `0-<threads-1>`, e.g. `0-5` on the Jetson, `0-7`
  on Server A, `0-47` on Server B and `0-127` on Server C.
- `REPETITIONS` sets the number of runs per batch size and mode. The default is 5, and the paper
  reports medians over 3 runs (`REPETITIONS=3`).
- `SWEEP_B` lists the batch sizes. The default is `"1 2 4 6 8 10 12 15 18"`, and batch sizes that
  do not divide the height are skipped. 4K takes hours per run, so trim the list there, e.g.
  `SWEEP_B="1 4 10"`.

Results go to `benchmarks/supernova/<MACHINE>/sweep/`, one CSV per size and mode, named
`<W>x<H>-{cpu,gpu}-<commit>.csv`. Each size also gets `collection.log` and, if any run failed,
`failures.log`. In a copy without git history (a downloaded archive), `<commit>` and the git
columns of each row read `unknown`. A run that fails (for example by running out of memory)
leaves no row, and the sweep continues.

To benchmark a single size, use `scripts/collect_supernova_benchmarks.sh --width W --height H
[--out-dir DIR] [--cpu-only]`, which `sweep_sizes.sh` calls for each size.

### Tables and figures

Rows of `tab:sizes` (median proving time at the fastest `B`), one sweep directory per machine in
column order:

```sh
python3 scripts/sizes_table.py benchmarks/supernova/{Jetson,ServerA,ServerB,ServerC}/sweep
```

Run with `--mem-budget-mib N`, it picks the fastest `B` whose median peak memory fits in `N` MiB.

Proving time and memory against `B`, CPU and GPU runs of one size:

```sh
python3 scripts/plot_supernova_cpu_gpu.py \
    --cpu benchmarks/supernova/Jetson/sweep/1280x720-cpu-<commit>.csv \
    --gpu benchmarks/supernova/Jetson/sweep/1280x720-gpu-<commit>.csv \
    --out charts/jetson-720p
```

Both scripts need only Python 3; the plot also needs matplotlib and numpy. They also run inside the
container: `docker compose -f scripts/compose.yaml run --rm rust python scripts/sizes_table.py ...`.

## Docker through a custom command

Some managed servers do not give users the standard Docker socket. If `docker info` fails,
`dev.sh` runs the command in `DOCKER_ALIAS_CMD` instead:

```sh
export DOCKER_ALIAS_CMD="sudo docker -H unix:///var/run/docker-<user>.sock"
./scripts/dev.sh run
```

The value must be a plain command with space-separated arguments, without pipes, redirects or
other shell syntax.

## License

MIT, see [LICENSE](LICENSE). The forks under
`src/supernova/third-party/` keep their own licenses.
