# Setup, reproduction and live-demo guide

← [Back to the README](../README.md)

Everything you need to **build**, **verify**, **reproduce** the results and **run the live demo**.

**Contents:** [Requirements](#requirements) · [Build and verify](#1-build-and-verify) · [Run one benchmark](#2-run-one-benchmark) · [Reproduce all results](#3-reproduce-all-results) · [Live demo](#4-live-demo) · [W04 experiments](#5-w04-experiments) · [Regenerate figures and video](#6-regenerate-the-readme-figures-and-video) · [Measurement notes](#measurement-notes) · [Versions](#versions)

---

## Requirements

| | Needed |
|---|---|
| **OS** | Linux (tested on Ubuntu 24.04) |
| **GPU** | NVIDIA, compute capability **8.9** (for another GPU, change `sm_89` in the `Makefile`) |
| **Toolchain** | CUDA Toolkit 12+ (`nvcc`, cuBLAS), `g++` with OpenMP, GNU `make`, Python 3 |
| **Optional** | matplotlib (figures), ffmpeg (video), Nsight Compute / Nsight Systems / Intel VTune (profiling) |

Tested configuration: i7-13650HX · RTX 4060 Laptop (55 W) · g++ 13.3 · CUDA 13.4 · driver 595.99.

## 1. Build and verify

```bash
git clone https://github.com/Fe3w/GPU-CPU-Performance-Profiling_and-Optimization.git
cd GPU-CPU-Performance-Profiling_and-Optimization

make legacy w04                        # W01–W03 and W04 binaries, into build/
bash tests/test_w04_correctness.sh     # every W04 stage vs a double-precision reference (~1 min)
```

- `make w01`, `make w02`, `make w03`, `make w04` build **one workload** each.
- Each binary is named after its source file: `src/gpu/W03_GEMM/W03_GEMM_GPU_S3_HWACC_TENSOR_CORE.cu` → `build/W03_GEMM_GPU_S3_HWACC_TENSOR_CORE`.
- **File names encode the stage:** `W04_ATTENTION_GPU_S2b_SWOPT_FUSED` = workload 4 · GPU · stage 2b · software optimization · fused.
- CPU S1 stages use `-O0 -g`; CPU S2/S3 use `-O3`. W02/W03/W01-S2 turn auto-vectorization off, so the AVX2 stage is the vectorized one.

## 2. Run one benchmark

Every binary prints a summary, writes one CSV file and reports correctness (`Correct : YES|NO`, or a `correct` column in the CSV).

**W04: one set of named flags for every binary**

```bash
mkdir -p results/run
build/W04_ATTENTION_GPU_S3_HWACC_TENSOR_CORE_FP16 --n 2048 --bh 8 --d 64 \
    --iters 20 --reps 5 --warmup 2 --out results/run/w04_s3_fp16.csv
OMP_PROC_BIND=close OMP_PLACES=cores \
build/W04_ATTENTION_CPU_S3_HWACC_AVX2 --n 1024 --bh 8 --d 64 \
    --iters 20 --reps 5 --warmup 2 --out results/run/w04_cpu_s3.csv
```

Tunable W04 stages also take `--br` / `--bc` (tile sizes) and, on the GPU, `--warps`. Without them a binary uses the tuned defaults in `src/common/w04_tuned.hpp`.

**W01–W03: original command lines.** `N` = size, `I` = timed iterations, `R` = repetitions, `W` = warm-ups, `T` = threads.

```bash
N=33554432
# W01: CPU named flags (threads from OMP_NUM_THREADS), GPU positional N I R W [BLOCK] FILE
OMP_NUM_THREADS=6 OMP_PROC_BIND=close OMP_PLACES=cores \
  build/W01_VECTOR_ADD_CPU_S2_SWOPT_OMP --size $N --iterations 20 --repetitions 5 --warmup 2 --output results/run/w01_cpu_s2.csv
build/W01_VECTOR_ADD_GPU_S1_BASE_NAIVE $N 20 5 2 results/run/w01_gpu_s1.csv
build/W01_VECTOR_ADD_GPU_S2_SWOPT_COALESCED $N 20 5 2 128 results/run/w01_gpu_s2.csv
# W02: CPU positional N I R W [T] FILE; GPU S1 named flags, GPU S2 positional
build/W02_DOT_PRODUCT_CPU_S1_BASE $N 20 5 2 results/run/w02_cpu_s1.csv
OMP_PROC_BIND=close OMP_PLACES=cores build/W02_DOT_PRODUCT_CPU_S3_HWACC_AVX2_FMA $N 20 5 2 6 results/run/w02_cpu_s3.csv
build/W02_DOT_PRODUCT_GPU_S1_BASE_NAIVE --size $N --iterations 20 --repetitions 5 --warmup 2 --output results/run/w02_gpu_s1.csv
build/W02_DOT_PRODUCT_GPU_S2_SWOPT_BLOCK_REDUCE $N 20 5 2 results/run/w02_gpu_s2.csv
# W03: positional N I R W FILE (threads from OMP_NUM_THREADS)
OMP_NUM_THREADS=8 OMP_PROC_BIND=close OMP_PLACES=cores \
  build/W03_GEMM_CPU_S3_HWACC_AVX2_FMA 1024 20 5 2 results/run/w03_cpu_s3.csv
build/W03_GEMM_GPU_S3_HWACC_TENSOR_CORE 1024 20 5 2 results/run/w03_gpu_s3.csv
```

The exact command for **every** binary is in [`scripts/run_all.sh`](../scripts/run_all.sh).

## 3. Reproduce all results

**One command** builds everything, runs the W04 correctness test as a gate, then runs every stage of W01–W04 in one session into `results/raw/FINAL/<timestamp>/`:

```bash
bash scripts/run_all.sh               # full re-test, ~25 min
bash scripts/run_all.sh --smoke       # tiny sizes, ~2 min: checks every command works (not a measurement)
bash scripts/run_all.sh --only w01,w03
```

**Before a full run:** mains power, the Performance power mode, and no other GPU applications (browsers included).

What the script does for you:
- waits for the CPU to cool to ≤ 70 °C before each CPU stage
- gives each short W01–W03 GPU stage one untimed priming run
- logs GPU clocks and power every 250 ms
- refuses to record results from an uncommitted tree unless `ALLOW_DIRTY=1` is set

Regenerate the tables and figures from a run folder:

```bash
python3 tools/final_summary.py results/raw/FINAL/<timestamp>                 # tables, CSV and figure in the run folder
python3 tools/final_summary.py results/raw/FINAL/<timestamp> --publish-docs  # also docs/RESULTS_SUMMARY.md, docs/results/, docs/figures/final/
```

Comparison with earlier runs uses `docs/results/reference_values.csv`; changes over 10% are flagged and explained in `docs/results/deviation_notes.md`.

## 4. Live demo

`demo_kit/` runs the benchmarks live, prints a speedup scoreboard and draws race charts. It uses the binaries in `build/`, so it works from a clone after `make legacy w04`.

```bash
./demo_kit/demo.sh check              # preflight: binaries, GPU, profilers
./demo_kit/demo.sh race w04           # W04 GPU race, N = 2048 (~10 s); also: all, gpu, w01, w02, w03, w04cpu
ROUNDS=3 ./demo_kit/demo.sh race gpu  # 3 interleaved rounds, median over rounds
./demo_kit/demo.sh fusion --backup    # W04 DRAM bytes S2a vs S2b from the recorded Nsight Compute run
./demo_kit/demo.sh plot               # charts and tables for the newest race, into results/live/<timestamp>/
```

- `./demo_kit/demo.sh fusion` without `--backup`, and `./demo_kit/demo.sh ncu`, run Nsight Compute. That needs GPU performance counters: root, or the driver option `NVreg_RestrictProfilingOnlyToAdminUsers=0` with `NCU_SUDO=""`.
- Paths and tools are set in `demo_kit/demo.conf`; override any of them from the environment (`PROJECT=/other/checkout`, `NCU=/path/to/ncu`, `VTUNE=...`).
- Presenter's guide: [`demo_kit/DEMO_RUNBOOK.md`](../demo_kit/DEMO_RUNBOOK.md).

## 5. W04 experiments

```bash
bash scripts/w04_sweep.sh cpu-all        # CPU N sweep, tiles, 1–20 thread sweep, compiler-flag study (make w04-flagstudy first)
bash scripts/w04_sweep.sh gpu-nsweep     # GPU N sweep, 3 interleaved rounds, clocks logged
bash scripts/w04_sweep.sh gpu-autotune   # Br × Bc × warps screen + finals
bash scripts/w04_sweep.sh gpu-fusion     # fusion alone: S2a vs S2b with the same work per thread
python3 tools/w04_summary.py results/raw/W04_ATTENTION/<run folder>
python3 tools/plot_w04.py results/raw/W04_ATTENTION/<gpu run>   # figures into docs/figures/w04/
bash scripts/w04_profile.sh e3           # writes the Nsight Compute commands (they need root) into profiling/
bash scripts/w04_profile.sh e5           # Nsight Systems launch gaps
```

## 6. Regenerate the README figures and video

```bash
python3 tools/readme_figures.py        # hero_speedup.png, comparison.png, roofline.png, figure_values.csv → docs/figures/readme/
python3 tools/readme_video.py --poster docs/media/comparison_video_poster.png   # 72 s video → docs/media/ (needs ffmpeg)
python3 tools/readme_video.py --poster-only                                     # poster only
```

Both scripts read only `docs/results/final_results.csv` and `docs/figures/w04/plot_data.csv`. Every plotted value is written to `docs/figures/readme/figure_values.csv`.

---

## Measurement notes

- **Laptop power and clocks.** The GPU runs at its default 55 W limit, so each stage runs at its own clock (1600–2700 MHz SM); GPU ratios hold at equal power, not equal clock. Against the memory-bound cuBLAS reference, ratios change by ~30% at a locked SM clock. Memory-bound stages follow the memory clock (7001 or 8001 MHz). The CPU reaches 72–96 °C under load; long CPU runs can differ by up to ~24% between sessions. Clocks were not locked (that needs root).
- **Roofline ceilings are nominal**: vendor per-SM rates at the measured clock, not microbenchmarked.
- **One machine**: one laptop, one driver, one compiler. Other hardware gives other ratios.
- **W03 GPU S3 is not a tuned GEMM**: a 32×32-block WMMA kernel (4 warps of 16×16, TF32, 0.004% error vs double). cuBLAS-class kernels use larger register tiles, double buffering and asynchronous copies.
- **Earlier October records** that could not be reproduced within 10% (W02 GPU S2, W03 GPU S1/S2, W01 CPU S1) are listed with known causes in [`docs/RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md).

## Versions

| Tag | Content |
|---|---|
| `v1.1` | current: optimized W03 GPU S3 kernel, demo kit runnable from a clone, full re-test `20261106-151143` |
| `v1.0` | first release: re-test `20261106-141652` with the first W03 GPU S3 kernel; its run is kept in `Comparison final/runs/final/20261106-141652_previous/` |

`git checkout v1.0` gives the first release.
