# CPU vs GPU Performance: Profiling and Optimization

How far does each optimization take a workload on a laptop CPU and GPU, and *why*?

This repo implements four workloads in progressively optimized stages. Each stage is benchmarked, checked against a reference for correctness, and explained with hardware profilers (NVIDIA Nsight Compute, Intel VTune) and a Roofline model.

| | Hardware |
|---|---|
| CPU | Intel Core i7-13650HX: 6 P-cores + 8 E-cores, 20 threads, AVX2/FMA |
| GPU | NVIDIA GeForce RTX 4060 Laptop: Ada, sm_89, 24 SMs, 8 GB GDDR6, Tensor Cores |

## Workloads and optimization stages

Every workload follows the same ladder:
- **S1 Baseline:** the simplest correct code.
- **S2 Software optimization:** parallelism, tiling, coalescing, better reductions, fusion.
- **S3 Hardware acceleration:** AVX2/FMA on the CPU, Tensor Cores on the GPU.

| Workload | What it stresses | CPU stages | GPU stages |
|---|---|---|---|
| **W01 Vector Add** `C = A + B` | memory bandwidth | scalar → OpenMP → AVX2 | naive → coalesced |
| **W02 Dot Product** `s = Σ aᵢbᵢ` | reductions | scalar → OpenMP → AVX2/FMA | naive → hierarchical block reduction |
| **W03 GEMM** `C = A × B` | compute + data reuse | naive → cache-blocked + OpenMP → AVX2/FMA | naive → shared-memory tiling → TF32 Tensor Cores |
| **W04 Attention** `softmax(QKᵀ/√d)·V` | memory traffic and kernel fusion | naive → fused + blocked + OpenMP → AVX2/FMA | naive → tiled (unfused) → fused FlashAttention-style → Tensor Cores |

## Key results so far

Measured on the hardware above (mean of 5 repetitions). W04 CPU numbers come from one session on a laptop that reached ~97 °C under load, so differences below ~10–15% between experiments are not significant. W04 GPU numbers are medians of 3 interleaved rounds with the GPU at its default 55 W power limit, so stages are compared at equal power, not at equal clock:

| Workload | Result | Why |
|---|---|---|
| W01 GPU | S2 ≈ S1 (~1.0×), both at ~236 GB/s | Already at ~92% of DRAM bandwidth. Memory-bound, so faster code can't help. |
| W02 GPU | ~28× faster (43.3 → 1.55 ms) | Restructuring the reduction keeps the GPU busy. |
| W03 CPU | ~121× faster (0.71 → 86 GFLOP/s) | Cache blocking × threads × 8-wide FMA |
| W03 GPU | ~3.5× faster (2.83 → 0.80 ms) | Shared-memory reuse, then Tensor Cores (`HMMA` verified in SASS) |
| W04 CPU | ~62× faster at N=1024 (285.8 → 4.59 ms; 7.5 → 469 GFLOP/s) | Fusion + blocking (S1→S2 28.8×), then 8-wide FMA and a vector `exp` (S2→S3 2.2×), on 20 threads |
| W04 GPU (unfused) | S1→S2a ~3.4× at N=1024 (7.63 → 2.23 ms); cuBLAS-based reference 0.75 ms | Tiled GEMMs and a warp-per-row softmax; S2a's GEMMs are limited by shared-memory loads. With cuBLAS GEMMs the N×N softmax becomes the largest kernel (46% at N=8192): the cost that fusion removes |
| W04 GPU (fused) | S2a→S2b 2.0–2.1× (32×32 tile) and 2.9–3.0× (64×64) at N=2048–8192; at N=4096: 38.75 → 18.77 / 13.33 ms. S2b 64×64 matches or beats the cuBLAS-based unfused reference from N=2048 | One kernel, no N×N intermediate: measured DRAM traffic drops ~33–42× at N=4096 (≈2.2 GB → 52–68 MB) and operational intensity rises from ~15 to 500–650 FLOP/B. The speedup also includes register-tiled inner GEMMs (fewer shared-memory loads per FMA), so it is not fusion alone |
| W04 GPU (Tensor Cores) | S2b 64×64 → S3: FP16 5.0–5.2×, TF32 1.7–1.8× at N=2048–8192; at N=4096: 13.36 → 2.56 ms (FP16, 13.4 TFLOP/s) / 7.57 ms (TF32, 4.5 TFLOP/s). S1 → S3 FP16 52.6× at N=8192. Converting Q, K, V to FP16 adds 2–6% | Both inner GEMMs on the Tensor Cores with FP32 accumulation; max relative error ≤ 3.7e-4 against a double-precision reference. Nsight Compute: tensor pipe 22% busy, `exp` unit 6%: the kernel is limited by latency at low occupancy (2 warps per scheduler), not by any execution unit. TF32 is ~3× slower than FP16 (half the Tensor Core rate, more instructions, half the occupancy) |

## Status

| Phase | Content | State |
|---|---|---|
| W01–W03 | benchmarks, profiling, analysis | done |
| W04 P0 | harness, double-precision reference, correctness tests | done |
| W04 P1 | CPU attention S1–S3, compiler-flag study, 1–20 thread sweep, tile-size choice | done: S1→S3 62× at N=1024 (7.5 → 469 GFLOP/s); default tiles S2 64×64, S3 32×64 from a temperature-logged rerun |
| W04 P2 | GPU S1, S2a unfused, cuBLAS reference | done: S1→S2a 3.3–4.3×; S2a reaches 31–38% of the cuBLAS-based reference (N=512–8192), with the GPU at its 55 W power limit |
| W04 P3 | GPU S2b fused kernel | done: one FlashAttention-2-style FP32 kernel (online softmax, no N×N buffer, peak device memory linear in N); faster than S2a at every N (1.5–2.1× at 32×32, 2.1–3.0× at 64×64, N=512–8192); DRAM bytes measured with Nsight Compute: ~21–42× fewer than S2a at N=2048–4096 |
| W04 P4 | GPU S3 Tensor Core kernel | done: FP16 (WMMA m16n16k16, `HMMA.16816.F32`) and TF32 (m16n16k8, `HMMA.1684.F32.TF32`) variants of the fused kernel, FP32 softmax, accumulation and output, 64×64 tiles; FP16 5.0–5.2× and TF32 1.7–1.8× faster than the FP32 fused kernel at the same tile (N=2048–8192); Nsight Compute shows no saturated pipe (tensor pipe 22% busy) — the kernel is latency-bound at low occupancy |
| W04 P5 | autotuning | planned |
| W04 P6 | profiling and plots | planned |
| W04 P7 | demo integration | planned |

## Requirements

- Linux (tested on Ubuntu 24.04)
- NVIDIA GPU with compute capability 8.9, or change `sm_89` in the Makefile
- CUDA Toolkit 12+ (`nvcc`), `g++` with OpenMP, GNU `make`
- Optional: Nsight Compute, Nsight Systems, Intel VTune (profiling); Python 3 + matplotlib (plots)

## Build and test

```bash
git clone https://github.com/<you>/GPU-CPU-Performance-Profiling_and-Optimization.git
cd GPU-CPU-Performance-Profiling_and-Optimization

make w04                               # build W04 binaries into build/
bash tests/test_w04_correctness.sh     # all stages vs a double-precision reference
```

W01–W03 binaries (one per stage, named after the source file):

```bash
make w01        # vector add:  CPU S1–S3, GPU S1–S2
make w02        # dot product: CPU S1–S3, GPU S1–S2
make w03        # GEMM:        CPU S1–S3, GPU S1–S3
make legacy     # all of the above
```

Each W01–W03 binary is built with the flags it records in its own CSV output. CPU S1 stages use `-O0 -g`. W02/W03 CPU S2/S3 use `-O3` with auto-vectorization turned off (`-fno-tree-vectorize -fno-tree-slp-vectorize`), so that the AVX2 stage is the only vectorized one. W03 CPU S2/S3 use a 128×128 tile.

W04 compiler-flag study binaries (`-O2`, `-O3 -march=native`, `-O3 -ffast-math`, `-O3 -march=native -ffast-math`):

```bash
make w04-flagstudy                     # into build/flagstudy/
```

## Run a benchmark

Every W04 binary takes the same flags and prints a summary plus `Correct : YES|NO`:

```bash
./build/<W04 binary> --n 1024 --bh 8 --d 64 --iters 20 --reps 5 --warmup 2 --out results/run.csv
```

Results are written as CSV, one row per repetition, including the git commit and a timestamp.
A build made outside a git checkout records `git_sha=nogit` and prints a warning.

CPU tunable stages also accept `--br` and `--bc` (query and key/value tile sizes). The CPU experiments (N sweep, tile grid, 1–20 thread sweep, compiler-flag study) run as one session:

```bash
bash scripts/w04_sweep.sh cpu-all      # writes results/raw/W04_ATTENTION/<YYYYMMDD-HHMMSS>/
```

GPU N sweep (S1, S2a, cuBLAS reference, S2b fused at 32×32 and 64×64 tiles, S3 Tensor Core FP16 and TF32; 3 interleaved rounds with GPU clock, power and temperature logged), and a per-stage summary of a run folder:

```bash
bash scripts/w04_sweep.sh gpu-nsweep
python3 tools/w04_summary.py results/raw/W04_ATTENTION/<run folder>
```

Measured DRAM bytes and operational intensity from Nsight Compute raw-page CSV exports (one export per kernel, named `<stage>_<kernel or tile>_n<N>.raw.csv`):

```bash
python3 tools/w04_e3_oi.py <folder with .raw.csv exports>
```

S2b tile size is selectable (`--br`, `--bc` in {16, 32, 64}), e.g.:

```bash
build/W04_ATTENTION_GPU_S2b_SWOPT_FUSED --n 1024 --br 64 --bc 64
```

The Tensor Core stage S3 has two binaries, one per input precision (default tile 64×64, same flags). The FP16 binary converts Q, K and V to half before the timed loop and reports that conversion time separately (`prepare` line, CSV column `prep_ms`):

```bash
build/W04_ATTENTION_GPU_S3_HWACC_TENSOR_CORE_FP16 --n 1024
build/W04_ATTENTION_GPU_S3_HWACC_TENSOR_CORE_TF32 --n 1024
```

## Repository layout

```
src/cpu/<workload>/   CPU stages (one file per stage)
src/gpu/<workload>/   CUDA stages (one file per stage)
src/common/           shared CLI, reference implementation, CSV writer
tests/                correctness tests
scripts/  tools/      sweep, profiling and plotting helpers
docs/                 design notes (W04 plan)
```

File names encode the stage: `W04_ATTENTION_GPU_S2b_SWOPT_FUSED.cu` = workload 4, GPU, stage 2b, software optimization, fused.

## Method notes

- **Correctness first:** every stage is compared with a double-precision reference (absolute and relative error). W04 includes a large-logits input that breaks an unstable softmax.
- **Timing:** 2 warm-ups, 20 timed iterations, 5 repetitions, reported as mean ± SD. Stages are only compared within the same session.
- **Profiler numbers explain, benchmarks measure:** profiler runs replay kernels, so their durations are never used as speedups.
- **Roofline:** points use measured operational intensity (FLOPs ÷ bytes). Fusion in W04 moves points *right* by cutting bytes; Tensor Cores move them *up*.

## References

- S. Boehm, *How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance*
- T. Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, NeurIPS 2022
- S. Williams, A. Waterman, D. Patterson, *Roofline*, CACM 2009
