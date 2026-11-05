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

Measured on the hardware above (mean of 5 repetitions):

| Workload | Result | Why |
|---|---|---|
| W01 GPU | S2 ≈ S1 (~1.0×), both at ~236 GB/s | Already at ~92% of DRAM bandwidth. Memory-bound, so faster code can't help. |
| W02 GPU | ~28× faster (43.3 → 1.55 ms) | Restructuring the reduction keeps the GPU busy. |
| W03 CPU | ~121× faster (0.71 → 86 GFLOP/s) | Cache blocking × threads × 8-wide FMA |
| W03 GPU | ~3.5× faster (2.83 → 0.80 ms) | Shared-memory reuse, then Tensor Cores (`HMMA` verified in SASS) |
| W04 | *in progress, see status* | Fusion removes the N×N intermediate from DRAM |

## Status

| Phase | Content | State |
|---|---|---|
| W01–W03 | benchmarks, profiling, analysis | done |
| W04 P0 | harness, double-precision reference, correctness tests | done |
| W04 P1 | CPU attention S1–S3 | planned |
| W04 P2 | GPU S1, S2a unfused, cuBLAS reference | planned |
| W04 P3 | GPU S2b fused kernel | planned |
| W04 P4 | GPU S3 Tensor Core kernel | planned |
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

TODO: list the build targets for W01–W03 here once they are in the Makefile.

## Run a benchmark

Every W04 binary takes the same flags and prints a summary plus `Correct : YES|NO`:

```bash
./build/<W04 binary> --n 1024 --bh 8 --d 64 --iters 20 --reps 5 --warmup 2 --out results/run.csv
```

Results are written as CSV, one row per repetition, including the git commit and a timestamp.

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
