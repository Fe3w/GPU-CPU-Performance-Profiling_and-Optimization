# Results summary — final paired re-test of W01–W04

| | |
|---|---|
| Run | `20261106-151143` (one session, `scripts/run_all.sh`) |
| Commit | `7f0e5f6cb244` (W04 build `7f0e5f6cb244`) |
| Date | 2026-11-06T15:12:18+05:30 |
| CPU | 13th Gen Intel(R) Core(TM) i7-13650HX (6 P + 8 E cores, 20 threads, AVX2/FMA), governor powersave, EPP performance, AC 1 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU (sm_89, 24 SMs, 8 GiB GDDR6 128-bit), driver 595.99.02, power limit 55.00 W |
| Toolchain | g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0 / Build cuda_13.4.r13.4/compiler.38657139_0 |
| Protocol | warmup=2 iters=20 reps=5 rounds(GPU)=3 cool-down<=70C (max 120s) |
| Sizes / threads | W01/W02 N=33554432  W03 N=1024  W04 N=1024 2048 --bh 8 --d 64; W01 6  W02 6  W03 8  W04 20  (OMP_PROC_BIND=close OMP_PLACES=cores) |
| GPU state (loaded samples) | SM clock median 2670 MHz (1785–2700), power median 33.4 W, max 67 °C (282 samples, 250 ms) |
| CPU state | package mean temperature per run 74–96 °C (cool-down to ≤ 70 °C before each run) |

Times are ms per iteration: mean ± SD over the 5 repetitions of a run (20 timed iterations each, after 2 warm-ups). GPU stages ran in 3 interleaved round(s): median of the round means, SD = mean within-round SD, spread = (max − min) / median of the round means. CPU stages ran once each. × vs S1 = t(S1)/t; × prev = t(previous stage)/t (S3_TF32's previous stage is S2b); saved = (1 − t/t(S1))·100. Δ earlier = change vs the earlier recorded run (`docs/results/reference_values.csv`); |Δ| > 10% is marked ⚠ and explained below.

## W01 Vector Add — GPU, N = 33,554,432

C = A + B, FP32, N = 33,554,432.

| Stage | | ms (mean ± SD) | GB/s | × vs S1 | × prev | saved | Correct | error vs double | Δ earlier | SM / mem MHz |
|---|---|---|---|---|---|---|---|---|---|---|
| S1 | naive, 256 threads/block | 1.796 ± 0.0020 (spread 0.1%) | 224.2 | **1.00×** | – | 0.0% | YES | – | +4.7% | 2685 / 8001 |
| S2 | coalesced, 128 threads/block | 1.796 ± 0.0014 (spread 0.1%) | 224.2 | **1.00×** | 1.00× | -0.0% | YES | – | +5.7% | 2685 / 8001 |

## W01 Vector Add — CPU, N = 33,554,432

C = A + B, FP32, N = 33,554,432.

| Stage | | ms (mean ± SD) | GB/s | × vs S1 | × prev | saved | Correct | error vs double | Δ earlier |
|---|---|---|---|---|---|---|---|---|---|
| S1 | scalar, -O0 | 79.942 ± 0.2988 | 5.0 | **1.00×** | – | 0.0% | YES | – | -20.7% ⚠ |
| S2 | OpenMP, 6 threads | 7.771 ± 0.0486 | 51.8 | **10.29×** | 10.29× | 90.3% | YES | – | -3.6% |
| S3 | OpenMP + AVX2, 6 threads | 7.819 ± 0.0245 | 51.5 | **10.22×** | 0.99× | 90.2% | YES | – | -1.7% |

Factored: S1→S2 10.29× = **10.3×**

## W02 Dot Product — GPU, N = 33,554,432

s = Σ aᵢbᵢ, FP32 inputs, N = 33,554,432.

| Stage | | ms (mean ± SD) | GB/s | × vs S1 | × prev | saved | Correct | error vs double | Δ earlier | SM / mem MHz |
|---|---|---|---|---|---|---|---|---|---|---|
| S1 | naive (double atomics) | 43.439 ± 0.0169 (spread 0.3%) | 6.2 | **1.00×** | – | 0.0% | YES | 0.0e+00% | -0.6% | 2685 / 8001 |
| S2 | hierarchical block reduction | 1.258 ± 0.0728 (spread 2.1%) | 214.5 | **34.54×** | 34.54× | 97.1% | YES | 2.7e-05% | -18.6% ⚠ | 2685 / 8001 |

Factored: S1→S2 34.54× = **34.5×**

## W02 Dot Product — CPU, N = 33,554,432

s = Σ aᵢbᵢ, FP32 inputs, N = 33,554,432.

| Stage | | ms (mean ± SD) | GB/s | × vs S1 | × prev | saved | Correct | error vs double | Δ earlier |
|---|---|---|---|---|---|---|---|---|---|
| S1 | scalar, double accumulator | 63.394 ± 0.0674 | 4.2 | **1.00×** | – | 0.0% | YES | 2.1e-09% | +12.7% ⚠ |
| S2 | OpenMP, double partials, 6 threads | 4.789 ± 0.0382 | 56.1 | **13.24×** | 13.24× | 92.4% | YES | 0.0e+00% | -1.3% |
| S3 | OpenMP + AVX2/FMA (FP32 lanes), 6 threads | 3.960 ± 0.0346 | 67.8 | **16.01×** | 1.21× | 93.8% | YES | 0.018% | -6.6% |

Factored: S1→S2 13.24× · S2→S3 1.21× = **16.0×**

## W03 GEMM — GPU, N = 1,024

C = A × B, FP32, N × N matrices, N = 1,024.

| Stage | | ms (mean ± SD) | GFLOP/s | × vs S1 | × prev | saved | Correct | error vs double | Δ earlier | SM / mem MHz |
|---|---|---|---|---|---|---|---|---|---|---|
| S1 | naive | 2.456 ± 0.2675 (spread 2.3%) | 874.3 | **1.00×** | – | 0.0% | YES | 1.5e-04% | -16.5% ⚠ | 2655 / 8001 |
| S2 | 16×16 shared-memory tile | 1.801 ± 0.0616 (spread 10.0%) | 1192.1 | **1.36×** | 1.36× | 26.7% | YES | 1.5e-04% | -21.7% ⚠ | 2528 / 8001 |
| S3 | TF32 Tensor Cores (WMMA) | 0.4719 ± 0.0003 (spread 11.2%) | 4551.1 | **5.21×** | 3.82× | 80.8% | YES | 4.1e-03% | -41.1% ⚠ | 2580 / 8001 |

Factored: S1→S2 1.36× · S2→S3 3.82× = **5.2×**

## W03 GEMM — CPU, N = 1,024

C = A × B, FP32, N × N matrices, N = 1,024.

| Stage | | ms (mean ± SD) | GFLOP/s | × vs S1 | × prev | saved | Correct | error vs double | Δ earlier |
|---|---|---|---|---|---|---|---|---|---|
| S1 | naive triple loop, -O0 | 2931.9 ± 10.197 | 0.7 | **1.00×** | – | 0.0% | YES | 1.6e-04% | -3.4% |
| S2 | 128-blocked + OpenMP, 8 threads | 89.045 ± 0.1722 | 24.1 | **32.93×** | 32.93× | 97.0% | YES | 1.6e-04% | -0.1% |
| S3 | AVX2/FMA + blocked + OpenMP, 8 threads | 25.735 ± 0.0230 | 83.4 | **113.93×** | 3.46× | 99.12% | YES | 1.5e-04% | +3.1% |

Factored: S1→S2 32.93× · S2→S3 3.46× = **113.9×**

## W04 Attention — GPU, N = 1,024

O = softmax(QKᵀ/√d)·V, FP32 inputs, N = 1,024.

| Stage | | ms (mean ± SD) | GFLOP/s | × vs S1 | × prev | saved | Correct | error vs double | Δ earlier | SM / mem MHz |
|---|---|---|---|---|---|---|---|---|---|---|
| S1 | naive, 3 kernels | 7.505 ± 0.1201 (spread 2.0%) | 286.2 | **1.00×** | – | 0.0% | YES | 1.8e-04% | +0.3% | 2595 / 8001 |
| S2a | tiled, unfused (3 kernels) | 2.200 ± 0.1553 (spread 1.8%) | 979.2 | **3.41×** | 3.41× | 70.7% | YES | 1.6e-04% | -0.1% | 2625 / 8001 |
| S2b | fused, FlashAttention-style | 0.7918 ± 0.0428 (spread 0.4%) | 2718.8 | **9.48×** | 2.78× | 89.4% | YES | 2.0e-04% | +0.7% | 2160 / 8001 |
| S3_FP16 | fused, FP16 Tensor Cores | 0.1738 ± 0.0004 (spread 0.2%) | 12358.7 | **43.19×** | 4.56× | 97.7% | YES | 0.038% | -0.1% | 2550 / 8001 |
| S3_TF32 | fused, TF32 Tensor Cores | 0.2772 ± 0.0004 (spread 0.1%) | 7748.1 | **27.08×** | 2.86× | 96.3% | YES | 0.038% | -0.1% | 2430 / 8001 |
| REF (ref.) | cuBLAS GEMMs + softmax (reference) | 0.6483 ± 0.0306 (spread 14.3%) | 3312.4 | 11.58× | – | – | YES | 1.6e-04% | -10.3% ⚠ | 2685 / 8001 |

Factored: S1→S2a 3.41× · S2a→S2b 2.78× · S2b→S3_FP16 4.56× = **43.2×**

## W04 Attention — GPU, N = 2,048

O = softmax(QKᵀ/√d)·V, FP32 inputs, N = 2,048.

| Stage | | ms (mean ± SD) | GFLOP/s | × vs S1 | × prev | saved | Correct | error vs double | Δ earlier | SM / mem MHz |
|---|---|---|---|---|---|---|---|---|---|---|
| S1 | naive, 3 kernels | 32.090 ± 0.0842 (spread 0.7%) | 267.7 | **1.00×** | – | 0.0% | YES | 2.7e-04% | -1.3% | 2445 / 8001 |
| S2a | tiled, unfused (3 kernels) | 9.491 ± 0.2154 (spread 1.3%) | 905.4 | **3.38×** | 3.38× | 70.4% | YES | 1.7e-04% | -0.1% | 2130 / 8001 |
| S2b | fused, FlashAttention-style | 3.011 ± 0.1220 (spread 2.2%) | 2855.1 | **10.66×** | 3.15× | 90.6% | YES | 2.1e-04% | -0.5% | 2685 / 8001 |
| S3_FP16 | fused, FP16 Tensor Cores | 0.6403 ± 0.0451 (spread 2.8%) | 13485.0 | **50.12×** | 4.70× | 98.0% | YES | 0.03% | -0.9% | 2340 / 8001 |
| S3_TF32 | fused, TF32 Tensor Cores | 1.140 ± 0.0948 (spread 4.4%) | 7567.7 | **28.15×** | 2.64× | 96.4% | YES | 0.03% | -0.3% | 2250 / 8001 |
| REF (ref.) | cuBLAS GEMMs + softmax (reference) | 3.349 ± 0.2699 (spread 1.8%) | 2572.4 | 9.58× | – | – | YES | 1.7e-04% | -2.1% | 2340 / 7501 |

Factored: S1→S2a 3.38× · S2a→S2b 3.15× · S2b→S3_FP16 4.70× = **50.1×**

## W04 Attention — CPU, N = 1,024

O = softmax(QKᵀ/√d)·V, FP32 inputs, N = 1,024.

| Stage | | ms (mean ± SD) | GFLOP/s | × vs S1 | × prev | saved | Correct | error vs double | Δ earlier |
|---|---|---|---|---|---|---|---|---|---|
| S1 | scalar, full S row | 304.1 ± 1.258 | 7.1 | **1.00×** | – | 0.0% | YES | 1.9e-04% | +6.4% |
| S2 | fused + blocked + OpenMP, 20 threads | 9.928 ± 0.4919 | 216.7 | **30.63×** | 30.63× | 96.7% | YES | 2.1e-04% | -0.0% |
| S3 | AVX2/FMA + vector exp, 20 threads | 3.330 ± 0.0696 | 645.1 | **91.33×** | 2.98× | 98.9% | YES | 1.9e-04% | -27.4% ⚠ |

Factored: S1→S2 30.63× · S2→S3 2.98× = **91.3×**

## W04 Attention — CPU, N = 2,048

O = softmax(QKᵀ/√d)·V, FP32 inputs, N = 2,048.

| Stage | | ms (mean ± SD) | GFLOP/s | × vs S1 | × prev | saved | Correct | error vs double | Δ earlier |
|---|---|---|---|---|---|---|---|---|---|
| S1 | scalar, full S row | 1187.7 ± 2.157 | 7.2 | **1.00×** | – | 0.0% | YES | 2.7e-04% | +5.7% |
| S2 | fused + blocked + OpenMP, 20 threads | 37.752 ± 0.3963 | 227.6 | **31.46×** | 31.46× | 96.8% | YES | 1.9e-04% | -4.7% |
| S3 | AVX2/FMA + vector exp, 20 threads | 12.889 ± 0.3147 | 666.8 | **92.15×** | 2.93× | 98.9% | YES | 2.1e-04% | -28.9% ⚠ |

Factored: S1→S2 31.46× · S2→S3 2.93× = **92.1×**

## CPU vs GPU (same N, same session)

| Workload | CPU S1 → GPU S1 | best CPU → best GPU | CPU S1 → best GPU |
|---|---|---|---|
| W01 Vector Add | 44.5× | S2 7.771 → S1 1.796 ms: **4.3×** | 45× |
| W02 Dot Product | 1.5× | S3 3.960 → S2 1.258 ms: **3.1×** | 50× |
| W03 GEMM | 1193.7× | S3 25.735 → S3 0.4719 ms: **54.5×** | 6214× |
| W04 Attention N=1024 | 40.5× | S3 3.330 → S3_FP16 0.1738 ms: **19.2×** | 1750× |
| W04 Attention N=2048 | 37.0× | S3 12.889 → S3_FP16 0.6403 ms: **20.1×** | 1855× |

## W01 reconciliation (GPU S1 vs S2)

From this run's CSV rows only (5 repetitions per round, every round), never from console output:

| | rounds × reps | per-round means (ms) | median ms | SD | GB/s | % of 256 GB/s nominal | memory clock per round (MHz) |
|---|---|---|---|---|---|---|---|
| S1 | 3 × 5 | 1.796, 1.796, 1.795 | 1.796 | 0.002 | 224.2 | 88% | 7001, 8001, 8001 |
| S2 | 3 × 5 | 1.796, 1.795, 1.797 | 1.796 | 0.001 | 224.2 | 88% | 8001, 8001, 8001 |

**Final W01 GPU: S1 1.796 ms, S2 1.796 ms, S2/S1 speedup 1.000×** (within ±3%: both kernels stream at the DRAM limit).

Earlier W01 GPU figures and why they differ:

- S1: 1.715 ms over 5 row(s), `results/demo/csv/W01_S1.csv` (2026-10-21): 5 per-rep rows, B256, 234.8 GB/s; 6% faster than this run: a different session, memory clock not logged
- S1: 35.126 ms over 1 row(s), `results/raw/W01_VECTOR_ADD/W01_VECTOR_ADD_GPU_S1_BASE_NAIVE.csv` (2026-10-02): one row, repetition 1 only, 11 GB/s: 19x slower than every other W01 GPU record, so not a DRAM-limited run (cause unknown); not used
- S2: 1.699 ms over 5 row(s), `results/demo/csv/W01_S2.csv` (2026-10-21): 5 per-rep rows, B128, 237.0 GB/s; 7% faster than this run: a different session, memory clock not logged
- S2: 1.859 ms over 5 row(s), `results/raw/W01_VECTOR_ADD/W01_VECTOR_ADD_GPU_S2_SWOPT_COALESCED_B128.csv` (2026-10-02): 5 per-rep rows: rep 1 2.08 ms, reps 2-5 1.80-1.82 ms, the memory-clock ramp seen in run 20261106-134741
- Console output of an earlier session showed 2.011 / 1.779 ms. These are console lines, not recorded per-repetition rows, and are not used. [ASSUMPTION] The S1 value matches S1 at the lower 7001 MHz memory clock (2.065 ms in run 20261106-134741), so it is probably the same memory-clock effect.

## Correctness (one line per stage run)

Correct is each binary's own verdict (and exit status 0). The error column is the binary's reported error against its double-precision reference, where it reports one. No tolerance was changed for this run.

| Stage | N | Correct | error vs double | binary's check |
|---|---|---|---|---|
| W01_GPU_S1 | 33554432 | YES | – | every element |C − (A+B)| ≤ 1e-6 |
| W01_GPU_S2 | 33554432 | YES | – | every element |C − (A+B)| ≤ 1e-6 |
| W01_CPU_S1 | 33554432 | YES | – | every element |C − (A+B)| ≤ 1e-6 |
| W01_CPU_S2 | 33554432 | YES | – | every element |C − (A+B)| ≤ 1e-6 |
| W01_CPU_S3 | 33554432 | YES | – | every element |C − (A+B)| ≤ 1e-6 |
| W02_GPU_S1 | 33554432 | YES | 0% | relative error vs the double-precision sum ≤ 0.1% (every W02 stage) |
| W02_GPU_S2 | 33554432 | YES | 2.69e-05% | relative error vs the double-precision sum ≤ 0.1% (every W02 stage) |
| W02_CPU_S1 | 33554432 | YES | 2.07e-09% | relative error vs the double-precision sum ≤ 0.1% (every W02 stage) |
| W02_CPU_S2 | 33554432 | YES | 0% | relative error vs the double-precision sum ≤ 0.1% (every W02 stage) |
| W02_CPU_S3 | 33554432 | YES | 0.0181% | relative error vs the double-precision sum ≤ 0.1% (every W02 stage) |
| W03_GPU_S1 | 1024 | YES | 0.000154% | max relative error vs CPU reference < 0.1% |
| W03_GPU_S2 | 1024 | YES | 0.000154% | max relative error vs CPU reference < 0.1% |
| W03_GPU_S3 | 1024 | YES | 0.00415% | max relative error vs CPU reference < 0.1% |
| W03_CPU_S1 | 1024 | YES | 0.000161% | max relative error vs CPU reference < 0.1% |
| W03_CPU_S2 | 1024 | YES | 0.000161% | max relative error vs CPU reference < 0.1% |
| W03_CPU_S3 | 1024 | YES | 0.000154% | max relative error vs CPU reference < 0.1% |
| W04_GPU_S1 | 1024 | YES | 0.000183% | max rel. error vs double ≤ 1e-4 (FP32), 5e-3 (TF32), 1e-2 (FP16) |
| W04_GPU_S2a | 1024 | YES | 0.000164% | max rel. error vs double ≤ 1e-4 (FP32), 5e-3 (TF32), 1e-2 (FP16) |
| W04_GPU_S2b | 1024 | YES | 0.000197% | max rel. error vs double ≤ 1e-4 (FP32), 5e-3 (TF32), 1e-2 (FP16) |
| W04_GPU_S3_FP16 | 1024 | YES | 0.0383% | max rel. error vs double ≤ 1e-4 (FP32), 5e-3 (TF32), 1e-2 (FP16) |
| W04_GPU_S3_TF32 | 1024 | YES | 0.0384% | max rel. error vs double ≤ 1e-4 (FP32), 5e-3 (TF32), 1e-2 (FP16) |
| W04_GPU_REF | 1024 | YES | 0.000164% | max rel. error vs double ≤ 1e-4 (FP32), 5e-3 (TF32), 1e-2 (FP16) |
| W04_GPU_S1 | 2048 | YES | 0.000266% | max rel. error vs double ≤ 1e-4 (FP32), 5e-3 (TF32), 1e-2 (FP16) |
| W04_GPU_S2a | 2048 | YES | 0.000175% | max rel. error vs double ≤ 1e-4 (FP32), 5e-3 (TF32), 1e-2 (FP16) |
| W04_GPU_S2b | 2048 | YES | 0.000206% | max rel. error vs double ≤ 1e-4 (FP32), 5e-3 (TF32), 1e-2 (FP16) |
| W04_GPU_S3_FP16 | 2048 | YES | 0.03% | max rel. error vs double ≤ 1e-4 (FP32), 5e-3 (TF32), 1e-2 (FP16) |
| W04_GPU_S3_TF32 | 2048 | YES | 0.0295% | max rel. error vs double ≤ 1e-4 (FP32), 5e-3 (TF32), 1e-2 (FP16) |
| W04_GPU_REF | 2048 | YES | 0.000175% | max rel. error vs double ≤ 1e-4 (FP32), 5e-3 (TF32), 1e-2 (FP16) |
| W04_CPU_S1 | 1024 | YES | 0.000189% | max rel. error vs double ≤ 1e-4 (FP32), 5e-3 (TF32), 1e-2 (FP16) |
| W04_CPU_S2 | 1024 | YES | 0.000208% | max rel. error vs double ≤ 1e-4 (FP32), 5e-3 (TF32), 1e-2 (FP16) |
| W04_CPU_S3 | 1024 | YES | 0.000188% | max rel. error vs double ≤ 1e-4 (FP32), 5e-3 (TF32), 1e-2 (FP16) |
| W04_CPU_S1 | 2048 | YES | 0.000266% | max rel. error vs double ≤ 1e-4 (FP32), 5e-3 (TF32), 1e-2 (FP16) |
| W04_CPU_S2 | 2048 | YES | 0.000188% | max rel. error vs double ≤ 1e-4 (FP32), 5e-3 (TF32), 1e-2 (FP16) |
| W04_CPU_S3 | 2048 | YES | 0.000206% | max rel. error vs double ≤ 1e-4 (FP32), 5e-3 (TF32), 1e-2 (FP16) |

**W02 accuracy.** Error vs the double-precision sum: CPU S1 2.07e-09%; CPU S2 0%; CPU S3 0.0181%; GPU S1 0%; GPU S2 2.69e-05%. Until P8, CPU S1, CPU S2 and GPU S1 accumulated in FP32 and were 6.75%, 0.41% and 6.8% off; each binary also used a different check (≤ 0.01% vs double, finite only, or vs an FP32 serial sum with the same error). P8 made them accumulate in double (FP32 inputs) and gave every W02 stage one check: ≤ 0.1% vs double. CPU S3 (FP32 AVX2 lanes) and GPU S2 (block reduction) were already within it and are unchanged. Their earlier times are therefore not directly comparable for CPU S1/S2 and GPU S1.

## Deviations > 10% from the earlier recorded runs

- ⚠ **W01_CPU_S1**: 79.942 ms now vs 100.8 ms (`results/raw/W01_VECTOR_ADD/W01_VECTOR_ADD_CPU_S1_BASE.csv`, 2026-10-02): -20.7%.
- ⚠ **W02_GPU_S2**: 1.258 ms now vs 1.544 ms (`results/raw/W02_DOT_PRODUCT/W02_DOT_PRODUCT_GPU_S2_SWOPT_BLOCK_REDUCE_V2.csv`, 2026-10-03): -18.6%.
- ⚠ **W02_CPU_S1**: 63.394 ms now vs 56.24 ms (`results/raw/W02_DOT_PRODUCT/W02_DOT_PRODUCT_CPU_S1_BASE_V2.csv`, 2026-10-03): +12.7%.
- ⚠ **W03_GPU_S1**: 2.456 ms now vs 2.941 ms (`results/raw/W03_GEMM/W03_GEMM_GPU_S1_BASE_NAIVE_V1.csv`, 2026-10): -16.5%.
- ⚠ **W03_GPU_S2**: 1.801 ms now vs 2.3 ms (`results/raw/W03_GEMM/W03_GEMM_GPU_S2_SWOPT_SHARED_TILE_V1.csv`, 2026-10): -21.7%.
- ⚠ **W03_GPU_S3**: 0.4719 ms now vs 0.8015 ms (`results/raw/W03_GEMM/W03_GEMM_GPU_S3_HWACC_TENSOR_CORE_V2.csv`, 2026-10): -41.1%.
- ⚠ **W04_GPU_REF_N1024**: 0.6483 ms now vs 0.7226 ms (`results/raw/W04_ATTENTION/20261106-114338 (median of 3 rounds, tools/w04_summary.py)`, 2026-11-06): -10.3%.
- ⚠ **W04_CPU_S3_N1024**: 3.330 ms now vs 4.585 ms (`results/raw/W04_ATTENTION/20261105-174743 nsweep (tools/w04_summary.py)`, 2026-11-05): -27.4%.
- ⚠ **W04_CPU_S3_N2048**: 12.889 ms now vs 18.13 ms (`results/raw/W04_ATTENTION/20261105-174743 nsweep (tools/w04_summary.py)`, 2026-11-05): -28.9%.

### Why each flagged stage differs

| Stage | Δ | Explanation | Evidence |
|---|---|---|---|
| W03 GPU S3 | −41% | **Different code: the S3 kernel was optimized after the first release.** The earlier value (0.8015 ms) is the first S3 kernel (k-tile 8, generic fragment loads `LD.E`, failed at N not a multiple of 16). The current kernel stages a 32×32 k-tile 32 block in padded shared memory, already rounded to TF32, and loads fragments with shared-memory loads (`LD.E` 8 → 0 in SASS, 4 MMAs per barrier pair). Part of the −41% is the October session (see W03 S1/S2 below); the kernel alone is **1.26× at N = 1024 and 1.19× at N = 2048**, measured as a paired A/B of the old and new binaries at matched clocks | A/B `results/raw/W03_GEMM/20261106-150800-s3ab`, `…150856-s3ab`: 0.764 → 0.604 ms (N = 1024), 5.445 → 4.564 ms (N = 2048), SM ~1890–2010 MHz both. Error 0.0686% → 0.0041% |
| W03 GPU S1 / S2 | −16% / −22% | **Not attributable for the earlier run.** The earlier values are single summary rows from October (no clock or driver record). The computation is the same: S1/S2 match the earlier error to the last digit (0.000154%). Today's values repeat across the three rounds; S2's 10% spread follows the SM clock | Rounds: S1 2.456 / 2.452 / 2.508 ms, S2 1.802 / 1.635 / 1.815 ms (SM 2355 / 2685 / 2475 MHz); driver 595.99.02 |
| W01 CPU S1 | −21% | **Session difference, not the code or the build.** Both builds use the same flags (`-O0 -std=c++17 -g`, read from the DWARF producer of both binaries), and the earlier binary runs at 77 ms today, this build at 80 ms. The earlier records of this stage themselves disagree (100.8 ms in the per-repetition CSV, 123.6 ms in its text log). The cause of the slower October sessions is not identified | Paired run of the two binaries, 2026-11-06 |
| W02 GPU S2 | −19% | **Not attributable for the earlier run.** The earlier value (1.544 ms) is a single summary row from 2026-10-03 with no clock log. Its timed window is only ~0.15 s, the case where the memory clock may not have risen yet; this run primes it. Today's value repeats: 1.258 / 1.280 / 1.253 ms in the three rounds, and 1.278 ms twice in the demo races of 2026-11-06 | `gpu_state.csv`: SM 2685 MHz, memory 8001 MHz (7501 in round 2); 214 GB/s = 84% of the 256 GB/s nominal |
| W02 CPU S1 | +13% | **Expected: different computation.** Since this re-test, S1 accumulates in double (it was 6.75% off the double sum with an FP32 accumulator). The extra cost at -O0 is the double add. | Error vs double 6.75% → 2e-9%. CPU S2 (also switched to double partials) is unchanged in time, because it is limited by memory bandwidth, not by the add |
| W04 GPU REF, N = 1024 | −10% | **Clock-dependent, within today's own range.** cuBLAS at N = 1024 is a ~0.65 ms job that follows the SM clock. Rounds 1 and 3 ran at 2685 MHz (0.648 / 0.647 ms), round 2 at 2295 MHz (0.740 ms); the earlier 0.723 ms lies between them. The reference is not a project stage, and no speedup in this summary depends on it | Rounds 0.648 / 0.740 / 0.647 ms; spread 14.3% |
| W04 CPU S3, N = 1024 / 2048 | −27% / −29% | **The earlier run was thermally loaded.** The reference is the N sweep of `20261105-174743`, the first part of a long CPU session. The S3 code has not changed since (only a comment). The temperature-logged rerun of the same evening (`20261105-181924-controlled`, same 32×64 tile, 20 threads) measured 3.37–3.46 ms at N = 1024, within 4% of today's 3.33 ms. The CPU counter run of 2026-11-06 (`perf stat`, 618 GFLOP/s ≈ 13.9 ms) and the demo race (14.0 ms) agree with today's 12.9 ms at N = 2048 | Same source and tile; controlled rerun 3.45 ms median |

Not flagged but worth stating: W01 GPU S1 and S2 are equal (1.796 / 1.796 ms, spread 0.1%). In the first full run of this re-test (`20261106-134741`, superseded), S1 ran at 7001 MHz memory clock and was 14% slower, exactly the memory-clock ratio (0.875). The priming run before each W01–W03 GPU stage removed that effect.

This run (`20261106-151143`) replaces `20261106-141652` as the release result. The two agree within 4% on every stage except W03 GPU S3 (new kernel: 0.603 → 0.472 ms, −22%, at a higher SM clock than the A/B) and three short W04 GPU runs that follow the SM clock: S3 FP16 at N = 2048 (0.592 → 0.640 ms, +8%), S2b at N = 1024 (0.746 → 0.792 ms, +6%) and REF at N = 1024 (0.733 → 0.648 ms, −12%).

## Files

- consolidated CSV: `docs/results/final_results.csv` (one row per stage and N)
- earlier values: `docs/results/reference_values.csv`
- raw CSVs, logs, `runs.csv`, `gpu_state.csv`: `results/raw/FINAL/20261106-151143/` (private repository)
- reproduce: `bash scripts/run_all.sh`, then `python3 tools/final_summary.py results/raw/FINAL/<run> --publish-docs`

