# Results summary — final paired re-test of W01–W04

| | |
|---|---|
| Run | `20261106-141652` (one session, `scripts/run_all.sh`) |
| Commit | `5b61fa315aae` (W04 build `5b61fa315aae`) |
| Date | 2026-11-06T14:17:28+05:30 |
| CPU | 13th Gen Intel(R) Core(TM) i7-13650HX (6 P + 8 E cores, 20 threads, AVX2/FMA), governor powersave, EPP performance, AC 1 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU (sm_89, 24 SMs, 8 GiB GDDR6 128-bit), driver 595.99.02, power limit 55.00 W |
| Toolchain | g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0 / Build cuda_13.4.r13.4/compiler.38657139_0 |
| Protocol | warmup=2 iters=20 reps=5 rounds(GPU)=3 cool-down<=70C (max 120s) |
| Sizes / threads | W01/W02 N=33554432  W03 N=1024  W04 N=1024 2048 --bh 8 --d 64; W01 6  W02 6  W03 8  W04 20  (OMP_PROC_BIND=close OMP_PLACES=cores) |
| GPU state (loaded samples) | SM clock median 2685 MHz (1605–2700), power median 32.6 W, max 65 °C (281 samples, 250 ms) |
| CPU state | package mean temperature per run 72–96 °C (cool-down to ≤ 70 °C before each run) |

Times are ms per iteration: mean ± SD over the 5 repetitions of a run (20 timed iterations each, after 2 warm-ups). GPU stages ran in 3 interleaved round(s): median of the round means, SD = mean within-round SD, spread = (max − min) / median of the round means. CPU stages ran once each. × vs S1 = t(S1)/t; × prev = t(previous stage)/t (S3_TF32's previous stage is S2b); saved = (1 − t/t(S1))·100. Δ earlier = change vs the earlier recorded run (`docs/results/reference_values.csv`); |Δ| > 10% is marked ⚠ and explained below.

## W01 Vector Add — GPU, N = 33,554,432

C = A + B, FP32, N = 33,554,432.

| Stage | | ms (mean ± SD) | GB/s | × vs S1 | × prev | saved | Correct | error vs double | Δ earlier | SM / mem MHz |
|---|---|---|---|---|---|---|---|---|---|---|
| S1 | naive, 256 threads/block | 1.824 ± 0.0032 (spread 0.0%) | 220.8 | **1.00×** | – | 0.0% | YES | – | +6.3% | 2490 / 8001 |
| S2 | coalesced, 128 threads/block | 1.823 ± 0.0030 (spread 0.0%) | 220.9 | **1.00×** | 1.00× | 0.0% | YES | – | +7.3% | 2685 / 8001 |

Factored: S1→S2 1.00× = **1.0×**

## W01 Vector Add — CPU, N = 33,554,432

C = A + B, FP32, N = 33,554,432.

| Stage | | ms (mean ± SD) | GB/s | × vs S1 | × prev | saved | Correct | error vs double | Δ earlier |
|---|---|---|---|---|---|---|---|---|---|
| S1 | scalar, -O0 | 81.054 ± 0.1245 | 5.0 | **1.00×** | – | 0.0% | YES | – | -19.6% ⚠ |
| S2 | OpenMP, 6 threads | 7.809 ± 0.1280 | 51.6 | **10.38×** | 10.38× | 90.4% | YES | – | -3.1% |
| S3 | OpenMP + AVX2, 6 threads | 7.799 ± 0.0198 | 51.6 | **10.39×** | 1.00× | 90.4% | YES | – | -1.9% |

Factored: S1→S2 10.38× · S2→S3 1.00× = **10.4×**

## W02 Dot Product — GPU, N = 33,554,432

s = Σ aᵢbᵢ, FP32 inputs, N = 33,554,432.

| Stage | | ms (mean ± SD) | GB/s | × vs S1 | × prev | saved | Correct | error vs double | Δ earlier | SM / mem MHz |
|---|---|---|---|---|---|---|---|---|---|---|
| S1 | naive (double atomics) | 43.424 ± 0.0104 (spread 0.6%) | 6.2 | **1.00×** | – | 0.0% | YES | 0.0e+00% | -0.6% | 2685 / 8001 |
| S2 | hierarchical block reduction | 1.274 ± 0.0323 (spread 12.7%) | 211.8 | **34.10×** | 34.10× | 97.1% | YES | 2.7e-05% | -17.5% ⚠ | 2640 / 8001 |

Factored: S1→S2 34.10× = **34.1×**

## W02 Dot Product — CPU, N = 33,554,432

s = Σ aᵢbᵢ, FP32 inputs, N = 33,554,432.

| Stage | | ms (mean ± SD) | GB/s | × vs S1 | × prev | saved | Correct | error vs double | Δ earlier |
|---|---|---|---|---|---|---|---|---|---|
| S1 | scalar, double accumulator | 64.157 ± 0.2136 | 4.2 | **1.00×** | – | 0.0% | YES | 2.1e-09% | +14.1% ⚠ |
| S2 | OpenMP, double partials, 6 threads | 4.769 ± 0.0207 | 56.3 | **13.45×** | 13.45× | 92.6% | YES | 0.0e+00% | -1.7% |
| S3 | OpenMP + AVX2/FMA (FP32 lanes), 6 threads | 3.960 ± 0.0362 | 67.8 | **16.20×** | 1.20× | 93.8% | YES | 0.018% | -6.6% |

Factored: S1→S2 13.45× · S2→S3 1.20× = **16.2×**

## W03 GEMM — GPU, N = 1,024

C = A × B, FP32, N × N matrices, N = 1,024.

| Stage | | ms (mean ± SD) | GFLOP/s | × vs S1 | × prev | saved | Correct | error vs double | Δ earlier | SM / mem MHz |
|---|---|---|---|---|---|---|---|---|---|---|
| S1 | naive | 2.537 ± 0.1855 (spread 3.5%) | 846.4 | **1.00×** | – | 0.0% | YES | 1.5e-04% | -13.7% ⚠ | 2632 / 8001 |
| S2 | 16×16 shared-memory tile | 1.750 ± 0.1124 (spread 3.4%) | 1226.9 | **1.45×** | 1.45× | 31.0% | YES | 1.5e-04% | -23.9% ⚠ | 2685 / 8001 |
| S3 | TF32 Tensor Cores (WMMA) | 0.6034 ± 0.0441 (spread 2.6%) | 3559.2 | **4.21×** | 2.90× | 76.2% | YES | 0.069% | -24.7% ⚠ | 2685 / 8001 |

Factored: S1→S2 1.45× · S2→S3 2.90× = **4.2×**

## W03 GEMM — CPU, N = 1,024

C = A × B, FP32, N × N matrices, N = 1,024.

| Stage | | ms (mean ± SD) | GFLOP/s | × vs S1 | × prev | saved | Correct | error vs double | Δ earlier |
|---|---|---|---|---|---|---|---|---|---|
| S1 | naive triple loop, -O0 | 2922.6 ± 5.194 | 0.7 | **1.00×** | – | 0.0% | YES | 1.6e-04% | -3.7% |
| S2 | 128-blocked + OpenMP, 8 threads | 89.191 ± 0.1061 | 24.1 | **32.77×** | 32.77× | 96.9% | YES | 1.6e-04% | +0.0% |
| S3 | AVX2/FMA + blocked + OpenMP, 8 threads | 25.697 ± 0.0209 | 83.6 | **113.73×** | 3.47× | 99.12% | YES | 1.5e-04% | +2.9% |

Factored: S1→S2 32.77× · S2→S3 3.47× = **113.7×**

## W04 Attention — GPU, N = 1,024

O = softmax(QKᵀ/√d)·V, FP32 inputs, N = 1,024.

| Stage | | ms (mean ± SD) | GFLOP/s | × vs S1 | × prev | saved | Correct | error vs double | Δ earlier | SM / mem MHz |
|---|---|---|---|---|---|---|---|---|---|---|
| S1 | naive, 3 kernels | 7.400 ± 0.0309 (spread 1.4%) | 290.2 | **1.00×** | – | 0.0% | YES | 1.8e-04% | -1.1% | 2648 / 8001 |
| S2a | tiled, unfused (3 kernels) | 2.200 ± 0.1335 (spread 0.8%) | 979.3 | **3.36×** | 3.36× | 70.3% | YES | 1.6e-04% | -0.0% | 2400 / 8001 |
| S2b | fused, FlashAttention-style | 0.7461 ± 0.0133 (spread 5.4%) | 2878.4 | **9.92×** | 2.95× | 89.9% | YES | 2.0e-04% | -5.1% | 2280 / 8001 |
| S3_FP16 | fused, FP16 Tensor Cores | 0.1742 ± 0.0003 (spread 0.1%) | 12330.3 | **42.49×** | 4.28× | 97.6% | YES | 0.038% | +0.2% | 2415 / 8001 |
| S3_TF32 | fused, TF32 Tensor Cores | 0.2771 ± 0.0002 (spread 0.1%) | 7751.1 | **26.71×** | 2.69× | 96.3% | YES | 0.038% | -0.1% | 2685 / 8001 |
| REF (ref.) | cuBLAS GEMMs + softmax (reference) | 0.7333 ± 0.0583 (spread 12.4%) | 2961.0 | 10.09× | – | – | YES | 1.6e-04% | +1.5% | 2685 / 8001 |

Factored: S1→S2a 3.36× · S2a→S2b 2.95× · S2b→S3_FP16 4.28× = **42.5×**

## W04 Attention — GPU, N = 2,048

O = softmax(QKᵀ/√d)·V, FP32 inputs, N = 2,048.

| Stage | | ms (mean ± SD) | GFLOP/s | × vs S1 | × prev | saved | Correct | error vs double | Δ earlier | SM / mem MHz |
|---|---|---|---|---|---|---|---|---|---|---|
| S1 | naive, 3 kernels | 31.994 ± 0.0903 (spread 0.5%) | 268.5 | **1.00×** | – | 0.0% | YES | 2.7e-04% | -1.6% | 2535 / 8001 |
| S2a | tiled, unfused (3 kernels) | 9.447 ± 0.1967 (spread 0.9%) | 909.8 | **3.39×** | 3.39× | 70.5% | YES | 1.7e-04% | -0.6% | 2145 / 8001 |
| S2b | fused, FlashAttention-style | 3.022 ± 0.0987 (spread 0.6%) | 2845.5 | **10.59×** | 3.13× | 90.6% | YES | 2.1e-04% | -0.1% | 2355 / 8001 |
| S3_FP16 | fused, FP16 Tensor Cores | 0.5924 ± 0.0090 (spread 4.6%) | 14500.6 | **54.01×** | 5.10× | 98.1% | YES | 0.03% | -8.3% | 2685 / 8001 |
| S3_TF32 | fused, TF32 Tensor Cores | 1.158 ± 0.1001 (spread 6.8%) | 7469.5 | **27.63×** | 2.61× | 96.4% | YES | 0.03% | +1.3% | 2685 / 8001 |
| REF (ref.) | cuBLAS GEMMs + softmax (reference) | 3.430 ± 0.2607 (spread 6.2%) | 2507.4 | 9.33× | – | – | YES | 1.7e-04% | +0.2% | 2468 / 8001 |

Factored: S1→S2a 3.39× · S2a→S2b 3.13× · S2b→S3_FP16 5.10× = **54.0×**

## W04 Attention — CPU, N = 1,024

O = softmax(QKᵀ/√d)·V, FP32 inputs, N = 1,024.

| Stage | | ms (mean ± SD) | GFLOP/s | × vs S1 | × prev | saved | Correct | error vs double | Δ earlier |
|---|---|---|---|---|---|---|---|---|---|
| S1 | scalar, full S row | 308.0 ± 0.5350 | 7.0 | **1.00×** | – | 0.0% | YES | 1.9e-04% | +7.8% |
| S2 | fused + blocked + OpenMP, 20 threads | 9.658 ± 0.2837 | 222.5 | **31.89×** | 31.89× | 96.9% | YES | 2.1e-04% | -2.7% |
| S3 | AVX2/FMA + vector exp, 20 threads | 3.286 ± 0.0502 | 653.7 | **93.74×** | 2.94× | 98.9% | YES | 1.9e-04% | -28.3% ⚠ |

Factored: S1→S2 31.89× · S2→S3 2.94× = **93.7×**

## W04 Attention — CPU, N = 2,048

O = softmax(QKᵀ/√d)·V, FP32 inputs, N = 2,048.

| Stage | | ms (mean ± SD) | GFLOP/s | × vs S1 | × prev | saved | Correct | error vs double | Δ earlier |
|---|---|---|---|---|---|---|---|---|---|
| S1 | scalar, full S row | 1205.6 ± 1.249 | 7.1 | **1.00×** | – | 0.0% | YES | 2.7e-04% | +7.3% |
| S2 | fused + blocked + OpenMP, 20 threads | 38.062 ± 0.6796 | 225.7 | **31.67×** | 31.67× | 96.8% | YES | 1.9e-04% | -3.9% |
| S3 | AVX2/FMA + vector exp, 20 threads | 12.938 ± 0.2929 | 664.2 | **93.18×** | 2.94× | 98.9% | YES | 2.1e-04% | -28.6% ⚠ |

Factored: S1→S2 31.67× · S2→S3 2.94× = **93.2×**

## CPU vs GPU (same N, same session)

| Workload | CPU S1 → GPU S1 | best CPU → best GPU | CPU S1 → best GPU |
|---|---|---|---|
| W01 Vector Add | 44.4× | S3 7.799 → S2 1.823 ms: **4.3×** | 44× |
| W02 Dot Product | 1.5× | S3 3.960 → S2 1.274 ms: **3.1×** | 50× |
| W03 GEMM | 1151.9× | S3 25.697 → S3 0.6034 ms: **42.6×** | 4844× |
| W04 Attention N=1024 | 41.6× | S3 3.286 → S3_FP16 0.1742 ms: **18.9×** | 1768× |
| W04 Attention N=2048 | 37.7× | S3 12.938 → S3_FP16 0.5924 ms: **21.8×** | 2035× |

## W01 reconciliation (GPU S1 vs S2)

From this run's CSV rows only (5 repetitions per round, every round), never from console output:

| | rounds × reps | per-round means (ms) | median ms | SD | GB/s | % of 256 GB/s nominal | memory clock per round (MHz) |
|---|---|---|---|---|---|---|---|
| S1 | 3 × 5 | 1.824, 1.824, 1.823 | 1.824 | 0.003 | 220.8 | 86% | 8001, 8001, 8001 |
| S2 | 3 × 5 | 1.823, 1.822, 1.823 | 1.823 | 0.003 | 220.9 | 86% | 8001, 8001, 8001 |

**Final W01 GPU: S1 1.824 ms, S2 1.823 ms, S2/S1 speedup 1.000×** (within ±3%: both kernels stream at the DRAM limit).

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
| W03_GPU_S3 | 1024 | YES | 0.0686% | max relative error vs CPU reference < 0.1% |
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

- ⚠ **W01_CPU_S1**: 81.054 ms now vs 100.8 ms (`results/raw/W01_VECTOR_ADD/W01_VECTOR_ADD_CPU_S1_BASE.csv`, 2026-10-02): -19.6%.
- ⚠ **W02_GPU_S2**: 1.274 ms now vs 1.544 ms (`results/raw/W02_DOT_PRODUCT/W02_DOT_PRODUCT_GPU_S2_SWOPT_BLOCK_REDUCE_V2.csv`, 2026-10-03): -17.5%.
- ⚠ **W02_CPU_S1**: 64.157 ms now vs 56.24 ms (`results/raw/W02_DOT_PRODUCT/W02_DOT_PRODUCT_CPU_S1_BASE_V2.csv`, 2026-10-03): +14.1%.
- ⚠ **W03_GPU_S1**: 2.537 ms now vs 2.941 ms (`results/raw/W03_GEMM/W03_GEMM_GPU_S1_BASE_NAIVE_V1.csv`, 2026-10): -13.7%.
- ⚠ **W03_GPU_S2**: 1.750 ms now vs 2.3 ms (`results/raw/W03_GEMM/W03_GEMM_GPU_S2_SWOPT_SHARED_TILE_V1.csv`, 2026-10): -23.9%.
- ⚠ **W03_GPU_S3**: 0.6034 ms now vs 0.8015 ms (`results/raw/W03_GEMM/W03_GEMM_GPU_S3_HWACC_TENSOR_CORE_V2.csv`, 2026-10): -24.7%.
- ⚠ **W04_CPU_S3_N1024**: 3.286 ms now vs 4.585 ms (`results/raw/W04_ATTENTION/20261105-174743 nsweep (tools/w04_summary.py)`, 2026-11-05): -28.3%.
- ⚠ **W04_CPU_S3_N2048**: 12.938 ms now vs 18.13 ms (`results/raw/W04_ATTENTION/20261105-174743 nsweep (tools/w04_summary.py)`, 2026-11-05): -28.6%.

### Why each flagged stage differs

| Stage | Δ | Explanation | Evidence |
|---|---|---|---|
| W02 CPU S1 | +14% | **Expected: different computation.** Since this re-test, S1 accumulates in double (it was 6.75% off the double sum with an FP32 accumulator). The extra cost at -O0 is the double add. | Error vs double 6.75% → 2e-9%. CPU S2 (also switched to double partials) is unchanged in time (4.77 vs 4.85 ms), because it is limited by memory bandwidth, not by the add |
| W01 CPU S1 | −20% | **Session difference, not the code or the build.** Both builds use the same flags (`-O0 -std=c++17 -g`, read from the DWARF producer of both binaries), and the earlier binary runs at 77 ms today, this build at 80 ms. The earlier records of this stage themselves disagree (100.8 ms in the per-repetition CSV, 123.6 ms in its text log). The cause of the slower October sessions is not identified | Paired run of the two binaries, 2026-11-06 |
| W02 GPU S2 | −18% | **Not attributable for the earlier run.** The earlier value (1.544 ms) is a single summary row from 2026-10-03 with no clock log. Its timed window is only ~0.15 s, the case where the memory clock may not have risen yet; this run primes it and records 8001 MHz. Today's value repeats: 1.12 / 1.28 / 1.27 ms in the three rounds, and 1.278 ms twice in the demo races of 2026-11-06 | `gpu_state.csv`: memory clock 8001 MHz, SM 2557–2685 MHz in all three rounds; 212 GB/s = 83% of the 256 GB/s nominal |
| W03 GPU S1 / S2 / S3 | −14% / −24% / −25% | **Not attributable for the earlier run.** The earlier values are single summary rows from October (no clock or driver record). The computation is the same: S3's relative error is identical to the earlier row (0.0686%), and S1/S2 match it to the last digit (0.000154%). Today's values repeat within 3.5% across the three rounds, and agree with the demo races of 2026-11-06 (2.46 / 1.78 / 0.61 ms) | Rounds: S1 2.50 / 2.59 / 2.54, S2 1.75 / 1.81 / 1.75, S3 0.599 / 0.615 / 0.603 ms; SM clock 2145–2685 MHz; driver 595.99.02 |
| W04 CPU S3, N = 1024 / 2048 | −28% / −29% | **The earlier run was thermally loaded.** The reference is the N sweep of `20261105-174743`, the first part of a long CPU session. The S3 code has not changed since (only a comment). The temperature-logged rerun of the same evening (`20261105-181924-controlled`, same 32×64 tile, 20 threads) measured 3.37–3.46 ms at N = 1024, within 5% of today's 3.29 ms. The CPU counter run of 2026-11-06 (`perf stat`, 618 GFLOP/s ≈ 13.9 ms) and the demo race (14.0 ms) agree with today's 12.9 ms at N = 2048 | Same source and tile; controlled rerun 3.45 ms median |

Not flagged but worth stating: W01 GPU S1 and S2 are now equal (1.824 / 1.823 ms, both runs at 8001 MHz memory clock). In the first full run of this re-test (`20261106-134741`, superseded), S1 ran at 7001 MHz and was 14% slower, exactly the memory-clock ratio (0.875). The priming run before each W01–W03 GPU stage removed that effect.

## Files

- consolidated CSV: `docs/results/final_results.csv` (one row per stage and N)
- earlier values: `docs/results/reference_values.csv`
- raw CSVs, logs, `runs.csv`, `gpu_state.csv`: `results/raw/FINAL/20261106-141652/` (private repository)
- reproduce: `bash scripts/run_all.sh`, then `python3 tools/final_summary.py results/raw/FINAL/<run> --publish-docs`

