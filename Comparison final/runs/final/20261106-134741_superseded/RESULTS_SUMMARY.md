# Results summary — final paired re-test of W01–W04

| | |
|---|---|
| Run | `20261106-134741` (one session, `scripts/run_all.sh`) |
| Commit | `55b980e2dd56` (W04 build `55b980e2dd56`) |
| Date | 2026-11-06T13:48:17+05:30 |
| CPU | 13th Gen Intel(R) Core(TM) i7-13650HX (6 P + 8 E cores, 20 threads, AVX2/FMA), governor powersave, EPP performance, AC 1 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU (sm_89, 24 SMs, 8 GiB GDDR6 128-bit), driver 595.99.02, power limit 55.00 W |
| Toolchain | g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0 / Build cuda_13.4.r13.4/compiler.38657139_0 |
| Protocol | warmup=2 iters=20 reps=5 rounds(GPU)=3 cool-down<=70C (max 120s) |
| Sizes / threads | W01/W02 N=33554432  W03 N=1024  W04 N=1024 2048 --bh 8 --d 64; W01 6  W02 6  W03 8  W04 20  (OMP_PROC_BIND=close OMP_PLACES=cores) |
| GPU state (loaded samples) | SM clock median 2655 MHz (1815–2700), power median 36.5 W, max 66 °C (237 samples, 250 ms) |
| CPU state | package mean temperature per run 70–96 °C (cool-down to ≤ 70 °C before each run) |

Times are ms per iteration: mean ± SD over the 5 repetitions of a run (20 timed iterations each, after 2 warm-ups). GPU stages ran in 3 interleaved round(s): median of the round means, SD = mean within-round SD, spread = (max − min) / median of the round means. CPU stages ran once each. × vs S1 = t(S1)/t; × prev = t(previous stage)/t (S3_TF32's previous stage is S2b); saved = (1 − t/t(S1))·100. Δ earlier = change vs the earlier recorded run (`docs/results/reference_values.csv`); |Δ| > 10% is marked ⚠ and explained below.

## W01 Vector Add — GPU, N = 33,554,432

C = A + B, FP32, N = 33,554,432.

| Stage | | ms (mean ± SD) | GB/s | × vs S1 | × prev | saved | Correct | error vs double | Δ earlier | SM / mem MHz |
|---|---|---|---|---|---|---|---|---|---|---|
| S1 | naive, 256 threads/block | 2.065 ± 0.0443 (spread 6.2%) | 195.0 | **1.00×** | – | 0.0% | YES | – | +20.4% ⚠ | – / – |
| S2 | coalesced, 128 threads/block | 1.803 ± 0.0019 (spread 0.1%) | 223.3 | **1.15×** | 1.15× | 12.7% | YES | – | +6.1% | – / – |

Factored: S1→S2 1.15× = **1.1×**

## W01 Vector Add — CPU, N = 33,554,432

C = A + B, FP32, N = 33,554,432.

| Stage | | ms (mean ± SD) | GB/s | × vs S1 | × prev | saved | Correct | error vs double | Δ earlier |
|---|---|---|---|---|---|---|---|---|---|
| S1 | scalar, -O0 | 80.029 ± 0.3686 | 5.0 | **1.00×** | – | 0.0% | YES | – | -20.6% ⚠ |
| S2 | OpenMP, 6 threads | 20.413 ± 0.3495 | 19.7 | **3.92×** | 3.92× | 74.5% | YES | – | +153.3% ⚠ |
| S3 | OpenMP + AVX2, 6 threads | 10.026 ± 0.0586 | 40.2 | **7.98×** | 2.04× | 87.5% | YES | – | +26.1% ⚠ |

Factored: S1→S2 3.92× · S2→S3 2.04× = **8.0×**

## W02 Dot Product — GPU, N = 33,554,432

s = Σ aᵢbᵢ, FP32 inputs, N = 33,554,432.

| Stage | | ms (mean ± SD) | GB/s | × vs S1 | × prev | saved | Correct | error vs double | Δ earlier | SM / mem MHz |
|---|---|---|---|---|---|---|---|---|---|---|
| S1 | naive (double atomics) | 43.462 ± 0.0138 (spread 0.6%) | 6.2 | **1.00×** | – | 0.0% | YES | 0.0e+00% | -0.5% | – / – |
| S2 | hierarchical block reduction | 1.259 ± 0.0953 (spread 1.7%) | 214.3 | **34.51×** | 34.51× | 97.1% | YES | 2.7e-05% | -18.4% ⚠ | – / – |

Factored: S1→S2 34.51× = **34.5×**

## W02 Dot Product — CPU, N = 33,554,432

s = Σ aᵢbᵢ, FP32 inputs, N = 33,554,432.

| Stage | | ms (mean ± SD) | GB/s | × vs S1 | × prev | saved | Correct | error vs double | Δ earlier |
|---|---|---|---|---|---|---|---|---|---|
| S1 | scalar, double accumulator | 63.861 ± 0.5462 | 4.2 | **1.00×** | – | 0.0% | YES | 2.1e-09% | +13.6% ⚠ |
| S2 | OpenMP, double partials, 6 threads | 4.803 ± 0.0309 | 55.9 | **13.30×** | 13.30× | 92.5% | YES | 0.0e+00% | -1.0% |
| S3 | OpenMP + AVX2/FMA (FP32 lanes), 6 threads | 3.993 ± 0.0198 | 67.2 | **15.99×** | 1.20× | 93.7% | YES | 0.018% | -5.8% |

Factored: S1→S2 13.30× · S2→S3 1.20× = **16.0×**

## W03 GEMM — GPU, N = 1,024

C = A × B, FP32, N × N matrices, N = 1,024.

| Stage | | ms (mean ± SD) | GFLOP/s | × vs S1 | × prev | saved | Correct | error vs double | Δ earlier | SM / mem MHz |
|---|---|---|---|---|---|---|---|---|---|---|
| S1 | naive | 2.458 ± 0.2314 (spread 2.9%) | 873.8 | **1.00×** | – | 0.0% | YES | 1.5e-04% | -16.4% ⚠ | – / – |
| S2 | 16×16 shared-memory tile | 1.806 ± 0.0983 (spread 0.3%) | 1189.1 | **1.36×** | 1.36× | 26.5% | YES | 1.5e-04% | -21.5% ⚠ | – / – |
| S3 | TF32 Tensor Cores (WMMA) | 0.5686 ± 0.0156 (spread 5.6%) | 3776.7 | **4.32×** | 3.18× | 76.9% | YES | 0.069% | -29.1% ⚠ | – / – |

Factored: S1→S2 1.36× · S2→S3 3.18× = **4.3×**

## W03 GEMM — CPU, N = 1,024

C = A × B, FP32, N × N matrices, N = 1,024.

| Stage | | ms (mean ± SD) | GFLOP/s | × vs S1 | × prev | saved | Correct | error vs double | Δ earlier |
|---|---|---|---|---|---|---|---|---|---|
| S1 | naive triple loop, -O0 | 2931.8 ± 1.623 | 0.7 | **1.00×** | – | 0.0% | YES | 1.6e-04% | -3.4% |
| S2 | 128-blocked + OpenMP, 8 threads | 88.960 ± 0.0584 | 24.1 | **32.96×** | 32.96× | 97.0% | YES | 1.6e-04% | -0.2% |
| S3 | AVX2/FMA + blocked + OpenMP, 8 threads | 25.592 ± 0.0183 | 83.9 | **114.56×** | 3.48× | 99.13% | YES | 1.5e-04% | +2.5% |

Factored: S1→S2 32.96× · S2→S3 3.48× = **114.6×**

## W04 Attention — GPU, N = 1,024

O = softmax(QKᵀ/√d)·V, FP32 inputs, N = 1,024.

| Stage | | ms (mean ± SD) | GFLOP/s | × vs S1 | × prev | saved | Correct | error vs double | Δ earlier | SM / mem MHz |
|---|---|---|---|---|---|---|---|---|---|---|
| S1 | naive, 3 kernels | 7.412 ± 0.0275 (spread 1.4%) | 289.7 | **1.00×** | – | 0.0% | YES | 1.8e-04% | -1.0% | – / – |
| S2a | tiled, unfused (3 kernels) | 2.173 ± 0.1741 (spread 4.8%) | 993.3 | **3.41×** | 3.41× | 70.7% | YES | 1.6e-04% | -1.3% | – / – |
| S2b | fused, FlashAttention-style | 0.7879 ± 0.0263 (spread 6.1%) | 2731.1 | **9.41×** | 2.76× | 89.4% | YES | 2.0e-04% | +0.2% | – / – |
| S3_FP16 | fused, FP16 Tensor Cores | 0.1741 ± 0.0003 (spread 0.2%) | 12332.5 | **42.57×** | 4.52× | 97.7% | YES | 0.038% | +0.1% | – / – |
| S3_TF32 | fused, TF32 Tensor Cores | 0.2773 ± 0.0008 (spread 0.3%) | 7744.5 | **26.73×** | 2.84× | 96.3% | YES | 0.038% | -0.0% | – / – |
| REF (ref.) | cuBLAS GEMMs + softmax (reference) | 0.7154 ± 0.0398 (spread 11.0%) | 3018.7 | 10.36× | – | – | YES | 1.6e-04% | -1.0% | – / – |

Factored: S1→S2a 3.41× · S2a→S2b 2.76× · S2b→S3_FP16 4.52× = **42.6×**

## W04 Attention — GPU, N = 2,048

O = softmax(QKᵀ/√d)·V, FP32 inputs, N = 2,048.

| Stage | | ms (mean ± SD) | GFLOP/s | × vs S1 | × prev | saved | Correct | error vs double | Δ earlier | SM / mem MHz |
|---|---|---|---|---|---|---|---|---|---|---|
| S1 | naive, 3 kernels | 32.163 ± 0.0887 (spread 0.7%) | 267.1 | **1.00×** | – | 0.0% | YES | 2.7e-04% | -1.1% | – / – |
| S2a | tiled, unfused (3 kernels) | 9.474 ± 0.1867 (spread 0.8%) | 906.9 | **3.39×** | 3.39× | 70.5% | YES | 1.7e-04% | -0.3% | – / – |
| S2b | fused, FlashAttention-style | 3.018 ± 0.0893 (spread 0.1%) | 2848.6 | **10.66×** | 3.14× | 90.6% | YES | 2.1e-04% | -0.2% | – / – |
| S3_FP16 | fused, FP16 Tensor Cores | 0.5984 ± 0.0027 (spread 6.9%) | 14356.4 | **53.75×** | 5.04× | 98.1% | YES | 0.03% | -7.4% | – / – |
| S3_TF32 | fused, TF32 Tensor Cores | 1.163 ± 0.0968 (spread 4.3%) | 7431.2 | **27.65×** | 2.59× | 96.4% | YES | 0.03% | +1.7% | – / – |
| REF (ref.) | cuBLAS GEMMs + softmax (reference) | 3.351 ± 0.2601 (spread 5.5%) | 2568.9 | 9.60× | – | – | YES | 1.7e-04% | -2.1% | – / – |

Factored: S1→S2a 3.39× · S2a→S2b 3.14× · S2b→S3_FP16 5.04× = **53.7×**

## W04 Attention — CPU, N = 1,024

O = softmax(QKᵀ/√d)·V, FP32 inputs, N = 1,024.

| Stage | | ms (mean ± SD) | GFLOP/s | × vs S1 | × prev | saved | Correct | error vs double | Δ earlier |
|---|---|---|---|---|---|---|---|---|---|
| S1 | scalar, full S row | 303.2 ± 0.7169 | 7.1 | **1.00×** | – | 0.0% | YES | 1.9e-04% | +6.1% |
| S2 | fused + blocked + OpenMP, 20 threads | 9.774 ± 0.3072 | 219.9 | **31.03×** | 31.03× | 96.8% | YES | 2.1e-04% | -1.6% |
| S3 | AVX2/FMA + vector exp, 20 threads | 3.283 ± 0.0295 | 654.1 | **92.36×** | 2.98× | 98.9% | YES | 1.9e-04% | -28.4% ⚠ |

Factored: S1→S2 31.03× · S2→S3 2.98× = **92.4×**

## W04 Attention — CPU, N = 2,048

O = softmax(QKᵀ/√d)·V, FP32 inputs, N = 2,048.

| Stage | | ms (mean ± SD) | GFLOP/s | × vs S1 | × prev | saved | Correct | error vs double | Δ earlier |
|---|---|---|---|---|---|---|---|---|---|
| S1 | scalar, full S row | 1197.7 ± 1.163 | 7.2 | **1.00×** | – | 0.0% | YES | 2.7e-04% | +6.6% |
| S2 | fused + blocked + OpenMP, 20 threads | 37.970 ± 0.7656 | 226.3 | **31.54×** | 31.54× | 96.8% | YES | 1.9e-04% | -4.1% |
| S3 | AVX2/FMA + vector exp, 20 threads | 12.866 ± 0.3060 | 668.0 | **93.09×** | 2.95× | 98.9% | YES | 2.1e-04% | -29.0% ⚠ |

Factored: S1→S2 31.54× · S2→S3 2.95× = **93.1×**

## CPU vs GPU (same N, same session)

| Workload | CPU S1 → GPU S1 | best CPU → best GPU | CPU S1 → best GPU |
|---|---|---|---|
| W01 Vector Add | 38.8× | S3 10.026 → S2 1.803 ms: **5.6×** | 44× |
| W02 Dot Product | 1.5× | S3 3.993 → S2 1.259 ms: **3.2×** | 51× |
| W03 GEMM | 1193.0× | S3 25.592 → S3 0.5686 ms: **45.0×** | 5156× |
| W04 Attention N=1024 | 40.9× | S3 3.283 → S3_FP16 0.1741 ms: **18.9×** | 1741× |
| W04 Attention N=2048 | 37.2× | S3 12.866 → S3_FP16 0.5984 ms: **21.5×** | 2001× |

## W01 reconciliation (GPU S1 vs S2)

From this run's CSV rows only (5 repetitions per round, every round), never from console output:

| | rounds × reps | per-round means (ms) | median ms | SD | GB/s | % of 256 GB/s nominal | memory clock per round (MHz) |
|---|---|---|---|---|---|---|---|
| S1 | 3 × 5 | 2.065, 1.938, 2.066 | 2.065 | 0.044 | 195.0 | 76% | –, –, – |
| S2 | 3 × 5 | 1.803, 1.803, 1.804 | 1.803 | 0.002 | 223.3 | 87% | –, –, – |

**Final W01 GPU: S1 2.065 ms, S2 1.803 ms, S2/S1 speedup 1.145×** (outside ±3%: both kernels stream at the DRAM limit).

Earlier W01 GPU figures and why they differ:

- S1: 1.715 ms over 5 row(s), `results/demo/csv/W01_S1.csv` (2026-10-21): 5 per-rep rows (B256)
- S1: 35.126 ms over 1 row(s), `results/raw/W01_VECTOR_ADD/W01_VECTOR_ADD_GPU_S1_BASE_NAIVE.csv` (2026-10-02): single row (repetition 1 only); 11 GB/s
- S2: 1.699 ms over 5 row(s), `results/demo/csv/W01_S2.csv` (2026-10-21): 5 per-rep rows (B128)
- S2: 1.859 ms over 5 row(s), `results/raw/W01_VECTOR_ADD/W01_VECTOR_ADD_GPU_S2_SWOPT_COALESCED_B128.csv` (2026-10-02): 5 per-rep rows, rep 1 2.08 ms
- Console output of an earlier session showed 2.011 / 1.779 ms. Console lines are not the recorded per-repetition rows and are not used.

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

- ⚠ **W01_GPU_S1**: 2.065 ms now vs 1.715 ms (`results/demo/csv/W01_S1.csv`, 2026-10-21): +20.4%.
- ⚠ **W01_CPU_S1**: 80.029 ms now vs 100.8 ms (`results/raw/W01_VECTOR_ADD/W01_VECTOR_ADD_CPU_S1_BASE.csv`, 2026-10-02): -20.6%.
- ⚠ **W01_CPU_S2**: 20.413 ms now vs 8.06 ms (`results/raw/W01_VECTOR_ADD/W01_VECTOR_ADD_CPU_S2_SWOPT_OMP_T6.csv`, 2026-10-02): +153.3%.
- ⚠ **W01_CPU_S3**: 10.026 ms now vs 7.952 ms (`results/raw/W01_VECTOR_ADD/W01_VECTOR_ADD_CPU_S3_HWACC_AVX2_T6.csv`, 2026-10-02): +26.1%.
- ⚠ **W02_GPU_S2**: 1.259 ms now vs 1.544 ms (`results/raw/W02_DOT_PRODUCT/W02_DOT_PRODUCT_GPU_S2_SWOPT_BLOCK_REDUCE_V2.csv`, 2026-10-03): -18.4%.
- ⚠ **W02_CPU_S1**: 63.861 ms now vs 56.24 ms (`results/raw/W02_DOT_PRODUCT/W02_DOT_PRODUCT_CPU_S1_BASE_V2.csv`, 2026-10-03): +13.6%.
- ⚠ **W03_GPU_S1**: 2.458 ms now vs 2.941 ms (`results/raw/W03_GEMM/W03_GEMM_GPU_S1_BASE_NAIVE_V1.csv`, 2026-10): -16.4%.
- ⚠ **W03_GPU_S2**: 1.806 ms now vs 2.3 ms (`results/raw/W03_GEMM/W03_GEMM_GPU_S2_SWOPT_SHARED_TILE_V1.csv`, 2026-10): -21.5%.
- ⚠ **W03_GPU_S3**: 0.5686 ms now vs 0.8015 ms (`results/raw/W03_GEMM/W03_GEMM_GPU_S3_HWACC_TENSOR_CORE_V2.csv`, 2026-10): -29.1%.
- ⚠ **W04_CPU_S3_N1024**: 3.283 ms now vs 4.585 ms (`results/raw/W04_ATTENTION/20261105-174743 nsweep (tools/w04_summary.py)`, 2026-11-05): -28.4%.
- ⚠ **W04_CPU_S3_N2048**: 12.866 ms now vs 18.13 ms (`results/raw/W04_ATTENTION/20261105-174743 nsweep (tools/w04_summary.py)`, 2026-11-05): -29.0%.

_Explanations: not written yet (docs/results/deviation_notes.md)._

## Files

- consolidated CSV: `docs/results/final_results.csv` (one row per stage and N)
- earlier values: `docs/results/reference_values.csv`
- raw CSVs, logs, `runs.csv`, `gpu_state.csv`: `results/raw/FINAL/20261106-134741/` (private repository)
- reproduce: `bash scripts/run_all.sh`, then `python3 tools/final_summary.py results/raw/FINAL/<run> --publish-docs`

