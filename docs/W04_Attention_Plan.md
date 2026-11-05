# W04 Plan: Fused Attention (Deep-Learning Workload)

**Project:** GPU-CPU-Performance-Profiling_and-Optimization
**Status:** P0–P2 done (CPU stages; GPU S1, S2a, REF_CUBLAS); P3 step 1 (S2b code and tests) done, sweep and E3 next. See `docs/W04_PROGRESS.md` for what is measured
**Hardware:** Intel Core i7-13650HX (6P+8E, AVX2/FMA) · RTX 4060 Laptop GPU (sm_89, 24 SMs, 8 GiB, 128-bit GDDR6)
**Date:** 4 Nov 2026

---

## 1. Why attention is W04

W01–W03 each show one behaviour: W01 is memory-bound, W02 is reduction-bound, and W03 is compute-bound with data reuse. Scaled dot-product attention combines all three in one operator:

```
S = Q·Kᵀ / √d        → GEMM (W03 behaviour)
P = softmax(S)       → row max + row sum reductions (W02) and element-wise exp (W01)
O = P·V              → GEMM (W03)
```

The new lesson is **fusion**. In W01–W03, operational intensity (OI) is fixed by the algorithm, so optimization only moves a point *up* the Roofline. Attention is different. The naive version writes the N×N matrix S/P to DRAM and reads it back. The fused (FlashAttention) version keeps tiles of it in shared memory. That removes most of the DRAM traffic and moves the point *right* on the Roofline. This is the IO-aware principle of the FlashAttention paper, and it is the most important optimization that deep-learning compilers perform automatically.

**Alternative considered:** Conv2D, the original W04 idea, implemented as implicit GEMM. It is still relevant, but it mostly repeats the W03 GEMM story. Keep it as a possible W05.

## 2. What the best similar projects do well, and what W04 takes from them

| Project | Its best quality | Adopted in W04 |
|---|---|---|
| siboehm, *CUDA matmul worklog* (SGEMM_CUDA) | One kernel per optimization step, each measured against cuBLAS, which reaches 100%. Lower bounds are computed before any code is written. Autotuning is its own step. | Each stage reports % of a **vendor reference** (cuBLAS batched GEMM, optionally PyTorch SDPA). Theoretical lower bounds are written before implementation. Autotuning is a separate experiment. |
| Dao et al., *FlashAttention* (NeurIPS 2022) | IO-complexity analysis: count HBM bytes, not just FLOPs. Reports about 7.6× on the attention computation for GPT-2. | Measured DRAM bytes from Nsight for every stage, and a **measured-traffic Roofline** where fused points move right. |
| tspeterkim, *flash-attention-minimal* | About 100 readable lines and paper notation. Honest caveats: thread-per-row matmul gets slow at long sequence lengths. | Readable S2b kernel using paper notation, a "known limitations" section per stage, and S3 fixing the slow inner matmul with Tensor Cores. |
| Karpathy, *llm.c* `dev/cuda` | A library of kernels for each layer, from simple to fast, plus parity tests against a PyTorch reference. | Kernel versions kept side by side under the S1/S2/S3 naming, plus a correctness harness that every stage must pass. |
| OpenAI, *Triton fused-attention tutorial* | Compiler-generated FlashAttention-2 with autotuned block sizes. | Optional S4: the same shapes in Triton or `torch.compile`, comparing hand-written CUDA with compiler-generated code. This is the main compiler-job link. |

## 3. Workload definition

| Item | Value |
|---|---|
| Operation | Forward pass, non-causal (causal is an optional variant): `O = softmax(QKᵀ/√d)·V` |
| Shapes | batch×heads `BH = 8`; head dim `d = 64`; sequence length `N ∈ {512, 1024, 2048, 4096}` (8192 if memory allows) |
| Layout | Row-major `[BH][N][d]`, FP32 inputs (FP16 copies for S3) |
| FLOPs (model) | `4·BH·N²·d` (two GEMMs); softmax counted separately as about `5·BH·N²` |
| Minimum bytes | `4 tensors × BH·N·d × 4 B` (Q, K, V, O). **Algorithmic OI = N/4** (256 FLOP/B at N=1024) |
| Naive bytes | Minimum plus about `4 × BH·N² × 4 B` (write S, read S, write P, read P). **OI ≈ d/4 ≈ 16 FLOP/B**, independent of N. As implemented (decision C17, P2): S2a/REF make 5 passes (≈ 20·BH·N² B, OI ≈ 12.8) and GPU S1 makes 7 (≈ 28·BH·N² B, OI ≈ 9.1), before L2 hits. P6 uses these counts |

The gap between those two OI values (≈16 → ≈256 at N=1024) is the headline quantity W04 measures.

## 4. Stages

**Cache sizes (decision C7, measured from sysfs in P1):** P-core L1d 48 KiB and L2 1.25 MiB (shared by the core's two SMT threads); E-core L1d 32 KiB and L2 2 MiB per 4-core cluster; L3 24 MiB. An earlier version of this plan assumed 2 MB of L2 per P-core.

All stages use the existing ID convention: `W04_ATTENTION_<CPU|GPU>_<S#>_<BASE|SWOPT|HWACC>_<VARIANT>`.

### GPU

| Stage | Implementation | What it isolates |
|---|---|---|
| **S1_BASE_NAIVE** | 3 kernels: naive GEMM (QKᵀ) → one-thread-per-row softmax → naive GEMM (PV). Materializes S in global memory. | Baseline |
| **S2a_SWOPT_UNFUSED** | Same 3-kernel structure, but with W03-S2 shared-memory tiled GEMMs and a warp-shuffle softmax (one warp per row). | Effect of better kernels **without** fusion |
| **S2b_SWOPT_FUSED** | One kernel, FlashAttention-2 style. Each block owns `Br` query rows, loops over `Bc` K/V tiles in shared memory, uses online softmax (running max m and sum l), and writes only O. | Effect of **fusion**, compared with S2a |
| **S3_HWACC_TENSOR_CORE** | S2b with WMMA for both inner GEMMs: FP16 inputs, FP32 accumulation, softmax kept in FP32. A TF32 variant keeps it consistent with W03. | Effect of Tensor Cores and reduced precision |
| *REF_CUBLAS* (not a stage) | `cublasSgemmStridedBatched` + S2a softmax + batched GEMM | Vendor unfused ceiling |
| *REF_SDPA* (optional) | PyTorch `scaled_dot_product_attention` | Vendor fused ceiling |
| *S4_COMPILER* (optional) | Triton fused-attention tutorial, or `torch.compile`, on the same shapes | Hand-written vs compiler-generated |

**Why S2 is split into a and b:** without S2a, the S1→S2b speedup mixes "better GEMM" with "fusion". With it, the speedup factors into two parts: kernel quality (S1→S2a) times fusion (S2a→S2b). This is the main methodological improvement over W01–W03.

### CPU

| Stage | Implementation |
|---|---|
| **S1_BASE** | Scalar triple loops; full S row per query in a buffer (per head N×N); `-O2`, no OpenMP |
| **S2_SWOPT_BLOCKED_OMP** | Fused and blocked: OpenMP over (head, query-block); K/V tiles of Bc rows sized for L1d (48 KiB per P-core), with one head's K and V (512·N B at d=64) re-read from L2 (**1.25 MiB per P-core**, measured; it fits up to N ≈ 2048); online softmax; no N×N buffer |
| **S3_HWACC_AVX2** | S2 with AVX2/FMA intrinsics for both dot-product loops and a vectorized polynomial `exp` (8 lanes). Confirm `vfmadd*ps` with objdump. |

**Compiler study (CPU S3):** `-O2` vs `-O3 -march=native` vs `-O3 -ffast-math`, auto-vectorized vs hand intrinsics. Record speed and error for each. On GPU, do the same with `--use_fast_math` (`expf` → `__expf`, which appears as `MUFU.EX2` in SASS).

## 5. Correctness rules

- Reference: double-precision CPU attention with numerically stable softmax (subtract the row max), using OpenMP.
- Report both `max_abs_err` and `max_rel_err` against double, as the W02 lesson requires.
- Tolerances: FP32 stages ≤ 1e-4 relative; TF32 ≤ 5e-3; FP16 ≤ 1e-2. Record the actual values; tolerances only decide the YES/NO flag.
- **Stability test:** a "large logits" input (Q, K scaled ×12; changed from ×30 in P0, decision C2b: at ×30 the FP32 rounding of Q·K alone exceeds the 1e-4 FP32 tolerance). Naive softmax without max subtraction overflows, while online softmax must pass. This tests the algorithm, not just the arithmetic.
- Every stage also checks that each output row of `P` sums to 1 (±tol) on a sampled subset.

## 6. Measurements and profiling

| Experiment | Tool | Output |
|---|---|---|
| E1 Runtime vs N, all stages | benchmark harness (20 iterations, 5 reps, 2 warmup) | log-log plot; S1 vs fused scaling |
| E2 Speedups at N=1024 and 2048 | harness | factored bars: kernel quality × fusion × Tensor Core |
| E3 **Measured DRAM bytes** | `ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum` | measured OI per stage → **measured-traffic Roofline** (points move right) |
| E4 Kernel diagnosis | ncu sections: SpeedOfLight, MemoryWorkloadAnalysis, ComputeWorkloadAnalysis, Occupancy, WarpStateStats, SourceCounters | stall reasons (Barrier, MIO Throttle, Long Scoreboard), shared-memory bank conflicts, tensor-pipe utilization |
| E5 Launch overhead | `nsys profile` on S1/S2a (3 kernels) vs S2b (1 kernel) | timeline gaps: first real use of Nsight Systems in the project |
| E6 Peak memory vs N | `cudaMemGetInfo` before/after | O(N²) vs O(N) footprint plot |
| E7 Autotune | sweep `Br, Bc ∈ {16, 32, 64}` and warps per block | heatmap; selected configuration |
| E8 CPU | VTune hotspots, memory-access, uarch-exploration; thread sweep 1–20 | LLC-miss drop S1→S2; L2 hit rate of S2 at N=2048 (K/V fits the 1.25 MiB P-core L2) vs N=4096 (does not); FMA/vector use in S3 |
| E9 *Decode case* (optional) | `N_q = 1`, `N_kv = 4096` | inference decode is memory-bound, which ties back to W01/W02 |

**Hypotheses, written before measuring:**
- H1: S1 time is dominated by S/P traffic and softmax, not by the GEMMs.
- H2: S2a→S2b gains grow with N, because the bytes fusion saves grow as N².
- H3: S3 is limited by softmax/`exp` and shared-memory traffic, not by Tensor Core throughput. W03-S3 showed the tensor pipe only ~10% active.
- H4: CPU S2 gains come mainly from L2 reuse (per-head K/V resident in the 1.25 MiB P-core L2 up to N ≈ 2048); S3 gains depend on vectorizing `exp`.

Each one is confirmed or rejected with a specific counter.

**Shared-memory budget (sm_89):** with FP32 and `Br = Bc = 64, d = 64`, the Q, K and V tiles take 48 KB, right at the default per-block limit. Start with `Bc = 32` or opt in to more with `cudaFuncSetAttribute`. In FP16 (S3) the same tiles take 24 KB.

## 7. Files and harness (new code only in the clean project)

```
src/common/attention_ref.hpp        double reference, input generator, error metrics
src/common/bench_cli.hpp            ONE parser for every W04 binary (named flags):
                                    --n --bh --d --iters --reps --warmup --out --seed
src/cpu/W04_ATTENTION/W04_ATTENTION_CPU_S1_BASE.cpp
src/cpu/W04_ATTENTION/W04_ATTENTION_CPU_S2_SWOPT_BLOCKED_OMP.cpp
src/cpu/W04_ATTENTION/W04_ATTENTION_CPU_S3_HWACC_AVX2.cpp
src/gpu/W04_ATTENTION/W04_ATTENTION_GPU_S1_BASE_NAIVE.cu
src/gpu/W04_ATTENTION/W04_ATTENTION_GPU_S2a_SWOPT_UNFUSED.cu
src/gpu/W04_ATTENTION/W04_ATTENTION_GPU_S2b_SWOPT_FUSED.cu
src/gpu/W04_ATTENTION/W04_ATTENTION_GPU_S3_HWACC_TENSOR_CORE.cu
src/gpu/W04_ATTENTION/W04_ATTENTION_GPU_REF_CUBLAS.cu
scripts/w04_sweep.sh                N-sweep + autotune, timestamped output folders
scripts/w04_profile.sh              ncu/nsys/vtune, one kernel per command
tools/plot_w04.py                   E1, E2, E3 Roofline, E6, E7 plots
tests/test_w04_correctness.sh       small-N all-stage pass/fail (runs in under 1 min)
```

**CSV schema** (one row per repetition): `workload, stage, variant, device, precision, n, bh, d, iters, rep, runtime_ms, gflops, alg_bytes, max_abs_err, max_rel_err, correct, peak_mem_bytes, br, bc, compiler_flags, timestamp, git_sha`

Lessons from W01–W03 built in: one CLI for every binary (no positional/named mix-ups), timestamped and never-overwritten outputs (no repeat of the W01 console/CSV conflict), the git SHA in every row, and paired runs within one session.

## 8. Execution phases

| Phase | Deliverable | Done when |
|---|---|---|
| P0 | Harness, CLI, double reference, CSV writer, Makefile targets | `make w04` builds; reference self-test passes |
| P1 | CPU S1 → S2 → S3 + compiler-flag study | all pass the correctness test, including large logits; thread sweep CSV |
| P2 | GPU S1, S2a, REF_CUBLAS | pass; S2a within a stated % of the cuBLAS reference |
| P3 | GPU S2b fused | passes; ncu shows reduced DRAM bytes vs S2a |
| P4 | GPU S3 Tensor Core (FP16; TF32 variant) | passes tolerance; `HMMA` present in SASS |
| P5 | Autotune (E7), optional S4 Triton / SDPA | selected configuration recorded |
| P6 | Profiling E3–E6, E8, plots | each hypothesis H1–H4 marked supported or rejected, with the metric |
| P7 | Add W04 to the demo kit race and `plot_race.py`; update docs | `./demo.sh race` shows W04 |

**Commits (clean private repo):**
1. `feat(w04): attention harness, reference and correctness tests`
2. `feat(w04-cpu): S1-S3 attention + compiler-flag study`
3. `feat(w04-gpu): S1, S2a unfused, cuBLAS reference`
4. `feat(w04-gpu): S2b fused FlashAttention-style kernel`
5. `feat(w04-gpu): S3 Tensor Core attention`
6. `data(w04): sweeps, profiles, plots`
7. `docs(w04): analysis + demo integration`

After that comes the combined re-test of W01–W04, then the README, then the sanitized public copy with one clean commit.

## 9. Risks

| Risk | Mitigation |
|---|---|
| The WMMA fragment layout is opaque, so softmax can't act on fragments directly | Store the accumulator to shared memory, run softmax there, reload. Simple first, faster later. |
| S2b is slower than S2a at small N (as the minimal repo found) | Expected and worth reporting. Show the crossover N. |
| Laptop clocks and power cap drift | Mains power, Performance mode, record `nvidia-smi` clocks per run, paired runs. **Measured:** CPU runs at 81–96 °C under load (P1); the GPU is held at its 55 W default power limit, so each GPU stage runs at its own clock (P2). Mitigation in use: interleaved rounds with per-run temperature/clock logging (`w04_controlled.sh`, `gpu-nsweep`); fixed clocks need root (user decision) |
| A 4096-size double reference is slow | Full reference up to N=2048; sampled rows above that |
| Scope creep (backward pass, causal, Triton) | Backward pass out of scope; causal, S4 and E9 optional |

## Sources

- S. Boehm, *How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance* — https://siboehm.com/articles/22/CUDA-MMM · code https://github.com/siboehm/SGEMM_CUDA
- T. Dao et al., *FlashAttention*, NeurIPS 2022 — https://arxiv.org/abs/2205.14135
- P. Kim, *flash-attention-minimal* — https://github.com/tspeterkim/flash-attention-minimal
- A. Karpathy, *llm.c* — https://github.com/karpathy/llm.c
- OpenAI Triton, *Fused Attention tutorial* — https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
- NVIDIA job postings: Senior AI Compiler Engineer (MLIR); Senior Software Engineer, DL Compilers
