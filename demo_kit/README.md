# Demo kit — how to run it and read the results

The kit re-runs the project's benchmarks live and prints a speedup scoreboard. It also draws charts, and it
shows the profiler evidence behind the speedups. `DEMO_RUNBOOK.md` is the stage script (what to say at each
cue). This file is the how-to.

| Cue | Command | Time | What it shows |
|---|---|---|---|
| Race | `./demo.sh race [workloads]` | 10–45 s | every stage re-run, scoreboard (×, %), charts |
| Fusion | `./demo.sh fusion` | ~60 s | W04: DRAM bytes S2a vs S2b (ncu, sudo), then the time fusion buys |
| Nsight Compute | `./demo.sh ncu`, `ncu-gui`, `ncu-side` | ~40 s | W03: Tensor Core SASS, counters S1/S2/S3 |
| VTune | `./demo.sh vtune`, `vtune-gui`, `vtune-side` | ~40 s | W03 CPU: naive hotspot vs AVX2/FMA + OpenMP |
| Memory wall | `./demo.sh wall` | ~15 s | W01: DRAM near peak, SM idle |
| Hardware | `./demo.sh hw` | ~2 s | SMs, registers, L2, caches |
| Charts | `./demo.sh plot [run_dir]` | ~3 s | charts + tables for the newest (or a given) race |

`./demo.sh` with no argument opens a menu. `./demo.sh check` is the preflight.

## 1. Setup

The kit uses the binaries of the repository it lives in, so a fresh clone works:

```bash
git clone https://github.com/Fe3w/GPU-CPU-Performance-Profiling_and-Optimization.git
cd GPU-CPU-Performance-Profiling_and-Optimization
make legacy w04             # W01-W03 and W04 binaries, into build/
./demo_kit/demo.sh check    # every binary, ncu/vtune, GPU state, fusion backup
```

Every path and tool is set in `demo.conf` and can be overridden from the environment:

| Variable | Default | Use |
|---|---|---|
| `PROJECT` | the repository that contains `demo_kit/` | another checkout or build |
| `BUILD` | `$PROJECT/build` | binaries somewhere else |
| `NCU`, `NCU_UI` | `/opt/nvidia/nsight-compute/<newest>/ncu`, else `ncu` on `PATH` | another Nsight Compute |
| `VTUNE`, `VTUNE_GUI` | `vtune`, `vtune-gui` on `PATH` | VTune not on `PATH` |
| `NCU_SUDO` | `sudo` | `NCU_SUDO=""` if counters are open to users (`NVreg_RestrictProfilingOnlyToAdminUsers=0`) |
| `ROUNDS` | 1 | interleaved rounds per race |

Example: `PROJECT=/data/other-checkout NCU=/usr/local/cuda/bin/ncu ./demo_kit/demo.sh race w03`.
Race CSVs go to `$PROJECT/results/live/<timestamp>/`, profiler reports to `$PROJECT/profiling/demo/`
(both git-ignored). `fusion --backup` reads the recorded Nsight Compute run from `profiling/` or, in the
public repository, from `Comparison final/runs/profiling/`.

Before timing anything: plug in the charger, select the Performance power mode, and close the browser and
other GPU apps.

## 2. Run one workload, several, or all

```bash
./demo_kit/demo.sh race w01          # W01 Vector Add, GPU S1 vs S2
./demo_kit/demo.sh race w02          # W02 Dot Product, GPU S1 vs S2
./demo_kit/demo.sh race w03          # W03 GEMM N=1024, GPU S1, S2, S3
./demo_kit/demo.sh race w04          # W04 Attention N=2048, GPU S1, S2a, S2b, S3_FP16, S3_TF32 + cuBLAS REF
./demo_kit/demo.sh race w04cpu       # W04 Attention N=2048, CPU S1, S2, S3
./demo_kit/demo.sh race w03 w04      # any combination
./demo_kit/demo.sh race gpu          # W01-W04 GPU
./demo_kit/demo.sh race              # all: W01-W04 GPU + W04 CPU, plus CPU vs GPU comparisons
ROUNDS=3 ./demo_kit/demo.sh race w04 # accurate mode: 3 interleaved rounds, median over rounds
NOPLOT=1 ./demo_kit/demo.sh race w04 # scoreboard only, no charts
```
W04 GPU stages use their tuned tile defaults (`src/common/w04_tuned.hpp`) and the project protocol
(2 warm-ups, 20 iterations, 5 repetitions). W04 CPU runs at the same N, so CPU and GPU can be compared directly.
Only CPU S1 runs 1 iteration × 3 repetitions, because one pass takes ~1.2 s. Shapes and protocols are set in
`demo.conf` (`W04_N`, `W04_ARGS`, `W04_CPU_*`).

## 3. Reading the scoreboard

| Column | Meaning |
|---|---|
| ms ± SD | mean time per iteration ± SD over repetitions (with rounds: median over rounds, plus the round spread) |
| rate | GFLOP/s (compute) or GB/s (memory-bound workloads) |
| **×S1** | how many times faster than the workload's S1: t(S1) / t(stage) |
| **×prev** | how many times faster than the previous stage. S3_TF32 is compared with S2b, not with S3_FP16 |
| **saved** | % less time than S1: (1 − t / t(S1)) · 100. 10× faster = 90% saved; 50× = 98% |
| **%REF** | speed as a % of the cuBLAS reference (W04): 100% = as fast, 556% = 5.56× faster |
| factored | the chain of ×prev factors to the fastest stage; their product is its ×S1 |
| best stage per workload | one line per workload, when more than one ran |
| cross comparisons | W04 CPU vs GPU at the same N (when both `w04` and `w04cpu` ran) |

Example. These are measured live numbers (race `20261106_123456`, one round, RTX 4060 Laptop at its 55 W cap,
i7-13650HX), not formal results:

| W04 N=2048 | ms | ×S1 | ×prev | saved |
|---|---|---|---|---|
| GPU S1 naive | 33.12 | 1.00× | – | 0% |
| GPU S2a unfused, tiled | 9.73 | 3.40× | 3.40× | 70.6% |
| GPU S2b fused, 64×32 W4 | 3.01 | 11.0× | 3.23× | 90.9% |
| GPU S3 FP16 Tensor Core | 0.600 | 55.2× | 5.01× | 98.2% |
| GPU S3 TF32 Tensor Core | 1.05 | 31.5× | 2.87× | 96.8% |
| REF cuBLAS (reference) | 3.46 | 9.58× | – | – |

Factored: kernel quality 3.40× · fusion + register tiling 3.23× · Tensor Cores 5.01× = 55.2×.
CPU vs GPU at N=2048: S1 36.5×, CPU S3 AVX2 (14.0 ms) → GPU S3 FP16 23.4×, CPU S1 → GPU S3 FP16 2015×.

## 4. Charts and tables

Every race ends by running `tools/plot_race.py` on its folder. That writes:
- `race_<stamp>.png`: runtime per stage, one panel per workload. The axis is logarithmic when a workload spans
  more than 20×. The labels show ms, ×S1 and −% time. The cuBLAS reference is a dashed line.
- `race_speedup_<stamp>.png`: every stage of every workload as × vs its own S1, on one log axis.
- `race_<stamp>.md` and `race_<stamp>.csv`: the full tables (all columns above, the factored chain, best stage,
  CPU vs GPU).

```bash
./demo_kit/demo.sh plot                       # newest race
./demo_kit/demo.sh plot <PROJECT>/results/live/<stamp>
python3 demo_kit/tools/plot_race.py --compare # every run side by side: run-to-run drift
python3 demo_kit/tools/plot_race.py --selftest
```
Only files named `W0<k>[CPU]_<stage>[_r<round>].csv` are read, for example `W04_S2a.csv`,
`W04_S3_TF32_r2.csv` or `W04CPU_S1.csv`. Profiler and preflight CSVs in the same folder are ignored. S2a and
S2b, and S3_FP16 and S3_TF32, stay separate.

## 5. The fusion cue (W04)

```bash
./demo_kit/demo.sh fusion --print    # only print the ncu commands
./demo_kit/demo.sh fusion            # live: prints the commands, asks for sudo once, runs them
./demo_kit/demo.sh fusion --backup   # no sudo: the recorded P6 profile + the live timing
FUSION_TILE16=1 ./demo_kit/demo.sh fusion   # also profile S2b 16x16 W8
```
1. ncu reads the GPU's hardware counters. That needs root, so the cue first prints every command (one kernel
   each, `-c 1`): S2a's three kernels (`qk_tiled`, `softmax_rows_warp`, `pv_tiled`) and S2b's single
   `flash_fwd_fp32`.
2. **What fusion removes**: DRAM bytes per forward pass, the live run next to the recorded one. Recorded
   (P6, N=2048): S2a reads 298.9 MB and writes 263.1 MB. S2b reads 19.5 MB. That is **28.8× fewer bytes**
   (24.1× with writes raised to their compulsory minimum), and operational intensity goes from 15 to ~360
   FLOP/B. Measured writes are a lower bound, because dirty L2 lines at kernel end are not counted.
   Live run `20261106_125049`: S2a 289.9 + 244.7 MB, S2b 12.7 + 5.2 MB, so **29.8× fewer raw, 31.4×
   write-corrected** (96.8% of the bytes removed). S2b's read bytes vary between runs (12.6–21.2 MB, because
   K/V re-reads sometimes hit L2), and so does whether O's 4.2 MB reaches DRAM before the kernel ends (5.2 MB
   live vs 0 MB recorded).
3. **What fusion buys**: benchmark times of S2a, S2b 16×16 W8 (fused, same per-thread work as S2a) and S2b tuned.
   - S2a → S2b 16×16 is **fusion alone**, ~1.13–1.16× (live `20261106_125049`: 9.46 → 8.18 ms, 1.16×).
   - S2b 16×16 → tuned is **register tiling**, ~2.75–2.78× (live: 8.18 → 2.94 ms, 2.78×).
   - The bytes drop ~25–31×, but the time drops only ~13–15% from fusion, because only S2a's softmax is DRAM-bound
     (86% of peak); its GEMMs are not (~11%).
   - ncu durations are at base clock with flushed caches, so they are not benchmark times.

## 6. Caveats

- The laptop GPU runs at its 55 W power cap, so each stage runs at its own clock. Ratios against the cuBLAS
  reference hold at the cap only: at a locked 1500 MHz SM clock they fall ~30%. Ratios among S2b and S3 move
  ≤ 6%.
- The first repetition can be up to 17% faster, before the cap engages. Use `ROUNDS=3` for numbers you quote.
- Live runs differ from the formal results by a few % (clocks, temperature). The CPU runs at 81–96 °C under
  load. Compare stages within one run, not across runs.
- The live race writes to `$PROJECT/results/live/<timestamp>/` and nothing is overwritten. The fusion
  profiles go to `$PROJECT/profiling/demo/live_<timestamp>_fusion/`.
