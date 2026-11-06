# Comparison final — index

Everything needed to understand the project and check the CPU/GPU comparison, in one place. Start with **`Complete Brief.md`**.

| Path | What it is |
|---|---|
| `Complete Brief.md` | definition, goals and objectives, method, workloads, what was done, results, explanations, comparison with earlier records, fixes, limitations, conclusions |
| **`data/`** | |
| `data/final_results.csv` | the consolidated table: one row per stage and size (34 rows), with mean, SD, rate, × vs S1, × vs previous stage, % saved, % of the cuBLAS reference, correctness, error vs double, earlier value and Δ, median SM and memory clock |
| `data/RESULTS_SUMMARY.md` | the same results as readable tables, plus CPU vs GPU, the W01 reconciliation, the correctness table and the deviation explanations |
| `data/reference_values.csv` | earlier recorded values with their source files (used for the Δ column) |
| `data/deviation_notes.md` | why each stage that differs > 10% from its earlier record differs |
| `data/W04_phase_results.md` | W04 results phase by phase (P1–P6), as recorded during development |
| `data/w04_plot_data.csv` | data behind the W04 development figures (Roofline points, factored speedups) |
| **`graphs/`** | |
| `graphs/comparison/1_runtime_every_stage.png` | time of every stage, CPU and GPU, per workload |
| `graphs/comparison/2_best_cpu_vs_best_gpu.png` | fastest CPU stage vs fastest GPU stage per workload |
| `graphs/comparison/3_step_speedups.png` | speedup of every optimization step |
| `graphs/comparison/4_final_vs_earlier.png` | final run vs earlier records, ±10% band |
| `graphs/final/speedup_all.png` | every stage as × vs its own S1 |
| `graphs/w04_development/` | W04 runtime vs N (e1), factored speedups (e2), measured-traffic Roofline (e3), memory vs N (e6), autotune heatmaps (e7) |
| **`runs/`** (CSV and text only; binary profiler reports are not included; in the public repository, the W01 VTune summaries and the E5 Nsight Systems console logs are also left out because they hold machine metadata) | |
| `runs/final/20261106-151143/` | **the final paired re-test** (after the W03 GPU S3 optimization): every CSV, every log, `runs.csv` (per run: threads, temperature, clocks, exit code), `gpu_state.csv`, `run_info.txt`, `commands.log` |
| `runs/final/20261106-141652_previous/` | the first release run (v1.0), with the first W03 GPU S3 kernel; agrees with the final run within a few % on every other stage |
| `runs/w03_s3_ab/` | paired A/B of the first and the optimized W03 GPU S3 kernel at N = 1024 and 2048 (3 alternating rounds, clocks per run, `summary.txt`) |
| `runs/final/20261106-134741_superseded/` | first full run of the re-test: W01 GPU S1 at the lower memory clock, W01 CPU S2/S3 built -O0 (see the brief, §8) |
| `runs/w04_development/20261106-114338_gpu_nsweep/` | W04 GPU N sweep 512–8192, all stages, 3 rounds (source of the W04 figures) |
| `runs/w04_development/20261106-115441_gpu_locked_clock_1500MHz/` | the same sweep at a locked 1500 MHz SM clock (equal-clock check) |
| `runs/w04_development/20261106-103416_gpu_fusion_only/` | fusion alone: S2a vs S2b 16×16 W8 vs S2b tuned |
| `runs/w04_development/20261106-031127_gpu_autotune_E7/` | tile × warps autotune screen and finals |
| `runs/w04_development/20261105-174743_cpu_all_nsweep_tiles_threads_flags/` | W04 CPU N sweep, tiles, 1–20 thread sweep, compiler-flag study |
| `runs/w04_development/20261105-181924-controlled_cpu_controlled_rerun/` | temperature-logged CPU tile rerun |
| `runs/profiling/*_E3_dram_bytes/` | Nsight Compute DRAM bytes per stage and kernel (raw / details CSV) |
| `runs/profiling/*_E4_stalls_occupancy/`, `*_E4x_tf32_fix_and_stall_sources/` | warp stalls, occupancy, pipe use, SASS-level stall sources |
| `runs/profiling/*_E5_nsys_launch_gaps/` | Nsight Systems launch gaps |
| `runs/profiling/*_E8_cpu_counters/` | `perf stat` counters, VTune hotspot exports |
| `runs/profiling/*_live_fusion_bytes/` | live Nsight Compute fusion comparison (demo) |
| `runs/earlier_W01-W03/` | the October W01–W03 records (raw CSVs, processed summaries, demo CSVs) the final run is compared with |
| **`scripts/make_graphs.py`** | regenerates `graphs/comparison/` from `data/` |

Hardware: Intel Core i7-13650HX + NVIDIA RTX 4060 Laptop (55 W), Ubuntu 24.04, CUDA 13.4, driver 595.99. Final run: `20261106-151143`, commit `7f0e5f6`, public release tag `v1.1` (the first release, tag `v1.0`, is run `20261106-141652`). The live demo kit that re-runs these benchmarks is `demo_kit/` at the repository root.
