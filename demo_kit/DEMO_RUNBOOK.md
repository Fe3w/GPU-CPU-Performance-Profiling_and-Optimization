# Live Demo Runbook — CPU/GPU Profiling Project

The stage script for the live part of the talk: what to run, what to say, and what to do when something fails. `README.md` in this folder is the how-to (setup, options, reading the scoreboard). Every cue has a fallback, so nothing on stage depends on a single live collection.

| Cue | Command | Time | What the audience sees | If it fails |
|---|---|---|---|---|
| A. The race | `./demo.sh race` | ~45 s | W01–W04 GPU stages and W04 CPU re-run live, with a speedup scoreboard and charts | `./demo.sh plot <earlier race folder>`, or the recorded video (`./demo.sh video`, if `VIDEO` points to one) |
| A2. W04 only | `./demo.sh race w04` | ~10 s | W04 GPU S1 → S2a → S2b → S3 FP16/TF32 + cuBLAS reference, factored speedups | as above |
| B. Nsight Compute | `./demo.sh ncu`, then `./demo.sh ncu-gui` | ~60 s | HMMA instructions in SASS, a live profile of the Tensor Core kernel, a counter table S1 / S2 / S3 | The table falls back to the `prep.sh` backup reports; use the screenshot slide |
| C. VTune | `./demo.sh vtune`, then `./demo.sh vtune-gui` | ~60 s | Naive GEMM keeping one core busy, against S3 OpenMP workers running FMA | The backup `vtune_W03_S1` and S3 results; the screenshot slide |
| D. Fusion | `./demo.sh fusion` | ~60 s | ncu DRAM bytes S2a vs S2b (~25–31× fewer), then the time it buys (fusion alone ~1.15×) | `./demo.sh fusion --backup` (recorded profile, no sudo) |
| Bonus | `./demo.sh wall` | ~15 s | W01 with DRAM near peak and the SMs idle: the memory wall | Skip it |

---

## 1. Setup (once)

```bash
cd GPU-CPU-Performance-Profiling_and-Optimization     # the repository root
make legacy w04                                        # every binary the kit uses, into build/
./demo_kit/demo.sh check
source /opt/intel/oneapi/setvars.sh                    # only if vtune is not on PATH
```

All paths and tools come from `demo.conf` and can be overridden from the environment (`PROJECT`, `BUILD`, `NCU`, `VTUNE`, `VIDEO`, ...; see `README.md`). Edit `demo.conf`, not the scripts, if a binary name or argument differs.

## 2. The day before

```bash
./demo.sh check      # every binary, profiler, GPU, and which kernel ncu will catch
./prep.sh            # captures backup reports into profiling/demo/ (ncu needs sudo)
./gui_doctor.sh      # only if ncu-ui closes by itself
```

Check these in the `check` output:

- **Kernel name** printed under "Checking which kernel ncu will catch". It must be the Tensor Core GEMM kernel. If it is an init or convert kernel, set `NCU_KFILTER="regex:<part of the GEMM kernel name>"` in `demo.conf` and run `check` again.
- **perf_event_paranoid**: VTune hotspots needs ≤ 2. If it is higher: `sudo sysctl kernel.perf_event_paranoid=1`.

After `prep.sh`, open both GUIs and **take the screenshots** it lists. They become the fallback slides.

Nsight Compute GUI: on Ubuntu 24.04 the usual cause of a GUI that closes at start is a missing `libxcb-cursor0` (`sudo apt install libxcb-cursor0`). `ncu-gui` already sets `QT_QPA_PLATFORM=xcb`. `gui_doctor.sh` starts the GUI with Qt debug logging under four settings and reports which one stays open.

## 3. Thirty minutes before

```bash
./demo.sh check
nvidia-smi -q -d PERFORMANCE | grep -A3 "Clocks Event"   # make sure nothing is throttling
```

- Plug in the charger and select the **Performance** power mode. Laptop clocks follow the power mode, and the numbers drift with them.
- Close browsers and other GPU applications (`nvidia-smi` lists them).
- Terminal font **18–20 pt**, dark theme, window maximised. The scripts use 256-colour output.
- Have one terminal ready in `demo_kit/`, and run `sudo -v` beforehand, so the ncu password prompt does not appear on stage.

## 4. On stage: what to say at each cue

### Cue A: the race (after the results slides)

> "These numbers aren't from a spreadsheet. I'll re-run every stage right now on this laptop."

Run `./demo.sh race`. While it runs, say:

> "Same inputs, same machine. Only the code changes between S1, S2 and S3."

Point out on the scoreboard (formal values from run `20261106-151143`; live values differ by a few %):

1. **W01: about 1.0×.** "Optimising vector add did nothing, and that is the finding. Both kernels already stream ~224 GB/s, 88% of the memory bandwidth. Hold that thought for Nsight."
2. **W02: about 34×.** "Same arithmetic. Only the reduction structure changed."
3. **W03: about 5.2×**: tiling 1.36×, then TF32 Tensor Cores 3.8×. "Tensor Cores. Let's see *why*."
4. **W04 attention: about 50×**, read as a product: S1→S2a ~3.4× (better kernels) · S2a→S2b ~3.2× (fusion + register tiling) · S2b→S3 FP16 ~4.7× (Tensor Cores). "S2b already beats the cuBLAS-based reference at this power cap; the Tensor Core version is ~5× faster than it."
5. **CPU vs GPU at the same N** (W04, N = 2048): GPU S3 FP16 is ~20× faster than the AVX2 CPU S3, and ~1900× faster than the naive CPU S1.

> Say: "Within a few percent of the recorded run, which is why I report the mean of five repetitions, and the median of three rounds for the formal numbers."

### Cue B: Nsight Compute (why W03 S3 is faster)

Run `./demo.sh ncu`. It pauses between its three steps; press Enter to continue.

1. **SASS grep** shows `HMMA.1684.F32.TF32`. "Before profiling anything, I prove the Tensor Core instruction is actually in the binary. A speedup alone doesn't prove which hardware ran."
2. **Live profile** of one kernel launch. "Nsight replays the kernel several times to read hardware counters, so this run is slower than normal and isn't used for timing."
3. **Comparison table** for S1, S2 and S3:
   - S1/S2: SM throughput ~97–99% with DRAM only ~1–4%. "The ordinary FP32 pipeline is saturated, so the kernel is compute-bound."
   - S3: the tensor pipe is active, registers per thread are 56 (occupancy capped at 75%), and the fragment loads are shared-memory `LDS`, with no generic `LD.E`. "The first version of this kernel loaded its Tensor Core fragments through generic addresses. Switching to explicit shared-memory loads and a 32-deep k-tile made it 1.26× faster."

Optionally run `./demo.sh ncu-gui`. Open S3, right-click the S1 tab, and choose **Set Baseline**, so the diff bars show up. If the GUI doesn't open, say "here's the same view I captured earlier" and move to the screenshot slide.

### Cue D: fusion (W04, why S2b is faster)

Run `./demo.sh fusion`. It prints the four ncu commands first (one kernel each), so the audience sees them. Then
it asks for the sudo password once.

1. **Bytes.** "Unfused, S2a writes the 2048×2048 score matrix S to DRAM, reads it back, writes P, and reads it
   back. Fused, S2b never leaves the chip." Point at the ×-fewer-bytes line (~25–31×; the live rehearsal gave 31.4×, 96.8% of the bytes removed).
2. **Time.** "Bytes fell ~25×, so fusion should be 25× faster? No. S2b 16×16 (fused, same work per thread)
   is only ~1.15× faster. The remaining 2.7× is register tiling." The S2a softmax is at 86% of DRAM peak, its
   GEMMs at 11%: only the softmax was memory-bound.
3. "That is why every speedup in this project comes with a counter that explains it."

If ncu fails: `./demo.sh fusion --backup` shows the recorded Nsight Compute profile and still runs the timing live.

### Cue C: VTune (where the CPU time goes)

Run `./demo.sh vtune`.

1. **Live collection.** "VTune samples the CPU every few milliseconds and records which function was running."
2. **Summary for S1.** Effective core utilization is low; the formal record shows ~24.6%. "14 cores available, and the naive code keeps essentially one busy."
3. **Top function.** One GEMM loop takes all the time.
4. **Comparison with S3.** The OpenMP worker functions now dominate, and CPU time is spread across 8 threads. "Blocking, then threads, then AVX2 FMA, for about 114× in total."

Optionally run `./demo.sh vtune-gui`, open **Bottom-up**, and double-click the hotspot to show the source line.

## 5. If things go wrong

| Symptom | Do this on stage |
|---|---|
| `sudo` asks for a password mid-demo | Type it. Next time, run `sudo -v` beforehand. |
| `==ERROR== ERR_NVGPUCTRPERM` | You forgot sudo for ncu. Re-run, or show the backup with `python3 tools/ncu_compare.py $NCU S1=../profiling/demo/W03_S1.ncu-rep S3=../profiling/demo/W03_S3.ncu-rep` |
| ncu profiles the wrong kernel | Show the backup table (above), then fix `NCU_KFILTER` afterwards |
| VTune: "paranoid" or "ptrace" error | Show the backup result with `vtune -report summary -r ../profiling/demo/vtune_W03_S1` |
| GUI doesn't open | Move to the screenshot slide. Don't debug on stage. |
| Race shows `MISSING` | The binary is not built: run `make legacy w04` afterwards (or fix its path in `demo.conf`). Show the chart of an earlier race with `./demo.sh plot` |
| Anything hangs > 30 s | Press Ctrl-C and say "I have this captured", then switch to the backup. |

## 6. What the kit writes (nothing is ever overwritten)

- `results/live/<timestamp>/*.csv` for every race run
- `profiling/demo/live_<timestamp>_*.ncu-rep` and `vtune_live_<timestamp>_*` for every live profile
- `profiling/demo/W03_S1.ncu-rep` … `vtune_W03_S3/`, the backups from `prep.sh`. Older backups move to `old_<timestamp>/`.

## 7. GUI navigation and what each number means

### Nsight Compute (`./demo.sh ncu-gui`)

Show these in order. The **Page** drop-down at the top left switches between Summary, Details, Source and Raw. Expand a section by clicking its header, and hover over any metric name to see NVIDIA's definition.

| Step | Where to click | What to point at | What it means |
|---|---|---|---|
| 1 | Details › **GPU Speed Of Light Throughput** | Compute (SM) % vs Memory %, Duration | Which side is busy. A high SM % means the kernel is compute-bound. |
| 2 | Same section › **GPU Throughput Rooflines** chart | Where the dot sits under the roof | Nsight's own Roofline, drawn from device peaks. |
| 3 | **Compute Workload Analysis** › pipe utilisation | Tensor pipe on S3, FMA pipe on S1/S2 | Which execution unit did the math |
| 4 | **Launch Statistics** + **Occupancy** | Registers/thread = 56, Theoretical 75%, the "Block Limit Registers" row | Registers cap how many warps fit |
| 5 | **Warp State Statistics** | The longest bar | Why warps waited. *Long Scoreboard* = waiting for L1/L2/DRAM data, *Barrier* = waiting at `__syncthreads`, *MIO Throttle* = the shared-memory pipe is full. |
| 6 | Page › **Source**, view SASS | HMMA lines and the sampling bars next to them | Where the time is spent, instruction by instruction |
| 7 | With the S1 tab active, click **Add Baseline**, then open the S3 tab | The change shown next to every value | The direct before/after comparison |

**Memory Throughput %** is the busiest of L1, L2 and DRAM, so it does not mean DRAM traffic. Always read it together with **DRAM Throughput %**.

### VTune (`./demo.sh vtune-gui`)

| Step | Where to click | What to point at | What it means |
|---|---|---|---|
| 1 | **Summary** tab | Elapsed Time vs CPU Time | When CPU Time ≈ Elapsed, one thread did the work. When CPU Time is much larger than Elapsed, the work was parallel. |
| 2 | Summary › **Effective CPU Utilization** histogram, Effective Physical Core Utilization | ~24.6% for W03 S1 | How many cores were actually busy |
| 3 | Summary › **Top Hotspots** | First row | The function that consumed the time |
| 4 | **Bottom-up** tab, Grouping: *Function / Call Stack* | Sort by CPU Time | Full ranked list of functions |
| 5 | Timeline at the bottom of Bottom-up | One row per thread; green = running, red = spinning or overhead | Load balance across OpenMP threads |
| 6 | Double-click the hot function, then the **Assembly** button | CPU time per line and `vfmadd213ps` | Proof that the AVX2 FMA path runs in S3 |
| 7 | **Compare** button on the toolbar | S1 result vs S3 result | Side-by-side differences |

**Spin Time** means threads waiting at barriers or locks. **Overhead Time** is time spent in the OpenMP runtime itself. Small values for both mean the work is well balanced.

## 8. Side by side (both tools)

```bash
sudo apt install wmctrl        # optional: lets the kit tile windows automatically
./demo.sh ncu-side             # Nsight: naive S1 report left, Tensor Core S3 right
./demo.sh vtune-side           # VTune: terminal comparison report, then S1 left / S3 right
```

Without wmctrl, both windows still open. Press **Super+←** on one and **Super+→** on the other.

- **Nsight Compute:** both windows on Page › Details. Expand *GPU Speed Of Light* and *Compute Workload Analysis* in each. For a single combined view, use Add Baseline instead (S1 tab active, Add Baseline, then the S3 tab).
- **VTune:** the terminal part runs `vtune -report summary -r S1 -r S3` and `-report hotspots -r S1 -r S3`. Passing two `-r` flags gives VTune's built-in comparison, with both values and the difference in columns. In the GUIs, show the Summary tab in both windows first, then Bottom-up.
