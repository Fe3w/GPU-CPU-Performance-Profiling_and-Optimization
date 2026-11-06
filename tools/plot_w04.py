#!/usr/bin/env python3
"""W04 plots: E1 runtime vs N, E2 factored speedups, E3 measured-traffic Roofline, E6 memory vs N.

    python3 tools/plot_w04.py <gpu_run> [--cpu-run <run>] [--e3 <dir> ...] [--e3-replace <dir> ...] [--out docs/figures/w04]

<gpu_run>  a results/raw/W04_ATTENTION/<TS> folder with gpu_nsweep/ (decision C35:
           every GPU point of E1, E2, E6 and the Roofline's y values come from this
           one session)
--cpu-run  a run folder with nsweep/ (CPU S1-S3, the paired P1 N sweep); adds the
           CPU panel of E1
--e6-run   run folder whose gpu_nsweep/ gives the E6 memory plot (default: <gpu_run>).
           peak_mem_bytes comes from cudaMemGetInfo, which is device-wide: other
           processes that allocate GPU memory during a run (browser, desktop)
           change it, so E6 may need a session without them (P6: the P4 sweep)
--e3       ncu E3 folders (profiling/W04_ATTENTION/e3/<TS>); without them the
           Roofline shows the algorithmic operational intensity only. A (stage, N)
           found in several folders is averaged over them
--e3-replace  ncu folders whose (stage, N) entries replace those from --e3 instead
           of being averaged with them (a stage rebuilt after the --e3 runs, e.g.
           the TF32 fragment-load fix, issue 17)
--out      output folder (default docs/figures/w04)

Times: for every (stage CSV, N) each process run (one timestamp, `reps` rows)
gives one run mean; the plotted value is the median over runs (rounds), the
same rule as tools/w04_summary.py. GFLOP/s = 4·BH·N²·d / time.

Roofline (E3): x = FLOPs / bytes, y = benchmark GFLOP/s at the same N.
  hollow marker  model bytes (plan §4): Q, K, V, O once (4·BH·N·d·4 B), plus
                 4 passes over the N×N matrix (16·BH·N² B) for the unfused
                 stages S1, S2a, REF. (The CSV alg_bytes column holds only the
                 compulsory Q, K, V, O bytes, the same for every stage.)
  filled marker  measured DRAM bytes from ncu (read + max(write, compulsory
                 write), the write-corrected value of tools/w04_e3_oi.py)
Ceilings (C36): every value in CEILINGS carries its source. Values marked
estimate=True are drawn dashed and labelled [ESTIMATE]; the nominal DRAM
bandwidth is drawn dashed too (a spec, not a measurement). Compute ceilings are
per-SM-per-clock rates times 24 SMs times the median SM clock of the sweep
(gpu_state.csv), because the GPU runs power-capped at 55 W, not at boost clock.

Output: e1_runtime_vs_n.png, e2_factored_speedup.png, e3_roofline.png,
e6_memory_vs_n.png and plot_data.csv (every plotted value, the table view).
"""
import argparse
import csv
import glob
import os
import statistics as st
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import w04_e3_oi  # noqa: E402

# Categorical slots in fixed order (reference palette, light mode); one slot per entity.
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

G = "W04_ATTENTION_GPU_"
# (label, CSV names in order of preference, slot, marker, unfused?)
GPU_STAGES = [
    ("S1 naive", [G + "S1_BASE_NAIVE"], 0, "o", True),
    ("S2a unfused", [G + "S2a_SWOPT_UNFUSED"], 1, "s", True),
    ("REF cuBLAS", [G + "REF_CUBLAS"], 2, "D", True),
    ("S2b fused (tuned)", [G + "S2b_SWOPT_FUSED_tuned", G + "S2b_SWOPT_FUSED_64x64"], 3, "^", False),
    ("S3 FP16 (tuned)", [G + "S3_HWACC_TENSOR_CORE_FP16_tuned", G + "S3_HWACC_TENSOR_CORE_FP16"], 4, "v", False),
    ("S3 TF32 (tuned)", [G + "S3_HWACC_TENSOR_CORE_TF32_tuned", G + "S3_HWACC_TENSOR_CORE_TF32"], 5, "P", False),
]
C = "W04_ATTENTION_CPU_"
CPU_STAGES = [
    ("CPU S1 scalar, 1 thread", [C + "S1_BASE"], 0, "o"),
    ("CPU S2 fused + OpenMP", [C + "S2_SWOPT_BLOCKED_OMP"], 1, "s"),
    ("CPU S3 AVX2/FMA", [C + "S3_HWACC_AVX2"], 2, "D"),
]
# E3 folder stage key -> sweep CSV whose time gives the Roofline y value
E3_TO_CSV = {
    "S1": G + "S1_BASE_NAIVE", "S2a": G + "S2a_SWOPT_UNFUSED", "REF": G + "REF_CUBLAS",
    "S2b_32x32": G + "S2b_SWOPT_FUSED", "S2b_64x64": G + "S2b_SWOPT_FUSED_64x64",
    "S2b_64x32w4": G + "S2b_SWOPT_FUSED_tuned",
    "S3FP16_tuned": G + "S3_HWACC_TENSOR_CORE_FP16_tuned", "S3TF32_tuned": G + "S3_HWACC_TENSOR_CORE_TF32_tuned",
}
N_SMS = 24
# Ceilings (C36). Compute: FLOP per SM per clock. Memory: bytes per second.
CEILINGS = [
    {"label": "DRAM 236 GB/s (W01 measured)", "kind": "mem", "value": 236e9, "estimate": False,
     "source": "W01 vector add, GPU S1/S2 measured on this machine (~92% of nominal)"},
    {"label": "DRAM 256 GB/s nominal", "kind": "mem", "value": 256e9, "estimate": False,
     "source": "nvidia-smi max memory clock 8001 MHz x 2 (GDDR6, 16 Gbps) x 128-bit bus (cudaDeviceProp) / 8 "
               "= 256.0 GB/s; matches the published RTX 4060 Laptop spec (checked 2026-11-06)"},
    # Per-SM rates from the NVIDIA Ada GPU Architecture whitepaper V2.02, Appendix A (RTX 4090:
    # 128 SMs, 2520 MHz boost; checked 2026-11-06): peak = rate x SMs x clock. The RTX 4080 rows
    # (76 SMs, 2505 MHz) give the same per-SM rates, so they are properties of the Ada SM.
    {"label": "FP32 FMA", "kind": "flop", "value": 256, "estimate": False,
     "source": "Ada whitepaper V2.02 App. A: 82.6 TFLOPS FP32 / (128 SMs x 2.52 GHz) = 256 FLOP/clk/SM"},
    {"label": "TF32 tensor", "kind": "flop", "value": 256, "estimate": False,
     "source": "Ada whitepaper V2.02 App. A: 82.6 TFLOPS TF32 dense / (128 x 2.52 GHz) = 256 FLOP/clk/SM"},
    {"label": "FP16 tensor, FP32 acc.", "kind": "flop", "value": 512, "estimate": False,
     "source": "Ada whitepaper V2.02 App. A: 165.2 TFLOPS FP16 w/ FP32 accumulate dense / (128 x 2.52 GHz) "
               "= 512 FLOP/clk/SM"},
]


# ------------------------------------------------------------------ data
def load_sweep(folder):
    """{csv name: {N: {"ms": median, "flops": F, "alg_bytes": B, "peak_mem": M, "runs": k}}}"""
    out = {}
    for path in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        name = os.path.basename(path)[:-4]
        runs = defaultdict(lambda: defaultdict(list))
        mem = defaultdict(dict)   # N -> {timestamp: peak bytes} (one value per process run)
        meta = {}
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                n = int(r["n"])
                runs[n][r["timestamp"]].append(float(r["runtime_ms"]))
                mem[n][r["timestamp"]] = float(r["peak_mem_bytes"] or "nan")
                meta[n] = (4.0 * int(r["bh"]) * n * n * int(r["d"]), float(r["alg_bytes"] or "nan"))
        out[name] = {}
        for n, by_ts in runs.items():
            means = [st.mean(v) for v in by_ts.values()]
            fl, ab = meta[n]
            out[name][n] = {"ms": st.median(means), "flops": fl, "alg_bytes": ab,
                            "peak_mem": st.median(mem[n].values()), "runs": len(means)}
    return out


def pick(sweep, names):
    for nm in names:
        if nm in sweep:
            return nm
    return None


def median_sm_clock(run):
    path = os.path.join(run, "gpu_state.csv")
    if not os.path.exists(path):
        return None
    vals = []
    with open(path) as f:
        next(f, None)
        for row in csv.reader(f):
            try:
                sm = float(row[1].split()[0])
                pw = float(row[3].split()[0])
            except (ValueError, IndexError):
                continue
            if pw > 20:   # samples under load only (idle ~4 W)
                vals.append(sm)
    return st.median(vals) if vals else None


def load_e3(dirs):
    """{(stage key, N): (read, write, compulsory write)} averaged over the folders that have it."""
    acc = defaultdict(list)
    for d in dirs:
        per = defaultdict(lambda: [0.0, 0.0])
        for path in sorted(glob.glob(os.path.join(d, "*.raw.csv"))):
            m = w04_e3_oi.NAME.match(os.path.basename(path))
            if not m:
                continue
            unf = m.group(1) in w04_e3_oi.UNFUSED
            key = (m.group(1) if unf else f"{m.group(1)}_{m.group(2)}", int(m.group(3)))
            met = w04_e3_oi.read_export(path)
            per[key][0] += met["dram__bytes_read.sum"]
            per[key][1] += met["dram__bytes_write.sum"]
        for key, (rd, wr) in per.items():
            acc[key].append((rd, wr))
    out = {}
    for (stage, n), vals in acc.items():
        bh, d = w04_e3_oi.BH, w04_e3_oi.D
        o_b, s_b = bh * n * d * 4.0, bh * n * n * 4.0
        comp = 2 * s_b + o_b if stage in w04_e3_oi.UNFUSED else o_b
        out[(stage, n)] = (st.mean(v[0] for v in vals), st.mean(v[1] for v in vals), comp)
    return out


# ------------------------------------------------------------------ style
def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK2, labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def footnote(fig, text):
    fig.text(0.01, 0.005, text, fontsize=7.5, color=INK2, ha="left", va="bottom")


TABLE = []   # rows of plot_data.csv


def series_plot(ax, sweep, stages, key, scale=1.0, label_ends=True):
    for label, names, slot, marker, *_ in stages:
        nm = pick(sweep, names)
        if not nm:
            continue
        ns = sorted(sweep[nm])
        ys = [sweep[nm][n][key] * scale for n in ns]
        ax.plot(ns, ys, color=SLOT[slot], linewidth=2, marker=marker, markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, zorder=3)
        if label_ends:
            ax.annotate(label, (ns[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK)


def fig_e1(gpu, cpu, out, gpu_run, cpu_run):
    panels = 2 if cpu else 1
    fig, axes = plt.subplots(1, panels, figsize=(7.5 * panels, 5.2), facecolor=SURFACE, squeeze=False)
    ax = axes[0][0]
    series_plot(ax, gpu, GPU_STAGES, "ms")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    style(ax, "GPU runtime vs N (BH=8, d=64)", "sequence length N", "time per forward pass (ms, log)")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.margins(x=0.25)
    for label, names, *_ in GPU_STAGES:
        nm = pick(gpu, names)
        for n, v in sorted(gpu.get(nm, {}).items()):
            TABLE.append(("E1", "GPU", label, n, "ms", v["ms"], v["runs"], gpu_run))
    if cpu:
        ax = axes[0][1]
        series_plot(ax, cpu, CPU_STAGES, "ms")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        style(ax, "CPU runtime vs N (another session; not paired with the GPU)", "sequence length N",
              "time per forward pass (ms, log)")
        ax.legend(fontsize=8, frameon=False, loc="upper left")
        ax.margins(x=0.25)
        for label, names, *_ in CPU_STAGES:
            nm = pick(cpu, names)
            for n, v in sorted(cpu.get(nm, {}).items()):
                TABLE.append(("E1", "CPU", label, n, "ms", v["ms"], v["runs"], cpu_run))
    footnote(fig, f"Median over rounds of the run means. GPU: {gpu_run}, 55 W power cap (equal power, not equal clock)."
             + (f"  CPU: {cpu_run}, 20 threads (S1: 1)." if cpu else ""))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path = os.path.join(out, "e1_runtime_vs_n.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_e2(gpu, out, gpu_run):
    chain = [("kernel quality\nS1 → S2a", 0, 1), ("fusion + tiling\nS2a → S2b", 1, 3),
             ("Tensor Cores\nS2b → S3 FP16", 3, 4)]
    ns = [n for n in (1024, 2048) if all(pick(gpu, GPU_STAGES[i][1]) and n in gpu[pick(gpu, GPU_STAGES[i][1])]
                                          for i in (0, 1, 3, 4))]
    if not ns:
        return None
    fig, axes = plt.subplots(1, len(ns), figsize=(6 * len(ns), 4.6), facecolor=SURFACE, squeeze=False, sharey=True)
    for ax, n in zip(axes[0], ns):
        t = {i: gpu[pick(gpu, GPU_STAGES[i][1])][n]["ms"] for i in (0, 1, 3, 4)}
        fac = [t[a] / t[b] for _, a, b in chain]
        xs = range(len(chain))
        ax.bar(xs, fac, width=0.6, color=SLOT[0], edgecolor=SURFACE, linewidth=2, zorder=3)
        for x, f in zip(xs, fac):
            ax.annotate(f"×{f:.2f}", (x, f), xytext=(0, 4), textcoords="offset points", ha="center",
                        fontsize=9, color=INK)
        ax.set_xticks(list(xs), [c[0] for c in chain])
        total = t[0] / t[4]
        tag = " (L2-resident)" if n <= 1024 else ""
        style(ax, f"N = {n}{tag}: S1 → S3 FP16 total ×{total:.1f}", "", "speedup factor of the step")
        ax.axhline(1, color=INK2, linewidth=1, zorder=4)
        ax.grid(False, axis="x")
        for (lab, a, b), f in zip(chain, fac):
            TABLE.append(("E2", "GPU", lab.replace("\n", " "), n, "factor", f, "", gpu_run))
        TABLE.append(("E2", "GPU", "total S1 -> S3 FP16", n, "factor", total, "", gpu_run))
    footnote(fig, f"Ratios of median times, same session ({gpu_run}); the factors multiply to the total. "
             "S2a → S2b also includes register tiling, not fusion alone (P3).")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path = os.path.join(out, "e2_factored_speedup.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_e3(gpu, e3, out, gpu_run, clock_mhz):
    fig, ax = plt.subplots(figsize=(9, 6), facecolor=SURFACE)
    xs_all = []
    # One CSV per stage: the first of its names present in the sweep (the later
    # names are fallbacks for older sessions, not extra points).
    stage_slot = {}
    for lab, names, slot, mk, _ in GPU_STAGES:
        nm = pick(gpu, names)
        if nm:
            stage_slot[nm] = (lab, slot, mk)
    extra = {G + "S2b_SWOPT_FUSED": ("S2b 32×32", 6, "^"), G + "S2b_SWOPT_FUSED_64x64": ("S2b 64×64", 7, "^")}
    stage_slot.update(extra)
    for n in (2048, 4096):
        for nm, (lab, slot, mk) in stage_slot.items():
            if nm not in gpu or n not in gpu[nm]:
                continue
            v = gpu[nm][n]
            y = v["flops"] / (v["ms"] * 1e-3) / 1e9
            alpha = 1.0 if n == 4096 else 0.55
            unfused = nm in (G + "S1_BASE_NAIVE", G + "S2a_SWOPT_UNFUSED", G + "REF_CUBLAS")
            model = 4 * 8 * n * 64 * 4.0 + (16 * 8 * n * n if unfused else 0.0)
            if nm not in extra:
                xa = v["flops"] / model
                ax.scatter([xa], [y], s=64, facecolors="none", edgecolors=SLOT[slot], linewidths=1.6,
                           marker=mk, alpha=alpha, zorder=3)
                xs_all.append(xa)
                TABLE.append(("E3", "GPU", lab, n, "OI_model", xa, "", gpu_run))
            key = next((k for k, c in E3_TO_CSV.items() if c == nm), None)
            if n == 4096 and nm not in extra and not (key and (key, n) in e3):
                ax.annotate(lab + " (model only)", (xa, y), xytext=(6, -3), textcoords="offset points",
                            fontsize=8, color=INK2)
            if key and (key, n) in e3:
                rd, wr, comp = e3[(key, n)]
                xm = v["flops"] / (rd + max(wr, comp))
                ax.scatter([xm], [y], s=64, color=SLOT[slot], marker=mk, edgecolors=SURFACE, linewidths=1.2,
                           alpha=alpha, zorder=4)
                if n == 4096:
                    ax.annotate(lab, (xm, y), xytext=(6, -3), textcoords="offset points", fontsize=8, color=INK)
                xs_all.append(xm)
                TABLE.append(("E3", "GPU", lab, n, "OI_measured", xm, "", gpu_run))
            TABLE.append(("E3", "GPU", lab, n, "GFLOP/s", y, "", gpu_run))
    if not xs_all:
        plt.close(fig)
        return None
    lo, hi = min(xs_all) / 3, max(xs_all) * 3
    xr = [lo * (hi / lo) ** (i / 200) for i in range(201)]
    clk = (clock_mhz or 0) * 1e6
    mems = [c for c in CEILINGS if c["kind"] == "mem"]
    # Compute roofs with the same peak share one line and one label.
    roofs = defaultdict(list)
    for c in CEILINGS:
        if c["kind"] == "flop" and clk:
            roofs[round(c["value"] * N_SMS * clk / 1e9)].append(c)
    top = max(roofs) if roofs else None
    for i, c in enumerate(mems):
        ls = "--" if c["estimate"] or "nominal" in c["label"] else "-"
        tag = " [ESTIMATE]" if c["estimate"] else ""
        x_end = top / (c["value"] / 1e9) if top else hi   # up to the highest ridge point
        seg = [x for x in xr if x <= x_end] + [x_end]
        ys = [x * c["value"] / 1e9 for x in seg]
        ax.plot(seg, ys, color=INK2, linewidth=1.2, linestyle=ls, zorder=1)
        ax.annotate(c["label"] + tag, (seg[0], ys[0]), xytext=(2, 8 if c["value"] == max(m["value"] for m in mems) else -12),
                    textcoords="offset points", fontsize=7.5, color=INK2)
    mem = max(m["value"] for m in mems) / 1e9
    for peak, cs in sorted(roofs.items()):
        est = any(c["estimate"] for c in cs)
        ls, tag = ("--", " [ESTIMATE]") if est else ("-", "")
        ax.plot([peak / mem, hi], [peak, peak], color=INK2, linewidth=1.2, linestyle=ls, zorder=1)
        names = " = ".join(c["label"] for c in cs)
        ax.annotate(f"{names}: {peak / 1e3:.1f} TFLOP/s at {clock_mhz:.0f} MHz{tag}", (hi, peak),
                    xytext=(-4, 4), textcoords="offset points", ha="right", fontsize=7.5, color=INK2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    style(ax, "Roofline: model traffic (hollow) vs measured DRAM traffic (filled); N=4096, N=2048 faded",
          "operational intensity (FLOP / DRAM byte, log)", "achieved GFLOP/s (benchmark, log)")
    footnote(fig, f"y: benchmark GFLOP/s, {gpu_run}. Measured bytes: ncu E3, write-corrected (tools/w04_e3_oi.py).\n"
             "Compute ceilings at the median loaded SM clock of the sweep (55 W cap); sources in tools/plot_w04.py CEILINGS.")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    path = os.path.join(out, "e3_roofline.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_e6(gpu, out, gpu_run):
    fig, ax = plt.subplots(figsize=(8, 5.2), facecolor=SURFACE)
    # Device memory does not depend on the tile, so the labels drop "(tuned)" (an older
    # session may only have the pinned-tile entries).
    stages = [(lab.replace(" (tuned)", ""),) + tuple(rest) for lab, *rest in GPU_STAGES]
    series_plot(ax, gpu, stages, "peak_mem", scale=1 / 2 ** 20, label_ends=False)
    ns = sorted({n for nm in gpu for n in gpu[nm]})
    if not ns:
        plt.close(fig)
        return None
    bh, d = 8, 64
    ax.plot(ns, [(4 * bh * n * d * 4 + bh * n * n * 4) / 2 ** 20 for n in ns], color=INK, linestyle="--",
            linewidth=1, label="model: Q, K, V, O + one N×N matrix", zorder=5)
    ax.plot(ns, [4 * bh * n * d * 4 / 2 ** 20 for n in ns], color=INK, linestyle=":", linewidth=1.2,
            label="model: Q, K, V, O only", zorder=5)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    style(ax, "Peak device memory vs N: O(N²) unfused vs O(N) fused", "sequence length N",
          "peak device memory (MiB, log₂)")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.margins(x=0.25)
    for label, names, *_ in stages:
        nm = pick(gpu, names)
        for n, v in sorted(gpu.get(nm, {}).items()):
            TABLE.append(("E6", "GPU", label, n, "peak_mem_MiB", v["peak_mem"] / 2 ** 20, "", gpu_run))
    footnote(fig, f"cudaMemGetInfo before/after (CSV peak_mem_bytes), median over runs, {gpu_run}. S1 and S2a coincide (one N×N buffer each); "
             "FP16 adds its half copies of Q, K, V.")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path = os.path.join(out, "e6_memory_vs_n.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gpu_run")
    ap.add_argument("--cpu-run")
    ap.add_argument("--e6-run")
    ap.add_argument("--e3", nargs="*", default=[])
    ap.add_argument("--e3-replace", nargs="*", default=[])
    ap.add_argument("--out", default="docs/figures/w04")
    a = ap.parse_args()
    gdir = os.path.join(a.gpu_run, "gpu_nsweep")
    if not os.path.isdir(gdir):
        print(f"no gpu_nsweep/ in {a.gpu_run}")
        return 1
    gpu = load_sweep(gdir)
    cpu = load_sweep(os.path.join(a.cpu_run, "nsweep")) if a.cpu_run else None
    e3 = load_e3(a.e3)
    if a.e3_replace:
        e3.update(load_e3(a.e3_replace))
    clock = median_sm_clock(a.gpu_run)
    os.makedirs(a.out, exist_ok=True)
    made = [fig_e1(gpu, cpu, a.out, a.gpu_run, a.cpu_run), fig_e2(gpu, a.out, a.gpu_run),
            fig_e3(gpu, e3, a.out, a.gpu_run, clock)]
    if a.e6_run:
        made.append(fig_e6(load_sweep(os.path.join(a.e6_run, "gpu_nsweep")), a.out, a.e6_run))
    else:
        made.append(fig_e6(gpu, a.out, a.gpu_run))
    with open(os.path.join(a.out, "plot_data.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["figure", "device", "series", "n", "quantity", "value", "runs", "source"])
        w.writerows(TABLE)
    for p in made:
        if p:
            print(p)
    print(os.path.join(a.out, "plot_data.csv"))
    print(f"median loaded SM clock: {clock if clock else 'n/a'} MHz; ceilings:")
    for c in CEILINGS:
        print(f"  {c['label']:<30} {'[ESTIMATE] ' if c['estimate'] else ''}{c['source']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
