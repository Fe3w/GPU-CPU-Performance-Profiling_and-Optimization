#!/usr/bin/env python3
"""README figures: the CPU/GPU comparison graph and the Roofline graph.

Every plotted number is read from files in this repository:
  docs/results/final_results.csv    final paired re-test (run 20261106-151143): runtimes, rates
  docs/figures/w04/plot_data.csv    W04 measured operational intensity (Nsight Compute DRAM bytes)
                                    and GFLOP/s from the W04 GPU sweep (run 20261106-114338)
Hardware ceilings are documented constants (see CEILINGS below, with their sources).

Usage:  python3 tools/readme_figures.py [--out docs/figures/readme]
Output: hero_speedup.png, comparison.png, roofline.png, and figure_values.csv (every value drawn, for checking)
"""
import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINAL = os.path.join(ROOT, "docs", "results", "final_results.csv")
W04PLOT = os.path.join(ROOT, "docs", "figures", "w04", "plot_data.csv")

# ---------------------------------------------------------------- style
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
CPU_C, GPU_C, REF_C = "#2a78d6", "#eb6834", "#9a9893"            # categorical slots 1, 2 + neutral
FP32_C, TF32_C, FP16_C = "#2a78d6", "#1baf7a", "#eb6834"          # slots 1, 3, 2 (validated all-pairs)
for f in ("Inter", "DejaVu Sans"):
    if any(f in x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "font.size": 11,
})

VALUES = []   # (figure, item, quantity, value, source) for figure_values.csv


def load_final():
    rows = {}
    with open(FINAL, newline="") as f:
        for r in csv.DictReader(f):
            rows[(r["workload"], r["device"], r["stage"], int(r["n"]))] = r
    return rows


def fmt_ms(ms):
    if ms >= 100:
        return f"{ms:,.0f} ms"
    if ms >= 10:
        return f"{ms:.1f} ms"
    if ms >= 1:
        return f"{ms:.2f} ms"
    return f"{ms:.3f} ms"


def tick_ms(v, _):
    if v >= 1:
        return f"{v:,.0f}"
    return f"{v:g}"


# ---------------------------------------------------------------- comparison graph
PANELS = [
    ("W01", 33554432, "W01 · Vector Add", "C = A + B · N = 33.5 M floats · memory-bound (0.083 FLOP/byte)", [
        ("CPU", "S1", "S1 scalar, 1 thread"),
        ("CPU", "S2", "S2 OpenMP, 6 threads"),
        ("CPU", "S3", "S3 AVX2 + OpenMP, 6 threads"),
        ("GPU", "S1", "S1 naive"),
        ("GPU", "S2", "S2 coalesced"),
    ]),
    ("W02", 33554432, "W02 · Dot Product", "s = Σ aᵢbᵢ · N = 33.5 M floats · reduction (0.25 FLOP/byte)", [
        ("CPU", "S1", "S1 scalar, 1 thread"),
        ("CPU", "S2", "S2 OpenMP, 6 threads"),
        ("CPU", "S3", "S3 AVX2/FMA + OpenMP, 6 thr."),
        ("GPU", "S1", "S1 naive (atomics)"),
        ("GPU", "S2", "S2 block reduction"),
    ]),
    ("W03", 1024, "W03 · GEMM", "C = A × B · 1024 × 1024 · compute-bound (171 FLOP/byte)", [
        ("CPU", "S1", "S1 naive loop, 1 thread"),
        ("CPU", "S2", "S2 blocked + OpenMP, 8 thr."),
        ("CPU", "S3", "S3 AVX2/FMA + blocked, 8 thr."),
        ("GPU", "S1", "S1 naive"),
        ("GPU", "S2", "S2 shared-memory tiles"),
        ("GPU", "S3", "S3 TF32 Tensor Cores"),
    ]),
    ("W04", 2048, "W04 · Attention", "softmax(Q·K^T/√d)·V · N = 2048, 8 heads, d = 64 · fusion", [
        ("CPU", "S1", "S1 scalar, 1 thread"),
        ("CPU", "S2", "S2 fused + OpenMP, 20 thr."),
        ("CPU", "S3", "S3 AVX2/FMA + vector exp, 20 thr."),
        ("GPU", "S1", "S1 naive, 3 kernels"),
        ("GPU", "S2a", "S2a tiled, unfused"),
        ("GPU", "S2b", "S2b fused (FlashAttention-style)"),
        ("GPU", "S3_TF32", "S3 TF32 Tensor Cores"),
        ("GPU", "S3_FP16", "S3 FP16 Tensor Cores"),
        ("GPU", "REF", "cuBLAS reference (not a stage)"),
    ]),
]


def comparison(rows, out):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.4), gridspec_kw={"height_ratios": [5.2, 9]})
    for ax, (wl, n, title, sub, stages) in zip(axes.flat, PANELS):
        names, times, colors, labels, devs = [], [], [], [], []
        best = {}
        for dev, st, name in stages:
            r = rows[(wl, dev, st, n)]
            ms = float(r["mean_ms"])
            x = float(r["x_vs_s1"])
            names.append(name)
            times.append(ms)
            devs.append(dev)
            colors.append(REF_C if st == "REF" else (CPU_C if dev == "CPU" else GPU_C))
            labels.append(fmt_ms(ms) + ("" if st == "REF" else f"   {x:.2f}×" if x < 10 else f"   {x:.1f}×"))
            if st != "REF" and (dev not in best or ms < best[dev][0]):
                best[dev] = (ms, name)
            VALUES.append(("comparison", f"{wl} {dev} {st} N={n}", "mean_ms", ms, f"final_results.csv run {r['run']}"))
            VALUES.append(("comparison", f"{wl} {dev} {st} N={n}", "x_vs_s1", x, f"final_results.csv run {r['run']}"))
        y = list(range(len(names)))[::-1]
        bars = ax.barh(y, times, height=0.62, color=colors, edgecolor=SURFACE, linewidth=2, zorder=3)
        for b, st in zip(bars, [s for _, s, _ in stages]):
            if st == "REF":
                b.set_hatch("////")
                b.set_edgecolor(SURFACE)
        ax.set_xscale("log")
        lo, hi = min(times) / 2.2, max(times) * 16
        ax.set_xlim(lo, hi)
        for yi, t, lab, st in zip(y, times, labels, [s for _, s, _ in stages]):
            is_best = any(abs(t - best[d][0]) < 1e-12 for d in best)
            ax.annotate(lab, (t, yi), xytext=(5, 0), textcoords="offset points", va="center", fontsize=10.3,
                        color=INK if st != "REF" else INK2, fontweight="bold" if is_best else "normal", zorder=4)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=10.5)
        # separator between the CPU and GPU groups
        n_cpu = sum(1 for d in devs if d == "CPU")
        ax.axhline(y[n_cpu - 1] - 0.5, color=GRID, linewidth=1, zorder=1)
        ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
        ax.tick_params(axis="y", length=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(tick_ms))
        ax.set_xlabel("time per iteration (ms, log scale) · shorter bar = faster", fontsize=10.3, color=INK2)
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=22)
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=10.3, color=INK2, va="bottom")
        ratio = best["CPU"][0] / best["GPU"][0]
        ax.annotate(f"best GPU vs best CPU: {ratio:.1f}×", (1, 1), xycoords="axes fraction", xytext=(0, 22),
                    textcoords="offset points", ha="right", va="baseline", fontsize=11.5, color=INK,
                    fontweight="bold", bbox=dict(boxstyle="round,pad=0.35", fc="#f1f0ec", ec="none"))
        VALUES.append(("comparison", f"{wl} best CPU / best GPU", "ratio", ratio, "derived from mean_ms"))
    handles = [plt.Rectangle((0, 0), 1, 1, color=CPU_C), plt.Rectangle((0, 0), 1, 1, color=GPU_C),
               plt.Rectangle((0, 0), 1, 1, fc=REF_C, hatch="////", ec=SURFACE)]
    fig.legend(handles, ["CPU · Intel Core i7-13650HX", "GPU · NVIDIA RTX 4060 Laptop (55 W)",
                         "vendor reference (cuBLAS)"],
               loc="upper left", bbox_to_anchor=(0.006, 0.952), ncol=3, frameon=False, fontsize=11.3)
    fig.suptitle("Every optimization stage, CPU vs GPU", x=0.012, y=0.985, ha="left", fontsize=17,
                 fontweight="bold")
    fig.text(0.012, 0.012,
             "Labels: mean time per iteration and speedup vs the same device's S1 (bold = fastest on that device). "
             "All 34 stage runs passed their correctness check vs a double-precision reference.\n"
             "One session, run 20261106-151143: 2 warm-ups, 20 timed iterations, 5 repetitions; GPU stages median of "
             "3 interleaved rounds. Source: docs/results/final_results.csv.",
             fontsize=9.8, color=MUTED, va="bottom")
    fig.tight_layout(rect=(0, 0.045, 1, 0.915), h_pad=3.2, w_pad=3.5)
    path = os.path.join(out, "comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- roofline
SM_CLOCK_MHZ = 2670    # median loaded SM clock of run 20261106-151143 (docs/RESULTS_SUMMARY.md, GPU state)
N_SMS = 24             # RTX 4060 Laptop, cudaDeviceProp (W04 GPU self-test)
CEILINGS = {
    # FLOP per SM per clock, NVIDIA Ada GPU Architecture whitepaper V2.02, Appendix A
    # (RTX 4090: 82.6 TFLOPS FP32 and TF32 dense, 165.2 TFLOPS FP16 with FP32 accumulate, 128 SMs, 2.52 GHz)
    "fp32": 256, "tf32": 256, "fp16": 512,
}
BW_NOMINAL = 256e9     # 8001 MHz GDDR6 (16 Gbps) x 128-bit bus / 8 (nvidia-smi + cudaDeviceProp)
BW_MEASURED = 224.2e9  # W01 GPU S1/S2 streaming bandwidth, run 20261106-151143


def _line_angle(ax, x0, x1, slope_bw):
    """On-screen angle of the memory roof y = x * bw between x0 and x1."""
    p0 = ax.transData.transform((x0, x0 * slope_bw))
    p1 = ax.transData.transform((x1, x1 * slope_bw))
    import math
    return math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))


def roofline(rows, out):
    peak = {k: v * N_SMS * SM_CLOCK_MHZ * 1e6 / 1e9 for k, v in CEILINGS.items()}   # GFLOP/s
    w04 = {}
    with open(W04PLOT, newline="") as f:
        for r in csv.DictReader(f):
            if r["figure"] == "E3" and r["n"] == "2048":
                w04.setdefault(r["series"], {})[r["quantity"]] = float(r["value"])
    pts = []   # (label, OI, GFLOP/s, path, marker, filled, label offset, table name)

    def final_rate(wl, st, n):
        return float(rows[(wl, "GPU", st, n)]["rate"])

    # W01/W02: algorithmic OI (each byte is streamed once); GFLOP/s = GB/s x OI
    pts.append(("W01", 1 / 12, final_rate("W01", "S1", 33554432) / 12, "fp32", "o", False, (8, -2), "W01 S1 = S2 vector add"))
    pts.append(("W02 S1", 0.25, final_rate("W02", "S1", 33554432) * 0.25, "fp32", "s", False, (8, 0), "W02 S1 dot, atomics"))
    pts.append(("W02 S2", 0.25, final_rate("W02", "S2", 33554432) * 0.25, "fp32", "s", False, (8, -2), "W02 S2 block reduction"))
    for st, lab, path, off, tn in [("S1", "W03 S1", "fp32", (8, -5), "W03 S1 GEMM naive"),
                                   ("S2", "W03 S2", "fp32", (8, 1), "W03 S2 GEMM tiled"),
                                   ("S3", "W03 S3", "tf32", (8, 0), "W03 S3 GEMM TF32")]:
        pts.append((lab, 1024 / 6, final_rate("W03", st, 1024), path, "D", False, off, tn))
    for series, lab, path, off, tn in [("S1 naive", "W04 S1", "fp32", (8, 0), "W04 S1 naive, 3 kernels"),
                                       ("S2a unfused", "W04 S2a", "fp32", (8, -2), "W04 S2a tiled, unfused"),
                                       ("REF cuBLAS", "cuBLAS ref.", "fp32", (8, 2), "W04 cuBLAS reference"),
                                       ("S2b fused (tuned)", "W04 S2b", "fp32", (8, -2), "W04 S2b fused"),
                                       ("S3 TF32 (tuned)", "W04 S3 TF32", "tf32", (8, -2), "W04 S3 TF32"),
                                       ("S3 FP16 (tuned)", "W04 S3 FP16", "fp16", (8, -4), "W04 S3 FP16")]:
        d = w04[series]
        pts.append((lab, d["OI_measured"], d["GFLOP/s"], path, "^", True, off, tn))

    fig = plt.figure(figsize=(13.5, 8.4))
    ax = fig.add_axes([0.085, 0.14, 0.575, 0.73])
    tx = fig.add_axes([0.69, 0.13, 0.30, 0.74])
    tx.set_xlim(0, 1)
    tx.set_ylim(0, 1)
    tx.set_autoscale_on(False)
    tx.axis("off")
    xlo, xhi = 0.04, 4000
    ylo, yhi = 0.8, 1.6e5
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    xs = [xlo * (xhi / xlo) ** (i / 400) for i in range(401)]
    top = max(peak.values())
    for bw, ls in [(BW_NOMINAL, "--"), (BW_MEASURED, "-")]:
        seg = [x for x in xs if x * bw / 1e9 <= top]
        ax.plot(seg, [x * bw / 1e9 for x in seg], color=INK2, linewidth=1.4, linestyle=ls, zorder=2)
    VALUES.append(("roofline", "DRAM nominal", "bytes/s", BW_NOMINAL, "8001 MHz x 2 x 128 bit / 8"))
    VALUES.append(("roofline", "DRAM measured", "bytes/s", BW_MEASURED, "W01 GPU, run 20261106-151143"))
    ang = _line_angle(ax, 0.1, 1.0, BW_NOMINAL / 1e9)
    ax.text(2.5, 2.5 * BW_NOMINAL / 1e9 * 1.5, "DRAM 256 GB/s (nominal)", rotation=ang, rotation_mode="anchor",
            fontsize=9.8, color=INK2, ha="center", va="bottom")
    ax.text(2.5, 2.5 * BW_MEASURED / 1e9 / 1.6, "224 GB/s (measured, W01)", rotation=ang, rotation_mode="anchor",
            fontsize=9.8, color=INK2, ha="center", va="top")
    for key, lab, col in [("fp32", "FP32 / TF32", FP32_C),
                          ("fp16", "FP16 Tensor Cores", FP16_C)]:
        p = peak[key]
        ridge = p / (BW_NOMINAL / 1e9)
        ax.plot([ridge, xhi], [p, p], color=col, linewidth=2, zorder=2)
        ax.plot([ridge], [p], marker="o", ms=4.5, color=col, zorder=3)
        ax.text(xhi / 1.06, p * 1.12, f"{lab}: {p / 1e3:.1f} TFLOP/s  (ridge {ridge:.0f})", ha="right", va="bottom", fontsize=9.4, color=INK)
        VALUES.append(("roofline", lab, "GFLOP/s", p, f"{CEILINGS[key]} FLOP/clk/SM x {N_SMS} SMs x {SM_CLOCK_MHZ} MHz"))
    ax.text(0.06, 5e3, "memory-bound\nleft of the ridge: bytes limit speed", fontsize=10.8, color=MUTED, va="top")
    ax.text(xhi / 1.06, 7.5e4, "compute-bound: right of the ridge, arithmetic limits speed", fontsize=10.8,
            color=MUTED, ha="right", va="bottom")
    colors = {"fp32": FP32_C, "tf32": TF32_C, "fp16": FP16_C}
    table = []
    for lab, oi, g, path, mk, filled, off, tn in pts:
        c = colors[path]
        ax.scatter([oi], [g], s=95 if filled else 80, marker=mk, facecolors=c if filled else SURFACE,
                   edgecolors=c, linewidths=2, zorder=5)
        ax.annotate(lab, (oi, g), xytext=off, textcoords="offset points", fontsize=10.1, color=INK, va="center",
                    ha="left", zorder=6)
        mem_roof = oi * BW_NOMINAL / 1e9
        roof = min(peak[path], mem_roof)
        bound = "memory" if mem_roof < peak[path] else "compute"
        table.append((tn, oi, g, 100 * g / roof, bound, c))
        VALUES.append(("roofline", tn, "OI FLOP/byte", oi, "measured (ncu)" if filled else "algorithmic model"))
        VALUES.append(("roofline", tn, "GFLOP/s", g, "W04 sweep 20261106-114338" if filled else "final run 20261106-151143"))
        VALUES.append(("roofline", tn, "% of roof", 100 * g / roof, f"roof = {bound}"))
    a, b = w04["S2a unfused"], w04["S2b fused (tuned)"]
    ax.annotate("", xy=(b["OI_measured"] / 1.2, b["GFLOP/s"]), xytext=(a["OI_measured"] * 1.25, a["GFLOP/s"] * 1.1),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.3, connectionstyle="arc3,rad=-0.25"), zorder=4)
    ax.text(100, 300, f"fusion S2a → S2b:\n{b['OI_measured'] / a['OI_measured']:.0f}× fewer DRAM bytes per FLOP",
            fontsize=10.1, color=INK2, ha="center", va="bottom")
    ax.grid(color=GRID, linewidth=0.8, zorder=0)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}"))
    ax.set_xlabel("operational intensity (FLOP per DRAM byte, log scale)", fontsize=11.8)
    ax.set_ylabel("attained performance (GFLOP/s, log scale)", fontsize=11.8)
    handles = [
        Line2D([], [], marker="o", ls="", mfc=SURFACE, mec=INK2, mew=2, ms=8, label="W01 vector add"),
        Line2D([], [], marker="s", ls="", mfc=SURFACE, mec=INK2, mew=2, ms=8, label="W02 dot product"),
        Line2D([], [], marker="D", ls="", mfc=SURFACE, mec=INK2, mew=2, ms=7, label="W03 GEMM, N = 1024"),
        Line2D([], [], marker="^", ls="", mfc=INK2, mec=INK2, mew=2, ms=8, label="W04 attention, N = 2048"),
        Line2D([], [], marker="o", ls="", mfc=FP32_C, mec=FP32_C, ms=8, label="FP32 CUDA cores"),
        Line2D([], [], marker="o", ls="", mfc=TF32_C, mec=TF32_C, ms=8, label="TF32 Tensor Cores"),
        Line2D([], [], marker="o", ls="", mfc=FP16_C, mec=FP16_C, ms=8, label="FP16 Tensor Cores"),
    ]
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 0.12), frameon=True, facecolor=SURFACE,
              edgecolor=GRID, fontsize=10.1, ncol=2, title="shape = workload · colour = execution unit",
              title_fontsize=10.1)
    fig.text(0.06, 0.955, "Roofline · NVIDIA RTX 4060 Laptop GPU", fontsize=16, fontweight="bold", va="bottom")
    fig.text(0.06, 0.915, "attainable performance = min(peak compute, operational intensity × DRAM bandwidth)",
             fontsize=11.1, color=INK2, va="bottom")
    # table
    tx.text(0, 1.0, "How close each kernel gets to its roof", fontsize=12.3, fontweight="bold", va="top")
    hdr_y = 0.94
    cols = [(0.0, "kernel", "left"), (0.60, "FLOP/B", "right"), (0.80, "GFLOP/s", "right"), (1.0, "% roof", "right")]
    for x, h, ha in cols:
        tx.text(x, hdr_y, h, fontsize=9.9, color=INK2, ha=ha, va="top", fontweight="bold")
    tx.plot([0, 1], [hdr_y - 0.035, hdr_y - 0.035], color=GRID, lw=1, transform=tx.transAxes)
    y = hdr_y - 0.055
    for tn, oi, g, pct, bound, c in table:
        tx.scatter([0.008], [y - 0.012], s=22, color=c, transform=tx.transAxes, clip_on=False)
        tx.text(0.03, y, tn, fontsize=9.9, va="top")
        tx.text(0.60, y, f"{oi:.3g}" if oi < 10 else f"{oi:,.0f}", fontsize=9.9, va="top", ha="right")
        tx.text(0.80, y, f"{g:,.0f}" if g >= 10 else f"{g:.1f}", fontsize=9.9, va="top", ha="right")
        tx.text(1.0, y, f"{pct:.0f}% {bound[:3]}.", fontsize=9.9, va="top", ha="right")
        y -= 0.058
    tx.text(0, y - 0.01, "% roof = attained ÷ min(compute roof, OI × 256 GB/s);\n"
            "mem. / com. = which roof is lower at that intensity.", fontsize=9.3, color=MUTED, va="top")
    fig.text(0.012, 0.014,
             "Filled markers: OI from DRAM bytes measured with Nsight Compute (W04, GPU sweep 20261106-114338). "
             "Hollow: algorithmic bytes (W01–W03, final run 20261106-151143).\n"
             f"Compute ceilings: Ada whitepaper per-SM rates × {N_SMS} SMs × {SM_CLOCK_MHZ} MHz (median loaded SM clock). "
             "Nominal, not microbenchmarked. Values: docs/figures/readme/figure_values.csv.",
             fontsize=9.5, color=MUTED, va="bottom")
    path = os.path.join(out, "roofline.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- hero chart
HERO = [
    ("W04", 2048, "AI attention", "the core of large language models", "S3_FP16", "S3"),
    ("W03", 1024, "Matrix multiply", "most of the work in every neural network", "S3", "S3"),
    ("W02", 33554432, "Dot product", "similarity scores in search & recommendations", "S2", "S3"),
    ("W01", 33554432, "Vector add", "element-wise maths on big arrays", "S2", "S2"),
]


def hero(rows, out):
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    y0 = 0
    yt, ylab = [], []
    for wl, n, name, sub, gpu_best, cpu_best in HERO:
        cpu = float(rows[(wl, "CPU", cpu_best, n)]["x_vs_s1"])
        gpu = float(rows[(wl, "GPU", gpu_best, n)]["x_vs_s1"])
        for dy, v, col, dev in [(0.21, cpu, CPU_C, "CPU"), (-0.21, gpu, GPU_C, "GPU")]:
            ax.barh(y0 + dy, max(v, 1.0) - 0.8, left=0.8, height=0.36, color=col, zorder=3)
            lab = f"{v:.1f}×" if v < 100 else f"{v:.0f}×"
            note = "   already at the memory-bandwidth limit" if v < 1.05 else ""
            ax.annotate(lab, (max(v, 1.0), y0 + dy), xytext=(6, 0), textcoords="offset points", va="center",
                        fontsize=14, fontweight="bold", color=INK, zorder=4)
            if note:
                ax.annotate(note, (1.0, y0 + dy), xytext=(52, 0), textcoords="offset points", va="center",
                            fontsize=11, color=INK2, zorder=4)
            VALUES.append(("hero", f"{wl} {dev} best ({cpu_best if dev == 'CPU' else gpu_best}) vs S1", "x_vs_s1", v,
                           "final_results.csv run 20261106-151143"))
        cpu_ms = min(float(rows[(wl, "CPU", st, n)]["mean_ms"]) for st in ("S1", "S2", "S3"))
        gpu_ms = min(float(r["mean_ms"]) for (a, d, st, nn), r in rows.items()
                     if a == wl and d == "GPU" and nn == n and st != "REF")
        g = cpu_ms / gpu_ms
        ax.annotate(f"{g:.1f}×", (1.22, y0 + 0.07), xycoords=("axes fraction", "data"), ha="right", va="bottom", annotation_clip=False,
                    fontsize=17, fontweight="bold", color=GPU_C)
        ax.annotate("GPU faster than CPU", (1.22, y0 + 0.02), xycoords=("axes fraction", "data"), ha="right", va="top", annotation_clip=False,
                    fontsize=10.5, color=INK2)
        VALUES.append(("hero", f"{wl} best CPU ms / best GPU ms", "ratio", g, "derived from mean_ms"))
        yt.append(y0)
        ylab.append((name, sub))
        y0 -= 1.25
    ax.set_xscale("log")
    ax.set_xlim(0.8, 260)
    ax.set_ylim(y0 + 0.6, 0.75)
    ax.set_yticks(yt)
    ax.set_yticklabels(["" for _ in yt])
    for y, (name, sub) in zip(yt, ylab):
        ax.annotate(name, (0, y), xycoords=("axes fraction", "data"), xytext=(-14, 7), textcoords="offset points",
                    ha="right", va="center", fontsize=15, fontweight="bold", color=INK)
        ax.annotate(sub, (0, y), xycoords=("axes fraction", "data"), xytext=(-14, -14), textcoords="offset points",
                    ha="right", va="center", fontsize=10.5, color=INK2)
    ax.tick_params(axis="y", length=0)
    ax.annotate("best GPU vs best CPU", (1.22, 1.0), xycoords="axes fraction", ha="right", va="bottom",
                fontsize=11, fontweight="bold", color=INK2, annotation_clip=False)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    ax.set_xlabel("speedup of the best optimized version over the simple version on the same chip (log scale)",
                  fontsize=11, color=INK2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=CPU_C), plt.Rectangle((0, 0), 1, 1, color=GPU_C)]
    fig.legend(handles, ["CPU · Intel Core i7-13650HX", "GPU · NVIDIA RTX 4060 Laptop"], loc="upper left",
               bbox_to_anchor=(0.012, 0.905), ncol=2, frameon=False, fontsize=12)
    fig.text(0.012, 0.955, "How much faster each workload got after optimization", fontsize=19, fontweight="bold",
             va="center")
    fig.text(0.012, 0.03, "Same chip, same problem, same session (run 20261106-151143); every version verified "
             "against a double-precision reference. Source: docs/results/final_results.csv.",
             fontsize=9.5, color=MUTED)
    fig.subplots_adjust(left=0.28, right=0.80, top=0.80, bottom=0.17)
    path = os.path.join(out, "hero_speedup.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "figures", "readme"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = load_final()
    print(hero(rows, a.out))
    print(comparison(rows, a.out))
    print(roofline(rows, a.out))
    vpath = os.path.join(a.out, "figure_values.csv")
    with open(vpath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["figure", "item", "quantity", "value", "source"])
        w.writerows(VALUES)
    print(vpath)


if __name__ == "__main__":
    main()
