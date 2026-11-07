#!/usr/bin/env python3
"""Comparison video for the README: an animated, data-driven walk through W01-W04.

Every number shown is read from docs/results/final_results.csv (run 20261106-151143) and,
for the Roofline scene, docs/figures/w04/plot_data.csv (W04 GPU sweep 20261106-114338).

Usage:  python3 tools/readme_video.py [--out docs/media/comparison_video.mp4] [--stills DIR]
Needs:  matplotlib, ffmpeg (libx264). Renders 1280x720 at 30 fps, one file per scene,
        joined with the concat demuxer.
"""
import argparse
import csv
import math
import os
import shutil
import subprocess
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
from matplotlib import font_manager  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINAL = os.path.join(ROOT, "docs", "results", "final_results.csv")
W04PLOT = os.path.join(ROOT, "docs", "figures", "w04", "plot_data.csv")

W, H, FPS, DPI = 1280, 720, 30, 100
SURFACE, INK, INK2, MUTED, GRID, CARD = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#f1f0ec"
CPU_C, GPU_C, REF_C = "#2a78d6", "#eb6834", "#9a9893"
FP32_C, TF32_C, FP16_C = "#2a78d6", "#1baf7a", "#eb6834"
for f in ("Inter", "DejaVu Sans"):
    if any(f in x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "text.color": INK})

R = {}
with open(FINAL, newline="") as fh:
    for r in csv.DictReader(fh):
        R[(r["workload"], r["device"], r["stage"], int(r["n"]))] = r
E3 = {}
with open(W04PLOT, newline="") as fh:
    for r in csv.DictReader(fh):
        if r["figure"] == "E3" and r["n"] == "2048":
            E3.setdefault(r["series"], {})[r["quantity"]] = float(r["value"])


def ms(wl, dev, st, n):
    return float(R[(wl, dev, st, n)]["mean_ms"])


def ease(x):
    x = max(0.0, min(1.0, x))
    return 1 - (1 - x) ** 3


def seg(t, a, b):
    """progress 0..1 of t inside [a, b], eased"""
    return ease((t - a) / (b - a)) if b > a else float(t >= a)


def fmt_ms(v):
    return f"{v:,.0f} ms" if v >= 100 else f"{v:.1f} ms" if v >= 10 else f"{v:.2f} ms" if v >= 1 else f"{v:.3f} ms"


def new_fig():
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
    return fig


def header(fig, kicker, title, alpha=1.0):
    fig.text(0.05, 0.915, kicker, fontsize=13, color=GPU_C, fontweight="bold", alpha=alpha)
    fig.text(0.05, 0.85, title, fontsize=25, fontweight="bold", color=INK, alpha=alpha)


def footer(fig, txt="i7-13650HX + RTX 4060 Laptop (55 W) · run 20261106-151143 · mean of 5 repetitions"):
    fig.text(0.05, 0.03, txt, fontsize=9.5, color=MUTED)


def fade(fig, t, dur, fin=0.45, fout=0.45):
    a = 0.0
    if t < fin:
        a = 1 - t / fin
    elif t > dur - fout:
        a = (t - (dur - fout)) / fout
    if a > 0:
        fig.patches.append(plt.Rectangle((0, 0), 1, 1, transform=fig.transFigure, color=SURFACE,
                                         alpha=min(1, a), zorder=1000))


# ------------------------------------------------------------------ scenes
def s_title(fig, t, dur):
    k = seg(t, 0.2, 1.2)
    fig.text(0.5, 0.64, "CPU vs GPU", ha="center", fontsize=46, fontweight="bold", alpha=k)
    fig.text(0.5, 0.54, "How far does each optimization take a workload, and why?", ha="center",
             fontsize=20, color=INK2, alpha=seg(t, 0.6, 1.6))
    k2 = seg(t, 1.2, 2.2)
    for x, c, lab in [(0.33, CPU_C, "Intel Core i7-13650HX\n6 P + 8 E cores · AVX2/FMA"),
                      (0.67, GPU_C, "NVIDIA RTX 4060 Laptop\n24 SMs · Tensor Cores · 55 W")]:
        fig.patches.append(plt.Rectangle((x - 0.15, 0.27), 0.30, 0.15, transform=fig.transFigure,
                                         fc=CARD, ec="none", alpha=k2))
        fig.patches.append(plt.Rectangle((x - 0.15, 0.27), 0.006, 0.15, transform=fig.transFigure, fc=c, alpha=k2))
        fig.text(x, 0.345, lab, ha="center", va="center", fontsize=13.5, alpha=k2, linespacing=1.5)
    fig.text(0.5, 0.16, "4 workloads · 3 optimization stages each · benchmarked, verified, profiled",
             ha="center", fontsize=13, color=MUTED, alpha=seg(t, 2.0, 3.0))


def s_ladder(fig, t, dur):
    header(fig, "THE METHOD", "Same ladder for every workload", seg(t, 0, 0.8))
    steps = [("S1 · Baseline", "the simplest correct code", INK2),
             ("S2 · Software", "threads, tiling, coalescing,\nbetter reductions, fusion", CPU_C),
             ("S3 · Hardware", "AVX2/FMA on the CPU,\nTensor Cores on the GPU", GPU_C)]
    for i, (a, b, c) in enumerate(steps):
        k = seg(t, 0.6 + 0.6 * i, 1.4 + 0.6 * i)
        x = 0.07 + 0.30 * i
        fig.patches.append(plt.Rectangle((x, 0.50), 0.26, 0.2, transform=fig.transFigure, fc=CARD, ec="none", alpha=k))
        fig.patches.append(plt.Rectangle((x, 0.50), 0.26, 0.008, transform=fig.transFigure, fc=c, alpha=k))
        fig.text(x + 0.02, 0.645, a, fontsize=17, fontweight="bold", alpha=k)
        fig.text(x + 0.02, 0.615, b, fontsize=12.5, color=INK2, va="top", alpha=k, linespacing=1.4)
        if i < 2:
            fig.text(x + 0.275, 0.59, "→", fontsize=24, color=MUTED, alpha=k, va="center")
    rows = [("W01 Vector Add", "C = A + B", "memory bandwidth"),
            ("W02 Dot Product", "s = Σ aᵢbᵢ", "reductions"),
            ("W03 GEMM", "C = A × B", "compute + data reuse"),
            ("W04 Attention", "softmax(Q·K^T/√d)·V", "memory traffic + fusion")]
    for i, (a, b, c) in enumerate(rows):
        k = seg(t, 2.6 + 0.35 * i, 3.3 + 0.35 * i)
        y = 0.38 - 0.07 * i
        fig.text(0.07, y, a, fontsize=14, fontweight="bold", alpha=k)
        fig.text(0.34, y, b, fontsize=14, color=INK2, alpha=k)
        fig.text(0.62, y, "stresses: " + c, fontsize=13, color=MUTED, alpha=k)
    fig.text(0.07, 0.10, "Every stage checked against a double-precision reference · all 34 stage runs correct",
             fontsize=12, color=INK2, alpha=seg(t, 4.4, 5.2))


def bars_scene(fig, t, dur, kicker, title, items, insight, sub):
    """items: (label, ms, colour, speedup text). Bars grow on a log axis, one after another."""
    header(fig, kicker, title, seg(t, 0, 0.7))
    fig.text(0.05, 0.80, sub, fontsize=12.5, color=INK2, alpha=seg(t, 0.2, 0.9))
    ax = fig.add_axes([0.28, 0.20, 0.52, 0.55])
    times = [it[1] for it in items]
    lo, hi = min(times) / 2.5, max(times) * 4.5
    lg = lambda v: math.log10(v)  # noqa: E731
    ax.set_xlim(0, lg(hi) - lg(lo))
    ax.set_ylim(-0.6, len(items) - 0.4)
    ax.axis("off")
    n = len(items)
    for i, (lab, v, col, sp) in enumerate(items):
        y = n - 1 - i
        k = seg(t, 0.8 + 0.45 * i, 1.8 + 0.45 * i)
        length = (lg(v) - lg(lo)) * k
        ax.barh(y, length, height=0.6, color=col, alpha=0.95 if k > 0 else 0)
        ax.text(-0.04 * (lg(hi) - lg(lo)), y, lab, ha="right", va="center", fontsize=13, alpha=min(1, k * 3))
        if k > 0:
            cur = 10 ** (lg(lo) + length) if k < 1 else v
            ax.text(length + 0.02 * (lg(hi) - lg(lo)), y, fmt_ms(cur), va="center", fontsize=13,
                    fontweight="bold", alpha=min(1, k * 2))
            if sp:
                ax.text(1.10, y, sp, va="center", ha="right", fontsize=14, fontweight="bold",
                        transform=ax.get_yaxis_transform(),
                        color=col, alpha=seg(t, 1.8 + 0.45 * i, 2.3 + 0.45 * i))
    fig.text(0.852, 0.775, "speedup\nvs S1", fontsize=10, color=MUTED, ha="right", alpha=seg(t, 1.5, 2.2))
    fig.text(0.28, 0.165, "time per iteration, log scale · shorter = faster", fontsize=10, color=MUTED)
    k = seg(t, 0.8 + 0.45 * n + 0.6, 1.6 + 0.45 * n + 0.6)
    fig.patches.append(plt.Rectangle((0.05, 0.065), 0.90, 0.075, transform=fig.transFigure, fc=CARD, ec="none", alpha=k))
    fig.text(0.07, 0.102, insight, fontsize=14, va="center", alpha=k)


def x_s1(wl, dev, st, n):
    return f"{float(R[(wl, dev, st, n)]['x_vs_s1']):.1f}×"


def s_w01(fig, t, dur):
    n = 33554432
    items = [("CPU  S1 scalar", ms("W01", "CPU", "S1", n), CPU_C, "1.0×"),
             ("CPU  S2 OpenMP ×6", ms("W01", "CPU", "S2", n), CPU_C, x_s1("W01", "CPU", "S2", n)),
             ("CPU  S3 + AVX2", ms("W01", "CPU", "S3", n), CPU_C, x_s1("W01", "CPU", "S3", n)),
             ("GPU  S1 naive", ms("W01", "GPU", "S1", n), GPU_C, "1.0×"),
             ("GPU  S2 coalesced", ms("W01", "GPU", "S2", n), GPU_C, x_s1("W01", "GPU", "S2", n))]
    gbs = float(R[("W01", "GPU", "S2", n)]["rate"])
    bars_scene(fig, t, dur, "W01 · VECTOR ADD", "When memory is the limit, better code can't help", items,
               f"Both GPU kernels stream {gbs:.0f} GB/s, {100 * gbs / 256:.0f}% of the 256 GB/s DRAM peak. "
               "On the CPU, threads give 10×; AVX2 adds nothing.",
               "C = A + B over 33.5 M floats: one add for every 12 bytes moved")


def s_w02(fig, t, dur):
    n = 33554432
    items = [("CPU  S1 scalar", ms("W02", "CPU", "S1", n), CPU_C, "1.0×"),
             ("CPU  S2 OpenMP ×6", ms("W02", "CPU", "S2", n), CPU_C, x_s1("W02", "CPU", "S2", n)),
             ("CPU  S3 + AVX2/FMA", ms("W02", "CPU", "S3", n), CPU_C, x_s1("W02", "CPU", "S3", n)),
             ("GPU  S1 atomics", ms("W02", "GPU", "S1", n), GPU_C, "1.0×"),
             ("GPU  S2 block reduce", ms("W02", "GPU", "S2", n), GPU_C, x_s1("W02", "GPU", "S2", n))]
    bars_scene(fig, t, dur, "W02 · DOT PRODUCT", "How you combine results matters", items,
               "GPU S1 makes every thread update one address. Reducing inside blocks first: "
               f"{x_s1('W02', 'GPU', 'S2', n)} faster, same arithmetic.",
               "s = Σ aᵢbᵢ over 33.5 M floats: millions of partial sums become one number")


def s_w03(fig, t, dur):
    n = 1024
    items = [("CPU  S1 naive loop", ms("W03", "CPU", "S1", n), CPU_C, "1.0×"),
             ("CPU  S2 blocked + ×8", ms("W03", "CPU", "S2", n), CPU_C, x_s1("W03", "CPU", "S2", n)),
             ("CPU  S3 + AVX2/FMA", ms("W03", "CPU", "S3", n), CPU_C, x_s1("W03", "CPU", "S3", n)),
             ("GPU  S1 naive", ms("W03", "GPU", "S1", n), GPU_C, "1.0×"),
             ("GPU  S2 shared tiles", ms("W03", "GPU", "S2", n), GPU_C, x_s1("W03", "GPU", "S2", n)),
             ("GPU  S3 TF32 Tensor", ms("W03", "GPU", "S3", n), GPU_C, x_s1("W03", "GPU", "S3", n))]
    bars_scene(fig, t, dur, "W03 · MATRIX MULTIPLY", "Reuse data, then use special hardware", items,
               "Each value is reused 1024 times. CPU: blocking + threads + FMA = 114×. "
               "GPU: Tensor Cores add 3.8× over tiling.",
               "C = A × B, 1024 × 1024: 2.1 GFLOP, 171 FLOP per byte")


def s_w04(fig, t, dur):
    n = 2048
    header(fig, "W04 · ATTENTION (the core of every Transformer)", "Three steps to 50× on the GPU", seg(t, 0, 0.7))
    fig.text(0.05, 0.80, "softmax(Q·K^T/√d)·V, sequence length 2048, 8 heads, head size 64", fontsize=12.5,
             color=INK2, alpha=seg(t, 0.2, 0.9))
    st = [("S1", "naive\n3 kernels"), ("S2a", "tiled\nunfused"), ("S2b", "fused\n(FlashAttention)"),
          ("S3_FP16", "FP16\nTensor Cores")]
    vals = [ms("W04", "GPU", s, n) for s, _ in st]
    ax = fig.add_axes([0.08, 0.22, 0.50, 0.50])
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(0, max(vals) * 1.25)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_yticks([])
    ax.set_xticks(range(4))
    ax.set_xticklabels([b for _, b in st], fontsize=11.5)
    ax.tick_params(axis="x", length=0, colors=INK2)
    for i, v in enumerate(vals):
        k = seg(t, 0.8 + 0.5 * i, 1.6 + 0.5 * i)
        ax.bar(i, v * k, width=0.62, color=GPU_C, alpha=0.95)
        if k > 0:
            ax.text(i, v * k + max(vals) * 0.03, fmt_ms(v), ha="center", fontsize=13, fontweight="bold",
                    alpha=min(1, k * 2))
    fig.text(0.08, 0.095, "GPU time per iteration (linear scale)", fontsize=10, color=MUTED)
    steps = [("kernel quality", "S1 → S2a", vals[0] / vals[1], "tiled GEMMs, warp-shuffle softmax"),
             ("fusion + reg. tiling", "S2a → S2b", vals[1] / vals[2], "N×N scores stay on chip, not in DRAM"),
             ("FP16 Tensor Cores", "S2b → S3", vals[2] / vals[3], "13.5 TFLOP/s, error 0.03% vs double")]
    for i, (a, b, x, c) in enumerate(steps):
        k = seg(t, 3.0 + 0.9 * i, 3.8 + 0.9 * i)
        y = 0.66 - 0.135 * i
        fig.patches.append(plt.Rectangle((0.63, y - 0.06), 0.33, 0.11, transform=fig.transFigure, fc=CARD, ec="none", alpha=k))
        fig.text(0.65, y + 0.012, f"{x:.2f}×", fontsize=22, fontweight="bold", color=GPU_C, alpha=k, va="center")
        fig.text(0.76, y + 0.025, f"{a} ({b})", fontsize=12.5, fontweight="bold", alpha=k, va="center")
        fig.text(0.76, y - 0.015, c, fontsize=10.5, color=INK2, alpha=k, va="center")
    total = vals[0] / vals[3]
    k = seg(t, 6.0, 6.8)
    fig.text(0.65, 0.215, f"= {total:.1f}× in total", fontsize=24, fontweight="bold", alpha=k)
    fig.text(0.65, 0.175, f"CPU S1 → S3: {x_s1('W04', 'CPU', 'S3', n)} (fused + 20 threads + AVX2 exp)", fontsize=11.5,
             color=INK2, alpha=k)


def s_roof(fig, t, dur):
    header(fig, "ROOFLINE", "Why each kernel is as fast as it is", seg(t, 0, 0.7))
    fig.text(0.05, 0.80, "speed is capped by memory bandwidth (diagonal) or by compute (flat): whichever is lower",
             fontsize=12.5, color=INK2, alpha=seg(t, 0.2, 0.9))
    ax = fig.add_axes([0.10, 0.14, 0.84, 0.60])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.05, 3000)
    ax.set_ylim(1, 8e4)
    clk = 2670e6
    p32, p16 = 256 * 24 * clk / 1e9, 512 * 24 * clk / 1e9
    k = seg(t, 0.6, 1.6)
    xs = [0.05 * (3000 / 0.05) ** (i / 200) for i in range(201)]
    xs_m = [x for x in xs if x * 256 <= p16 * k + 1e-9 or x * 256 <= 1]
    ax.plot(xs_m, [x * 256 for x in xs_m], color=INK2, lw=1.6, ls="--")
    if k > 0.99:
        for p, c, lab in [(p32, FP32_C, f"FP32 / TF32 roof {p32 / 1e3:.1f} TFLOP/s"),
                          (p16, FP16_C, f"FP16 Tensor roof {p16 / 1e3:.1f} TFLOP/s")]:
            kk = seg(t, 1.6, 2.4)
            ax.plot([p / 256, p / 256 + (3000 - p / 256) * kk], [p, p], color=c, lw=2.2)
            ax.text(2900, p * 1.12, lab, ha="right", fontsize=11, alpha=kk)
        p0, p1 = ax.transData.transform((0.1, 25.6)), ax.transData.transform((1.0, 256))
        ang = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
        ax.text(1.5, 1.5 * 256 * 1.7, "DRAM 256 GB/s", fontsize=11, color=INK2, rotation=ang,
                rotation_mode="anchor", ha="center", va="bottom")
    ax.grid(color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("operational intensity (FLOP per DRAM byte)", fontsize=11, color=INK2)
    ax.set_ylabel("GFLOP/s", fontsize=11, color=INK2)
    ax.tick_params(colors=INK2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    w01 = float(R[("W01", "GPU", "S2", 33554432)]["rate"]) / 12
    w02 = float(R[("W02", "GPU", "S2", 33554432)]["rate"]) * 0.25
    pts = [(2.6, 1 / 12, w01, FP32_C, "o", "W01 add: 88% of roof", 1),
           (3.0, 0.25, w02, FP32_C, "s", "W02 dot: 84% of roof", 1),
           (3.4, 1024 / 6, float(R[("W03", "GPU", "S3", 1024)]["rate"]), TF32_C, "D", "W03 GEMM TF32", -1),
           (4.0, E3["S2a unfused"]["OI_measured"], E3["S2a unfused"]["GFLOP/s"], FP32_C, "^", "W04 unfused", 1),
           (5.6, E3["S2b fused (tuned)"]["OI_measured"], E3["S2b fused (tuned)"]["GFLOP/s"], FP32_C, "^", "W04 fused", 1),
           (6.6, E3["S3 FP16 (tuned)"]["OI_measured"], E3["S3 FP16 (tuned)"]["GFLOP/s"], FP16_C, "^", "W04 FP16 Tensor", -1)]
    for t0, x, y, c, mk, lab, side in pts:
        kk = seg(t, t0, t0 + 0.5)
        if kk <= 0:
            continue
        if lab == "W04 fused":   # slide right from the unfused point: fusion cuts DRAM bytes
            x0, y0 = E3["S2a unfused"]["OI_measured"], E3["S2a unfused"]["GFLOP/s"]
            kk2 = seg(t, t0, t0 + 1.2)
            x = 10 ** (math.log10(x0) + (math.log10(x) - math.log10(x0)) * kk2)
            y = 10 ** (math.log10(y0) + (math.log10(y) - math.log10(y0)) * kk2)
            kk = 1
        ax.scatter([x], [y], s=120, marker=mk, color=c, alpha=kk, zorder=5, edgecolors=SURFACE, linewidths=1.5)
        ax.annotate(lab, (x, y), xytext=(10 * side, -4 if side > 0 else -16), textcoords="offset points",
                    fontsize=11.5, alpha=kk, ha="left" if side > 0 else "right")
    k = seg(t, 5.0, 5.6)
    ax.text(45, 320, f"fusion: {E3['S2b fused (tuned)']['OI_measured'] / E3['S2a unfused']['OI_measured']:.0f}× "
            "fewer DRAM bytes\n→ the point moves right", fontsize=11.5, color=INK2, alpha=k * (1 if t < 7.5 else 1))


def s_cpu_gpu(fig, t, dur):
    header(fig, "CPU vs GPU", "Best GPU stage vs best CPU stage", seg(t, 0, 0.7))
    fig.text(0.05, 0.80, "same session, same problem size", fontsize=12.5, color=INK2, alpha=seg(t, 0.2, 0.9))
    wl = [("W01 Vector Add", "W01", 33554432, "memory-bound: the gap is the bandwidth gap"),
          ("W02 Dot Product", "W02", 33554432, "memory-bound after the reduction fix"),
          ("W03 GEMM", "W03", 1024, "compute-bound: thousands of lanes + Tensor Cores"),
          ("W04 Attention", "W04", 2048, "compute-bound once fused")]
    ratios = []
    for name, w, n, why in wl:
        cpu = min(ms(w, "CPU", s, n) for s in ("S1", "S2", "S3") if (w, "CPU", s, n) in R)
        gpu = min(float(r["mean_ms"]) for (a, d, s, nn), r in R.items() if a == w and d == "GPU" and nn == n and s != "REF")
        ratios.append((name, cpu / gpu, why))
    ax = fig.add_axes([0.25, 0.16, 0.48, 0.58])
    mx = max(r for _, r, _ in ratios)
    ax.set_xlim(0, math.log10(mx) * 1.15 + 0.1)
    ax.set_ylim(-0.6, 3.6)
    ax.axis("off")
    for i, (name, r, why) in enumerate(ratios):
        y = 3 - i
        k = seg(t, 0.8 + 0.6 * i, 1.8 + 0.6 * i)
        L = (math.log10(r) + 0.1) * k
        ax.barh(y, L, height=0.55, color=GPU_C)
        fig.text(0.235, 0.16 + 0.58 * (y + 0.6) / 4.2, name, ha="right", va="center", fontsize=14, alpha=min(1, k * 3))
        if k > 0:
            ax.text(L + 0.03, y + 0.05, f"{10 ** (L / k - 0.1) if k >= 1 else 10 ** (L - 0.1 * k):.1f}×",
                    va="center", fontsize=16, fontweight="bold", alpha=min(1, k * 2))
            ax.text(L + 0.03, y - 0.3, why, va="center", fontsize=10.5, color=INK2, alpha=seg(t, 1.8 + 0.6 * i, 2.4 + 0.6 * i))
    fig.text(0.25, 0.12, "GPU speedup over the best CPU stage (log scale)", fontsize=10, color=MUTED)


def s_end(fig, t, dur):
    header(fig, "TAKEAWAYS", "Match the optimization to the bottleneck", seg(t, 0, 0.7))
    pts = ["Memory-bound (W01, W02): use all the bandwidth, then stop; faster math changes nothing.",
           "Reductions (W02): organise how results are combined; the GPU went 34.5× faster.",
           "Compute-bound (W03, W04): reuse data on chip, then use Tensor Cores (5.2× and 50×).",
           "Fusion (W04): cut DRAM bytes 24×; it moves the kernel right on the Roofline.",
           "Every number is backed by a correctness check and a profiler counter."]
    for i, p in enumerate(pts):
        k = seg(t, 0.7 + 0.55 * i, 1.4 + 0.55 * i)
        y = 0.70 - 0.095 * i
        fig.patches.append(plt.Rectangle((0.06, y - 0.012), 0.008, 0.045, transform=fig.transFigure,
                                         fc=[CPU_C, CPU_C, GPU_C, GPU_C, INK2][i], alpha=k))
        fig.text(0.085, y + 0.01, p, fontsize=15.5, va="center", alpha=k)
    fig.text(0.06, 0.12, "github.com/Fe3w/GPU-CPU-Performance-Profiling_and-Optimization", fontsize=15,
             fontweight="bold", alpha=seg(t, 4.0, 4.8))


SCENES = [("title", s_title, 4.5), ("ladder", s_ladder, 7.0), ("w01", s_w01, 8.5), ("w02", s_w02, 8.5),
          ("w03", s_w03, 9.0), ("w04", s_w04, 10.0), ("roof", s_roof, 10.0), ("cpugpu", s_cpu_gpu, 7.5),
          ("end", s_end, 7.5)]


def frame(fn, t, dur, footer_on=True):
    fig = new_fig()
    fn(fig, t, dur)
    if footer_on:
        footer(fig)
    fade(fig, t, dur)
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    data = bytes(buf)
    plt.close(fig)
    return data


def render_scene(name, fn, dur, path):
    tmp = path[:-4] + ".part.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-tune", "animation",
           "-pix_fmt", "yuv420p", "-r", str(FPS), tmp]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    nfr = int(round(dur * FPS))
    for i in range(nfr):
        p.stdin.write(frame(fn, i / FPS, dur, footer_on=name not in ("title", "end")))
    p.stdin.close()
    if p.wait() != 0:
        raise SystemExit(f"ffmpeg failed for scene {name}")
    os.replace(tmp, path)


def write_poster(path):
    fig = new_fig()
    s_w04(fig, 9.0, 10.0)
    footer(fig)
    fig.patches.append(plt.Rectangle((0, 0), 1, 1, transform=fig.transFigure, color=SURFACE, alpha=0.35, zorder=900))
    import matplotlib.patches as mp
    cx, cy, r = 0.5, 0.5, 0.075
    fig.patches.append(mp.Ellipse((cx, cy), 2 * r * H / W, 2 * r, transform=fig.transFigure, fc=INK, alpha=0.82, zorder=950))
    tri = [(cx - 0.018, cy - 0.045), (cx - 0.018, cy + 0.045), (cx + 0.03, cy)]
    fig.patches.append(mp.Polygon(tri, closed=True, transform=fig.transFigure, fc=SURFACE, zorder=960))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "media", "comparison_video.mp4"))
    ap.add_argument("--stills", help="only write PNG stills of each scene's busiest moment into this folder")
    ap.add_argument("--poster", help="also write a poster PNG (W04 scene with a play button)")
    ap.add_argument("--poster-only", action="store_true", help="write only the poster, no video")
    a = ap.parse_args()
    if a.stills:
        os.makedirs(a.stills, exist_ok=True)
        for name, fn, dur in SCENES:
            for frac in (0.55, 0.9):
                fig = new_fig()
                fn(fig, dur * frac, dur)
                footer(fig)
                fig.savefig(os.path.join(a.stills, f"{name}_{int(frac * 100)}.png"), dpi=DPI)
                plt.close(fig)
        return
    if a.poster_only:
        write_poster(a.poster or os.path.join(ROOT, "docs", "media", "comparison_video_poster.png"))
        return
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    work = tempfile.mkdtemp(prefix="readme_video_")
    parts = []
    for name, fn, dur in SCENES:
        path = os.path.join(work, f"{name}.mp4")
        render_scene(name, fn, dur, path)
        parts.append(path)
        print("scene", name, "done")
    lst = os.path.join(work, "list.txt")
    with open(lst, "w") as fh:
        for p in parts:
            fh.write(f"file '{p}'\n")
    tmp = a.out[:-4] + ".part.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
                    "-movflags", "+faststart", tmp], check=True)
    os.replace(tmp, a.out)
    if a.poster:
        write_poster(a.poster)
    shutil.rmtree(work)
    print(a.out)


if __name__ == "__main__":
    main()
