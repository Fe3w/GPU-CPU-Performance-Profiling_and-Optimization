#!/usr/bin/env python3
"""make_graphs.py - comparison graphs for "Comparison final/graphs/comparison/".

    python3 "Comparison final/scripts/make_graphs.py"

Reads data/final_results.csv (the final paired re-test, run 20261106-151143) and
data/reference_values.csv (earlier records). Writes:
  1_runtime_every_stage.png   time per stage, CPU and GPU, one panel per workload (log axis)
  2_best_cpu_vs_best_gpu.png  fastest CPU stage vs fastest GPU stage per workload
  3_step_speedups.png         speedup of each optimization step (x vs the previous stage)
  4_final_vs_earlier.png      change of every stage vs its earlier recorded value (±10% band)
Colours: CPU blue, GPU orange (validated pair, direct labels on every bar).
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "graphs/comparison"
OUT.mkdir(parents=True, exist_ok=True)
COL = {"CPU": "#2a78d6", "GPU": "#eb6834"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
NAMES = {"W01": "W01 Vector Add", "W02": "W02 Dot Product", "W03": "W03 GEMM", "W04": "W04 Attention"}

rows = list(csv.DictReader(open(HERE / "data/final_results.csv")))
for r in rows:
    r["ms"] = float(r["mean_ms"])
refs = {r["key"]: float(r["mean_ms"]) for r in csv.DictReader(open(HERE / "data/reference_values.csv")) if r["primary"] == "1"}


def group(r):
    return r["workload"] + (f" N={r['n']}" if r["workload"] == "W04" else "")


def style(ax, grid_axis="y"):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def ms_txt(v):
    return f"{v:.0f}" if v >= 100 else f"{v:.1f}" if v >= 10 else f"{v:.2f}" if v >= 1 else f"{v:.3f}"


legend = [Patch(color=COL["CPU"], label="CPU i7-13650HX"), Patch(color=COL["GPU"], label="GPU RTX 4060 Laptop (55 W)")]
groups = []
for r in rows:
    if group(r) not in groups:
        groups.append(group(r))

# 1) runtime of every stage
ratios = [sum(1 for r in rows if group(r) == g and r["stage"] != "REF") + 0.5 for g in groups]
fig, axes = plt.subplots(1, len(groups), figsize=(0.62 * sum(ratios) + 1.5, 4.8), squeeze=False,
                         gridspec_kw={"width_ratios": ratios})
fig.patch.set_facecolor(SURF)
for ax, g in zip(axes[0], groups):
    rs = [r for r in rows if group(r) == g and r["stage"] != "REF"]
    rs.sort(key=lambda r: (r["device"] == "GPU",))
    labels = [f"{r['device']}\n{r['stage'].replace('_', chr(10))}" for r in rs]
    style(ax)
    bars = ax.bar(range(len(rs)), [r["ms"] for r in rs], color=[COL[r["device"]] for r in rs], edgecolor=SURF, linewidth=1.5)
    for b, r in zip(bars, rs):
        ax.annotate(ms_txt(r["ms"]), (b.get_x() + b.get_width() / 2, r["ms"]), ha="center", va="bottom", fontsize=7,
                    color=INK, xytext=(0, 2), textcoords="offset points")
    ref = next((r for r in rows if group(r) == g and r["stage"] == "REF"), None)
    if ref:
        ax.axhline(ref["ms"], color=INK2, linestyle="--", linewidth=1, label=f"GPU cuBLAS reference {ms_txt(ref['ms'])} ms")
        ax.legend(loc="upper right", fontsize=7, frameon=False, labelcolor=INK2)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_ylim(min(r["ms"] for r in rs) / 2, max(r["ms"] for r in rs) * 12)
    ax.set_xticks(range(len(rs)), labels, fontsize=6.8)
    ax.set_title(NAMES[g[:3]] + g[3:], fontsize=9.5, color=INK)
axes[0][0].set_ylabel("ms per iteration (log scale)", color=INK2, fontsize=8.5)
fig.legend(handles=legend, loc="lower center", ncol=2, frameon=False, fontsize=8.5)
fig.suptitle("Time of every stage, final paired re-test (lower is faster)", color=INK, fontsize=11)
fig.tight_layout(rect=(0, 0.06, 1, 0.95))
fig.savefig(OUT / "1_runtime_every_stage.png", dpi=150, facecolor=SURF)
plt.close(fig)

# 2) best CPU vs best GPU
fig, ax = plt.subplots(figsize=(8.5, 4.4))
fig.patch.set_facecolor(SURF)
style(ax)
w = 0.38
for i, g in enumerate(groups):
    best = {d: min((r for r in rows if group(r) == g and r["device"] == d and r["stage"] != "REF"), key=lambda r: r["ms"])
            for d in ("CPU", "GPU")}
    for j, d in enumerate(("CPU", "GPU")):
        b = best[d]
        x = i + (j - 0.5) * w
        ax.bar(x, b["ms"], width=w - 0.03, color=COL[d], edgecolor=SURF, linewidth=1.5)
        ax.annotate(f"{b['stage'].replace('_', ' ')}\n{ms_txt(b['ms'])} ms", (x, b["ms"]), ha="center", va="bottom",
                    fontsize=7, color=INK, xytext=(0, 2), textcoords="offset points")
    ratio = best["CPU"]["ms"] / best["GPU"]["ms"]
    ax.annotate(f"GPU {ratio:.1f}× faster", (i, max(best["CPU"]["ms"], best["GPU"]["ms"]) * 4.5), ha="center",
                fontsize=8.5, color=INK, fontweight="bold")
ax.set_yscale("log")
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.set_ylim(0.08, 300)
ax.set_xticks(range(len(groups)), [NAMES[g[:3]] + g[3:] for g in groups], fontsize=8.5)
ax.set_ylabel("ms per iteration (log scale)", color=INK2, fontsize=8.5)
ax.legend(handles=legend, loc="upper left", frameon=False, fontsize=8.5)
ax.set_title("Fastest CPU stage vs fastest GPU stage, same session", color=INK, fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "2_best_cpu_vs_best_gpu.png", dpi=150, facecolor=SURF)
plt.close(fig)

# 3) step speedups
steps = [r for r in rows if r["x_vs_prev"] and r["stage"] != "REF"]
fig, ax = plt.subplots(figsize=(8.5, 0.32 * len(steps) + 1.4))
fig.patch.set_facecolor(SURF)
style(ax, "x")
ys = list(range(len(steps)))[::-1]
vals = [float(r["x_vs_prev"]) for r in steps]
ax.barh(ys, vals, height=0.68, color=[COL[r["device"]] for r in steps], edgecolor=SURF, linewidth=1.5)
for y, v in zip(ys, vals):
    ax.annotate(f"{v:.2f}×", (v, y), va="center", ha="left", fontsize=7.5, color=INK, xytext=(3, 0), textcoords="offset points")
ax.set_yticks(ys, [f"{group(r)} {r['device']}  {r['prev']} → {r['stage']}" for r in steps], fontsize=7.5, color=INK)
ax.set_xscale("log")
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
ax.set_xlim(0.7, max(vals) * 2.5)
ax.axvline(1, color=INK2, linewidth=0.8)
ax.legend(handles=legend, loc="lower right", frameon=False, fontsize=8)
ax.set_xlabel("speedup of the step: time of the previous stage / time of this stage (log scale)", color=INK2, fontsize=8.5)
ax.set_title("What each optimization step bought", color=INK, fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "3_step_speedups.png", dpi=150, facecolor=SURF)
plt.close(fig)

# 4) final vs earlier
items = []
for r in rows:
    key = f"{r['workload']}_{r['device']}_{r['stage']}" + (f"_N{r['n']}" if r["workload"] == "W04" else "")
    if key in refs:
        items.append((key.replace("_", " "), (r["ms"] - refs[key]) / refs[key] * 100, r["device"]))
fig, ax = plt.subplots(figsize=(8.5, 0.28 * len(items) + 1.5))
fig.patch.set_facecolor(SURF)
style(ax, "x")
ys = list(range(len(items)))[::-1]
ax.axvspan(-10, 10, color=GRID, alpha=0.6, linewidth=0)
ax.barh(ys, [d for _, d, _ in items], height=0.66, color=[COL[dv] for _, _, dv in items], edgecolor=SURF, linewidth=1.5)
for y, (_, d, _) in zip(ys, items):
    ax.annotate(f"{d:+.0f}%" + (" ⚠" if abs(d) > 10 else ""), (d, y), va="center", ha="left" if d >= 0 else "right",
                fontsize=7, color=INK, xytext=(3 if d >= 0 else -3, 0), textcoords="offset points")
ax.set_yticks(ys, [k for k, _, _ in items], fontsize=7, color=INK)
ax.axvline(0, color=INK2, linewidth=0.8)
ax.set_xlim(min(-45, min(x for _, x, _ in items) - 12), 30)
ax.set_xlabel("time now vs the earlier recorded run (%; negative = faster now; grey band = ±10%)", color=INK2, fontsize=8.5)
ax.legend(handles=legend, loc="lower right", frameon=False, fontsize=8)
ax.set_title("Final re-test vs earlier records (every ⚠ is explained in data/deviation_notes.md)", color=INK, fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "4_final_vs_earlier.png", dpi=150, facecolor=SURF)
plt.close(fig)
print("written:", *sorted(p.name for p in OUT.glob("*.png")))
