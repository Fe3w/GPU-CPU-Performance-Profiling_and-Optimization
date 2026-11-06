#!/usr/bin/env python3
"""
plot_race.py - chart and tabulate a demo-kit race run.

Reads the race CSVs of one run folder (results/live/<timestamp>/). Only strictly
named files are used:  W0<k>[CPU]_<stage>[_r<round>].csv  with stage S1, S2, S2a,
S2b, S3, S3_FP16, S3_TF32 or REF (vendor reference). Other CSVs in the folder
(preflight, profiling runs) are ignored. Rounds: median over rounds of each
round's mean.

Writes into the run folder:
  race_<stamp>.png          runtime per stage, one panel per workload (log axis
                            when the stages span > 20×), labels: ms, × vs S1, % time saved
  race_speedup_<stamp>.png  every stage of every workload as × vs its own S1 (log axis)
  race_<stamp>.md           tables: ms, SD, × vs S1, × vs previous stage, % saved,
                            % of REF speed, tile, correct; factored chain; best per
                            workload; W04 CPU vs GPU at the same N
  race_<stamp>.csv          the same numbers, machine-readable

Usage:
  python3 plot_race.py                     # newest run folder
  python3 plot_race.py path/to/run_folder  # a specific run
  python3 plot_race.py --compare           # every run, to see run-to-run drift
  python3 plot_race.py --selftest          # file-name grammar check (no data needed)
"""
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import racelib  # noqa: E402

# Validated categorical slots (light surface): S2/S2a blue, S2b aqua, S3/FP16 orange, TF32 violet.
# S1 is the baseline and stays neutral grey; REF is a dashed ink line.
COLORS = {"S1": "#8f8d87", "S2": "#2a78d6", "S2a": "#2a78d6", "S2b": "#1baf7a",
          "S3": "#eb6834", "S3_FP16": "#eb6834", "S3_TF32": "#6250d6"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
WORKLOAD_ORDER = ["W01", "W02", "W03", "W04", "W04 CPU"]
NAMES = {"W01": "W01 Vector Add", "W02": "W02 Dot Product", "W03": "W03 GEMM",
         "W04": "W04 Attention (GPU)", "W04 CPU": "W04 Attention (CPU)"}
# CPU vs GPU pairs for W04 (slow, fast)
CROSS = [("CPU S1 naive → GPU S1 naive", ("W04 CPU", "S1"), ("W04", "S1")),
         ("CPU S2 blocked+OMP → GPU S2b fused", ("W04 CPU", "S2"), ("W04", "S2b")),
         ("CPU S3 AVX2 → GPU S3 FP16 Tensor Core", ("W04 CPU", "S3"), ("W04", "S3_FP16")),
         ("CPU S1 → GPU S3 FP16 (end to end)", ("W04 CPU", "S1"), ("W04", "S3_FP16"))]


def live_roots():
    """Run-folder roots, in order: $LIVE_ROOT, $PROJECT/results/live, the repository's results/live."""
    here = Path(__file__).resolve().parent
    cands = []
    if os.environ.get("LIVE_ROOT"):
        cands.append(Path(os.environ["LIVE_ROOT"]))
    if os.environ.get("PROJECT"):
        cands.append(Path(os.environ["PROJECT"]) / "results/live")
    cands.append(here.parent.parent / "results/live")
    return [c.resolve() for c in cands if c.is_dir()]


def newest_run():
    runs = [p for r in live_roots() for p in r.iterdir() if p.is_dir() and any(p.glob("W0*_*.csv"))]
    if not runs:
        sys.exit("No race run found. Run ./demo.sh race first, or pass a folder.")
    return max(runs, key=lambda p: p.name)


def wsort(w):
    return WORKLOAD_ORDER.index(w) if w in WORKLOAD_ORDER else 99


def analyse_all(data):
    out = {}
    for w in sorted(data, key=wsort):
        rows, ref, chain = racelib.analyse(data[w])
        if rows:
            out[w] = (rows, ref, chain)
    return out


def fmt(x, f="{:.2f}", none="–"):
    return f.format(x) if x is not None else none


def tables(res, title):
    L = [f"# {title}", "",
         "× vs S1 = t(S1) / t(stage). × vs prev = t(previous stage) / t(stage). "
         "Saved = (1 − t/t(S1))·100. %REF = t(REF) / t(stage)·100 (100% = as fast as the reference).", ""]
    flat = []
    for w, (rows, ref, chain) in res.items():
        n = rows[0]["n"]
        L += [f"## {NAMES.get(w, w)}" + (f" — N={n}" if n else ""), "",
              "| Stage | ms | ± SD | rounds | GFLOP/s | × vs S1 | × vs prev | saved | %REF | tile | correct |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            prev = f"{r['x_prev']:.2f}× (vs {r['prev']})" if r["x_prev"] else "–"
            L.append(f"| {r['stage']} | {r['ms']:.4f} | {r['sd']:.4f} | {r['rounds']} | "
                     f"{fmt(r['gflops'], '{:.1f}')} | **{r['x_s1']:.2f}×** | {prev} | {r['saved']:.1f}% | "
                     f"{fmt(r['pct_ref'], '{:.0f}%')} | {r['tile'] or '–'} | {r['ok']} |")
            flat.append((w, r))
        if ref:
            L.append(f"| REF (reference) | {ref['ms']:.4f} | {ref['sd']:.4f} | {ref['rounds']} | "
                     f"{fmt(ref['gflops'], '{:.1f}')} | {rows[0]['ms'] / ref['ms']:.2f}× | – | – | 100% | – | {ref['ok']} |")
        if chain:
            best = rows[0]["ms"] / min(r["ms"] for r in rows)
            L += ["", "Factored: " + " · ".join(f"{a}→{b} **{x:.2f}×**" for a, b, x in chain)
                  + f" = **{best:.1f}×**"]
        L.append("")
    L += ["## Best stage per workload", "", "| Workload | Best stage | ms | × vs S1 | time saved |",
          "|---|---|---|---|---|"]
    for w, (rows, _, _) in res.items():
        b = min(rows, key=lambda r: r["ms"])
        L.append(f"| {NAMES.get(w, w)} | {b['stage']} | {b['ms']:.4f} | **{b['x_s1']:.2f}×** | {b['saved']:.1f}% |")
    cross = []
    for label, (wa, sa), (wb, sb) in CROSS:
        A = next((r for r in res.get(wa, ([],))[0] if r["stage"] == sa), None)
        B = next((r for r in res.get(wb, ([],))[0] if r["stage"] == sb), None)
        if A and B and (not A["n"] or not B["n"] or A["n"] == B["n"]):
            cross.append((label, A, B))
    if cross:
        L += ["", f"## W04 CPU vs GPU (same N={cross[0][1]['n']})", "",
              "| Comparison | CPU ms | GPU ms | GPU × faster | time saved |", "|---|---|---|---|---|"]
        for label, A, B in cross:
            L.append(f"| {label} | {A['ms']:.4f} | {B['ms']:.4f} | **{A['ms'] / B['ms']:.1f}×** | "
                     f"{(1 - B['ms'] / A['ms']) * 100:.1f}% |")
    L += ["", "Live numbers: one session on a laptop GPU held at its 55 W power cap (each stage at its own "
          "clock); compare within one run, not across runs."]
    return "\n".join(L), flat


def write_csv(flat, path):
    cols = ["workload", "stage", "ms", "sd", "rounds", "gflops", "x_vs_s1", "x_vs_prev", "prev",
            "saved_pct", "pct_of_ref", "tile", "correct", "n"]
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(cols)
        for w, r in flat:
            wr.writerow([w, r["stage"], f"{r['ms']:.6f}", f"{r['sd']:.6f}", r["rounds"], fmt(r["gflops"], "{:.2f}", ""),
                         f"{r['x_s1']:.4f}", fmt(r["x_prev"], "{:.4f}", ""), r["prev"] or "",
                         f"{r['saved']:.2f}", fmt(r["pct_ref"], "{:.1f}", ""), r["tile"], r["ok"], r["n"]])


def ms_txt(v):
    return f"{v:.0f}" if v >= 100 else f"{v:.1f}" if v >= 10 else f"{v:.3g}"


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plots(res, stem, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib missing (sudo apt install python3-matplotlib); tables still written")
        return []
    from matplotlib.ticker import FuncFormatter
    plain = FuncFormatter(lambda v, _: f"{v:g}")
    out = []
    # 1) runtime per stage, one panel per workload
    works = list(res)
    ratios = [len(res[w][0]) + 0.6 for w in works]
    fig, axes = plt.subplots(1, len(works), figsize=(2.0 + 1.25 * sum(ratios), 4.8), squeeze=False,
                             gridspec_kw={"width_ratios": ratios})
    fig.patch.set_facecolor(SURFACE)
    for ax, w in zip(axes[0], works):
        rows, ref, _ = res[w]
        st = [r["stage"] for r in rows]
        ms = [r["ms"] for r in rows]
        vals = ms + ([ref["ms"]] if ref else [])
        logy = max(vals) / min(vals) > 20
        style(ax)
        bars = ax.bar(st, ms, width=0.62, color=[COLORS.get(s, "#777") for s in st],
                      edgecolor=SURFACE, linewidth=2)
        for b, r in zip(bars, rows):
            lab = f"{ms_txt(r['ms'])} ms" if r is rows[0] else \
                f"{ms_txt(r['ms'])} ms\n{r['x_s1']:.1f}×\n{-r['saved']:+.0f}%".replace("-", "−")
            ax.annotate(lab, (b.get_x() + b.get_width() / 2, r["ms"]), ha="center", va="bottom",
                        fontsize=8, color=INK, xytext=(0, 3), textcoords="offset points")
        if ref:
            ax.axhline(ref["ms"], color=INK2, linestyle="--", linewidth=1.2,
                       label=f"REF cuBLAS (reference) {ms_txt(ref['ms'])} ms")
            ax.legend(loc="upper right", fontsize=8, frameon=False, labelcolor=INK2)
        if logy:
            ax.set_yscale("log")
            ax.set_ylim(min(vals) / 2, max(vals) * 12)
            ax.yaxis.set_major_formatter(plain)
        else:
            ax.set_ylim(0, max(vals) * 1.45)
        n = rows[0]["n"]
        ax.set_title(NAMES.get(w, w) + (f"\nN={n}" if n else ""), fontsize=10, color=INK)
        ax.set_ylabel("ms per iteration" + (" (log)" if logy else ""), color=INK2, fontsize=9)
        ax.tick_params(axis="x", rotation=30 if len(st) > 3 else 0)
    fig.suptitle(f"{title}: runtime per stage (lower is faster)", color=INK, fontsize=11)
    fig.tight_layout()
    p1 = f"{stem}.png"
    fig.savefig(p1, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    out.append(p1)
    # 2) combined speedup vs own S1
    items = [(w, r) for w in works for r in res[w][0]]
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(items) + 1.4))
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ys = list(range(len(items)))[::-1]
    ax.barh(ys, [r["x_s1"] for _, r in items], height=0.66,
            color=[COLORS.get(r["stage"], "#777") for _, r in items], edgecolor=SURFACE, linewidth=2)
    for y, (w, r) in zip(ys, items):
        ax.annotate(f"{r['x_s1']:.2f}×", (r["x_s1"], y), va="center", ha="left", fontsize=8, color=INK,
                    xytext=(4, 0), textcoords="offset points")
    ax.set_yticks(ys, [f"{w} · {r['stage']}" for w, r in items], fontsize=8.5, color=INK)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.set_xlim(0.5, max(r["x_s1"] for _, r in items) * 3)
    ax.axvline(1, color=INK2, linewidth=0.8)
    ax.set_xlabel("speedup vs the workload's own S1 (log scale; 1× = baseline)", color=INK2, fontsize=9)
    ax.set_title(f"{title}: speedup of every stage", color=INK, fontsize=11)
    fig.tight_layout()
    p2 = stem.replace("race_", "race_speedup_") + ".png"
    fig.savefig(p2, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    out.append(p2)
    return out


def selftest():
    cases = {"W04_S2a.csv": ("W04", "S2a", 1), "W04_S2b.csv": ("W04", "S2b", 1),
             "W04_S3_FP16.csv": ("W04", "S3_FP16", 1), "W04_S3_TF32_r3.csv": ("W04", "S3_TF32", 3),
             "W04CPU_S1.csv": ("W04 CPU", "S1", 1), "W04_REF.csv": ("W04", "REF", 1),
             "W03_S3.csv": ("W03", "S3", 1), "W01_S2_r2.csv": ("W01", "S2", 2),
             "pre_s3.csv": None, "p_w03s1.csv": None, "ncu_s3.csv": None, "W04_S2ab.csv": None}
    bad = [(k, racelib.parse_name(k), v) for k, v in cases.items() if racelib.parse_name(k) != v]
    order = sorted(["S3_TF32", "S2b", "S1", "S3_FP16", "S2a"], key=racelib.stage_sort_key)
    if order != ["S1", "S2a", "S2b", "S3_FP16", "S3_TF32"]:
        bad.append(("order", order, None))
    rows, _, chain = racelib.analyse({s: {"ms": m, "n": "2048"} for s, m in
                                      [("S1", 32.0), ("S2a", 10.0), ("S2b", 3.0), ("S3_FP16", 0.6), ("S3_TF32", 1.2)]})
    prev = {r["stage"]: r["prev"] for r in rows}
    if prev != {"S1": None, "S2a": "S1", "S2b": "S2a", "S3_FP16": "S2b", "S3_TF32": "S2b"}:
        bad.append(("prev", prev, None))
    if [c[:2] for c in chain] != [("S1", "S2a"), ("S2a", "S2b"), ("S2b", "S3_FP16")]:
        bad.append(("chain", chain, None))
    for b in bad:
        print("FAIL", b)
    print("selftest:", "PASS" if not bad else f"{len(bad)} failure(s)")
    return 0 if not bad else 1


def main():
    args = sys.argv[1:]
    if "--selftest" in args:
        return selftest()
    if "--compare" in args:
        runs = sorted({p for r in live_roots() for p in r.iterdir() if p.is_dir() and any(p.glob("W0*_*.csv"))},
                      key=lambda p: p.name)
        per = {r.name: racelib.read_folder(r) for r in runs}
        keys = sorted({(w, s) for d in per.values() for w in d for s in d[w]},
                      key=lambda k: (wsort(k[0]), racelib.stage_sort_key(k[1])))
        print(f"{len(runs)} run(s)\n")
        print("run".ljust(18) + "".join(f"{w.replace(' ', '')}{s}".rjust(14) for w, s in keys))
        for name, d in per.items():
            print(name.ljust(18) + "".join((f"{d[w][s]['ms']:.4f}" if s in d.get(w, {}) else "-").rjust(14)
                                           for w, s in keys))
        return 0
    folder = Path(args[0]).resolve() if args else newest_run()
    res = analyse_all(racelib.read_folder(folder))
    if not res:
        sys.exit(f"No race CSVs (W0<k>_<stage>.csv) in {folder}")
    title = f"Race {folder.name}"
    md, flat = tables(res, title)
    stem = str(folder / f"race_{folder.name}")
    Path(stem + ".md").write_text(md + "\n")
    write_csv(flat, stem + ".csv")
    print(f"  table: {stem}.md\n  csv:   {stem}.csv")
    for p in plots(res, stem, title):
        print(f"  chart: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
