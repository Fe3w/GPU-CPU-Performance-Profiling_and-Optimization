#!/usr/bin/env python3
"""W04 E7 heatmaps: screen GFLOP/s per Br x Bc, one panel per warp count.

    python3 tools/w04_e7_heatmap.py <run_dir> [--out docs/figures/w04]

One PNG per stage and N (e7_<stage>_n<N>.png). Rows Br, columns Bc, panels
W = 2, 4, 8. Cell colour and label = screen median GFLOP/s (model FLOPs
4·BH·N²·d); one colour scale per figure, so the panels compare directly.
Finalists have a dark outline, the selected configuration (same rule as
tools/w04_e7_select.py report) a thick outline and "sel". Skipped cells are
grey and hatched with a short reason code: "W>Br/16" (S3 FA2 split), "smem"
(shared memory above the opt-in limit), "other". The table view of the same
numbers is `tools/w04_e7_select.py report <run_dir>`.

Needs matplotlib (no other dependencies).
"""
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import w04_e7_select as sel  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

TILES = [16, 32, 64]
WARPS = [2, 4, 8]
# Sequential single-hue ramp, light (low) -> dark (high).
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, MUTED, SKIP_FILL, SURFACE = "#1f1f1e", "#6b6a66", "#e4e3df", "#ffffff"


def reason_code(text):
    if "query rows per warp" in text:
        return "W>Br/16"
    if "shared memory" in text:
        return "smem"
    return "other"


def luminance(hex_color):
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    run = argv[1]
    out_dir = argv[argv.index("--out") + 1] if "--out" in argv else os.path.join("docs", "figures", "w04")
    os.makedirs(out_dir, exist_ok=True)
    per_run, meta = sel.load_runs(run)
    sids = sorted({k[1] for k in per_run})
    ns = sorted({k[2] for k in per_run})
    skipped = defaultdict(dict)
    sp = os.path.join(run, sel.SKIPPED_CSV)
    if os.path.exists(sp):
        with open(sp) as f:
            for r in csv.DictReader(f):
                skipped[(r["stage_id"], int(r["n"]))][(int(r["br"]), int(r["bc"]), int(r["warps"]))] = r["reason"]
    cmap = LinearSegmentedColormap.from_list("seq_blue", RAMP)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
    written = []
    for sid in sids:
        choice = sel.choose(per_run, sid, ns)
        for n in ns:
            flops = next(m[1] for k, m in meta.items() if k[1] == sid and k[2] == n)
            gf = {cfg: flops / ms / 1e6 for ms, cfg in sel.screen_table(per_run, sid, n)}
            if not gf:
                continue
            finalists = {x[3] for x in sel.final_table(per_run, sid, n)}
            chosen = choice.get(n, (None,))[0]
            lo, hi = min(gf.values()), max(gf.values())
            fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), constrained_layout=True)
            fig.patch.set_facecolor(SURFACE)
            for ax, w in zip(axes, WARPS):
                ax.set_facecolor(SURFACE)
                for i, br in enumerate(TILES):
                    for j, bc in enumerate(TILES):
                        cfg = (br, bc, w)
                        x, y = j, i
                        if cfg in gf:
                            t = (gf[cfg] - lo) / (hi - lo) if hi > lo else 1.0
                            color = matplotlib.colors.to_hex(cmap(t))
                            ax.add_patch(plt.Rectangle((x + 0.03, y + 0.03), 0.94, 0.94, facecolor=color,
                                                       edgecolor=SURFACE, linewidth=2))
                            ink = SURFACE if luminance(color) < 0.25 else INK
                            ax.text(x + 0.5, y + 0.5, f"{gf[cfg]:.0f}", ha="center", va="center",
                                    color=ink, fontsize=10, fontweight="bold" if cfg == chosen else "normal")
                            if cfg in finalists:
                                lw = 3.0 if cfg == chosen else 1.4
                                ax.add_patch(plt.Rectangle((x + 0.06, y + 0.06), 0.88, 0.88, fill=False,
                                                           edgecolor=INK, linewidth=lw))
                            if cfg == chosen:
                                ax.text(x + 0.5, y + 0.8, "sel", ha="center", va="center", color=ink, fontsize=8)
                        else:
                            why = skipped[(sid, n)].get(cfg, "not run")
                            ax.add_patch(plt.Rectangle((x + 0.03, y + 0.03), 0.94, 0.94, facecolor=SKIP_FILL,
                                                       edgecolor=SURFACE, linewidth=2, hatch="//"))
                            ax.text(x + 0.5, y + 0.5, reason_code(why), ha="center", va="center",
                                    color=MUTED, fontsize=8)
                ax.set_xlim(0, 3)
                ax.set_ylim(3, 0)
                ax.set_xticks([0.5, 1.5, 2.5], [str(t) for t in TILES])
                ax.set_yticks([0.5, 1.5, 2.5], [str(t) for t in TILES])
                ax.set_xlabel("Bc (keys per tile)")
                if w == WARPS[0]:
                    ax.set_ylabel("Br (queries per block)")
                ax.set_title(f"{w} warps per block", fontsize=10)
                ax.tick_params(length=0)
                for s in ax.spines.values():
                    s.set_visible(False)
                ax.set_aspect("equal")
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(lo, hi))
            cb = fig.colorbar(sm, ax=axes, shrink=0.8, pad=0.02)
            cb.set_label("screen median GFLOP/s")
            cb.outline.set_visible(False)
            short = sid.replace("W04_ATTENTION_GPU_", "")
            sel_txt = f"selected {sel.cfg_str(chosen)}" if chosen else "no finals"
            fig.suptitle(f"E7 autotune  {short}  N={n}, BH=8, d=64  ({sel_txt}; outlined = finalists)",
                         fontsize=10, color=INK)
            path = os.path.join(out_dir, f"e7_{short}_n{n}.png")
            fig.savefig(path, dpi=130, facecolor=SURFACE)
            plt.close(fig)
            written.append(path)
    for p in written:
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
