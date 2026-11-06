#!/usr/bin/env python3
"""Live race scoreboard: read the CSVs a race just wrote and print speedups.

Usage:
  summarize.py --group "W03 GEMM (GPU)" [--id W03] [--ordered] S1=dir/W03_S1 S2=dir/W03_S2 ...
               --group "W04 Attention"  --id W04 S1=... S2a=... S2b=... S3_FP16=... REF=...
               --pair "CPU S3 AVX2 -> GPU S3 FP16" W04C:S3 W04:S3_FP16

A path may be a CSV file or a base name without .csv: base.csv, or the round
files base_r1.csv, base_r2.csv ... (median over rounds of each round's mean).
--ordered keeps the stages in the order given (default: S1 < S2a < S2b < S3 ...).
REF is a vendor reference: shown with a dashed bar, not counted as a stage.

Columns per stage:
  ms ± SD      mean runtime per iteration (± mean rep SD); with rounds: median over rounds
  rate         GFLOP/s (compute workloads) or GB/s (memory workloads)
  ×S1          speedup vs the first stage of the workload
  ×prev        speedup vs the previous stage (S3_TF32's previous stage is S2b, not S3_FP16)
  saved        time saved vs S1, in % = (1 - t / t_S1) * 100
  %REF         speed as a % of the reference (100% = as fast as REF)
After each workload: the factored chain to its fastest stage. At the end: one
line per workload (best stage) and each --pair (how many × faster B is than A).
Missing files are shown as MISSING; the demo never crashes on them.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import racelib  # noqa: E402

TEAL, ORANGE, GREY, BOLD, DIM, RED, RESET = (
    "\033[38;5;37m", "\033[38;5;208m", "\033[38;5;245m",
    "\033[1m", "\033[2m", "\033[31m", "\033[0m")


def bar(frac, width, colour, dashed=False):
    frac = max(0.0, min(1.0, frac))
    full = max(int(frac * width), 1)
    ch = "╌" if dashed else "█"
    return colour + ch * full + RESET + GREY + "·" * (width - full) + RESET


def parse(argv):
    groups, pairs, cur = [], [], None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--group":
            cur = {"title": argv[i + 1], "id": None, "stages": [], "ordered": False}
            groups.append(cur)
            i += 2
        elif a == "--id" and cur is not None:
            cur["id"] = argv[i + 1]
            i += 2
        elif a == "--ordered" and cur is not None:
            cur["ordered"] = True
            i += 1
        elif a == "--pair":
            pairs.append(tuple(argv[i + 1:i + 4]))
            i += 4
        elif "=" in a and cur is not None:
            k, v = a.split("=", 1)
            cur["stages"].append((k, v))
            i += 1
        else:
            i += 1
    return groups, pairs


def fmt_rate(d, gid=""):
    """GB/s for the memory-bound workloads (W01, W02), GFLOP/s for the others."""
    memory_bound = (gid or "").startswith(("W01", "W02"))
    if memory_bound and d.get("bw"):
        return f"{d['bw']:9.1f} GB/s   "
    if d.get("gflops"):
        return f"{d['gflops']:9.1f} GFLOP/s"
    if d.get("bw"):
        return f"{d['bw']:9.1f} GB/s   "
    return " " * 17


def pct(x):
    """99.95% stays 99.95%, not a rounded 100.0%."""
    return f"{x:.2f}%" if x >= 99 else f"{x:.1f}%"


def main():
    groups, pairs = parse(sys.argv[1:])
    if not groups:
        print(__doc__)
        return 1
    width = 28
    results, best_lines = {}, []
    for g in groups:
        print(f"\n{BOLD}{g['title']}{RESET}")
        stages, missing = {}, []
        for k, path in g["stages"]:
            d = racelib.read_stage(path)
            if d is None:
                missing.append((k, path))
            else:
                stages[k] = d
        rows, ref, chain = racelib.analyse(stages, [k for k, _ in g["stages"]] if g["ordered"] else None)
        results[g["id"] or g["title"]] = {r["stage"]: r for r in rows}
        fastest = min([r["ms"] for r in rows] + ([ref["ms"]] if ref else []), default=None)
        any_ref = ref is not None
        hdr = (f"  {'stage':<10} {'':<{width}} {'ms':>9}  {'±SD':<7} {'rate':>17}  {'×S1':>7} "
               f"{'×prev':>6} {'saved':>6}" + (f" {'%REF':>6}" if any_ref else "") + "  tile / ok")
        print(DIM + hdr + RESET)
        for r in rows:
            colour = GREY if r is rows[0] else (ORANGE if r["ms"] == min(x["ms"] for x in rows) else TEAL)
            ok = r["ok"] if r["ok"] in ("YES", "n/a") else f"{RED}{r['ok']}{RESET}"
            xp = f"{r['x_prev']:5.2f}×" if r["x_prev"] else f"{'–':>6}"
            pr = f" {r['pct_ref']:5.0f}%" if any_ref else ""
            spread = f" rounds={r['rounds']} spread={r['spread']:.1f}%" if r["rounds"] > 1 else ""
            print(f"  {r['stage']:<10} {bar(fastest / r['ms'], width, colour)} {r['ms']:9.3f}  ±{r['sd']:<6.3f}"
                  f" {fmt_rate(r, g['id'])}  {BOLD}{r['x_s1']:6.2f}×{RESET} {xp} {r['saved']:5.1f}%{pr}"
                  f"  {r['tile'] or '-':<9} ok={ok}{DIM}{spread}{RESET}")
        if ref:
            x = rows[0]["ms"] / ref["ms"] if rows else 0
            print(f"  {GREY}{'REF':<10} {bar(fastest / ref['ms'], width, GREY, dashed=True)} {ref['ms']:9.3f}  "
                  f"±{ref['sd']:<6.3f} {fmt_rate(ref, g['id'])}  {x:6.2f}×  (reference, not a stage){RESET}")
        for k, path in missing:
            print(f"  {k:<10} {RED}MISSING{RESET} {DIM}{path}{RESET}")
        if chain:
            parts = " · ".join(f"{a}→{b} {x:.2f}×" for a, b, x in chain)
            best = rows[0]["ms"] / min(r["ms"] for r in rows)
            print(f"  {DIM}factored:{RESET} {parts} {BOLD}= {best:.1f}×{RESET}")
        if rows:
            b = min(rows, key=lambda r: r["ms"])
            best_lines.append((" · ".join(g["title"].split(" · ")[:2]), b))
    if len(best_lines) > 1:
        print(f"\n{BOLD}Best stage per workload{RESET}")
        for title, b in best_lines:
            print(f"  {title[:46]:<46} {b['stage']:<8} {b['ms']:9.3f} ms  {BOLD}{b['x_s1']:7.2f}×{RESET} vs S1"
                  f"  ({pct(b['saved'])} less time)")
    if pairs:
        print(f"\n{BOLD}Cross comparisons (same N){RESET}")
        for label, a, b in pairs:
            (ga, sa), (gb, sb) = a.split(":", 1), b.split(":", 1)
            da, db = results.get(ga, {}).get(sa), results.get(gb, {}).get(sb)
            if not (da and db):
                print(f"  {label:<46} {RED}n/a{RESET} (run both workloads)")
                continue
            if da["n"] and db["n"] and da["n"] != db["n"]:
                print(f"  {label:<46} {RED}different N ({da['n']} vs {db['n']}){RESET}")
                continue
            x = da["ms"] / db["ms"]
            print(f"  {label:<46} {da['ms']:9.3f} → {db['ms']:8.3f} ms  {BOLD}{x:7.2f}×{RESET}"
                  f"  ({pct((1 - db['ms'] / da['ms']) * 100)} less time)")
    print(f"\n{DIM}Bars: speed relative to the fastest row (longer = faster). ×S1 / saved are vs the "
          f"workload's S1; ×prev vs the previous stage.{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
