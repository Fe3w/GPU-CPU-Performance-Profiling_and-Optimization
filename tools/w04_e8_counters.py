#!/usr/bin/env python3
"""W04 E8: CPU counters per forward pass (perf stat, differential) and exp share (VTune).

    python3 tools/w04_e8_counters.py <e8_dir>

Inputs written by `scripts/w04_profile.sh e8`:
  <tag>_n<N>_p<P>_g<G>.perf.csv   `perf stat -x,` output, event group G, P timed passes
  vtune_<tag>_n<N>.hotspots.csv   `vtune -report hotspots -group-by function` (CSV)
  csv/<tag>.csv                   the binaries' own result rows (runtime, GFLOP/s)

Differential counting: for each (tag, N, group) there are two runs with P_A < P_B
timed passes and otherwise the same work (setup, double-reference check), so
    per pass = (count(P_B) - count(P_A)) / (P_B - P_A)
removes everything that is not the timed forward pass. Both runs use
--warmup 0 --reps 1, so a run makes exactly P timed passes.

Printed per (tag, N):
  GFLOP/s      median of the binary's own rows of the runs (all P)
  P-core       instructions, L2 hit/miss loads, L3 miss loads per pass;
               L2 miss % = l2_miss / (l2_hit + l2_miss) of retired loads that missed L1;
               L2 MPKI = L2-missing loads per 1000 instructions
  E-core       instructions and L2 miss % (mem_load_uops_retired)
  all cores    L2-missing loads of P + E cores per pass and per 1000 instructions
               (the OpenMP split of work between P- and E-cores can differ between
               runs; the sum does not depend on it)
  FP (P-core)  FP arithmetic instructions by width: scalar, 128-bit, 256-bit (% of all)
  exp share    VTune effective CPU time (spin excluded) of float exp functions (libm expf, libmvec) as % of the
               stage's CPU time; the double reference (attention_reference, double exp)
               is excluded. Inlined exp (S3 exp256_ps) is not a separate function, so it
               is reported as "inlined".

Standard library only. Reads CSVs; never writes.
"""
import csv
import glob
import os
import re
import statistics
import sys
from collections import defaultdict

REF_FUNCS = re.compile(r"attention_reference|__ieee754_exp\b|__ieee754_exp_|__GI___exp\b|__exp1|^__exp\b|^exp$")
EXPF_FUNCS = re.compile(r"expf|_ZGV\w*expf")


def read_perf(path):
    out = {}
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            p = line.rstrip("\n").split(",")
            if len(p) < 3:
                continue
            try:
                out[p[2]] = float(p[0])
            except ValueError:
                out[p[2]] = float("nan")   # <not counted> / <not supported>
    return out


def ev(d, name):
    for k, v in d.items():
        if k.rstrip("/") == name:
            return v
    return float("nan")


def gflops(e8_dir, tag, n):
    path = os.path.join(e8_dir, "csv", f"{tag}.csv")
    if not os.path.exists(path):
        return float("nan")
    with open(path, newline="") as f:
        vals = [float(r["gflops"]) for r in csv.DictReader(f) if int(r["n"]) == n]
    return statistics.median(vals) if vals else float("nan")


def exp_share(e8_dir, tag, n):
    path = os.path.join(e8_dir, f"vtune_{tag}_n{n}.hotspots.csv")
    if not os.path.exists(path):
        return None
    stage = expf = 0.0
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            fn = r.get("Function", "")
            t = r.get("CPU Time:Effective Time", r.get("CPU Time", ""))   # without OpenMP spin
            try:
                t = float(t)
            except ValueError:
                continue
            if REF_FUNCS.search(fn):
                continue
            stage += t
            if EXPF_FUNCS.search(fn):
                expf += t
    return (100.0 * expf / stage if stage else float("nan")), stage


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    d = sys.argv[1]
    runs = defaultdict(dict)   # (tag, n, g) -> {P: counts}
    for path in glob.glob(os.path.join(d, "*.perf.csv")):
        m = re.match(r"(.+)_n(\d+)_p(\d+)_g(\d)\.perf\.csv$", os.path.basename(path))
        if m:
            runs[(m.group(1), int(m.group(2)), int(m.group(4)))][int(m.group(3))] = read_perf(path)
    per_pass = {}
    for key, byp in runs.items():
        if len(byp) != 2:
            print(f"{key}: need exactly 2 pass counts, found {sorted(byp)}")
            continue
        pa, pb = sorted(byp)
        a, b = byp[pa], byp[pb]
        per_pass[key] = {k: (b[k] - a.get(k, float("nan"))) / (pb - pa) for k in b}
    keys = sorted({(t, n) for t, n, _ in per_pass}, key=lambda k: (k[0], k[1]))
    print("E8 CPU counters per forward pass (differential perf stat; P = P-core, E = E-core)")
    hdr = (f"{'stage':<9} {'N':>5} {'GFLOP/s':>8} {'P instr':>10} {'P L2 hit':>10} {'P L2 miss':>10} "
           f"{'L2 miss%':>8} {'L2 MPKI':>8} {'P L3 miss':>10} {'E instr':>10} {'E L2 miss%':>10} "
           f"{'all L2 miss':>11} {'all MPKI':>8}")
    print(hdr)
    for tag, n in keys:
        g1 = per_pass.get((tag, n, 1), {})
        pi = ev(g1, "cpu_core/instructions")
        h, mi, l3 = (ev(g1, "cpu_core/mem_load_retired.l2_hit"), ev(g1, "cpu_core/mem_load_retired.l2_miss"),
                     ev(g1, "cpu_core/mem_load_retired.l3_miss"))
        ei = ev(g1, "cpu_atom/instructions")
        eh, em = ev(g1, "cpu_atom/mem_load_uops_retired.l2_hit"), ev(g1, "cpu_atom/mem_load_uops_retired.l2_miss")
        print(f"{tag:<9} {n:>5} {gflops(d, tag, n):>8.1f} {pi:>10.3g} {h:>10.3g} {mi:>10.3g} "
              f"{100 * mi / (h + mi):>8.2f} {1000 * mi / pi:>8.3f} {l3:>10.3g} {ei:>10.3g} {100 * em / (eh + em):>10.2f} "
              f"{mi + em:>11.3g} {1000 * (mi + em) / (pi + ei):>8.3f}")
    print("\nFP arithmetic instructions on P-cores per pass (share of all FP arith instructions)")
    print(f"{'stage':<9} {'N':>5} {'scalar':>10} {'128-bit':>10} {'256-bit':>10} {'scalar%':>8} {'128b%':>7} "
          f"{'256b%':>7} {'exp share':>10}")
    for tag, n in keys:
        g2 = per_pass.get((tag, n, 2), {})
        s = ev(g2, "cpu_core/fp_arith_inst_retired.scalar_single")
        x = ev(g2, "cpu_core/fp_arith_inst_retired.128b_packed_single")
        y = ev(g2, "cpu_core/fp_arith_inst_retired.256b_packed_single")
        tot = s + x + y
        es = exp_share(d, tag, n)
        if es is None:
            est = "n/a"
        elif tag.startswith("S3"):
            est = f"inlined ({es[0]:.1f})"
        else:
            est = f"{es[0]:.1f} %"
        print(f"{tag:<9} {n:>5} {s:>10.3g} {x:>10.3g} {y:>10.3g} {100 * s / tot:>8.1f} {100 * x / tot:>7.1f} "
              f"{100 * y / tot:>7.1f} {est:>10}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
