#!/usr/bin/env python3
"""fusion_bytes.py - measured DRAM traffic, unfused S2a vs fused S2b (W04).

    fusion_bytes.py [--n 2048] LABEL=<dir> [LABEL=<dir> ...]

Each <dir> holds ncu raw-page exports (`ncu --import X --page raw --csv`):
  S2a_<kernel>[_h1]_n<N>.raw.csv   kernel = qk, softmax, pv (summed: one forward pass)
  S2b_<variant>_n<N>.raw.csv       one fused kernel per variant (tuned, 64x32w4, 16x16w8, ...)
with dram__bytes_read.sum, dram__bytes_write.sum, lts__t_bytes.sum, gpu__time_duration.sum.
Parsing reuses read_export() from tools/w04_e3_oi.py.

Printed per directory: DRAM read / write MB, L2 MB, OI raw and write-corrected,
and for each S2b variant: × fewer DRAM bytes and % of S2a's bytes removed (raw and
corrected). L2 is write-back: lines still dirty at kernel end are not counted as
DRAM writes, so measured writes are a lower bound. "Corrected" raises each write
to its compulsory minimum (S2a: S + P + O; S2b: O). Shape BH = 8, d = 64.
ncu durations run at base clock (--clock-control base): they are not benchmark times.
"""
import glob
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
try:
    from w04_e3_oi import read_export  # noqa: E402
except ImportError:
    sys.exit(f"tools/w04_e3_oi.py not found under {REPO}")

BH, D = 8, 64
BOLD, DIM, TEAL, ORANGE, RED, RESET = "\033[1m", "\033[2m", "\033[38;5;37m", "\033[38;5;208m", "\033[31m", "\033[0m"
RD, WR, L2, T = "dram__bytes_read.sum", "dram__bytes_write.sum", "lts__t_bytes.sum", "gpu__time_duration.sum"
PCT = "dram__throughput.avg.pct_of_peak_sustained_elapsed"
S2A_RE = re.compile(r"^S2a_(qk|softmax|pv)(?:_h1)?_n(\d+)\.raw\.csv$")
S2B_RE = re.compile(r"^S2b_(\w+?)_n(\d+)\.raw\.csv$")


def load(d, n):
    s2a, kernels, s2b = {RD: 0.0, WR: 0.0, L2: 0.0, T: 0.0}, {}, {}
    for p in sorted(glob.glob(os.path.join(d, "*.raw.csv"))):
        name = os.path.basename(p)
        if (m := S2A_RE.match(name)) and int(m.group(2)) == n:
            if m.group(1) in kernels:
                continue
            e = read_export(p)
            kernels[m.group(1)] = e
            for k in s2a:
                s2a[k] += e.get(k, 0.0)
        elif (m := S2B_RE.match(name)) and int(m.group(2)) == n:
            s2b[m.group(1)] = read_export(p)
    return (s2a if len(kernels) == 3 else None), kernels, s2b


def main():
    args = sys.argv[1:]
    n = 2048
    if "--n" in args:
        i = args.index("--n")
        n = int(args[i + 1])
        del args[i:i + 2]
    dirs = [a.split("=", 1) for a in args if "=" in a]
    if not dirs:
        print(__doc__)
        return 2
    flops = 4.0 * BH * n * n * D
    qkvo = BH * n * D * 4.0
    s_bytes = BH * n * n * 4.0
    comp_s2a_w, comp_s2b_w = 2 * s_bytes + qkvo, qkvo
    ok_any = False
    for label, d in dirs:
        s2a, kernels, s2b = load(d, n)
        print(f"\n{BOLD}{label}{RESET}  {DIM}{d}{RESET}")
        if s2a is None or not s2b:
            print(f"  {RED}incomplete: need S2a qk/softmax/pv and at least one S2b export at N={n}{RESET}")
            continue
        ok_any = True
        print(f"  {DIM}{'kernel(s)':<26} {'DRAM read':>10} {'write':>9} {'L2':>9} {'OI raw':>7} {'OI corr':>8}"
              f" {'ncu ms':>8}   (MB = 1e6 B; FLOP/B){RESET}")
        a_tot = s2a[RD] + s2a[WR]
        a_cor = s2a[RD] + max(s2a[WR], comp_s2a_w)

        def line(name, e, tot, cor, colour=""):
            print(f"  {colour}{name:<26}{RESET} {e[RD] / 1e6:10.1f} {e[WR] / 1e6:9.1f} {e[L2] / 1e6:9.1f}"
                  f" {flops / tot:7.1f} {flops / cor:8.1f} {e.get(T, 0) * 1e3:8.3f}")
        line("S2a unfused (3 kernels)", s2a, a_tot, a_cor, ORANGE)
        for k in ("qk", "softmax", "pv"):
            e = kernels[k]
            pct = f"   DRAM {e[PCT]:.0f}% of peak" if PCT in e else ""
            print(f"  {DIM}  └ {k:<22} {e[RD] / 1e6:10.1f} {e[WR] / 1e6:9.1f} {e[L2] / 1e6:9.1f}"
                  f" {'':>7} {'':>8} {e.get(T, 0) * 1e3:8.3f}{pct}{RESET}")
        for v, e in s2b.items():
            tot = e[RD] + e[WR]
            cor = e[RD] + max(e[WR], comp_s2b_w)
            line(f"S2b fused {v}", e, tot, cor, TEAL)
            print(f"    {BOLD}{a_tot / tot:5.1f}× fewer DRAM bytes{RESET} raw ({(1 - tot / a_tot) * 100:.1f}% removed),"
                  f" {BOLD}{a_cor / cor:5.1f}×{RESET} write-corrected ({(1 - cor / a_cor) * 100:.1f}% removed)")
    print(f"\n{DIM}Compulsory at N={n}: Q+K+V+O = {4 * qkvo / 1e6:.1f} MB; one N×N matrix (S or P) = "
          f"{s_bytes / 1e6:.1f} MB. Writes are lower bounds (dirty L2 lines at kernel end are not counted).\n"
          f"ncu durations are at base clock with caches flushed: not benchmark times.{RESET}")
    return 0 if ok_any else 1


if __name__ == "__main__":
    sys.exit(main())
