#!/usr/bin/env python3
"""W04 E3: measured DRAM bytes and operational intensity per stage.

    python3 tools/w04_e3_oi.py <e3_dir> [<e3_dir> ...]

Each <e3_dir> holds ncu raw-page exports named <STAGE>_<kernel>_n<N>.raw.csv
(S2a: one file per kernel, summed per stage) or <STAGE>_<BrxBc>_n<N>.raw.csv
(S2b), each with dram__bytes_read.sum, dram__bytes_write.sum and
lts__t_bytes.sum. Shape is BH = 8, d = 64 (the E3 commands).

Printed per (stage, N) and per directory:
  read, write    measured DRAM bytes (MB = 1e6 B, ncu units)
  L2             lts__t_bytes.sum
  OI raw         FLOPs / (read + write)
  OI corr        FLOPs / (read + max(write, compulsory write)), where the
                 compulsory write is |O| for a fused stage and |S| + |P| + |O|
                 for S2a. L2 is write-back: lines still dirty when the kernel
                 ends are not counted as DRAM writes by that kernel.
FLOPs = 4·BH·N²·d (the two GEMMs; softmax excluded, as in the gflops column).

Standard library only.
"""
import csv
import glob
import os
import re
import sys
from collections import defaultdict

BH, D = 8, 64
UNIT = {"byte": 1.0, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "Tbyte": 1e12}
METRICS = ("dram__bytes_read.sum", "dram__bytes_write.sum", "lts__t_bytes.sum")


def read_export(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    head, units, vals = rows[0], rows[1], rows[2:]
    if len(vals) != 1:
        raise SystemExit(f"{path}: expected 1 kernel row, found {len(vals)}")
    out = {}
    for m in METRICS:
        i = head.index(m)
        out[m] = float(vals[0][i].replace(",", "")) * UNIT[units[i]]
    return out


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    for d in sys.argv[1:]:
        # (stage, N) -> summed metrics
        acc = defaultdict(lambda: dict.fromkeys(METRICS, 0.0))
        kernels = defaultdict(int)
        for path in sorted(glob.glob(os.path.join(d, "*.raw.csv"))):
            m = re.match(r"(S2a|S2b)_(\w+?)_n(\d+)\.raw\.csv$", os.path.basename(path))
            if not m:
                continue
            stage = "S2a" if m.group(1) == "S2a" else f"S2b_{m.group(2)}"
            key = (stage, int(m.group(3)))
            for k, v in read_export(path).items():
                acc[key][k] += v
            kernels[key] += 1
        print(f"\n== {d} ==")
        print(f"{'stage':<10} {'N':>5} {'kern':>4} {'read MB':>9} {'write MB':>9} {'L2 MB':>9} "
              f"{'OI raw':>7} {'OI corr':>7} {'OI min':>7}")
        for (stage, n) in sorted(acc, key=lambda k: (k[1], k[0])):
            a = acc[(stage, n)]
            flops = 4.0 * BH * n * n * D
            o_bytes = BH * n * D * 4.0
            s_bytes = BH * n * n * 4.0
            comp_w = o_bytes if stage.startswith("S2b") else 2 * s_bytes + o_bytes
            rd, wr = a["dram__bytes_read.sum"], a["dram__bytes_write.sum"]
            oi_raw = flops / (rd + wr)
            oi_corr = flops / (rd + max(wr, comp_w))
            oi_min = flops / (4 * o_bytes)  # Q, K, V, O once: N/4
            print(f"{stage:<10} {n:>5} {kernels[(stage, n)]:>4} {rd / 1e6:>9.2f} {wr / 1e6:>9.2f} "
                  f"{a['lts__t_bytes.sum'] / 1e6:>9.1f} {oi_raw:>7.1f} {oi_corr:>7.1f} {oi_min:>7.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
