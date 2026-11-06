#!/usr/bin/env python3
"""W04 E3: measured DRAM bytes and operational intensity per stage.

    python3 tools/w04_e3_oi.py <e3_dir> [<e3_dir> ...]

Each <e3_dir> holds ncu raw-page exports named <STAGE>_<part>_n<N>.raw.csv,
each with dram__bytes_read.sum, dram__bytes_write.sum and lts__t_bytes.sum
(and, from P6, gpu__time_duration.sum). Shape is BH = 8, d = 64.
  unfused stages S1, S2a, REF: <part> is the kernel (qk, softmax, pv; P6 H1
      runs add "_h1"); the kernels are summed per stage
  fused stages S2b, S3FP16, S3TF32: <part> is the variant (64x64, 64x32w4,
      tuned), one kernel per file

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
OPTIONAL = ("gpu__time_duration.sum", "dram__throughput.avg.pct_of_peak_sustained_elapsed",
            "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum", "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
            "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum", "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum")
TIME_UNIT = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0, "nsecond": 1e-9, "usecond": 1e-6, "msecond": 1e-3, "second": 1.0}
UNFUSED = ("S1", "S2a", "REF")
NAME = re.compile(r"(S1|S2a|REF|S2b|S3FP16|S3TF32)_(\w+?)_n(\d+)\.raw\.csv$")


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
    for m in OPTIONAL:
        if m in head:
            i = head.index(m)
            v = float(vals[0][i].replace(",", ""))
            out[m] = v * TIME_UNIT[units[i]] if m.startswith("gpu__time") else v
    return out


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    for d in sys.argv[1:]:
        # (stage, N) -> summed metrics
        acc = defaultdict(lambda: dict.fromkeys(METRICS + ("gpu__time_duration.sum",), 0.0))
        kernels = defaultdict(int)
        per_kernel = []   # (stage, kernel, N, metrics) for the H1 table
        for path in sorted(glob.glob(os.path.join(d, "*.raw.csv"))):
            m = NAME.match(os.path.basename(path))
            if not m:
                continue
            unfused = m.group(1) in UNFUSED
            stage = m.group(1) if unfused else f"{m.group(1)}_{m.group(2)}"
            key = (stage, int(m.group(3)))
            met = read_export(path)
            for k in acc[key]:
                acc[key][k] += met.get(k, 0.0)
            kernels[key] += 1
            if unfused:
                per_kernel.append((m.group(1), m.group(2).replace("_h1", ""), int(m.group(3)), met))
        print(f"\n== {d} ==")
        print(f"{'stage':<10} {'N':>5} {'kern':>4} {'read MB':>9} {'write MB':>9} {'L2 MB':>9} "
              f"{'OI raw':>7} {'OI corr':>7} {'OI min':>7} {'ncu ms':>8}")
        for (stage, n) in sorted(acc, key=lambda k: (k[1], k[0])):
            a = acc[(stage, n)]
            flops = 4.0 * BH * n * n * D
            o_bytes = BH * n * D * 4.0
            s_bytes = BH * n * n * 4.0
            comp_w = 2 * s_bytes + o_bytes if stage in UNFUSED else o_bytes
            rd, wr = a["dram__bytes_read.sum"], a["dram__bytes_write.sum"]
            oi_raw = flops / (rd + wr)
            oi_corr = flops / (rd + max(wr, comp_w))
            oi_min = flops / (4 * o_bytes)  # Q, K, V, O once: N/4
            print(f"{stage:<10} {n:>5} {kernels[(stage, n)]:>4} {rd / 1e6:>9.2f} {wr / 1e6:>9.2f} "
                  f"{a['lts__t_bytes.sum'] / 1e6:>9.1f} {oi_raw:>7.1f} {oi_corr:>7.1f} {oi_min:>7.1f} "
                  f"{a['gpu__time_duration.sum'] * 1e3:>8.3f}")
        if any("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum" in k[3] for k in per_kernel):
            print("\nH1 per kernel (ncu at base clock; share = kernel time / stage time; "
                  "sectors per request: 4 = fully coalesced 32-bit warp access)")
            print(f"{'stage':<5} {'kernel':<8} {'N':>5} {'ncu ms':>8} {'share %':>8} {'DRAM %pk':>8} "
                  f"{'DRAM MB':>9} {'ld sec/req':>10} {'st sec/req':>10}")
            tot = defaultdict(float)
            for st, kn, n, met in per_kernel:
                tot[(st, n)] += met.get("gpu__time_duration.sum", 0.0)
            for st, kn, n, met in sorted(per_kernel, key=lambda k: (k[0], k[2], k[1])):
                if "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum" not in met:
                    continue
                t = met.get("gpu__time_duration.sum", float("nan"))
                rq = met.get("l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum", 0.0)
                sq = met.get("l1tex__t_requests_pipe_lsu_mem_global_op_st.sum", 0.0)
                ld = met["l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"] / rq if rq else float("nan")
                stv = met.get("l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum", 0.0) / sq if sq else float("nan")
                print(f"{st:<5} {kn:<8} {n:>5} {t * 1e3:>8.3f} {100 * t / tot[(st, n)]:>8.1f} "
                      f"{met.get('dram__throughput.avg.pct_of_peak_sustained_elapsed', float('nan')):>8.1f} "
                      f"{(met['dram__bytes_read.sum'] + met['dram__bytes_write.sum']) / 1e6:>9.1f} {ld:>10.2f} {stv:>10.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
