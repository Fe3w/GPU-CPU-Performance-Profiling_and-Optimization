#!/usr/bin/env python3
"""Side-by-side Nsight Compute metrics for several saved reports.

Usage: ncu_compare.py /path/to/ncu LABEL=report.ncu-rep [LABEL=report.ncu-rep ...]

Calls `ncu --import <rep> --csv --page raw` for each report and prints the few
metrics the talk relies on. Metrics a report did not collect show as "-".
"""
import csv
import io
import subprocess
import sys

METRICS = [
    ("Kernel", "Kernel Name", None),
    ("Duration (us)", "gpu__time_duration.sum", "us"),
    ("SM throughput %", "sm__throughput.avg.pct_of_peak_sustained_elapsed", None),
    ("Memory throughput %", "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", None),
    ("DRAM throughput %", "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", None),
    ("Achieved occupancy %", "sm__warps_active.avg.pct_of_peak_sustained_active", None),
    ("Registers / thread", "launch__registers_per_thread", None),
    ("Tensor pipe active %", "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active", None),
    ("Tensor (HMMA) instr", "sm__inst_executed_pipe_tensor_op_hmma.sum", None),
]
# Older/newer ncu builds name DRAM throughput differently; try both.
ALIASES = {"gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed":
           ["dram__throughput.avg.pct_of_peak_sustained_elapsed"]}


def load(ncu, rep):
    out = subprocess.run([ncu, "--import", rep, "--csv", "--page", "raw"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        return {"_error": out.stderr.strip().splitlines()[-1] if out.stderr else "import failed"}
    lines = [l for l in out.stdout.splitlines() if l.startswith('"')]
    rows = list(csv.reader(io.StringIO("\n".join(lines))))
    if len(rows) < 3:
        return {"_error": "no kernel rows"}
    header, units, first = rows[0], rows[1], rows[2]
    d = dict(zip(header, first))
    d["_units"] = dict(zip(header, units))
    return d


def value(d, key, unit):
    for k in [key] + ALIASES.get(key, []):
        if k in d and d[k] not in ("", "n/a"):
            v = d[k]
            u = d["_units"].get(k, "")
            try:
                f = float(v.replace(",", ""))
                if unit == "us":
                    f = f / 1000 if u in ("nsecond", "ns") else f * 1000 if u in ("msecond", "ms") else f
                return f"{f:,.2f}" if f < 1e6 else f"{f:,.0f}"
            except ValueError:
                return v[:28]
    return "-"


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    ncu, pairs = sys.argv[1], [a.split("=", 1) for a in sys.argv[2:]]
    data = [(label, load(ncu, rep)) for label, rep in pairs]
    w = max(len(m[0]) for m in METRICS) + 2
    cw = 30
    print("\n" + "Metric".ljust(w) + "".join(l.ljust(cw) for l, _ in data))
    print("-" * (w + cw * len(data)))
    for name, key, unit in METRICS:
        cells = []
        for _, d in data:
            cells.append(("ERR: " + d["_error"])[:cw - 2] if "_error" in d else value(d, key, unit))
        print(name.ljust(w) + "".join(c.ljust(cw) for c in cells))
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
