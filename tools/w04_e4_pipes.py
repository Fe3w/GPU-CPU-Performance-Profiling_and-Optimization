#!/usr/bin/env python3
"""E4 kernel diagnosis: pipe utilization and warp stall breakdown from ncu raw-page CSVs.

    python3 tools/w04_e4_pipes.py <folder with *.raw.csv exports>

Reads every `<name>.raw.csv` (from `ncu --import X.ncu-repz --page raw --csv`)
in the folder. A ComputeWorkloadAnalysis export gives the per-pipe utilization
(% of peak, active cycles); a WarpStateStats export gives the stall reasons in
cycles per issued instruction and their share of the total. Exports with
neither are skipped.

P6 additions:
- WarpStateStats exports that also carry the Occupancy / LaunchStats sections
  print theoretical and achieved occupancy, eligible and active warps per
  scheduler, registers and shared memory per block.
- `<name>_ld.raw.csv` exports (instruction-mix metrics) print warp-level
  shared-memory load instructions (LDS + LDSM), thread-level FFMA and their
  ratio: FFMA per shared load = 32 * FFMA_thread / (32 * shared_ld_warp)
  = FFMA_thread / (32 * shared_ld_warp) per thread-load.

Issue-rate check printed for WarpStateStats: with W warps per scheduler and
L cycles between two issues of one warp, a scheduler issues at most W / L
instructions per cycle, which is compared with the measured issue-slot use
from ComputeWorkloadAnalysis of the same stage (if present).

Standard library only. Reads CSVs; never writes.
"""
import csv
import glob
import os
import sys

PIPES = [  # (label, metric)
    ("tensor (HMMA) cycles active", "sm__pipe_tensor_op_hmma_cycles_active_v2.avg.pct_of_peak_sustained_active"),
    ("LSU (shared/global ld/st)", "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active"),
    ("ALU", "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active"),
    ("FMA", "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active"),
    ("XU (MUFU: exp, rcp)", "sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active"),
    ("issue slots busy", "sm__instruction_throughput.avg.pct_of_peak_sustained_elapsed"),
]
IPC = "sm__inst_executed.avg.per_cycle_active"
LAT = "smsp__average_warp_latency_per_inst_issued.ratio"
STALL_PREFIX = "smsp__average_warps_issue_stalled_"
STALL_SUFFIX = "_per_issue_active.ratio"
OCC = [  # (label, metric, format)
    ("theoretical occupancy %", "sm__maximum_warps_per_active_cycle_pct", "{:.1f}"),
    ("achieved occupancy %", "sm__warps_active.avg.pct_of_peak_sustained_active", "{:.1f}"),
    ("active warps / scheduler", "smsp__warps_active.avg.per_cycle_active", "{:.2f}"),
    ("eligible warps / scheduler", "smsp__warps_eligible.avg.per_cycle_active", "{:.2f}"),
    ("registers / thread", "launch__registers_per_thread", "{:.0f}"),
    ("smem / block (B, +1 KB reserved)", "launch__shared_mem_per_block", "{:.0f}"),
    ("blocks / SM (smem limit)", "launch__occupancy_limit_shared_mem", "{:.0f}"),
    ("blocks / SM (reg limit)", "launch__occupancy_limit_registers", "{:.0f}"),
]
LD_SHARED = "smsp__inst_executed_op_shared_ld.sum"
LD_LDSM = "smsp__inst_executed_op_ldsm.sum"
FFMA = "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum"
INST = "smsp__inst_executed.sum"


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) < 3:
        return {}
    scale = {"byte": 1.0, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9}   # also "Kbyte/block" etc.
    out = {}
    for h, u, v in zip(rows[0], rows[1], rows[2]):
        try:
            out[h] = float(v.replace(",", "")) * scale.get(u.split("/")[0], 1.0)
        except ValueError:
            pass
    return out


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    files = sorted(glob.glob(os.path.join(sys.argv[1], "*.raw.csv")))
    if not files:
        print(f"no *.raw.csv in {sys.argv[1]}")
        return 1
    issue_busy = {}
    for path in files:
        name = os.path.basename(path)[: -len(".raw.csv")]
        m = load(path)
        dur = m.get("gpu__time_duration.sum")
        if PIPES[0][1] in m:
            print(f"\n== {name}: pipe utilization (% of peak; profiler duration {dur:.3f} ms, not a benchmark time)")
            for label, key in PIPES:
                if key in m:
                    print(f"  {label:<30} {m[key]:6.1f} %")
            if IPC in m:
                print(f"  {'IPC (active, max 4)':<30} {m[IPC]:6.2f}")
            issue_busy[name.rsplit('_', 1)[0]] = m.get(PIPES[-1][1])
        elif LAT in m:
            lat = m[LAT]
            stalls = {k[len(STALL_PREFIX):-len(STALL_SUFFIX)]: v for k, v in m.items()
                      if k.startswith(STALL_PREFIX) and k.endswith(STALL_SUFFIX) and v > 0}
            total = sum(stalls.values())
            print(f"\n== {name}: warp cycles per issued instruction {lat:.2f} (stall sum {total:.2f})")
            for k, v in sorted(stalls.items(), key=lambda kv: -kv[1]):
                if v / lat >= 0.01:
                    print(f"  {k:<22} {v:6.2f} cycles  {100 * v / lat:5.1f} %")
            for label, key, fmt in OCC:
                if key in m:
                    print(f"  {label:<30} {fmt.format(m[key])}")
            base = name.rsplit('_', 1)[0]
            if base in issue_busy and issue_busy[base] is not None:
                print(f"  issue-slot use from ComputeWorkloadAnalysis: {issue_busy[base]:.1f} %; "
                      f"warps per scheduler needed for 100 %: {lat:.1f}")
        elif LD_SHARED in m:
            sh, ffma = m[LD_SHARED], m.get(FFMA, float("nan"))
            print(f"\n== {name}: instruction mix (whole kernel)")
            print(f"  {'shared loads (warp instr)':<30} {sh:,.0f}  (of which LDSM {m.get(LD_LDSM, 0):,.0f})")
            print(f"  {'FFMA (thread instr)':<30} {ffma:,.0f}")
            print(f"  {'warp instructions executed':<30} {m.get(INST, float('nan')):,.0f}")
            print(f"  {'FFMA per shared load (thread)':<30} {ffma / (32 * sh):.2f}")
            for label, key, fmt in OCC[2:4]:
                if key in m:
                    print(f"  {label:<30} {fmt.format(m[key])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
