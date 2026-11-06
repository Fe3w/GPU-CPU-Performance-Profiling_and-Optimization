#!/usr/bin/env python3
"""W04 sweep summary: one table per experiment folder of a run directory.

    python3 tools/w04_summary.py results/raw/W04_ATTENTION/<run> [experiment]

For every (stage, N), each process run (one CSV timestamp, `reps` rows) gives
one run mean; the table shows the median over runs (rounds), the min-max
spread of the run means in % of the median, and the mean SD across reps.
If the run folder has gpu_state.csv, each run is matched to the nvidia-smi
samples taken between its start (commands.log) and the next run's start, and
the median SM clock / power / max temperature over those samples are shown.
The first-rep column is rep 1's time relative to the mean of the other reps
(a clock ramp-up shows up as a positive value). On the GPU rep 1 is the
faster one (it runs before the 55 W power cap engages), so the steady column
gives the median over runs of the mean of reps 2..R, the power-capped steady
state (open issue 9, option b). Ratios and plots keep using the mean of all
reps (the protocol); quote steady next to it where rep 1 matters.
If a CSV has a non-empty prep_ms column (P4, decision C23: the untimed FP16
conversion of GPU S3), a second table shows the median prep time per (stage, N),
its size relative to runtime_ms, and runtime + prep.

Groups by (CSV name, N) only, so use it for experiments where nothing else varies
within one CSV (the GPU N sweep, where a second tile has its own CSV); Br/Bc or
thread sweeps would be mixed together.

Standard library only. Reads CSVs; never writes into the run folder.
"""
import csv
import datetime as dt
import glob
import os
import re
import statistics as st
import sys
from collections import defaultdict


def parse_ts(s):
    s = s.strip()
    for fmt in ("%Y/%m/%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            return dt.datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None


def load_gpu_state(run):
    path = os.path.join(run, "gpu_state.csv")
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        next(f, None)
        for row in csv.reader(f):
            if len(row) < 5:
                continue
            t = parse_ts(row[0])
            try:
                sm = float(row[1].split()[0])
                pw = float(row[3].split()[0])
                temp = float(row[4])
            except (ValueError, IndexError):
                continue
            out.append((t, sm, pw, temp))
    return out


def load_command_starts(run, experiment):
    """[(start_time, stage_id, n)] in execution order, for this experiment only."""
    path = os.path.join(run, "commands.log")
    starts = []
    if not os.path.exists(path):
        return starts
    with open(path) as f:
        for line in f:
            m = re.match(r"\[([^\]]+)\] .*?build/(\S+) .*--n (\d+) .*--out (\S+)", line)
            if not m:
                continue
            exp = os.path.basename(os.path.dirname(m.group(4)))
            # Stage id = CSV basename (same key as the tables), so two tiles of one
            # binary written to different CSVs are matched to their own samples.
            sid = os.path.basename(m.group(4))
            sid = sid[:-4] if sid.endswith(".csv") else sid
            starts.append((parse_ts(m.group(1)), sid, int(m.group(3)), exp))
    return starts


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    run = sys.argv[1].rstrip("/")
    only = sys.argv[2] if len(sys.argv) > 2 else None
    gpu = load_gpu_state(run)
    starts_all = load_command_starts(run, only)

    for exp_dir in sorted(glob.glob(os.path.join(run, "*/"))):
        exp = os.path.basename(exp_dir.rstrip("/"))
        if exp == "logs" or (only and exp != only):
            continue
        csvs = sorted(glob.glob(os.path.join(exp_dir, "*.csv")))
        if not csvs:
            continue
        # (stage_id, n) -> {timestamp: [(rep, ms)]}
        runs = defaultdict(lambda: defaultdict(list))
        prep = defaultdict(lambda: defaultdict(list))  # same keys; only rows with a prep_ms value
        flops = {}
        for path in csvs:
            sid = os.path.basename(path)[:-4]
            with open(path) as f:
                for r in csv.DictReader(f):
                    key = (sid, int(r["n"]))
                    runs[key][r["timestamp"]].append((int(r["rep"]), float(r["runtime_ms"])))
                    if r.get("prep_ms"):
                        prep[key][r["timestamp"]].append(float(r["prep_ms"]))
                    flops[key] = 4.0 * int(r["bh"]) * int(r["n"]) ** 2 * int(r["d"])

        # match runs to their sampling windows, in execution order
        starts = [s for s in starts_all if s[3] == exp]
        window = {}  # (sid, n, k-th run) -> (sm, power, tmax)
        if gpu and starts:
            seen = defaultdict(int)
            all_starts = [s[0] for s in starts_all]
            for (t0, sid, n, _) in starts:
                idx = all_starts.index(t0)
                t1 = all_starts[idx + 1] if idx + 1 < len(all_starts) else dt.datetime.max
                smp = [g for g in gpu if g[0] and t0 <= g[0] < t1]
                k = seen[(sid, n)]
                seen[(sid, n)] += 1
                if smp:
                    window[(sid, n, k)] = (st.median(g[1] for g in smp), st.median(g[2] for g in smp),
                                           max(g[3] for g in smp))

        print(f"\n== {exp} ({run}) ==")
        wid = max([38] + [len(k[0]) for k in runs])
        hdr = f"{'stage':<{wid}} {'N':>5} {'runs':>4} {'median ms':>11} {'steady ms':>10} {'GFLOP/s':>8} {'spread%':>8} {'repSD%':>7} {'rep1%':>6}"
        if window:
            hdr += f" {'SM MHz':>7} {'W':>5} {'Tmax':>5}"
        print(hdr)
        med_by = {}
        for key in sorted(runs, key=lambda k: (k[1], k[0])):
            sid, n = key
            per_run, steady = [], []
            sds, rep1 = [], []
            for k, ts in enumerate(sorted(runs[key])):
                reps = [ms for _, ms in sorted(runs[key][ts])]
                m = st.mean(reps)
                per_run.append(m)
                if len(reps) > 1:
                    sds.append(100 * st.stdev(reps) / m)
                    rest = st.mean(reps[1:])
                    rep1.append(100 * (reps[0] - rest) / rest)
                    steady.append(rest)
            med = st.median(per_run)
            med_by[key] = med
            spread = 100 * (max(per_run) - min(per_run)) / med
            sty = f"{st.median(steady):>10.4f}" if steady else f"{'-':>10}"
            line = (f"{sid:<{wid}} {n:>5} {len(per_run):>4} {med:>11.4f} {sty} {flops[key] / med / 1e6:>8.1f} "
                    f"{spread:>8.1f} {st.mean(sds) if sds else 0:>7.1f} {st.mean(rep1) if rep1 else 0:>6.1f}")
            if window:
                w = [window[(sid, n, k)] for k in range(len(per_run)) if (sid, n, k) in window]
                if w:
                    line += (f" {st.median(x[0] for x in w):>7.0f} {st.median(x[1] for x in w):>5.1f}"
                             f" {max(x[2] for x in w):>5.0f}")
            print(line)

        if prep:
            print(f"\n  prepare step (prep_ms, not included in runtime_ms), median over runs:")
            print(f"  {'stage':<{wid}} {'N':>5} {'prep ms':>9} {'% of run':>8} {'run+prep ms':>11}")
            for key in sorted(prep, key=lambda k: (k[1], k[0])):
                pm = st.median(st.mean(v) for v in prep[key].values())
                print(f"  {key[0]:<{wid}} {key[1]:>5} {pm:>9.4f} {100 * pm / med_by[key]:>8.1f} "
                      f"{med_by[key] + pm:>11.4f}")

        # ratios between stages at the same N (median of run means)
        ns = sorted({n for _, n in med_by})
        sids = sorted({s for s, _ in med_by})
        if len(sids) > 1:
            print(f"\n  time ratios (row stage / column stage), median of runs:")
            for a in sids:
                for b in sids:
                    if a >= b:
                        continue
                    vals = [f"N={n}: {med_by[(a, n)] / med_by[(b, n)]:.3f}" for n in ns
                            if (a, n) in med_by and (b, n) in med_by]
                    print(f"  {a.replace('W04_ATTENTION_', '')} / {b.replace('W04_ATTENTION_', '')}: "
                          + "  ".join(vals))
    return 0


if __name__ == "__main__":
    sys.exit(main())
