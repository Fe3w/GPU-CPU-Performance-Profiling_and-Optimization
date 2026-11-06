"""racelib.py - shared CSV reading and speedup logic for the demo race.

Used by summarize.py (terminal scoreboard) and plot_race.py (charts, tables).
Standard library only.

Stage keys: S1, S2, S2a, S2b, S3, S3_FP16, S3_TF32 (any S<digit>[letter][_suffix]),
plus REF (a vendor reference: shown, never counted as a stage).

Rounds: a stage may have several CSVs (one per interleaved round). Each CSV's
mean over its rows is one round value; the stage value is the median over rounds.
"""
import csv
import glob
import re
import statistics
from pathlib import Path

RUNTIME_COLS = ["runtime_mean_ms", "mean_runtime_ms", "mean_time_ms",
                "runtime_ms", "time_ms", "kernel_ms", "ms"]
SD_COLS = ["runtime_sd_ms", "runtime_std_ms", "std_ms", "sd_ms", "stddev_ms"]
GFLOP_COLS = ["gflops", "gflop_s"]
BW_COLS = ["bandwidth_gbs", "effective_algorithmic_bandwidth_gbs", "gbps"]
OK_COLS = ["correct", "execution_valid", "correctness", "valid", "validation", "correct_fp64", "is_correct", "pass", "status"]

# Race file names: W04_S2a.csv, W04_S3_FP16_r2.csv, W04CPU_S1.csv, W04_REF.csv
FILE_RE = re.compile(r"^(W0\d)(CPU)?_(S\d[a-z]?(?:_(?:FP16|TF32))?|REF)(?:_r(\d+))?\.csv$")
STAGE_RE = re.compile(r"^S(\d)([a-z]?)(?:_(\w+))?$")
SUFFIX_ORDER = {"": 0, "FP16": 1, "TF32": 2}


def stage_sort_key(stage):
    """S1 < S2 < S2a < S2b < S3 < S3_FP16 < S3_TF32; REF and unknown keys last."""
    m = STAGE_RE.match(stage)
    if not m:
        return (99, stage, 0, "")
    num, letter, suffix = m.group(1), m.group(2), m.group(3) or ""
    return (int(num), letter, SUFFIX_ORDER.get(suffix, 9), suffix)


def family(stage):
    """Precision variants of one stage share a family: S3_FP16 and S3_TF32 -> S3."""
    return re.sub(r"_(FP16|TF32)$", "", stage)


def parse_name(name):
    """File name -> (workload, stage, round) or None. Workload is 'W04' or 'W04 CPU'."""
    m = FILE_RE.match(name)
    if not m:
        return None
    w = m.group(1) + (" CPU" if m.group(2) else "")
    return w, m.group(3), int(m.group(4)) if m.group(4) else 1


def _pick(row, names):
    low = {k.strip().lower(): v for k, v in row.items() if k}
    for n in names:
        v = low.get(n)
        if v is not None and str(v).strip() not in ("", "nan"):
            return v
    return None


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def read_one(path):
    """One CSV -> dict for that round, or None if it has no runtime column."""
    p = Path(path)
    if not p.is_file():
        return None
    with p.open(newline="") as f:
        rows = list(csv.DictReader(f))
    rt = [x for r in rows if (x := _num(_pick(r, RUNTIME_COLS))) is not None]
    if not rt:
        return None
    sdcol = [x for r in rows if (x := _num(_pick(r, SD_COLS))) is not None]
    gf = [x for r in rows if (x := _num(_pick(r, GFLOP_COLS))) is not None]
    bw = [x for r in rows if (x := _num(_pick(r, BW_COLS))) is not None]
    ok = {str(v).strip().upper() for r in rows if (v := _pick(r, OK_COLS)) is not None}
    last = rows[-1]
    tile = ""
    br, bc, w = (_num(last.get(k)) for k in ("br", "bc", "warps"))
    if br and bc:
        tile = f"{int(br)}x{int(bc)}" + (f" W{int(w)}" if w else "")
    return {
        "mean": statistics.mean(rt),
        "sd": statistics.stdev(rt) if len(rt) > 1 else (sdcol[0] if sdcol else 0.0),
        "reps": len(rt),
        "gflops": statistics.mean(gf) if gf else None,
        "bw": statistics.mean(bw) if bw else None,
        "ok": ok,
        "tile": tile,
        "n": last.get("n", ""),
        "precision": last.get("precision", ""),
    }


def combine(rounds):
    """Round dicts -> one stage dict (median over rounds of the round means)."""
    rounds = [r for r in rounds if r]
    if not rounds:
        return None
    means = [r["mean"] for r in rounds]
    med = statistics.median(means)
    ok = set().union(*(r["ok"] for r in rounds))
    good = bool(ok) and ok <= {"YES", "PASS", "TRUE", "1"}
    gfl = [r["gflops"] for r in rounds if r["gflops"] is not None]
    bws = [r["bw"] for r in rounds if r["bw"] is not None]
    return {
        "ms": med,
        "sd": statistics.mean(r["sd"] for r in rounds),          # mean rep SD within a round
        "spread": (max(means) - min(means)) / med * 100 if len(means) > 1 else None,
        "rounds": len(rounds),
        "reps": sum(r["reps"] for r in rounds),
        "gflops": statistics.median(gfl) if gfl else None,
        "bw": statistics.median(bws) if bws else None,
        "ok": "YES" if good else ("/".join(sorted(ok)) if ok else "n/a"),
        "tile": rounds[-1]["tile"],
        "n": rounds[-1]["n"],
        "precision": rounds[-1]["precision"],
    }


def expand(base):
    """'dir/W04_S1' -> ['dir/W04_S1.csv'] or the round files dir/W04_S1_r<k>.csv."""
    if base.endswith(".csv"):
        return [base]
    single = base + ".csv"
    if Path(single).is_file():
        return [single]
    return sorted(glob.glob(base + "_r[0-9]*.csv"))


def read_stage(base):
    return combine([read_one(p) for p in expand(base)])


def read_folder(folder):
    """Race folder -> {workload: {stage: stage dict}} from strictly named files."""
    acc = {}
    for p in sorted(Path(folder).glob("*.csv")):
        parsed = parse_name(p.name)
        if not parsed:
            continue
        w, s, _ = parsed
        acc.setdefault(w, {}).setdefault(s, []).append(read_one(p))
    out = {}
    for w, stages in acc.items():
        d = {s: c for s, rs in stages.items() if (c := combine(rs))}
        if d:
            out[w] = d
    return out


def analyse(stages, order=None):
    """{stage: dict} -> ordered rows with x_s1, x_prev, prev, saved, pct_ref; plus the chain.

    prev = nearest earlier stage of a different family (S3_TF32 -> S2b, not S3_FP16).
    order: keep this stage order instead of the S1 < S2a < ... sort (e.g. the fusion triple).
    """
    order = [s for s in order if s in stages and s != "REF"] if order else \
        sorted((s for s in stages if s != "REF"), key=stage_sort_key)
    ref = stages.get("REF")
    base = stages.get(order[0])["ms"] if order else None
    rows = []
    for i, s in enumerate(order):
        d = dict(stages[s])
        prev = next((order[j] for j in range(i - 1, -1, -1) if family(order[j]) != family(s)), None)
        d.update(stage=s, prev=prev,
                 x_s1=base / d["ms"] if base else None,
                 x_prev=stages[prev]["ms"] / d["ms"] if prev else None,
                 saved=(1 - d["ms"] / base) * 100 if base else None,
                 pct_ref=ref["ms"] / d["ms"] * 100 if ref else None)
        rows.append(d)
    chain = []
    if rows:
        best = min(rows, key=lambda r: r["ms"])
        cur = best
        while cur and cur["prev"]:
            chain.append((cur["prev"], cur["stage"], cur["x_prev"]))
            cur = next(r for r in rows if r["stage"] == cur["prev"])
        chain.reverse()
    return rows, ref, chain
