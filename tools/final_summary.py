#!/usr/bin/env python3
"""final_summary.py - consolidated results of a scripts/run_all.sh run.

    python3 tools/final_summary.py <run folder>                 tables + files in the run folder
    python3 tools/final_summary.py <run folder> --publish-docs  also write docs/RESULTS_SUMMARY.md,
                                                                docs/results/final_results.csv and
                                                                docs/figures/final/speedup_all.png
    python3 tools/final_summary.py --check-recorded             parse the recorded CSVs listed in
                                                                docs/results/reference_values.csv with
                                                                the same adapters and compare

W01-W03 binaries write different CSV schemas; one adapter per schema turns each
CSV into: mean and SD of the runtime, a rate (GB/s or GFLOP/s), the binary's own
Correct verdict, and its error against the double-precision reference where the
binary reports one. Statistics:
  per-repetition CSVs  mean and SD over the repetitions of one run
  summary-row CSVs     the row's mean and SD (W02 GPU S2 writes the mean only; its SD
                       is read from its console line "Mean time (ms) : x ± sd")
  GPU, several rounds  median over rounds of the run means; SD = mean within-run SD;
                       spread = (max - min) / median of the run means
Speedups: x vs S1 = t(S1) / t; x vs prev = t(previous stage) / t, where the previous
stage of S3_TF32 is S2b (precision variants share a step). REF is a reference, not a stage.
Delta vs earlier: (t - t_earlier) / t_earlier against the primary row of
docs/results/reference_values.csv; |delta| > 10% is flagged.
Standard library only; matplotlib is optional (figure).
"""
import csv
import re
import shutil
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REF_FILE = REPO / "docs/results/reference_values.csv"
NOTES_FILE = REPO / "docs/results/deviation_notes.md"
FLAG_PCT = 10.0
REF_N = {"W01": "33554432", "W02": "33554432", "W03": "1024"}   # sizes of the earlier recorded W01-W03 runs

WORKLOADS = {
    "W01": ("W01 Vector Add", "C = A + B, FP32", "GB/s"),
    "W02": ("W02 Dot Product", "s = Σ aᵢbᵢ, FP32 inputs", "GB/s"),
    "W03": ("W03 GEMM", "C = A × B, FP32, N × N matrices", "GFLOP/s"),
    "W04": ("W04 Attention", "O = softmax(QKᵀ/√d)·V, FP32 inputs", "GFLOP/s"),
}
STAGE_TEXT = {
    ("W01", "CPU", "S1"): "scalar, -O0", ("W01", "CPU", "S2"): "OpenMP, 6 threads",
    ("W01", "CPU", "S3"): "OpenMP + AVX2, 6 threads", ("W01", "GPU", "S1"): "naive, 256 threads/block",
    ("W01", "GPU", "S2"): "coalesced, 128 threads/block",
    ("W02", "CPU", "S1"): "scalar, double accumulator", ("W02", "CPU", "S2"): "OpenMP, double partials, 6 threads",
    ("W02", "CPU", "S3"): "OpenMP + AVX2/FMA (FP32 lanes), 6 threads", ("W02", "GPU", "S1"): "naive (double atomics)",
    ("W02", "GPU", "S2"): "hierarchical block reduction",
    ("W03", "CPU", "S1"): "naive triple loop, -O0", ("W03", "CPU", "S2"): "128-blocked + OpenMP, 8 threads",
    ("W03", "CPU", "S3"): "AVX2/FMA + blocked + OpenMP, 8 threads", ("W03", "GPU", "S1"): "naive",
    ("W03", "GPU", "S2"): "16×16 shared-memory tile", ("W03", "GPU", "S3"): "TF32 Tensor Cores (WMMA)",
    ("W04", "CPU", "S1"): "scalar, full S row", ("W04", "CPU", "S2"): "fused + blocked + OpenMP, 20 threads",
    ("W04", "CPU", "S3"): "AVX2/FMA + vector exp, 20 threads", ("W04", "GPU", "S1"): "naive, 3 kernels",
    ("W04", "GPU", "S2a"): "tiled, unfused (3 kernels)", ("W04", "GPU", "S2b"): "fused, FlashAttention-style",
    ("W04", "GPU", "S3_FP16"): "fused, FP16 Tensor Cores", ("W04", "GPU", "S3_TF32"): "fused, TF32 Tensor Cores",
    ("W04", "GPU", "REF"): "cuBLAS GEMMs + softmax (reference)",
}
CHECK_TEXT = {
    "W01": "every element |C − (A+B)| ≤ 1e-6",
    "W02": "relative error vs the double-precision sum ≤ 0.1% (every W02 stage)",
    "W03": "max relative error vs CPU reference < 0.1%",
    "W04": "max rel. error vs double ≤ 1e-4 (FP32), 5e-3 (TF32), 1e-2 (FP16)",
}

KEY_RE = re.compile(r"^(W0\d)_(CPU|GPU)_(S\d[ab]?(?:_FP16|_TF32)?|REF)(?:_N(\d+))?$")


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def yes(v):
    s = str(v).strip().upper()
    return "YES" if s in ("YES", "1", "TRUE", "PASS") else ("NO" if s in ("NO", "0", "FALSE", "FAIL") else "?")


def rows_of(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def check_text(key):
    w = key[:3]
    return CHECK_TEXT.get(key[:10], CHECK_TEXT.get(w, ""))


# ------------------------------------------------------------------ adapters
def w02_reference(n):
    """Double-precision dot product of the W02 inputs, A[i] = (i % 1000)·0.001f, B[i] = (3i % 1000)·0.001f
    (identical in every W02 stage), with A and B rounded to float32 as the binaries store them."""
    import struct
    f32 = lambda x: struct.unpack("f", struct.pack("f", x))[0]
    a = [f32(f32(float(k)) * f32(0.001)) for k in range(1000)]
    full, rest = divmod(n, 1000)
    period = sum(a[k] * a[(3 * k) % 1000] for k in range(1000))
    return full * period + sum(a[k] * a[(3 * k) % 1000] for k in range(rest))


def w02_cpu_s1_rows(path):
    """W02 CPU S1 writes 19 fields under a 21-column header (reference_result and relative_error_pct
    are missing from its rows): read by position. Returns (runtime_ms, bandwidth_gbs, result, verdict, n)."""
    with open(path, newline="") as fh:
        rd = csv.reader(fh)
        head = next(rd)
        out = []
        for r in rd:
            if len(r) == len(head):
                d = dict(zip(head, r))
                out.append((fnum(d["runtime_ms"]), fnum(d["bandwidth_gbs"]), fnum(d["result"]), yes(d["execution_valid"]), int(d["N"])))
            else:
                out.append((fnum(r[9]), fnum(r[11]), fnum(r[15]), yes(r[16]), int(r[5])))
    return out


def parse(key, csv_path, log_path=None):
    """One CSV of one run -> dict, or None if the file is missing or empty."""
    p = Path(csv_path)
    if not p.is_file():
        return None
    rows = rows_of(p)
    if not rows:
        return None
    w, dev = key[:3], key[4:7]
    last = rows[-1]
    n_csv = last.get("N") or last.get("n") or ""
    out = {"sd_source": "reps", "err_pct": None}
    if key == "W02_CPU_S1":
        rr = w02_cpu_s1_rows(p)
        t = [x[0] for x in rr]
        ref = w02_reference(rr[0][4])
        verdicts = {x[3] for x in rr}
        out.update(mean=st.mean(t), sd=st.stdev(t) if len(t) > 1 else 0.0, reps=len(t), reps_ms=t,
                   rate=st.mean(x[1] for x in rr), n=str(rr[0][4]),
                   correct="YES" if verdicts == {"YES"} else ("NO" if "NO" in verdicts else "?"),
                   err_pct=max(abs(x[2] - ref) / ref * 100 for x in rr))
        return out
    if w == "W01" or (w == "W02" and dev == "CPU") or w == "W04":       # per-repetition rows
        t = [fnum(r["runtime_ms"]) for r in rows]
        out["mean"] = st.mean(t)
        out["sd"] = st.stdev(t) if len(t) > 1 else 0.0
        out["reps"] = len(t)
        out["reps_ms"] = t
        rate_col = "bandwidth_gbs" if w in ("W01", "W02") else "gflops"
        out["rate"] = st.mean(fnum(r[rate_col]) for r in rows)
        ok_col = "execution_valid" if w == "W02" else "correct"
        verdicts = {yes(r[ok_col]) for r in rows}
        out["correct"] = "YES" if verdicts == {"YES"} else ("NO" if "NO" in verdicts else "?")
        if w == "W02":
            out["err_pct"] = max(abs(fnum(r["relative_error_pct"])) for r in rows)
        if w == "W04":
            out["err_pct"] = max(fnum(r["max_rel_err"]) for r in rows) * 100
    elif w == "W02" and key.endswith("S1"):                              # W02 GPU S1: summary row
        r = rows[-1]
        out.update(mean=fnum(r["mean_time_ms"]), sd=fnum(r["stddev_time_ms"]), reps=int(r["repetitions"]),
                   rate=fnum(r["mean_bandwidth_gbs"]), correct=yes(r["correct"]),
                   err_pct=abs(fnum(r["relative_error_pct"])), sd_source="row")
    elif w == "W02":                                                     # W02 GPU S2: mean only
        r = rows[-1]
        out.update(mean=fnum(r["runtime_ms"]), sd=fnum(r.get("runtime_sd_ms")), reps=int(r["repetitions"]),
                   rate=fnum(r["bandwidth_gbs"]), correct=yes(r["execution_valid"]),
                   err_pct=abs(fnum(r["error_vs_double_pct"])), sd_source="row" if r.get("runtime_sd_ms") else "none")
        if out["sd"] is None and log_path and Path(log_path).is_file():   # older binaries: mean only in the CSV
            m = re.search(r"Mean time \(ms\)\s*:\s*([0-9.eE+-]+)\s*±\s*([0-9.eE+-]+)", Path(log_path).read_text())
            if m:
                out["sd"] = float(m.group(2))
                out["sd_source"] = "console"
    else:                                                                # W03: summary row
        r = rows[-1]
        out.update(mean=fnum(r["runtime_mean_ms"]), sd=fnum(r["runtime_std_ms"]), reps=int(r["repetitions"]),
                   rate=fnum(r["gflops"]), correct=yes(r["correct"]), err_pct=abs(fnum(r["max_rel_error_pct"])),
                   sd_source="row")
    out.setdefault("n", str(int(float(n_csv))) if n_csv else "")
    return out


# ------------------------------------------------------------------ run folder
def gpu_samples(run):
    """gpu_state.csv -> [(epoch s, SM MHz, memory MHz, W)]"""
    import datetime as dt
    p = run / "gpu_state.csv"
    out = []
    if not p.is_file():
        return out
    with open(p, newline="") as fh:
        for r in csv.reader(fh):
            if not r or r[0].startswith("timestamp"):
                continue
            try:
                t = dt.datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
                out.append((t, float(r[1].split()[0]), float(r[2].split()[0]), float(r[3].split()[0])))
            except (ValueError, IndexError):
                continue
    return out


def clocks(samples, t0, t1):
    """median SM and memory clock of the loaded samples (SM > 1000 MHz) inside one run."""
    if not (t0 and t1):
        return None, None
    w = [x for x in samples if t0 <= x[0] <= t1 and x[1] > 1000]
    if not w:
        return None, None
    return st.median(x[1] for x in w), st.median(x[2] for x in w)


def load_run(run):
    runs = rows_of(run / "runs.csv")
    samples = gpu_samples(run)
    by_key = {}
    for r in runs:
        by_key.setdefault(r["key"], []).append(r)
    stages = {}
    for key, rs in by_key.items():
        parsed, exits = [], []
        for r in rs:
            exits.append(int(r["exit_code"] or 0))
            d = parse(key, r["csv"], r["log"])
            if d:
                d["run"] = r
                d["sm_mhz"], d["mem_mhz"] = clocks(samples, fnum(r.get("t_start")), fnum(r.get("t_end")))
                parsed.append(d)
        m = KEY_RE.match(key)
        w, dev, stage, n = m.group(1), m.group(2), m.group(3), m.group(4)
        n = n or next((d["n"] for d in parsed if d.get("n")), "?")
        rec = {"key": key, "workload": w, "device": dev, "stage": stage, "n": n,
               "runs": len(rs), "exit_codes": exits}
        if not parsed:
            rec.update(mean=None, correct="MISSING")
        else:
            means = [d["mean"] for d in parsed]
            sds = [d["sd"] for d in parsed if d["sd"] is not None]
            med = st.median(means)
            verdicts = {d["correct"] for d in parsed}
            if any(e != 0 for e in exits) and verdicts == {"YES"}:
                verdicts = {"NO"}
            rec.update(mean=med, sd=st.mean(sds) if sds else None,
                       spread=(max(means) - min(means)) / med * 100 if len(means) > 1 else None,
                       rounds=len(parsed), reps=parsed[0]["reps"], rate=st.median(d["rate"] for d in parsed),
                       correct="YES" if verdicts == {"YES"} else ("NO" if "NO" in verdicts else "?"),
                       err_pct=max((d["err_pct"] for d in parsed if d["err_pct"] is not None), default=None),
                       sd_source=parsed[0]["sd_source"], parsed=parsed,
                       threads=rs[0]["threads"],
                       sm_mhz=st.median([d["sm_mhz"] for d in parsed if d["sm_mhz"]]) if any(d["sm_mhz"] for d in parsed) else None,
                       mem_mhz=st.median([d["mem_mhz"] for d in parsed if d["mem_mhz"]]) if any(d["mem_mhz"] for d in parsed) else None,
                       primed=all(r.get("primed") == "1" for r in rs),
                       cpu_c=[fnum(r["mean_c"]) for r in rs], cpu_start_c=[fnum(r["start_c"]) for r in rs],
                       p_mhz=[fnum(r["mean_p_mhz"]) for r in rs])
        stages[key] = rec
    for rec in stages.values():          # a stage without a CSV takes its workload's N
        if rec["n"] == "?":
            rec["n"] = next((x["n"] for x in stages.values() if x["workload"] == rec["workload"] and x["n"] != "?"), "0")
    return stages


def stage_order(s):
    if s == "REF":
        return (9, "", 0)
    m = re.match(r"S(\d)([ab]?)(?:_(FP16|TF32))?", s)
    return (int(m.group(1)), m.group(2), {"FP16": 1, "TF32": 2}.get(m.group(3), 0))


def family(s):
    return re.sub(r"_(FP16|TF32)$", "", s)


def groups(stages):
    """(workload, device, n) -> ordered list of stage records with speedups."""
    g = {}
    for rec in stages.values():
        g.setdefault((rec["workload"], rec["device"], rec["n"]), []).append(rec)
    out = {}
    for k in sorted(g, key=lambda k: (k[0], k[1] == "CPU", int(k[2]))):
        recs = sorted(g[k], key=lambda r: stage_order(r["stage"]))
        stages_only = [r for r in recs if r["stage"] != "REF"]
        base = stages_only[0]["mean"] if stages_only and stages_only[0]["mean"] else None
        ref = next((r for r in recs if r["stage"] == "REF"), None)
        for i, r in enumerate(stages_only):
            prev = next((stages_only[j] for j in range(i - 1, -1, -1)
                         if family(stages_only[j]["stage"]) != family(r["stage"])), None)
            ok = r["mean"] and base
            r["x_s1"] = base / r["mean"] if ok else None
            r["saved"] = (1 - r["mean"] / base) * 100 if ok else None
            r["prev"] = prev["stage"] if prev else None
            r["x_prev"] = prev["mean"] / r["mean"] if prev and prev["mean"] and r["mean"] else None
            r["pct_ref"] = ref["mean"] / r["mean"] * 100 if ref and ref["mean"] and r["mean"] else None
        if ref:
            ref.update(x_s1=base / ref["mean"] if base and ref["mean"] else None, saved=None, prev=None,
                       x_prev=None, pct_ref=100.0 if ref["mean"] else None)
        out[k] = recs
    return out


def references():
    """primary earlier value per key, and the other (non-primary) earlier rows per key."""
    refs, alts = {}, {}
    if REF_FILE.is_file():
        for r in rows_of(REF_FILE):
            if r["primary"] == "1":
                refs[r["key"]] = r
            else:
                alts.setdefault(r["key"], []).append(r)
    return refs, alts


# ------------------------------------------------------------------ output helpers
def f(v, fmt="{:.3f}", none="–"):
    return fmt.format(v) if v is not None else none


def ms(v):
    if v is None:
        return "–"
    return f"{v:.1f}" if v >= 100 else f"{v:.3f}" if v >= 1 else f"{v:.4f}"


def pct(v):
    return "–" if v is None else (f"{v:.2f}%" if abs(v) >= 99 else f"{v:.1f}%")


def info(run):
    d = {}
    p = run / "run_info.txt"
    if p.is_file():
        for line in p.read_text().splitlines():
            if " : " in line:
                k, v = line.split(" : ", 1)
                d[k.strip()] = v.strip()
    return d


def gpu_state(run):
    p = run / "gpu_state.csv"
    if not p.is_file():
        return None
    sm, pw, tc = [], [], []
    for r in rows_of(p):
        vals = {k.strip(): v.strip() for k, v in r.items() if k}
        s = fnum(vals.get("clocks.current.sm [MHz]", "").split()[0] if vals.get("clocks.current.sm [MHz]") else None)
        w = fnum(vals.get("power.draw [W]", "").split()[0] if vals.get("power.draw [W]") else None)
        t = fnum(vals.get("temperature.gpu"))
        if s and s > 1000:                      # loaded samples only
            sm.append(s); pw.append(w or 0); tc.append(t or 0)
    if not sm:
        return None
    return {"sm": st.median(sm), "sm_lo": min(sm), "sm_hi": max(sm), "w": st.median(pw), "t_max": max(tc),
            "samples": len(sm)}


# ------------------------------------------------------------------ report
def build(run, stages, publish):
    I = info(run)
    smoke = run.name.endswith("-smoke")
    g = groups(stages)
    refs, alts = references()
    flagged = []
    L = []
    L += ["# Results summary — final paired re-test of W01–W04", ""]
    if smoke:
        L += ["> **Smoke run** (tiny sizes, 1 iteration): checks that every command works. These numbers are not measurements.", ""]
    L += ["| | |", "|---|---|",
          f"| Run | `{run.name}` (one session, `scripts/run_all.sh`) |",
          f"| Commit | `{I.get('tree sha', '?')}` (W04 build `{I.get('w04 build', '?')}`) |",
          f"| Date | {I.get('date', '?')} |",
          f"| CPU | {I.get('cpu', '?')} (6 P + 8 E cores, 20 threads, AVX2/FMA), governor {I.get('governor', '?')}, EPP {I.get('epp', '?')}, AC {I.get('ac online', '?')} |",
          f"| GPU | {I.get('gpu', '?').split(',')[0]} (sm_89, 24 SMs, 8 GiB GDDR6 128-bit), driver {I.get('gpu', '?, ?').split(',')[1].strip() if ',' in I.get('gpu', '') else '?'}, power limit {I.get('gpu limits', '?').split(',')[0]} |",
          f"| Toolchain | {I.get('compiler', '?')} |",
          f"| Protocol | {I.get('protocol', '?')} |",
          f"| Sizes / threads | {I.get('sizes', '?')}; {I.get('threads', '?')} |"]
    gs = gpu_state(run)
    if gs:
        L.append(f"| GPU state (loaded samples) | SM clock median {gs['sm']:.0f} MHz ({gs['sm_lo']:.0f}–{gs['sm_hi']:.0f}), "
                 f"power median {gs['w']:.1f} W, max {gs['t_max']:.0f} °C ({gs['samples']} samples, 250 ms) |")
    cpu_t = [c for r in stages.values() if r["device"] == "CPU" and r.get("cpu_c") for c in r["cpu_c"] if c and c > 0]
    if cpu_t:
        L.append(f"| CPU state | package mean temperature per run {min(cpu_t):.0f}–{max(cpu_t):.0f} °C "
                 f"(cool-down to ≤ 70 °C before each run) |")
    proto = dict(kv.split("=") for kv in I.get("protocol", "").split() if "=" in kv)
    L += ["", f"Times are ms per iteration: mean ± SD over the {proto.get('reps', '?')} repetitions of a run "
          f"({proto.get('iters', '?')} timed iterations each, after {proto.get('warmup', '?')} warm-ups). GPU stages ran in "
          f"{proto.get('rounds(GPU)', '?')} interleaved round(s): median of the round means, SD = mean within-round SD, "
          "spread = (max − min) / median of the round means. CPU stages ran once each. × vs S1 = t(S1)/t; × prev = t(previous stage)/t (S3_TF32's previous stage is S2b); saved = (1 − t/t(S1))·100. "
          f"Δ earlier = change vs the earlier recorded run (`docs/results/reference_values.csv`); |Δ| > {FLAG_PCT:.0f}% is marked ⚠ "
          "and explained below.", ""]

    csv_rows = []
    for (w, dev, n), recs in g.items():
        name, what, unit = WORKLOADS[w]
        nlab = f", N = {int(n):,}"
        L += [f"## {name} — {dev}{nlab}", "", f"{what}{nlab}.", "",
              f"| Stage | | ms (mean ± SD) | {unit} | × vs S1 | × prev | saved | Correct | error vs double | Δ earlier |"
              + (" SM / mem MHz |" if dev == "GPU" else ""),
              "|---|---|---|---|---|---|---|---|---|---|" + ("---|" if dev == "GPU" else "")]
        for r in recs:
            sd = f" ± {ms(r.get('sd'))}" if r.get("sd") is not None else " (SD n/a)"
            if r.get("sd_source") == "console":
                sd += "ᶜ"
            spread = f" (spread {r['spread']:.1f}%)" if r.get("spread") is not None else ""
            ref = refs.get(r["key"])
            delta = None
            if ref and r["mean"] and not smoke and n == REF_N.get(w, n):
                delta = (r["mean"] - float(ref["mean_ms"])) / float(ref["mean_ms"]) * 100
            dtxt = "–" if delta is None else f"{delta:+.1f}%" + (" ⚠" if abs(delta) > FLAG_PCT else "")
            if delta is not None and abs(delta) > FLAG_PCT:
                flagged.append((r, ref, delta))
            err = r.get("err_pct")
            errtxt = "–" if err is None else (f"{err:.2g}%" if err >= 0.01 else f"{err:.1e}%")
            label = r["stage"] + (" (ref.)" if r["stage"] == "REF" else "")
            xs1 = f"**{r['x_s1']:.2f}×**" if r.get("x_s1") and r["stage"] != "REF" else (f"{r['x_s1']:.2f}×" if r.get("x_s1") else "–")
            L.append(f"| {label} | {STAGE_TEXT.get((w, dev, r['stage']), '')} | {ms(r['mean'])}{sd}{spread} | "
                     f"{f(r.get('rate'), '{:.1f}')} | {xs1} | {f(r.get('x_prev'), '{:.2f}×')} | {pct(r.get('saved'))} | "
                     f"{r['correct']} | {errtxt} | {dtxt} |"
                     + (f" {f(r.get('sm_mhz'), '{:.0f}')} / {f(r.get('mem_mhz'), '{:.0f}')} |" if dev == "GPU" else ""))
            csv_rows.append({
                "workload": w, "device": dev, "stage": r["stage"], "n": n, "variant": STAGE_TEXT.get((w, dev, r["stage"]), ""),
                "threads": r.get("threads", ""), "rounds": r.get("rounds", 0), "reps": r.get("reps", ""),
                "mean_ms": f(r["mean"], "{:.6f}", ""), "sd_ms": f(r.get("sd"), "{:.6f}", ""), "sd_source": r.get("sd_source", ""),
                "round_spread_pct": f(r.get("spread"), "{:.2f}", ""), "rate": f(r.get("rate"), "{:.3f}", ""), "rate_unit": unit,
                "x_vs_s1": f(r.get("x_s1"), "{:.4f}", ""), "x_vs_prev": f(r.get("x_prev"), "{:.4f}", ""), "prev": r.get("prev") or "",
                "saved_pct": f(r.get("saved"), "{:.2f}", ""), "pct_of_ref": f(r.get("pct_ref"), "{:.1f}", ""),
                "correct": r["correct"], "check": check_text(r["key"]), "err_vs_double_pct": f(err, "{:.6g}", ""),
                "earlier_ms": ref["mean_ms"] if ref else "", "delta_vs_earlier_pct": f(delta, "{:.2f}", ""),
                "median_sm_mhz": f(r.get("sm_mhz"), "{:.0f}", ""), "median_mem_mhz": f(r.get("mem_mhz"), "{:.0f}", ""),
                "exit_codes": " ".join(map(str, r["exit_codes"])), "run": run.name, "commit": I.get("tree sha", ""),
                "hardware": "i7-13650HX + RTX 4060 Laptop (55 W cap)"})
        st_only = [r for r in recs if r["stage"] != "REF" and r.get("x_prev")]
        if st_only:
            best = min((r for r in recs if r["stage"] != "REF" and r["mean"]), key=lambda r: r["mean"])
            chain, cur = [], best
            while cur and cur.get("prev"):
                chain.append(cur)
                cur = next(x for x in recs if x["stage"] == cur["prev"])
            chain.reverse()
            if chain:
                L += ["", "Factored: " + " · ".join(f"{c['prev']}→{c['stage']} {c['x_prev']:.2f}×" for c in chain)
                      + f" = **{best['x_s1']:.1f}×**"]
        L.append("")

    # CPU vs GPU
    pairs = []
    for w in ("W01", "W02", "W03", "W04"):
        ns = sorted({k[2] for k in g if k[0] == w}, key=int)
        for n in ns:
            c, gg = g.get((w, "CPU", n)), g.get((w, "GPU", n))
            if not (c and gg):
                continue
            cs = [r for r in c if r["mean"]]
            gst = [r for r in gg if r["mean"] and r["stage"] != "REF"]
            if not (cs and gst):
                continue
            cb, gb = min(cs, key=lambda r: r["mean"]), min(gst, key=lambda r: r["mean"])
            pairs.append((w, n, cs[0], gst[0], cb, gb))
    if pairs:
        L += ["## CPU vs GPU (same N, same session)", "",
              "| Workload | CPU S1 → GPU S1 | best CPU → best GPU | CPU S1 → best GPU |", "|---|---|---|---|"]
        for w, n, c1, g1, cb, gb in pairs:
            nl = f" N={n}" if w == "W04" else ""
            L.append(f"| {WORKLOADS[w][0]}{nl} | {c1['mean'] / g1['mean']:.1f}× | "
                     f"{cb['stage']} {ms(cb['mean'])} → {gb['stage']} {ms(gb['mean'])} ms: **{cb['mean'] / gb['mean']:.1f}×** | "
                     f"{c1['mean'] / gb['mean']:.0f}× |")
        L.append("")

    # W01 reconciliation
    w1 = [stages.get("W01_GPU_S1"), stages.get("W01_GPU_S2")]
    if all(x and x.get("parsed") for x in w1):
        L += ["## W01 reconciliation (GPU S1 vs S2)", "",
              "From this run's CSV rows only (5 repetitions per round, every round), never from console output:", "",
              "| | rounds × reps | per-round means (ms) | median ms | SD | GB/s | % of 256 GB/s nominal | memory clock per round (MHz) |",
              "|---|---|---|---|---|---|---|---|"]
        for x in w1:
            rm = ", ".join(f"{d['mean']:.3f}" for d in x["parsed"])
            mc = ", ".join(f(d.get("mem_mhz"), "{:.0f}") for d in x["parsed"])
            L.append(f"| {x['stage']} | {x['rounds']} × {x['reps']} | {rm} | {x['mean']:.3f} | {f(x['sd'])} | "
                     f"{x['rate']:.1f} | {x['rate'] / 256 * 100:.0f}% | {mc} |")
        ratio = w1[0]["mean"] / w1[1]["mean"]
        L += ["", f"**Final W01 GPU: S1 {w1[0]['mean']:.3f} ms, S2 {w1[1]['mean']:.3f} ms, S2/S1 speedup {ratio:.3f}×** "
              f"({'within' if abs(ratio - 1) < 0.03 else 'outside'} ±3%: both kernels stream at the DRAM limit).", "",
              "Earlier W01 GPU figures and why they differ:", ""]
        for k in ("W01_GPU_S1", "W01_GPU_S2"):
            for r in ([refs[k]] if k in refs else []) + alts.get(k, []):
                L.append(f"- {k[-2:]}: {float(r['mean_ms']):.3f} ms over {r['rows']} row(s), `{r['source']}` ({r['date']}): {r['note'] or 'primary earlier value'}")
        L += ["- Console output of an earlier session showed 2.011 / 1.779 ms. These are console lines, not recorded "
              "per-repetition rows, and are not used. [ASSUMPTION] The S1 value matches S1 at the lower 7001 MHz memory clock "
              "(2.065 ms in run 20261106-134741), so it is probably the same memory-clock effect.", ""]

    # correctness table
    L += ["## Correctness (one line per stage run)", "",
          "Correct is each binary's own verdict (and exit status 0). The error column is the binary's reported error "
          "against its double-precision reference, where it reports one. No tolerance was changed for this run.", "",
          "| Stage | N | Correct | error vs double | binary's check |", "|---|---|---|---|---|"]
    for (w, dev, n), recs in g.items():
        for r in recs:
            err = r.get("err_pct")
            L.append(f"| {r['key'].rsplit('_N', 1)[0] if w == 'W04' else r['key']} | {n} | {r['correct']} | "
                     f"{'–' if err is None else f'{err:.3g}%'} | {check_text(r['key'])} |")
    w02 = [r for r in stages.values() if r["workload"] == "W02" and r.get("err_pct") is not None]
    if w02:
        L += ["", "**W02 accuracy.** Error vs the double-precision sum: " + "; ".join(
              f"{r['device']} {r['stage']} {r['err_pct']:.3g}%" for r in sorted(w02, key=lambda r: (r["device"], r["stage"]))) +
              ". Until P8, CPU S1, CPU S2 and GPU S1 accumulated in FP32 and were 6.75%, 0.41% and 6.8% off; each binary "
              "also used a different check (≤ 0.01% vs double, finite only, or vs an FP32 serial sum with the same error). "
              "P8 made them accumulate in double (FP32 inputs) and gave every W02 stage one check: ≤ 0.1% vs double. "
              "CPU S3 (FP32 AVX2 lanes) and GPU S2 (block reduction) were already within it and are unchanged. Their "
              "earlier times are therefore not directly comparable for CPU S1/S2 and GPU S1."]
    L.append("")

    # deviations
    if flagged:
        L += [f"## Deviations > {FLAG_PCT:.0f}% from the earlier recorded runs", ""]
        for r, ref, d in flagged:
            L.append(f"- ⚠ **{r['key']}**: {ms(r['mean'])} ms now vs {float(ref['mean_ms']):.4g} ms "
                     f"(`{ref['source']}`, {ref['date']}): {d:+.1f}%.")
        if NOTES_FILE.is_file():
            L += ["", NOTES_FILE.read_text().strip()]
        else:
            L += ["", "_Explanations: not written yet (docs/results/deviation_notes.md)._"]
        L.append("")
    L += ["## Files", "", f"- consolidated CSV: `docs/results/final_results.csv` (one row per stage and N)",
          "- earlier values: `docs/results/reference_values.csv`",
          f"- raw CSVs, logs, `runs.csv`, `gpu_state.csv`: `results/raw/FINAL/{run.name}/` (private repository)",
          "- reproduce: `bash scripts/run_all.sh`, then `python3 tools/final_summary.py results/raw/FINAL/<run> --publish-docs`", ""]
    return "\n".join(L), csv_rows, g, flagged


def figure(g, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except ImportError:
        print("  matplotlib not installed: figure skipped")
        return False
    items = []
    for (w, dev, n), recs in g.items():
        for r in recs:
            if r["stage"] != "REF" and r.get("x_s1"):
                lab = f"{w} {dev}{' N=' + n if w == 'W04' else ''} · {r['stage']}"
                items.append((lab, r["x_s1"], dev))
    if not items:
        return False
    col = {"CPU": "#2a78d6", "GPU": "#eb6834"}
    ink, ink2, grid, surf = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    fig, ax = plt.subplots(figsize=(8.6, 0.30 * len(items) + 1.4))
    fig.patch.set_facecolor(surf)
    ax.set_facecolor(surf)
    ys = list(range(len(items)))[::-1]
    ax.barh(ys, [x for _, x, _ in items], height=0.7, color=[col[d] for _, _, d in items], edgecolor=surf, linewidth=1.5)
    for y, (_, x, _) in zip(ys, items):
        ax.annotate(f"{x:.2f}×" if x < 10 else f"{x:.1f}×", (x, y), va="center", ha="left", fontsize=7.5, color=ink,
                    xytext=(3, 0), textcoords="offset points")
    ax.set_yticks(ys, [lab for lab, _, _ in items], fontsize=7.5, color=ink)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.set_xlim(0.5, max(x for _, x, _ in items) * 3)
    ax.axvline(1, color=ink2, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(grid)
    ax.grid(axis="x", color=grid, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=ink2)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=col["CPU"], label="CPU (i7-13650HX)"), Patch(color=col["GPU"], label="GPU (RTX 4060 Laptop, 55 W)")],
              loc="lower right", fontsize=8, frameon=False)
    ax.set_xlabel("speedup vs the workload's own S1 on the same device (log scale)", color=ink2, fontsize=9)
    ax.set_title("Every stage of W01–W04, one session", color=ink, fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=surf)
    plt.close(fig)
    return True


def print_table(stages):
    g = groups(stages)
    print(f"\n{'stage':<24} {'N':>9} {'ms':>11} {'± SD':>9} {'× vs S1':>8}  Correct  error vs double")
    for (w, dev, n), recs in g.items():
        for r in recs:
            err = r.get("err_pct")
            print(f"{r['key']:<24} {n:>9} {ms(r['mean']):>11} {ms(r.get('sd')):>9} "
                  f"{f(r.get('x_s1'), '{:.2f}×'):>8}  {r['correct']:<7}  {'–' if err is None else f'{err:.3g}%'}")
    bad = [r["key"] for r in stages.values() if r["correct"] != "YES"]
    print(f"\n{len(stages)} stages, {len(stages) - len(bad)} Correct YES" + (f"; not YES: {', '.join(bad)}" if bad else ""))


def check_recorded():
    if not REF_FILE.is_file():
        sys.exit(f"missing {REF_FILE}")
    n_ok = n_skip = n_bad = 0
    for r in rows_of(REF_FILE):
        src = REPO / r["source"]
        if not src.is_file():
            n_skip += 1
            continue
        d = parse(r["key"], src)
        got = d["mean"] if d else None
        ok = got is not None and abs(got - float(r["mean_ms"])) <= 1e-3 * float(r["mean_ms"])
        n_ok += ok
        n_bad += not ok
        print(f"  {'ok ' if ok else 'BAD'} {r['key']:<14} recorded {float(r['mean_ms']):10.4f}  parsed {f(got, '{:10.4f}')}"
              f"  correct={d['correct'] if d else '?'}  {r['source']}")
    print(f"check-recorded: {n_ok} ok, {n_bad} mismatch, {n_skip} skipped (source not in this checkout, or a sweep-summary value)")
    return 1 if n_bad else 0


def main():
    a = sys.argv[1:]
    if not a or a[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if a[0] == "--check-recorded":
        return check_recorded()
    run = Path(a[0]).resolve()
    if not (run / "runs.csv").is_file():
        sys.exit(f"{run}: no runs.csv (not a run_all.sh folder)")
    stages = load_run(run)
    md, rows, g, flagged = build(run, stages, "--publish-docs" in a)
    (run / "RESULTS_SUMMARY.md").write_text(md + "\n")
    with open(run / "final_results.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        wr.writeheader()
        wr.writerows(rows)
    fig = run / "speedup_all.png"
    has_fig = figure(g, fig)
    print_table(stages)
    print(f"\nwritten: {run / 'RESULTS_SUMMARY.md'}, {run / 'final_results.csv'}" + (f", {fig}" if has_fig else ""))
    if flagged:
        print(f"deviations > {FLAG_PCT:.0f}% vs earlier runs: " + ", ".join(f"{r['key']} {d:+.0f}%" for r, _, d in flagged))
    if "--publish-docs" in a:
        if run.name.endswith("-smoke"):
            sys.exit("refusing --publish-docs for a smoke run")
        (REPO / "docs/results").mkdir(parents=True, exist_ok=True)
        shutil.copy(run / "final_results.csv", REPO / "docs/results/final_results.csv")
        (REPO / "docs/RESULTS_SUMMARY.md").write_text(md + "\n")
        if has_fig:
            (REPO / "docs/figures/final").mkdir(parents=True, exist_ok=True)
            shutil.copy(fig, REPO / "docs/figures/final/speedup_all.png")
        print("published: docs/RESULTS_SUMMARY.md, docs/results/final_results.csv, docs/figures/final/speedup_all.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
