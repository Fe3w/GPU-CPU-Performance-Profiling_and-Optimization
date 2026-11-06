#!/usr/bin/env python3
"""W04 E5: GPU idle gaps between the kernels of one forward pass (Nsight Systems).

    python3 tools/w04_nsys_gaps.py <e5_dir>

Reads every <name>_cuda_gpu_trace.csv in <e5_dir> (from `nsys stats --report
cuda_gpu_trace --format csv`). Kernel rows are the rows with a grid size;
memcpy/memset rows are ignored, and so is the FP16 prepare kernel
(convert_qkv_half), which is timed separately (prep_ms).

A forward pass starts at every launch of the first kernel of the trace
(S1/S2a/REF: Q·Kᵀ, then softmax, then P·V; S2b/S3: one kernel). For each pass:
  span    first kernel start -> last kernel end
  busy    sum of kernel durations
  intra   span - busy: idle time between the kernels of the pass
  inter   end of this pass -> start of the next pass
Which passes are timed follows the harness order (src/common/w04_gpu_stage.cuh,
time_protocol_gpu): WARMUP warm-up passes and a device sync, then the ITERS
timed passes launched back to back (one event pair around all of them), then
a per-phase breakdown pass set (an event before each kernel and a host sync
after every pass) and the correctness-check passes. Only passes
WARMUP .. WARMUP+ITERS-1 (the E5 runs use --warmup 2 --iters 20) are used for
the medians; the gap after the last timed pass is not counted.

Printed per run: passes found, kernels per pass, median span, busy, intra
and inter gaps (µs), and the gap share = (intra + inter) / (span + inter),
i.e. the fraction of back-to-back time in which the GPU runs no kernel.

Standard library only. Reads CSVs; never writes.
"""
import csv
import glob
import os
import statistics
import sys

WARMUP, ITERS = 2, 20   # the E5 protocol in scripts/w04_profile.sh
SKIP_KERNELS = ("convert_qkv_half",)


def kernels(path):
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if not r.get("GrdX"):
                continue
            name = r["Name"]
            if any(s in name for s in SKIP_KERNELS):
                continue
            start = int(r["Start (ns)"])
            out.append((start, start + int(r["Duration (ns)"]), name))
    out.sort()
    return out


def passes(ks):
    if not ks:
        return []
    first = ks[0][2]
    res, cur = [], []
    for k in ks:
        if k[2] == first and cur:
            res.append(cur)
            cur = []
        cur.append(k)
    res.append(cur)
    return res


def short(name):
    """'void w04::flash_fwd_tc<w04::PrecFP16, 64, 32, 4>(...)' -> 'flash_fwd_tc'"""
    base = name.replace("<unnamed>::", "").split("(")[0].split("<")[0].strip()
    return base.split("::")[-1].split()[-1] if base else name


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    files = sorted(glob.glob(os.path.join(sys.argv[1], "*_cuda_gpu_trace.csv")))
    if not files:
        print(f"no *_cuda_gpu_trace.csv in {sys.argv[1]}")
        return 1
    print(f"E5 launch gaps; medians over the {ITERS} timed passes (after {WARMUP} warm-ups); times in µs")
    print(f"{'run':<22} {'passes':>6} {'k/pass':>6} {'span':>10} {'busy':>10} {'intra':>8} {'inter':>8} "
          f"{'gap %':>6}  kernels")
    for path in files:
        name = os.path.basename(path)[: -len("_cuda_gpu_trace.csv")]
        ps = passes(kernels(path))
        if len(ps) < WARMUP + ITERS:
            print(f"{name:<22} only {len(ps)} passes found, need {WARMUP + ITERS}")
            continue
        tail = ps[WARMUP:WARMUP + ITERS]
        kpp = statistics.mode(len(p) for p in tail)
        span = [(p[-1][1] - p[0][0]) / 1e3 for p in tail]
        busy = [sum(e - s for s, e, _ in p) / 1e3 for p in tail]
        intra = [a - b for a, b in zip(span, busy)]
        inter = [(tail[i + 1][0][0] - tail[i][-1][1]) / 1e3 for i in range(len(tail) - 1)]
        med = statistics.median
        m_span, m_busy, m_intra = med(span), med(busy), med(intra)
        m_inter = med(inter) if inter else 0.0
        share = 100.0 * (m_intra + m_inter) / (m_span + m_inter)
        names = " > ".join(short(k[2]) for k in tail[-1])
        print(f"{name:<22} {len(ps):>6} {kpp:>6} {m_span:>10.1f} {m_busy:>10.1f} {m_intra:>8.2f} "
              f"{m_inter:>8.2f} {share:>6.2f}  {names}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
