#!/usr/bin/env python3
"""E4 source view: which SASS opcodes carry each warp-stall reason.

    python3 tools/w04_e4_source.py <name>.source.csv [--top 8]

Reads an ncu SourceCounters export (`ncu --import X --page source --csv
--print-source sass`). For the stall reasons below, sums the warp-state
samples per opcode (the mnemonic up to the first '.', e.g. HMMA, MUFU, FFMA)
and prints each opcode's share of that reason's samples, plus the opcode's
share of all samples and of executed warp instructions.

A sample is attributed to the instruction the warp is waiting to issue, so
a stall on an instruction names its *consumer*, not necessarily the unit
that is busy (e.g. a short-scoreboard stall on an FFMA waits for an earlier
MUFU or shared load).

Standard library only. Reads CSVs; never writes.
"""
import csv
import sys
from collections import defaultdict

REASONS = ("stall_math", "stall_short_sb", "stall_wait", "stall_long_sb", "stall_barrier", "stall_mio")


def num(s):
    try:
        return float(s.replace(",", ""))
    except (ValueError, AttributeError):
        return 0.0


def opcode(src):
    tok = src.strip().split()
    if not tok:
        return "?"
    if tok[0].startswith("@"):          # predicate guard, e.g. @!P0
        tok = tok[1:] or ["?"]
    return tok[0].split(".")[0]


def main():
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    top = 8
    if "--top" in args:
        i = args.index("--top")
        top = int(args[i + 1])
        del args[i:i + 2]
    rows = list(csv.reader(open(args[0], newline="")))
    hi = next(i for i, r in enumerate(rows) if "Source" in r and "Address" in r)
    h = rows[hi]
    col = {c: h.index(c) for c in h}
    for need in ("Source", "Instructions Executed", "Warp Stall Sampling (All Samples)") + REASONS:
        if need not in col:
            sys.exit(f"column '{need}' missing in {args[0]}")
    per = defaultdict(lambda: defaultdict(float))
    for r in rows[hi + 1:]:
        if len(r) < len(h):
            continue
        op = opcode(r[col["Source"]])
        per[op]["inst"] += num(r[col["Instructions Executed"]])
        per[op]["all"] += num(r[col["Warp Stall Sampling (All Samples)"]])
        for k in REASONS:
            per[op][k] += num(r[col[k]])
    tot = {k: sum(v[k] for v in per.values()) for k in ("inst", "all") + REASONS}
    print(f"== {args[0]}")
    print(f"total samples {tot['all']:,.0f}; warp instructions {tot['inst']:,.0f}")
    for k in REASONS:
        if tot[k] == 0:
            continue
        print(f"\n{k}: {tot[k]:,.0f} samples ({100 * tot[k] / tot['all']:.1f}% of all)")
        print(f"  {'opcode':10s} {'% of reason':>11s} {'% all smp':>10s} {'% instr':>8s}")
        for op, v in sorted(per.items(), key=lambda kv: -kv[1][k])[:top]:
            if v[k] == 0:
                break
            print(f"  {op:10s} {100 * v[k] / tot[k]:11.1f} {100 * v['all'] / tot['all']:10.1f}"
                  f" {100 * v['inst'] / max(tot['inst'], 1):8.1f}")


if __name__ == "__main__":
    main()
