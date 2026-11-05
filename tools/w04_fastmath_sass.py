#!/usr/bin/env python3
"""SASS comparison of two builds of one kernel (P5 GPU fast-math study).

    python3 tools/w04_fastmath_sass.py <accurate binary> <fast-math binary> <instance> [<instance> ...]

<instance> is "BRxBCwW", e.g. 64x64w8; it selects the S2b kernel
flash_fwd_fp32<Br, Bc, W> in each binary. For every instance and build it
prints the instruction counts that tell accurate and fast exp/division apart,
then the 6 instructions before the first two MUFU.EX2 of each build.

Rule 6 of the project: accurate expf also compiles to MUFU.EX2 (with a
range reduction around it), so "MUFU.EX2 present" proves nothing on its own;
the evidence is the shorter sequence around it and the count of supporting
instructions. Division: the IEEE-rounded x / y is a call to a slow-path
subroutine (CALL) around MUFU.RCP; -prec-div=false leaves MUFU.RCP + FMUL.

Standard library only; needs cuobjdump on PATH. Writes to stdout only.
"""
import re
import subprocess
import sys
from collections import Counter

CLASSES = ["MUFU.EX2", "MUFU.RCP", "FFMA", "FMUL", "FADD", "FSETP", "FSEL", "CALL", "BRA", "LDS", "SHFL", "STL"]


def functions(binary):
    """{mangled name: [opcode with modifiers, ...]} from cuobjdump -sass."""
    out = subprocess.run(["cuobjdump", "-sass", binary], check=True, capture_output=True, text=True).stdout
    funcs, cur = {}, None
    for line in out.splitlines():
        m = re.search(r"Function : (\S+)", line)
        if m:
            cur = funcs.setdefault(m.group(1), [])
            continue
        m = re.match(r"\s*/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)([^;]*);", line)
        if m and cur is not None:
            cur.append((m.group(1), (m.group(1) + m.group(2)).strip()))
    return funcs


def pick(funcs, br, bc, w):
    key = f"flash_fwd_fp32ILi{br}ELi{bc}ELi{w}E"
    names = [n for n in funcs if key in n]
    if len(names) != 1:
        raise SystemExit(f"instance {br}x{bc}w{w}: found {len(names)} kernels matching {key}")
    return funcs[names[0]]


def counts(instrs):
    c = Counter()
    for op, _ in instrs:
        for k in CLASSES:
            if op == k or op.startswith(k + "."):
                c[k] += 1
    return c


def main(argv):
    if len(argv) < 4:
        print(__doc__)
        return 2
    acc_bin, fm_bin = argv[1], argv[2]
    builds = [("accurate", functions(acc_bin)), ("fast-math", functions(fm_bin))]
    print(f"accurate : {acc_bin}\nfast-math: {fm_bin}")
    for inst in argv[3:]:
        m = re.fullmatch(r"(\d+)x(\d+)w(\d+)", inst)
        if not m:
            raise SystemExit(f"bad instance '{inst}' (expected e.g. 64x64w8)")
        br, bc, w = map(int, m.groups())
        print(f"\n== flash_fwd_fp32<{br}, {bc}, {w}> ==")
        print(f"  {'build':<10} {'total':>6} " + " ".join(f"{k:>9}" for k in CLASSES))
        per = {}
        for name, funcs in builds:
            ins = pick(funcs, br, bc, w)
            c = counts(ins)
            per[name] = ins
            print(f"  {name:<10} {len(ins):>6} " + " ".join(f"{c[k]:>9}" for k in CLASSES))
        for name, ins in per.items():
            idx = [i for i, (op, _) in enumerate(ins) if op.startswith("MUFU.EX2")][:2]
            for k, i in enumerate(idx, 1):
                print(f"  {name}: MUFU.EX2 #{k} and the 6 instructions before it")
                for _, text in ins[max(0, i - 6):i + 1]:
                    print(f"      {text}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
