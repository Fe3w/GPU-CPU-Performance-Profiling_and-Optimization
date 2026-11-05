#!/usr/bin/env python3
"""Compare two W04 output dumps (raw float32, written by W04_DUMP_O).

    python3 tools/w04_compare_o.py <O_a.bin> <O_b.bin> [--tol 1e-4]

Prints the max absolute difference and the normwise relative difference
max|O_a - O_b| / max|O_b| (the same normwise definition as max_rel_err, C3).
Exits 0 if both files have the same size, every value is finite and the
relative difference is <= tol; 1 otherwise. Standard library only.
"""
import argparse
import math
import sys
from array import array


def load(path):
    a = array("f")
    with open(path, "rb") as f:
        a.frombytes(f.read())
    if sys.byteorder != "little":
        a.byteswap()
    return a


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("a")
    ap.add_argument("b", help="reference side of the relative difference")
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()

    a, b = load(args.a), load(args.b)
    if len(a) != len(b) or len(a) == 0:
        print(f"compare : size mismatch or empty ({len(a)} vs {len(b)} floats)")
        print("Correct : NO")
        return 1
    nonfinite = 0
    max_abs = 0.0
    max_ref = 0.0
    for x, y in zip(a, b):
        if not (math.isfinite(x) and math.isfinite(y)):
            nonfinite += 1
            continue
        max_abs = max(max_abs, abs(x - y))
        max_ref = max(max_ref, abs(y))
    rel = math.inf if nonfinite or max_ref == 0.0 else max_abs / max_ref
    ok = rel <= args.tol
    print(f"compare : {len(a)} floats  max_abs={max_abs:.3e}  max_rel(normwise)={rel:.3e}  "
          f"nonfinite={nonfinite}  tol={args.tol:g}")
    print(f"Correct : {'YES' if ok else 'NO'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
