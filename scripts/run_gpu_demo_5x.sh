#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."

mkdir -p results/demo_5x/logs results/demo_5x/csv

run_benchmark() {
    local name="$1"
    shift

    echo
    echo "=========================================="
    echo "Running: $name"
    echo "=========================================="

    "$@" 2>&1 | tee "results/demo_5x/logs/${name}.log"
}

# W01: Vector Addition
run_benchmark W01_S1 \
    ./build/W01_VECTOR_ADD_GPU_S1_BASE_NAIVE \
    167772160 100 5 2 results/demo_5x/csv/W01_S1.csv

run_benchmark W01_S2 \
    ./build/W01_VECTOR_ADD_GPU_S2_SWOPT_COALESCED \
    167772160 100 5 2 128 results/demo_5x/csv/W01_S2.csv

# W02: Dot Product
run_benchmark W02_S1 \
    ./build/W02_DOT_PRODUCT_GPU_S1_BASE_NAIVE \
    --size 167772160 \
    --iterations 20 \
    --repetitions 5 \
    --warmup 2 \
    --output results/demo_5x/csv/W02_S1.csv

run_benchmark W02_S2 \
    ./build/W02_DOT_PRODUCT_GPU_S2_SWOPT_BLOCK_REDUCE \
    167772160 20 5 2 results/demo_5x/csv/W02_S2.csv

# W03: Matrix Multiplication (GEMM)
run_benchmark W03_S1 \
    ./build/W03_GEMM_GPU_S1_BASE_NAIVE \
    1792 20 5 2 results/demo_5x/csv/W03_S1.csv

run_benchmark W03_S2 \
    ./build/W03_GEMM_GPU_S2_SWOPT_SHARED_TILE \
    1792 20 5 2 results/demo_5x/csv/W03_S2.csv

run_benchmark W03_S3 \
    ./build/W03_GEMM_GPU_S3_HWACC_TENSOR_CORE \
    1792 20 5 2 results/demo_5x/csv/W03_S3.csv

# Extract timings from the console logs, not the last CSV row.
python3 - <<'PY'
import re
from pathlib import Path

entries = [
    ("W01_S1", r"Runtime/iter\s*:\s*([\d.]+)\s*ms"),
    ("W01_S2", r"Runtime/iter\s*:\s*([\d.]+)\s*ms"),
    ("W02_S1", r"Mean time\s*\(ms\)\s*:\s*([\d.]+)"),
    ("W02_S2", r"Mean time\s*\(ms\)\s*:\s*([\d.]+)"),
    ("W03_S1", r"Mean time\s*\(ms\)\s*:\s*([\d.]+)"),
    ("W03_S2", r"Mean time\s*\(ms\)\s*:\s*([\d.]+)"),
    ("W03_S3", r"Mean time\s*\(ms\)\s*:\s*([\d.]+)"),
]

print("\n\n========== DEMO TIMING SUMMARY ==========")
print(f"{'Benchmark':<12} {'Runtime (ms)':>14}")
print("-" * 28)

for name, pattern in entries:
    log = Path("results/demo_5x/logs") / f"{name}.log"
    matches = re.findall(pattern, log.read_text()) if log.exists() else []

    if matches:
        print(f"{name:<12} {float(matches[-1]):>14.4f}")
    else:
        print(f"{name:<12} {'NOT FOUND':>14}")

print("\nCSV files: results/demo_5x/csv/")
print("Logs:      results/demo_5x/logs/")
PY
