#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."

NCU="/opt/nvidia/nsight-compute/2026.3.0/ncu"
OUT="$PWD/profiling/gui_demo"

mkdir -p "$OUT/csv"

profile() {
    local name="$1"
    local launch_count="$2"
    shift 2

    echo
    echo "===== Nsight Compute: $name ====="
    "$NCU" --set basic -c "$launch_count" \
        -o "$OUT/$name" "$@"
}

# W01: Vector Add
profile W01_S1 1 \
    ./build/W01_VECTOR_ADD_GPU_S1_BASE_NAIVE \
    33554432 100 5 2 "$OUT/csv/W01_S1.csv"

profile W01_S2 1 \
    ./build/W01_VECTOR_ADD_GPU_S2_SWOPT_COALESCED \
    33554432 100 5 2 128 "$OUT/csv/W01_S2.csv"

# W02: Dot Product
profile W02_S1 1 \
    ./build/W02_DOT_PRODUCT_GPU_S1_BASE_NAIVE \
    --size 33554432 --iterations 20 --repetitions 5 \
    --warmup 2 --output "$OUT/csv/W02_S1.csv"

profile W02_S2 2 \
    ./build/W02_DOT_PRODUCT_GPU_S2_SWOPT_BLOCK_REDUCE \
    33554432 20 5 2 "$OUT/csv/W02_S2.csv"

# W03: GEMM
profile W03_S1 1 \
    ./build/W03_GEMM_GPU_S1_BASE_NAIVE \
    1024 20 5 2 "$OUT/csv/W03_S1.csv"

profile W03_S2 1 \
    ./build/W03_GEMM_GPU_S2_SWOPT_SHARED_TILE \
    1024 20 5 2 "$OUT/csv/W03_S2.csv"

profile W03_S3 1 \
    ./build/W03_GEMM_GPU_S3_HWACC_TENSOR_CORE \
    1024 20 5 2 "$OUT/csv/W03_S3.csv"

echo
echo "Reports saved under: $OUT"
find "$OUT" -maxdepth 1 -type f -name '*.ncu-rep*' -print
