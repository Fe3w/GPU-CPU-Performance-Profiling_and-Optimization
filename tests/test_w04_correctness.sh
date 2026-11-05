#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# W04 ATTENTION — correctness gate (must pass before any W04 commit)
#
#   1. make w04
#   2. CPU self-test of the shared harness (reference, generator, metrics, CLI, CSV,
#      float32 stable/unstable controls)
#   3. GPU self-test (device facts, rounding helpers vs CUDA, timing protocol)
#   4. every registered stage: n in {128, 256} x input in {normal, large_logits},
#      small bh, 1 iteration; requires exit 0 and the exact line "Correct : YES"
#
# Output goes to build/test_out/<timestamp>/ (git-ignored), never to results/.
# Must finish in under 60 s.
# -----------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."

START=$SECONDS
TS=$(date +%Y%m%d-%H%M%S)
OUT="build/test_out/$TS"
mkdir -p "$OUT"
PASS=0
FAIL=0

ok()  { echo "  PASS  $*"; PASS=$((PASS + 1)); }
bad() { echo "  FAIL  $*"; FAIL=$((FAIL + 1)); }

# Runs a command, logs it, and requires exit code 0 and that the LAST
# "Correct : ..." line is exactly "Correct : YES" (an earlier YES from a
# sub-check must not mask a final NO).
run_check() {
    local label=$1 log=$2; shift 2
    if "$@" >"$log" 2>&1 && [[ "$(grep -E '^Correct : ' "$log" | tail -n 1)" == "Correct : YES" ]]; then
        ok "$label"
    else
        bad "$label  (log: $log)"
        tail -n 15 "$log" | sed 's/^/        /'
    fi
}

echo "== build =="
make --no-print-directory w04

echo "== harness self-tests =="
run_check "CPU self-test" "$OUT/selftest.log" build/W04_ATTENTION_SELFTEST "$OUT/selftest"
run_check "GPU self-test" "$OUT/selftest_gpu.log" build/W04_ATTENTION_SELFTEST_GPU
echo "  -- float32 controls (CPU self-test) --"
grep -E "^\s+n=|negative controls" "$OUT/selftest.log" | sed 's/^/      /' || true
echo "  -- device facts (GPU self-test) --"
grep -E "L2 cache|smem per|SMs " "$OUT/selftest_gpu.log" | sed 's/^/      /' || true

# Registered stages, one per line: "<binary name> <tunable: 0|1>".
# Added phase by phase (P1: CPU, P2: GPU S1/S2a/REF_CUBLAS, P3: S2b, P4: S3).
# A registered stage whose binary is missing is a FAIL.
STAGES=(
)

echo "== stages (${#STAGES[@]} registered) =="
for entry in "${STAGES[@]}"; do
    read -r name tunable <<<"$entry"
    bin="build/$name"
    if [[ ! -x "$bin" ]]; then
        bad "$name  (binary missing: $bin)"
        continue
    fi
    for n in 128 256; do
        for input in normal large_logits; do
            log="$OUT/${name}_n${n}_${input}.log"
            run_check "$name n=$n $input" "$log" \
                "$bin" --n "$n" --bh 2 --d 64 --iters 1 --reps 1 --warmup 0 \
                --input "$input" --out "$OUT/${name}.csv"
        done
    done
done

ELAPSED=$((SECONDS - START))
echo "== summary: $PASS passed, $FAIL failed, ${ELAPSED}s (logs: $OUT) =="
if (( ELAPSED > 60 )); then
    echo "FAIL: correctness test took ${ELAPSED}s (limit 60s)"
    exit 1
fi
(( FAIL == 0 ))
