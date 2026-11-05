#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# W04 ATTENTION — correctness gate (must pass before any W04 commit)
#
#   1. make w04
#   2. CPU self-test of the shared harness (reference, generator, metrics, CLI, CSV,
#      float32 stable/unstable controls)
#   3. GPU self-test (device facts, rounding helpers vs CUDA, timing protocol)
#   4. AVX2 exp accuracy test (exhaustive on the softmax range)
#   5. every registered stage: n in {128, 256} x input in {normal, large_logits},
#      small bh, 1 iteration; requires exit 0 and the exact line "Correct : YES".
#      Tunable stages also run two ragged-tile cases (N not a multiple of Br/Bc,
#      one with d = 40)
#   6. machine-code evidence: CPU S3 must contain packed-single ymm FMAs
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
WARNINGS=$(grep -hE '^WARN' "$OUT/selftest.log" || true)

echo "== AVX2 exp =="
run_check "exp256_ps accuracy" "$OUT/exp_avx2.log" build/W04_EXP_AVX2_TEST
grep -E "ulp at" "$OUT/exp_avx2.log" | sed 's/^/      /' || true

# Registered stages, one per line: "<binary name> <tunable: 0|1>".
# Added phase by phase (P1: CPU, P2: GPU S1/S2a/REF_CUBLAS, P3: S2b, P4: S3).
# A registered stage whose binary is missing is a FAIL.
STAGES=(
    "W04_ATTENTION_CPU_S1_BASE 0"
    "W04_ATTENTION_CPU_S2_SWOPT_BLOCKED_OMP 1"
    "W04_ATTENTION_CPU_S3_HWACC_AVX2 1"
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
    if [[ "$tunable" == 1 ]]; then
        for shape in "--n 200 --d 64 --br 32 --bc 48" "--n 203 --d 40 --br 5 --bc 24"; do
            for input in normal large_logits; do
                tag=$(echo "$shape" | tr -d ' -' )
                log="$OUT/${name}_${tag}_${input}.log"
                # shellcheck disable=SC2086
                run_check "$name ragged ($shape) $input" "$log" \
                    "$bin" $shape --bh 2 --iters 1 --reps 1 --warmup 0 \
                    --input "$input" --out "$OUT/${name}.csv"
            done
        done
    fi
done

echo "== machine code =="
S3_BIN=build/W04_ATTENTION_CPU_S3_HWACC_AVX2
if [[ -x "$S3_BIN" ]]; then
    nfma=$(objdump -d --no-show-raw-insn "$S3_BIN" | grep -cE 'vfmadd[0-9]+ps.*ymm' || true)
    if (( nfma > 0 )); then ok "CPU S3 contains $nfma ymm vfmadd*ps instructions"
    else bad "CPU S3 contains no ymm vfmadd*ps instructions"; fi
fi

ELAPSED=$((SECONDS - START))
[[ -n "$WARNINGS" ]] && echo "$WARNINGS"
echo "== summary: $PASS passed, $FAIL failed, ${ELAPSED}s (logs: $OUT) =="
if (( ELAPSED > 60 )); then
    echo "FAIL: correctness test took ${ELAPSED}s (limit 60s)"
    exit 1
fi
(( FAIL == 0 ))
