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
#      Every stage also runs n = 203, d = 40 (N not a multiple of any tile, d not
#      a multiple of 16/64). Tunable stages also run ragged-tile cases
#      (N not a multiple of Br/Bc, one with d = 40). Kind 1 = any Br/Bc (CPU);
#      kind 2 = compiled tile set Br, Bc in {16, 32, 64}: all 9 tiles at n = 203
#      plus one d = 40 case; kind 3 = kind 2 with a warps axis (P5, C27; GPU
#      S2b, S3): every (Br, Bc, W) of Br, Bc in {16, 32, 64} x W in {1, 2, 4, 8}
#      is run at n = 203; refused cells (exit 2 + reason) are counted, and the
#      number of runnable cells must equal the expected count (S2b 27, S3 12
#      per precision), so a lost instance fails
#   5b. warps change only the thread mapping, not the arithmetic: at every tile
#      the O of each runnable W must be bitwise equal to the O of the native W
#      (S2b: 8, S3: Br/16). One measured exception: S2b at Br = 16, where nvcc
#      contracts multiply-adds differently for one row per thread (W = 8) than
#      for 2 or 4 (W = 4, 2); with -fmad=false those cells are bitwise equal
#      too (P5 step 1). They must agree within 1e-6 normwise and are listed
#   5c. tuned defaults (C29): a run without --br/--bc/--warps must use the
#      w04_tuned.hpp entry nearest to its n (checked at n = 256 and 4096)
#   6. direct stage-vs-stage checks at n = 1024 (O dumped via W04_DUMP_O,
#      compared by tools/w04_compare_o.py, normwise rel): GPU S2b vs S2a
#      (<= 1e-4); GPU S3 FP16 / TF32 vs S2b at 64x64 (<= the stage tolerance,
#      1e-2 / 5e-3) on normal inputs. On large_logits the S3-vs-S2b difference
#      is the input-rounding ("plain") error, which decision C2 reports but does
#      not gate (decision C26), so it is printed as INFO only; S3 large_logits
#      is gated against the rounded-input double reference in step 5
#   7. machine-code evidence: CPU S3 must contain packed-single ymm FMAs; GPU
#      S2b must use shared memory (LDS) and shuffles (SHFL), and no HMMA; GPU
#      S3 FP16 must contain HMMA.16816.F32 and no TF32 HMMA, S3 TF32 must
#      contain HMMA.1684.F32.TF32 and no other HMMA form
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

# Registered stages, one per line: "<binary name> <tunable: 0|1|2>".
# Added phase by phase (P1: CPU, P2: GPU S1/S2a/REF_CUBLAS, P3: S2b, P4: S3).
# A registered stage whose binary is missing is a FAIL.
STAGES=(
    "W04_ATTENTION_CPU_S1_BASE 0"
    "W04_ATTENTION_CPU_S2_SWOPT_BLOCKED_OMP 1"
    "W04_ATTENTION_CPU_S3_HWACC_AVX2 1"
    "W04_ATTENTION_GPU_S1_BASE_NAIVE 0"
    "W04_ATTENTION_GPU_S2a_SWOPT_UNFUSED 0"
    "W04_ATTENTION_GPU_REF_CUBLAS 0"
    "W04_ATTENTION_GPU_S2b_SWOPT_FUSED 3"
    "W04_ATTENTION_GPU_S3_HWACC_TENSOR_CORE_FP16 3"
    "W04_ATTENTION_GPU_S3_HWACC_TENSOR_CORE_TF32 3"
)
# Kind 3: runnable (Br, Bc, W) cells expected per binary, and the native warps of a tile.
declare -A EXPECTED_CELLS=(
    [W04_ATTENTION_GPU_S2b_SWOPT_FUSED]=27
    [W04_ATTENTION_GPU_S3_HWACC_TENSOR_CORE_FP16]=12
    [W04_ATTENTION_GPU_S3_HWACC_TENSOR_CORE_TF32]=12
)
native_warps() {  # <binary name> <br>
    if [[ "$1" == *S2b* ]]; then echo 8; else echo $(( $2 / 16 )); fi
}

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
    for input in normal large_logits; do
        log="$OUT/${name}_n203d40_${input}.log"
        run_check "$name n=203 d=40 $input" "$log" \
            "$bin" --n 203 --d 40 --bh 2 --iters 1 --reps 1 --warmup 0 \
            --input "$input" --out "$OUT/${name}.csv"
    done
    shapes=()
    if [[ "$tunable" == 1 ]]; then
        shapes=("--n 200 --d 64 --br 32 --bc 48" "--n 203 --d 40 --br 5 --bc 24")
    elif [[ "$tunable" == 2 ]]; then
        for br in 16 32 64; do for bc in 16 32 64; do shapes+=("--n 203 --d 64 --br $br --bc $bc"); done; done
        shapes+=("--n 203 --d 40 --br 64 --bc 16")
    elif [[ "$tunable" == 3 ]]; then
        shapes+=("--n 203 --d 40 --br 64 --bc 16" "--n 203 --d 40 --br 64 --bc 16 --warps 2")
        cells=0 refused=0
        for br in 16 32 64; do for bc in 16 32 64; do for w in 1 2 4 8; do
            tag="br${br}bc${bc}w${w}"
            probe="$OUT/${name}_${tag}_probe.log"
            "$bin" --n 203 --d 64 --bh 2 --br "$br" --bc "$bc" --warps "$w" --iters 1 --reps 1 --warmup 0 \
                --input normal --out "$OUT/${name}.csv" >"$probe" 2>&1 && rc=0 || rc=$?
            if (( rc == 2 )) && grep -q ': error: ' "$probe"; then
                refused=$((refused + 1))
                continue
            fi
            cells=$((cells + 1))
            for input in normal large_logits; do
                log="$OUT/${name}_${tag}_${input}.log"
                dump=()
                [[ "$input" == normal ]] && dump=(env "W04_DUMP_O=$OUT/O_${name}_${tag}.bin")
                run_check "$name (Br=$br Bc=$bc W=$w, n=203) $input" "$log" \
                    "${dump[@]}" "$bin" --n 203 --d 64 --bh 2 --br "$br" --bc "$bc" --warps "$w" \
                    --iters 1 --reps 1 --warmup 0 --input "$input" --out "$OUT/${name}.csv"
            done
        done; done; done
        if (( cells == EXPECTED_CELLS[$name] )); then
            ok "$name: $cells runnable (Br, Bc, W) cells, $refused refused with a reason (expected ${EXPECTED_CELLS[$name]})"
        else
            bad "$name: $cells runnable cells (expected ${EXPECTED_CELLS[$name]}), $refused refused"
        fi
        # 5b: O is bitwise independent of the warp count.
        same=0 diff=0 near=0
        for br in 16 32 64; do for bc in 16 32 64; do
            nat="$OUT/O_${name}_br${br}bc${bc}w$(native_warps "$name" "$br").bin"
            for w in 1 2 4 8; do
                f="$OUT/O_${name}_br${br}bc${bc}w${w}.bin"
                [[ -f "$f" && "$f" != "$nat" ]] || continue
                if [[ -f "$nat" ]] && cmp -s "$f" "$nat"; then
                    same=$((same + 1))
                elif [[ -f "$nat" && "$name" == *S2b* && "$br" == 16 ]] &&
                     python3 tools/w04_compare_o.py "$f" "$nat" --tol 1e-6 >"$OUT/cmp_w_${name}_${br}_${bc}_${w}.log" 2>&1; then
                    near=$((near + 1))
                    echo "  INFO  $name Br=$br Bc=$bc W=$w vs W=8: not bitwise (FMA contraction)," \
                        "$(grep -oE 'max_rel\(normwise\)=[^ ]+' "$OUT/cmp_w_${name}_${br}_${bc}_${w}.log")"
                else
                    diff=$((diff + 1))
                fi
            done
        done; done
        if (( diff == 0 && same > 0 )); then
            ok "$name: O bitwise equal across warp counts ($same cells; $near listed Br=16 cells within 1e-6)"
        else bad "$name: O differs from the native warp count in $diff cells ($same equal, $near within 1e-6)"; fi
    fi
    if (( ${#shapes[@]} > 0 )); then
        for shape in "${shapes[@]}"; do
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

echo "== S2b vs S2a (n=1024, direct O comparison) =="
G2A_BIN=build/W04_ATTENTION_GPU_S2a_SWOPT_UNFUSED
G2B_BIN=build/W04_ATTENTION_GPU_S2b_SWOPT_FUSED
if [[ -x "$G2A_BIN" && -x "$G2B_BIN" ]]; then
    for input in normal large_logits; do
        for b in "$G2A_BIN" "$G2B_BIN"; do
            nm=$(basename "$b")
            W04_DUMP_O="$OUT/O_${nm}_n1024_${input}.bin" run_check "$nm n=1024 $input (dump O)" \
                "$OUT/${nm}_n1024_${input}.log" \
                "$b" --n 1024 --bh 2 --d 64 --iters 1 --reps 1 --warmup 0 --input "$input" --out "$OUT/${nm}.csv"
        done
        run_check "S2b vs S2a n=1024 $input (normwise rel <= 1e-4)" "$OUT/cmp_s2b_s2a_n1024_${input}.log" \
            python3 tools/w04_compare_o.py "$OUT/O_W04_ATTENTION_GPU_S2b_SWOPT_FUSED_n1024_${input}.bin" \
            "$OUT/O_W04_ATTENTION_GPU_S2a_SWOPT_UNFUSED_n1024_${input}.bin" --tol 1e-4
        grep -E '^compare' "$OUT/cmp_s2b_s2a_n1024_${input}.log" | sed 's/^/      /' || true
    done
else
    bad "S2b vs S2a comparison (binary missing)"
fi

echo "== S3 vs S2b 64x64 (n=1024, direct O comparison, stage tolerance) =="
G3_FP16_BIN=build/W04_ATTENTION_GPU_S3_HWACC_TENSOR_CORE_FP16
G3_TF32_BIN=build/W04_ATTENTION_GPU_S3_HWACC_TENSOR_CORE_TF32
if [[ -x "$G2B_BIN" && -x "$G3_FP16_BIN" && -x "$G3_TF32_BIN" ]]; then
    for input in normal large_logits; do
        ref="$OUT/O_S2b_64x64_n1024_${input}.bin"
        W04_DUMP_O="$ref" run_check "S2b 64x64 n=1024 $input (dump O)" "$OUT/S2b_64x64_n1024_${input}.log" \
            "$G2B_BIN" --n 1024 --bh 2 --d 64 --br 64 --bc 64 --iters 1 --reps 1 --warmup 0 --input "$input" \
            --out "$OUT/S2b_64x64.csv"
        for pair in "FP16 $G3_FP16_BIN 1e-2" "TF32 $G3_TF32_BIN 5e-3"; do
            read -r prec b tol <<<"$pair"
            nm=$(basename "$b")
            W04_DUMP_O="$OUT/O_${nm}_n1024_${input}.bin" run_check "$nm n=1024 $input (dump O)" \
                "$OUT/${nm}_n1024_${input}.log" \
                "$b" --n 1024 --bh 2 --d 64 --iters 1 --reps 1 --warmup 0 --input "$input" --out "$OUT/${nm}.csv"
            cmplog="$OUT/cmp_s3_${prec}_s2b_n1024_${input}.log"
            if [[ "$input" == normal ]]; then
                run_check "S3 $prec vs S2b 64x64 n=1024 $input (normwise rel <= $tol)" "$cmplog" \
                    python3 tools/w04_compare_o.py "$OUT/O_${nm}_n1024_${input}.bin" "$ref" --tol "$tol"
                grep -E '^compare' "$cmplog" | sed 's/^/      /' || true
            else
                python3 tools/w04_compare_o.py "$OUT/O_${nm}_n1024_${input}.bin" "$ref" --tol "$tol" >"$cmplog" 2>&1 || true
                echo "  INFO  S3 $prec vs S2b 64x64 n=1024 $input (input rounding, not gated, C26):" \
                    "$(grep -E '^compare' "$cmplog" | sed -E 's/.*(max_rel\(normwise\)=[^ ]+).*/\1/')"
            fi
        done
    done
else
    bad "S3 vs S2b comparison (binary missing)"
fi

echo "== tuned defaults (w04_tuned.hpp, nearest tuned N) =="
# Expected entry: the table line for the stage whose N is nearest to n (ties to the smaller N).
expected_entry() {  # <stage id> <n>  ->  "br bc warps"
    sed -nE "s/^ *\{\"$1\", ([0-9]+), ([0-9]+), ([0-9]+), ([0-9]+)\},.*/\1 \2 \3 \4/p" src/common/w04_tuned.hpp |
        awk -v n="$2" '{d = $1 > n ? $1 - n : n - $1; if (!seen || d < bd || (d == bd && $1 < bn)) {seen = 1; bd = d; bn = $1; e = $2 " " $3 " " $4}} END {print e}'
}
for name in W04_ATTENTION_GPU_S2b_SWOPT_FUSED W04_ATTENTION_GPU_S3_HWACC_TENSOR_CORE_FP16 \
            W04_ATTENTION_GPU_S3_HWACC_TENSOR_CORE_TF32; do
    for n in 256 4096; do
        log="$OUT/${name}_default_n${n}.log"
        run_check "$name n=$n without tile flags" "$log" \
            "build/$name" --n "$n" --bh 1 --d 64 --iters 1 --reps 1 --warmup 0 --out "$OUT/${name}_default.csv"
        read -r ebr ebc ew <<<"$(expected_entry "$name" "$n")"
        got=$(grep -E '^tile    : ' "$log" | sed -E 's/.*Br=([0-9]+) Bc=([0-9]+), [0-9]+ threads \(([0-9]+) warps\).*/\1 \2 \3/')
        if [[ -n "$ebr" && "$got" == "$ebr $ebc $ew" ]] && grep -qE '^config  : .*\(tuned N=' "$log"; then
            ok "$name n=$n default = tuned entry (Br=$ebr Bc=$ebc W=$ew)"
        else
            bad "$name n=$n default: got '$got', expected '$ebr $ebc $ew' (log: $log)"
        fi
    done
done

echo "== machine code =="
S3_BIN=build/W04_ATTENTION_CPU_S3_HWACC_AVX2
if [[ -x "$S3_BIN" ]]; then
    nfma=$(objdump -d --no-show-raw-insn "$S3_BIN" | grep -cE 'vfmadd[0-9]+ps.*ymm' || true)
    if (( nfma > 0 )); then ok "CPU S3 contains $nfma ymm vfmadd*ps instructions"
    else bad "CPU S3 contains no ymm vfmadd*ps instructions"; fi
fi
if [[ -x "$G2B_BIN" ]]; then
    sass=$(cuobjdump -sass "$G2B_BIN")
    nlds=$(grep -cE '\bLDS' <<<"$sass" || true)
    nshfl=$(grep -c 'SHFL' <<<"$sass" || true)
    nhmma=$(grep -c 'HMMA' <<<"$sass" || true)
    if (( nlds > 0 && nshfl > 0 && nhmma == 0 )); then
        ok "GPU S2b SASS: $nlds LDS, $nshfl SHFL, 0 HMMA (SIMT FP32, shared-memory tiles, shuffle reductions)"
    else bad "GPU S2b SASS: LDS=$nlds SHFL=$nshfl HMMA=$nhmma (expected LDS>0, SHFL>0, HMMA=0)"; fi
fi
# S3: each variant must use its own Tensor Core instruction and no other HMMA form.
for pair in "FP16 HMMA.16816.F32" "TF32 HMMA.1684.F32.TF32"; do
    read -r prec want <<<"$pair"
    b="build/W04_ATTENTION_GPU_S3_HWACC_TENSOR_CORE_$prec"
    [[ -x "$b" ]] || { bad "GPU S3 $prec SASS (binary missing)"; continue; }
    forms=$(cuobjdump -sass "$b" | grep -oE 'HMMA\.[A-Z0-9.]+' | sort | uniq -c | awk '{printf "%s %s; ", $2, $1}')
    nwant=$(cuobjdump -sass "$b" | grep -oE 'HMMA\.[A-Z0-9.]+' | grep -cxF "$want" || true)
    nother=$(cuobjdump -sass "$b" | grep -oE 'HMMA\.[A-Z0-9.]+' | grep -cvxF "$want" || true)
    if (( nwant > 0 && nother == 0 )); then ok "GPU S3 $prec SASS: ${forms}(only $want)"
    else bad "GPU S3 $prec SASS: ${forms:-no HMMA} (expected $want only)"; fi
done

ELAPSED=$((SECONDS - START))
[[ -n "$WARNINGS" ]] && echo "$WARNINGS"
echo "== summary: $PASS passed, $FAIL failed, ${ELAPSED}s (logs: $OUT) =="
if (( ELAPSED > 60 )); then
    echo "FAIL: correctness test took ${ELAPSED}s (limit 60s)"
    exit 1
fi
(( FAIL == 0 ))
