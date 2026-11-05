#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# W04 ATTENTION — benchmark sweeps (CPU part: P1)
#
#   bash scripts/w04_sweep.sh cpu-nsweep    runtime vs N, S1/S2/S3
#   bash scripts/w04_sweep.sh cpu-tiles     Br x Bc grid for S2/S3 at N = 1024, 4096
#   bash scripts/w04_sweep.sh cpu-threads   S2/S3 at T = 1..20, N = 1024, 2048
#   bash scripts/w04_sweep.sh cpu-flags     compiler-flag study (build/flagstudy/)
#   bash scripts/w04_sweep.sh cpu-all       all of the above in one session folder
#
# Output: results/raw/W04_ATTENTION/<YYYYMMDD-HHMMSS>/ (refused if it exists)
#   run_info.txt  machine state at the start (CPU, caches, governor, power, GPU clocks)
#   commands.log  every command with its OpenMP environment
#   <experiment>/<STAGE_ID>.csv   append-only, one row per repetition
#   logs/         stdout/stderr of every run
# Codegen evidence of the flag study (text): profiling/W04_ATTENTION/codegen/<same TS>/
#
# All CPU runs use OMP_PROC_BIND=close OMP_PLACES=cores (20 threads unless the
# thread sweep sets fewer). Binaries built from a "-dirty" tree or without git
# are refused unless ALLOW_DIRTY=1, so every CSV row points to a real commit.
#
# Environment overrides (for smoke tests): QUICK=1 (tiny sizes, 1 iteration),
# RESULTS_ROOT, PROFILING_ROOT, ALLOW_DIRTY=1.
# Exit status 1 if any run did not end with "Correct : YES".
# -----------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."

usage() { sed -n '4,9p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
[[ $# -eq 1 ]] || usage
CMD=$1

QUICK=${QUICK:-0}
RESULTS_ROOT=${RESULTS_ROOT:-results/raw/W04_ATTENTION}
PROFILING_ROOT=${PROFILING_ROOT:-profiling/W04_ATTENTION/codegen}
TS=$(date +%Y%m%d-%H%M%S)
RUN="$RESULTS_ROOT/$TS"

S1=W04_ATTENTION_CPU_S1_BASE
S2=W04_ATTENTION_CPU_S2_SWOPT_BLOCKED_OMP
S3=W04_ATTENTION_CPU_S3_HWACC_AVX2
NTHREADS_ALL=20

# Protocol (plan): 2 warmups, 20 iterations, 5 repetitions, BH = 8, d = 64.
PROTO=(--warmup 2 --iters 20 --reps 5)
PROTO_S1_LARGE=(--warmup 1 --iters 1 --reps 3)   # S1 at N >= 2048 (risk C5)
SHAPE=(--bh 8 --d 64)
NS_S1=(512 1024 2048 4096)
NS_FUSED=(512 1024 2048 4096 8192)
NS_TILES=(1024 4096)
NS_THREADS=(1024 2048)
NS_FLAGS=(1024 2048)
BRS=(16 32 64 128)
BCS=(32 64 128 256)
THREADS=$(seq 1 20)
if [[ "$QUICK" == 1 ]]; then
    PROTO=(--warmup 0 --iters 1 --reps 2); PROTO_S1_LARGE=("${PROTO[@]}")
    SHAPE=(--bh 2 --d 64)
    NS_S1=(128 256); NS_FUSED=(128 256); NS_TILES=(256); NS_THREADS=(256); NS_FLAGS=(256)
    BRS=(16 32); BCS=(32 64); THREADS="1 2 20"
fi

FAILS=0

# ------------------------------------------------------------------ helpers
require_bin() { [[ -x "build/$1" ]] || { echo "missing build/$1 (run: make w04)"; exit 1; }; }

check_sha() {
    local sha
    sha=$(sed -n 's/^#define W04_GIT_SHA "\(.*\)"/\1/p' build/w04_build_info.h 2>/dev/null || true)
    [[ -n "$sha" ]] || { echo "build/w04_build_info.h missing (run: make w04)"; exit 1; }
    if [[ "$sha" == nogit || "$sha" == *-dirty ]]; then
        if [[ "${ALLOW_DIRTY:-0}" != 1 ]]; then
            echo "refusing to record results from build '$sha' (commit, rebuild, or set ALLOW_DIRTY=1)"
            exit 1
        fi
        echo "WARNING: recording results from build '$sha' (ALLOW_DIRTY=1)"
    fi
    BUILD_SHA=$sha
}

start_run() {
    [[ -e "$RUN" ]] && { echo "output folder exists: $RUN"; exit 1; }
    mkdir -p "$RUN/logs"
    {
        echo "date        : $(date -Iseconds)"
        echo "command     : $0 $CMD (QUICK=$QUICK)"
        echo "build sha   : $BUILD_SHA"
        echo "compiler    : $(g++ --version | head -n 1)"
        echo "kernel      : $(uname -r)"
        echo "ac online   : $(cat /sys/class/power_supply/A*/online 2>/dev/null | head -n 1 || echo n/a)"
        echo "governor    : $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo n/a)"
        echo "epp         : $(cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference 2>/dev/null || echo n/a)"
        echo "no_turbo    : $(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo n/a)"
        echo "gpu         : $(nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv,noheader 2>/dev/null || echo n/a)"
        echo "--- per-CPU caches (sysfs) ---"
        for c in /sys/devices/system/cpu/cpu[0-9]*; do
            printf '%s L1d=%s L2=%s (shared %s) max_khz=%s\n' "$(basename "$c")" \
                "$(cat "$c/cache/index0/size" 2>/dev/null)" "$(cat "$c/cache/index2/size" 2>/dev/null)" \
                "$(cat "$c/cache/index2/shared_cpu_list" 2>/dev/null)" "$(cat "$c/cpufreq/cpuinfo_max_freq" 2>/dev/null)"
        done
        echo "--- lscpu ---"
        lscpu
    } > "$RUN/run_info.txt"
    echo "run folder: $RUN"
}

# run <csv> <log-name> <threads> <binary> args...
run() {
    local csv=$1 logname=$2 threads=$3 bin=$4; shift 4
    local log="$RUN/logs/$logname.log"
    mkdir -p "$(dirname "$csv")" "$(dirname "$log")"
    local envs=(OMP_NUM_THREADS="$threads" OMP_PROC_BIND=close OMP_PLACES=cores)
    [[ -n "${AFFINITY:-}" ]] && envs+=(OMP_DISPLAY_AFFINITY=TRUE)
    echo "${envs[*]} $bin $* --out $csv" >> "$RUN/commands.log"
    if env "${envs[@]}" "$bin" "$@" --out "$csv" > "$log" 2>&1 \
        && [[ "$(grep -E '^Correct : ' "$log" | tail -n 1)" == "Correct : YES" ]]; then
        printf '  ok    %-60s %s\n' "$logname" "$(grep -oE '\| [0-9.]+ ± [0-9.]+ ms \| [0-9.]+ GFLOP/s' "$log" | head -n 1)"
    else
        printf '  FAIL  %s (log: %s)\n' "$logname" "$log"
        FAILS=$((FAILS + 1))
    fi
}

# ------------------------------------------------------------- experiments
cpu_nsweep() {
    echo "== N sweep =="
    for n in "${NS_S1[@]}"; do
        local proto=("${PROTO[@]}")
        (( n >= 2048 )) && proto=("${PROTO_S1_LARGE[@]}")
        run "$RUN/nsweep/$S1.csv" "nsweep/${S1}_n$n" 1 "build/$S1" --n "$n" "${SHAPE[@]}" "${proto[@]}"
    done
    for st in "$S2" "$S3"; do
        for n in "${NS_FUSED[@]}"; do
            run "$RUN/nsweep/$st.csv" "nsweep/${st}_n$n" "$NTHREADS_ALL" "build/$st" --n "$n" "${SHAPE[@]}" "${PROTO[@]}"
        done
    done
}

cpu_tiles() {
    echo "== Br x Bc tiles =="
    for st in "$S2" "$S3"; do
        for n in "${NS_TILES[@]}"; do
            for br in "${BRS[@]}"; do
                for bc in "${BCS[@]}"; do
                    run "$RUN/tiles/$st.csv" "tiles/${st}_n${n}_br${br}_bc${bc}" "$NTHREADS_ALL" "build/$st" \
                        --n "$n" "${SHAPE[@]}" "${PROTO[@]}" --br "$br" --bc "$bc"
                done
            done
        done
    done
}

cpu_threads() {
    echo "== thread sweep =="
    for st in "$S2" "$S3"; do
        for n in "${NS_THREADS[@]}"; do
            for t in $THREADS; do
                AFFINITY=1 run "$RUN/threads/$st.csv" "threads/${st}_n${n}_t$t" "$t" "build/$st" \
                    --n "$n" "${SHAPE[@]}" "${PROTO[@]}"
            done
        done
    done
}

codegen_summary() {   # <binary> -> one line of instruction counts
    local b=$1 dis
    dis=$(objdump -d --no-show-raw-insn "$b")
    printf '%-58s vfmadd_ps_ymm=%-5s vmulps_ymm=%-5s mulps_xmm=%-5s call_expf=%-3s libmvec_avx2_expf=%-3s libmvec_sse_expf=%s\n' \
        "$(basename "$b")" \
        "$(grep -cE 'vfmadd[0-9]+ps.*ymm' <<<"$dis" || true)" \
        "$(grep -cE 'vmulps.*ymm' <<<"$dis" || true)" \
        "$(grep -cE '\s(v)?mulps\s.*xmm' <<<"$dis" || true)" \
        "$(grep -cE 'call.*<expf(@plt)?>' <<<"$dis" || true)" \
        "$(grep -cE 'call.*_ZGVdN8v_expf' <<<"$dis" || true)" \
        "$(grep -cE 'call.*_ZGVbN4v_expf' <<<"$dis" || true)"
}

cpu_flags() {
    echo "== compiler-flag study =="
    make --no-print-directory w04-flagstudy
    local fs=build/flagstudy
    local cg="$PROFILING_ROOT/$TS"
    mkdir -p "$cg"
    selftest_fastmath
    for st in "$S2" "$S3"; do
        for tag in O2 O3n O3f O3nf; do
            local b="$fs/$st.$tag"
            for input in normal large_logits; do
                run "$RUN/flags_check/$st.csv" "flags_check/${st}.${tag}_$input" "$NTHREADS_ALL" "$b" \
                    --n 256 --bh 2 --d 64 --iters 1 --reps 1 --warmup 0 --input "$input"
            done
            for n in "${NS_FLAGS[@]}"; do
                run "$RUN/flags/$st.csv" "flags/${st}.${tag}_n$n" "$NTHREADS_ALL" "$b" --n "$n" "${SHAPE[@]}" "${PROTO[@]}"
            done
        done
    done
    {
        echo "# instruction counts in the whole binary (objdump -d); build sha $BUILD_SHA"
        for b in "build/$S1" "build/$S2" "build/$S3" "$fs/$S2".O2 "$fs/$S2".O3n "$fs/$S2".O3f "$fs/$S2".O3nf \
                 "$fs/$S3".O2 "$fs/$S3".O3n "$fs/$S3".O3f "$fs/$S3".O3nf; do
            codegen_summary "$b"
        done
    } > "$cg/codegen_summary.txt"
    cp "$fs"/*.optinfo "$cg/"
    echo "  codegen evidence: $cg"
}

# The fast-math self-test takes a directory argument, not the bench CLI.
selftest_fastmath() {
    local log="$RUN/logs/flags/selftest_O3nf.log"
    mkdir -p "$(dirname "$log")"
    echo "build/flagstudy/W04_ATTENTION_SELFTEST.O3nf $RUN/logs/flags/selftest_out" >> "$RUN/commands.log"
    if build/flagstudy/W04_ATTENTION_SELFTEST.O3nf "$RUN/logs/flags/selftest_out" > "$log" 2>&1 \
        && [[ "$(grep -E '^Correct : ' "$log" | tail -n 1)" == "Correct : YES" ]]; then
        echo "  ok    fast-math self-test (reference integrity under -O3 -march=native -ffast-math)"
    else
        echo "  FAIL  fast-math self-test (log: $log)"; FAILS=$((FAILS + 1))
    fi
}

# ------------------------------------------------------------------- main
case "$CMD" in
    cpu-nsweep|cpu-tiles|cpu-threads|cpu-flags|cpu-all) ;;
    *) usage ;;
esac
make --no-print-directory w04 >/dev/null
for b in "$S1" "$S2" "$S3"; do require_bin "$b"; done
check_sha
start_run
case "$CMD" in
    cpu-nsweep)  cpu_nsweep ;;
    cpu-tiles)   cpu_tiles ;;
    cpu-threads) cpu_threads ;;
    cpu-flags)   cpu_flags ;;
    cpu-all)     cpu_nsweep; cpu_tiles; cpu_threads; cpu_flags ;;
esac
echo "== done: $FAILS failed runs; results in $RUN =="
(( FAILS == 0 ))
