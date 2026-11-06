#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# W04 ATTENTION — benchmark sweeps (CPU: P1, GPU: P2+)
#
#   bash scripts/w04_sweep.sh cpu-nsweep    runtime vs N, S1/S2/S3
#   bash scripts/w04_sweep.sh cpu-tiles     Br x Bc grid for S2/S3 at N = 1024, 4096
#   bash scripts/w04_sweep.sh cpu-threads   S2/S3 at T = 1..20, N = 1024, 2048
#   bash scripts/w04_sweep.sh cpu-flags     compiler-flag study (build/flagstudy/)
#   bash scripts/w04_sweep.sh cpu-all       all CPU experiments in one session folder
#   bash scripts/w04_sweep.sh gpu-nsweep    runtime vs N, GPU S1/S2a/REF_CUBLAS/S2b (32x32, 64x64)/S3 (FP16, TF32), + tuned defaults
#                                           (GPU_NS="<N list>" replaces the default N = 512 1024 2048 4096 8192)
#   bash scripts/w04_sweep.sh gpu-autotune  E7: Br x Bc x warps screen + finals for S2b, S3 FP16/TF32 (P5)
#   bash scripts/w04_sweep.sh gpu-flags     S2b with and without --use_fast_math: checks, speed, SASS (P5)
#   bash scripts/w04_sweep.sh gpu-fusion    S2a vs S2b 16x16 W8 (same per-thread work), paired (P6)
#
# Output: results/raw/W04_ATTENTION/<YYYYMMDD-HHMMSS>/ (refused if it exists)
#   run_info.txt  machine state at the start (CPU, caches, governor, power, GPU clocks)
#   commands.log  every command with its OpenMP environment
#   <experiment>/<STAGE_ID>.csv   append-only, one row per repetition
#   logs/         stdout/stderr of every run
#   gpu_state.csv (GPU experiments) nvidia-smi every 250 ms: SM/memory clock,
#                 power, temperature, clock-event reasons; commands.log lines
#                 carry a start time so each run can be matched to its samples
#
# GPU sweep: every (N, stage) runs once per round, GPU_ROUNDS rounds (default 3);
# the stage order inside each N is rotated by one per round, so no stage is
# always first after a size change. Stages of one N are adjacent (paired).
# Codegen evidence of the flag study (text): profiling/W04_ATTENTION/codegen/<same TS>/
#
# All CPU runs use OMP_PROC_BIND=close OMP_PLACES=cores (20 threads unless the
# thread sweep sets fewer). Binaries built from a "-dirty" tree or without git
# are refused unless ALLOW_DIRTY=1, so every CSV row points to a real commit.
#
# E7 autotune (gpu-autotune, decisions C27-C30): every (Br, Bc, W) of
# Br, Bc in {16, 32, 64} x W in {2, 4, 8} is probed once with a tiny run whose
# output goes to /dev/null; cells the binary refuses (exit 2) are written to
# e7/skipped.csv with the binary's reason, once per N. At each N in NS_E7 the
# valid cells of all three stages run once in one shuffled order (seed E7_SEED,
# recorded in run_info.txt) with --iters 10 --reps 3 (screen); then the top 3
# per stage and N plus the incumbent default (tools/w04_e7_select.py
# finalists) run with the full protocol in GPU_ROUNDS interleaved rounds. All
# rows go to one CSV, e7/W04_E7_AUTOTUNE.csv (screen and final rows differ in
# iters/reps). Registers, stack and shared memory per instance:
# profiling/W04_ATTENTION/codegen/<TS>/e7_resources.csv.
#
# Fast-math (gpu-flags, C31): build/flagstudy/<S2b>.fastmath vs build/<S2b>.
# Correctness of both on every (Br, Bc, W) at n = 203 and the default at
# n = 128/256, both inputs (flags_check/); a direct O comparison fast vs
# accurate at n = 1024 (logs/flags/compare_*); speed at NS_FM for the configs
# in FM_CONFIGS ("br bc w;..."; default 32x32 W8 and 64x64 W8) in GPU_ROUNDS
# interleaved rounds, one CSV per (build, config) in flags/; SASS counts and
# MUFU.EX2 windows: profiling/W04_ATTENTION/codegen/<TS>/s2b_fastmath_sass.txt.
#
# Fusion only (gpu-fusion, C35): S2a's tiled GEMMs give each thread one
# output (2 shared loads per FMA); S2b at 16x16 with 8 warps does the same, so
# the S2a -> S2b 16x16 W8 ratio isolates fusion from register tiling. S2b at
# its tuned default runs alongside as the reference. NS_FUSION, GPU_ROUNDS
# interleaved rounds, stage order rotated per round, gpu_state.csv logged.
#
# Environment overrides (for smoke tests): QUICK=1 (tiny sizes, 1 iteration),
# RESULTS_ROOT, PROFILING_ROOT, ALLOW_DIRTY=1, E7_SEED, FM_CONFIGS.
# Exit status 1 if any run did not end with "Correct : YES".
# -----------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."

usage() { sed -n '4,13p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
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
G1=W04_ATTENTION_GPU_S1_BASE_NAIVE
G2A=W04_ATTENTION_GPU_S2a_SWOPT_UNFUSED
GREF=W04_ATTENTION_GPU_REF_CUBLAS
G2B=W04_ATTENTION_GPU_S2b_SWOPT_FUSED
G3H=W04_ATTENTION_GPU_S3_HWACC_TENSOR_CORE_FP16
G3T=W04_ATTENTION_GPU_S3_HWACC_TENSOR_CORE_TF32
NTHREADS_ALL=20   # also the host threads for the (untimed) double reference of GPU runs
GPU_ROUNDS=${GPU_ROUNDS:-3}

# Protocol (plan): 2 warmups, 20 iterations, 5 repetitions, BH = 8, d = 64.
PROTO=(--warmup 2 --iters 20 --reps 5)
PROTO_S1_LARGE=(--warmup 1 --iters 1 --reps 3)   # S1 at N >= 2048 (risk C5)
SHAPE=(--bh 8 --d 64)
NS_S1=(512 1024 2048 4096)
NS_FUSED=(512 1024 2048 4096 8192)
NS_TILES=(1024 4096)
NS_THREADS=(1024 2048)
NS_FLAGS=(1024 2048)
NS_GPU=(${GPU_NS:-512 1024 2048 4096 8192})   # GPU_NS="2048 4096" overrides (e.g. the locked-clock check, issue 11)
NS_E7=(2048 4096)
NS_FM=(1024 2048 4096)
NS_FUSION=(1024 2048 4096)
E7_SEED=${E7_SEED:-20261106}
FM_CONFIGS=${FM_CONFIGS:-"32 32 8;64 64 8"}
PROTO_SCREEN=(--warmup 2 --iters 10 --reps 3)
BRS=(16 32 64 128)
BCS=(32 64 128 256)
THREADS=$(seq 1 20)
if [[ "$QUICK" == 1 ]]; then
    PROTO=(--warmup 0 --iters 1 --reps 2); PROTO_S1_LARGE=("${PROTO[@]}")
    SHAPE=(--bh 2 --d 64)
    NS_S1=(128 256); NS_FUSED=(128 256); NS_TILES=(256); NS_THREADS=(256); NS_FLAGS=(256); NS_GPU=(128 256)
    NS_E7=(128 256); NS_FM=(128 256); NS_FUSION=(128 256); PROTO_SCREEN=(--warmup 0 --iters 1 --reps 3)
    GPU_ROUNDS=2
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
        echo "gpu limits  : $(nvidia-smi --query-gpu=enforced.power.limit,power.default_limit,power.max_limit,clocks.max.sm --format=csv,noheader 2>/dev/null || echo n/a) (enforced, default, max power; max SM clock)"
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
    echo "[$(date +%FT%T.%3N)] ${envs[*]} $bin $* --out $csv" >> "$RUN/commands.log"
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

# GPU state sampler: runs in the background for the whole GPU experiment.
GPU_SAMPLER_PID=
start_gpu_sampler() {
    nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,pstate,clocks_event_reasons.active \
        --format=csv -lms 250 > "$RUN/gpu_state.csv" 2>/dev/null &
    GPU_SAMPLER_PID=$!
    trap 'stop_gpu_sampler' EXIT
}
stop_gpu_sampler() {
    [[ -n "$GPU_SAMPLER_PID" ]] && kill "$GPU_SAMPLER_PID" 2>/dev/null && wait "$GPU_SAMPLER_PID" 2>/dev/null
    GPU_SAMPLER_PID=
}

gpu_nsweep() {
    echo "== GPU N sweep ($GPU_ROUNDS rounds, stage order rotated per round) =="
    start_gpu_sampler
    # Entries are "<name>|<binary>|<extra args>". The name is the CSV and log
    # name, so a second tile of the same binary gets its own summary row
    # (decision C21: S2b at its default 32x32 tile and at 64x64). P4 adds the
    # two S3 Tensor Core variants at their default 64x64 tile (C24). P5 (C29)
    # pins those entries to explicit tiles, so each CSV name keeps its P3/P4
    # meaning after the defaults change, and adds one "_tuned" entry per
    # tunable stage that runs without tile flags (w04_tuned.hpp defaults).
    local entries=("$G1|$G1|" "$G2A|$G2A|" "$GREF|$GREF|" "$G2B|$G2B|--br 32 --bc 32 --warps 8"
                   "${G2B}_64x64|$G2B|--br 64 --bc 64 --warps 8" "$G3H|$G3H|--br 64 --bc 64 --warps 4"
                   "$G3T|$G3T|--br 64 --bc 64 --warps 4"
                   "${G2B}_tuned|$G2B|" "${G3H}_tuned|$G3H|" "${G3T}_tuned|$G3T|")
    local k=${#entries[@]}
    for ((round = 1; round <= GPU_ROUNDS; round++)); do
        for n in "${NS_GPU[@]}"; do
            for ((s = 0; s < k; s++)); do
                local name bin extra
                IFS='|' read -r name bin extra <<< "${entries[$(( (s + round - 1) % k ))]}"
                local -a xa=()
                [[ -n "$extra" ]] && read -r -a xa <<< "$extra"
                run "$RUN/gpu_nsweep/$name.csv" "gpu_nsweep/${name}_n${n}_r$round" "$NTHREADS_ALL" "build/$bin" \
                    --n "$n" "${SHAPE[@]}" "${PROTO[@]}" "${xa[@]}"
            done
        done
    done
    stop_gpu_sampler
}

gpu_fusion() {
    echo "== GPU fusion-only comparison ($GPU_ROUNDS rounds, stage order rotated per round) =="
    start_gpu_sampler
    local entries=("$G2A|$G2A|" "${G2B}_16x16w8|$G2B|--br 16 --bc 16 --warps 8" "${G2B}_tuned|$G2B|")
    local k=${#entries[@]} round n s
    for ((round = 1; round <= GPU_ROUNDS; round++)); do
        for n in "${NS_FUSION[@]}"; do
            for ((s = 0; s < k; s++)); do
                local name bin extra
                IFS='|' read -r name bin extra <<< "${entries[$(( (s + round - 1) % k ))]}"
                local -a xa=()
                [[ -n "$extra" ]] && read -r -a xa <<< "$extra"
                run "$RUN/gpu_fusion/$name.csv" "gpu_fusion/${name}_n${n}_r$round" "$NTHREADS_ALL" "build/$bin" \
                    --n "$n" "${SHAPE[@]}" "${PROTO[@]}" "${xa[@]}"
            done
        done
    done
    stop_gpu_sampler
    echo "  summary: python3 tools/w04_summary.py $RUN"
}

# ------------------------------------------------------------- E7 autotune
# The binary's validate() decides which cells exist; this only records its answer.
e7_probe() {   # <stage id> <br> <bc> <w>  -> "" if runnable, else the reason
    local out rc
    out=$("build/$1" --n 64 --bh 1 --d 64 --iters 1 --reps 1 --warmup 0 --br "$2" --bc "$3" --warps "$4" \
          --out /dev/null 2>&1) && rc=0 || rc=$?
    if (( rc == 2 )) && grep -q ': error: ' <<<"$out"; then
        grep -m1 ': error: ' <<<"$out" | sed -E 's/^[^:]*: error: //'
    elif (( rc != 0 )); then
        echo "PROBE FAILED (exit $rc)"
    fi
}

gpu_autotune() {
    echo "== E7 autotune (screen: ${PROTO_SCREEN[*]}; finals: ${PROTO[*]} x $GPU_ROUNDS rounds; seed $E7_SEED) =="
    {   # tools/w04_e7_select.py tells screen and final rows apart by these
        echo "e7 seed     : $E7_SEED"
        echo "e7 screen   : ${PROTO_SCREEN[*]}"
        echo "e7 final    : ${PROTO[*]}"
        echo "e7 resources: $PROFILING_ROOT/$TS/e7_resources.csv"
    } >> "$RUN/run_info.txt"
    local e7="$RUN/e7" csv="$RUN/e7/W04_E7_AUTOTUNE.csv" cg="$PROFILING_ROOT/$TS"
    mkdir -p "$e7" "$cg"
    local stages=("$G2B" "$G3H" "$G3T")
    local -a cells=()   # "stage br bc w" runnable
    local -a skips=()   # "stage,br,bc,w,reason"
    local st br bc w why
    echo "stage_id,br,bc,warps,threads,smem_dynamic_B,regs,stack_B,local_B" > "$cg/e7_resources.csv"
    for st in "${stages[@]}"; do
        local res
        res=$(python3 tools/w04_e7_select.py resources "build/$st")
        for br in 16 32 64; do for bc in 16 32 64; do for w in 2 4 8; do
            why=$(e7_probe "$st" "$br" "$bc" "$w")
            if [[ -z "$why" ]]; then
                cells+=("$st $br $bc $w")
                local tile smem r
                tile=$("build/$st" --n 64 --bh 1 --d 64 --iters 1 --reps 1 --warmup 0 --br "$br" --bc "$bc" \
                       --warps "$w" --out /dev/null 2>&1 | grep -m1 '^tile    : ')
                smem=$(sed -E 's/.*, ([0-9]+) B dynamic.*/\1/' <<<"$tile")
                r=$(grep -m1 "^$br,$bc,$w," <<<"$res" | cut -d, -f4-)
                echo "$st,$br,$bc,$w,$((32 * w)),$smem,$r" >> "$cg/e7_resources.csv"
            elif [[ "$why" == "PROBE FAILED"* ]]; then
                echo "  FAIL  probe $st Br=$br Bc=$bc W=$w: $why"; FAILS=$((FAILS + 1))
            else
                skips+=("$st,$br,$bc,$w,\"$why\"")
            fi
        done; done; done
    done
    echo "stage_id,n,br,bc,warps,reason" > "$e7/skipped.csv"
    local n s
    for n in "${NS_E7[@]}"; do
        for s in "${skips[@]}"; do
            IFS=, read -r st br bc w why <<<"$s"
            echo "$st,$n,$br,$bc,$w,$why" >> "$e7/skipped.csv"
        done
    done
    echo "  ${#cells[@]} runnable cells, ${#skips[@]} skipped per N (reasons: $e7/skipped.csv)"
    cuobjdump --dump-resource-usage "${stages[@]/#/build/}" > "$cg/e7_resources.txt"

    start_gpu_sampler
    echo "-- screen --"
    for n in "${NS_E7[@]}"; do
        local -a order=()
        mapfile -t order < <(printf '%s\n' "${cells[@]}" |
            python3 -c "import random,sys; c=[l.rstrip() for l in sys.stdin]; random.Random($E7_SEED + $n).shuffle(c); print(*c, sep='\n')")
        for c in "${order[@]}"; do
            read -r st br bc w <<<"$c"
            run "$csv" "e7/screen_n${n}_${st}_br${br}_bc${bc}_w$w" "$NTHREADS_ALL" "build/$st" \
                --n "$n" "${SHAPE[@]}" "${PROTO_SCREEN[@]}" --br "$br" --bc "$bc" --warps "$w"
        done
    done
    echo "-- finals --"
    local -A fin=()
    for n in "${NS_E7[@]}"; do
        local -a list=()
        for st in "${stages[@]}"; do
            local picks
            picks=$(python3 tools/w04_e7_select.py finalists "$RUN" "$n" "$st" --top 3) ||
                { echo "  FAIL  finalists for $st at N=$n"; exit 1; }
            while read -r br bc w; do list+=("$st $br $bc $w"); done <<<"$picks"
        done
        fin[$n]=$(printf '%s;' "${list[@]}")
        echo "  N=$n finalists: ${#list[@]}"
    done
    local round
    for ((round = 1; round <= GPU_ROUNDS; round++)); do
        for n in "${NS_E7[@]}"; do
            local -a list=()
            IFS=';' read -r -a list <<<"${fin[$n]%;}"
            local k=${#list[@]} i
            for ((i = 0; i < k; i++)); do
                read -r st br bc w <<<"${list[$(( (i + round - 1) % k ))]}"
                run "$csv" "e7/final_n${n}_${st}_br${br}_bc${bc}_w${w}_r$round" "$NTHREADS_ALL" "build/$st" \
                    --n "$n" "${SHAPE[@]}" "${PROTO[@]}" --br "$br" --bc "$bc" --warps "$w"
            done
        done
    done
    stop_gpu_sampler
    echo "  report: python3 tools/w04_e7_select.py report $RUN"
}

# --------------------------------------------------------- GPU fast-math
gpu_flags() {
    echo "== GPU fast-math study (S2b) =="
    make --no-print-directory w04-gpu-flagstudy >/dev/null
    local acc="build/$G2B" fm="build/flagstudy/$G2B.fastmath" cg="$PROFILING_ROOT/$TS"
    [[ -x "$fm" ]] || { echo "missing $fm"; exit 1; }
    mkdir -p "$cg" "$RUN/logs/flags"
    local -a builds=("accurate|$acc" "fastmath|$fm")
    local b tag bin br bc w input n
    echo "-- checks (every Br, Bc, W at n=203; default at n=128/256; both inputs) --"
    for b in "${builds[@]}"; do
        IFS='|' read -r tag bin <<<"$b"
        for br in 16 32 64; do for bc in 16 32 64; do for w in 2 4 8; do for input in normal large_logits; do
            run "$RUN/flags_check/$G2B.$tag.csv" "flags_check/$G2B.${tag}_br${br}_bc${bc}_w${w}_$input" \
                "$NTHREADS_ALL" "$bin" --n 203 --bh 2 --d 64 --iters 1 --reps 1 --warmup 0 \
                --br "$br" --bc "$bc" --warps "$w" --input "$input"
        done; done; done; done
        for n in 128 256; do for input in normal large_logits; do
            run "$RUN/flags_check/$G2B.$tag.csv" "flags_check/$G2B.${tag}_default_n${n}_$input" \
                "$NTHREADS_ALL" "$bin" --n "$n" --bh 2 --d 64 --iters 1 --reps 1 --warmup 0 --input "$input"
        done; done
    done
    local -a cfgs=()
    IFS=';' read -r -a cfgs <<<"$FM_CONFIGS"
    echo "-- fast vs accurate O (n=1024, bh=2) --"
    local c
    for c in "${cfgs[@]}"; do
        read -r br bc w <<<"$c"
        for input in normal large_logits; do
            for b in "${builds[@]}"; do
                IFS='|' read -r tag bin <<<"$b"
                W04_DUMP_O="$RUN/logs/flags/O_${tag}_br${br}_bc${bc}_w${w}_$input.bin" \
                    run "$RUN/flags_check/$G2B.$tag.csv" "flags/dump_${tag}_br${br}_bc${bc}_w${w}_$input" \
                    "$NTHREADS_ALL" "$bin" --n 1024 --bh 2 --d 64 --iters 1 --reps 1 --warmup 0 \
                    --br "$br" --bc "$bc" --warps "$w" --input "$input"
            done
            local log="$RUN/logs/flags/compare_br${br}_bc${bc}_w${w}_$input.log"
            python3 tools/w04_compare_o.py "$RUN/logs/flags/O_fastmath_br${br}_bc${bc}_w${w}_$input.bin" \
                "$RUN/logs/flags/O_accurate_br${br}_bc${bc}_w${w}_$input.bin" --tol 1e-4 > "$log" 2>&1 || true
            echo "  ${br}x${bc} W$w $input: $(grep -E '^compare' "$log" || echo 'compare failed')"
        done
    done
    rm -f "$RUN"/logs/flags/O_*.bin   # raw dumps (MBs): only the comparison logs are kept
    echo "-- speed ($GPU_ROUNDS interleaved rounds) --"
    start_gpu_sampler
    local -a entries=()
    for c in "${cfgs[@]}"; do
        read -r br bc w <<<"$c"
        for b in "${builds[@]}"; do entries+=("$b|$br $bc $w"); done
    done
    local k=${#entries[@]} round i
    for ((round = 1; round <= GPU_ROUNDS; round++)); do
        for n in "${NS_FM[@]}"; do
            for ((i = 0; i < k; i++)); do
                IFS='|' read -r tag bin c <<<"${entries[$(( (i + round - 1) % k ))]}"
                read -r br bc w <<<"$c"
                run "$RUN/flags/$G2B.${tag}_${br}x${bc}w$w.csv" "flags/$G2B.${tag}_${br}x${bc}w${w}_n${n}_r$round" \
                    "$NTHREADS_ALL" "$bin" --n "$n" "${SHAPE[@]}" "${PROTO[@]}" --br "$br" --bc "$bc" --warps "$w"
            done
        done
    done
    stop_gpu_sampler
    local -a inst=()
    for c in "${cfgs[@]}"; do read -r br bc w <<<"$c"; inst+=("${br}x${bc}w$w"); done
    python3 tools/w04_fastmath_sass.py "$acc" "$fm" "${inst[@]}" > "$cg/s2b_fastmath_sass.txt"
    echo "  SASS: $cg/s2b_fastmath_sass.txt"
    echo "  speed: python3 tools/w04_summary.py $RUN flags"
}

# ------------------------------------------------------------------- main
case "$CMD" in
    cpu-nsweep|cpu-tiles|cpu-threads|cpu-flags|cpu-all) NEED=("$S1" "$S2" "$S3") ;;
    gpu-nsweep) NEED=("$G1" "$G2A" "$GREF" "$G2B" "$G3H" "$G3T") ;;
    gpu-autotune) NEED=("$G2B" "$G3H" "$G3T") ;;
    gpu-flags) NEED=("$G2B") ;;
    gpu-fusion) NEED=("$G2A" "$G2B") ;;
    *) usage ;;
esac
make --no-print-directory w04 >/dev/null
for b in "${NEED[@]}"; do require_bin "$b"; done
check_sha
start_run
case "$CMD" in
    cpu-nsweep)  cpu_nsweep ;;
    cpu-tiles)   cpu_tiles ;;
    cpu-threads) cpu_threads ;;
    cpu-flags)   cpu_flags ;;
    cpu-all)     cpu_nsweep; cpu_tiles; cpu_threads; cpu_flags ;;
    gpu-nsweep)  gpu_nsweep ;;
    gpu-autotune) gpu_autotune ;;
    gpu-flags)   gpu_flags ;;
    gpu-fusion)  gpu_fusion ;;
esac
echo "== done: $FAILS failed runs; results in $RUN =="
(( FAILS == 0 ))
