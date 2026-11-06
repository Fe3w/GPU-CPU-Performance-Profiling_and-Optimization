#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# run_all.sh — final paired re-test: every stage of W01–W04, one session
#
#   bash scripts/run_all.sh              full run (~20 min): build, correctness gate,
#                                        GPU stages in 3 interleaved rounds, CPU stages
#                                        with a cool-down before each run, summary
#   bash scripts/run_all.sh --smoke      tiny sizes, 1 iteration, 2 repetitions, 1 round,
#                                        no gate (~2 min): checks that every command works;
#                                        results are tagged smoke and are not measurements
#   bash scripts/run_all.sh --only w01,w04    a subset of workloads
#
# Protocol (all stages): 2 warm-ups, 20 timed iterations, 5 repetitions.
# Sizes are the recorded headline configurations: W01/W02 N=33,554,432,
# W03 N=1024, W04 BH=8 d=64 N in {1024, 2048} (GPU tiles: tuned defaults).
# Threads: W01/W02 S2/S3 6, W03 S2/S3 8, W04 S2/S3 20;
# W01-W03 GPU stages get an untimed priming run first (memory clock, see run_one);
# OMP_PROC_BIND=close OMP_PLACES=cores for every CPU run.
#
# The W01–W03 binaries keep their own command lines (listed in stage_cmd below,
# taken from their argument parsing); this script only wraps them.
#
# Output: results/raw/FINAL/<YYYYMMDD-HHMMSS>[-smoke]/
#   csv/<KEY>[_r<k>].csv  the binary's own CSV (never edited)
#   logs/<KEY>[_r<k>].log stdout + stderr of each run; logs/build.log, logs/gate.log
#   runs.csv              one line per run: round, key, threads, exit code, wall time,
#                         CPU temperature and P/E-core frequency during the run
#   gpu_state.csv         nvidia-smi every 250 ms during the GPU block
#   run_info.txt          machine state at the start; commands.log every command
# Then tools/final_summary.py <run folder> prints the tables (see its --help).
#
# Overrides: ROUNDS (3), START_C (70, cool-down target in °C), WAIT_S (120),
# RESULTS_ROOT, ALLOW_DIRTY=1 (record results from an uncommitted tree).
# -----------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$0")/.."

SMOKE=0
ONLY="w01,w02,w03,w04"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke) SMOKE=1 ;;
        --only) ONLY="$2"; shift ;;
        -h|--help) sed -n '3,32p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $1 (see --help)"; exit 2 ;;
    esac
    shift
done

ROUNDS=${ROUNDS:-3}
START_C=${START_C:-70}
WAIT_S=${WAIT_S:-120}
RESULTS_ROOT=${RESULTS_ROOT:-results/raw/FINAL}

# Protocol and sizes
I=20; R=5; W=2
N_VEC=33554432; N_GEMM=1024; W04_NS=(1024 2048); W04_SHAPE=(--bh 8 --d 64)
T_W01=6; T_W02=6; T_W03=8; T_W04=20
if [[ "$SMOKE" == 1 ]]; then
    I=1; R=2; W=1
    N_VEC=1048576; N_GEMM=256; W04_NS=(256); W04_SHAPE=(--bh 2 --d 64)
    ROUNDS=1; START_C=200
fi

TS=$(date +%Y%m%d-%H%M%S)
RUN="$RESULTS_ROOT/$TS$([[ $SMOKE == 1 ]] && echo -smoke)"
[[ -e "$RUN" ]] && { echo "output folder exists: $RUN"; exit 1; }
mkdir -p "$RUN/csv" "$RUN/logs"

want() { [[ ",$ONLY," == *",$1,"* ]]; }

# ---------------------------------------------------------------- build + sha
echo "== build (make legacy w04), log: $RUN/logs/build.log"
if ! make --no-print-directory legacy w04 > "$RUN/logs/build.log" 2>&1; then
    tail -20 "$RUN/logs/build.log"; echo "build failed"; exit 1
fi
SHA=$(sed -n 's/^#define W04_GIT_SHA "\(.*\)"/\1/p' build/w04_build_info.h 2>/dev/null)
TREE_SHA=$(git rev-parse --short=12 HEAD 2>/dev/null || echo nogit)
[[ -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" ]] && TREE_SHA="$TREE_SHA-dirty"
if [[ "$TREE_SHA" == nogit || "$TREE_SHA" == *-dirty || "$SHA" == *-dirty ]] && [[ "${ALLOW_DIRTY:-0}" != 1 && "$SMOKE" != 1 ]]; then
    echo "refusing to record results from tree '$TREE_SHA' (W04 build '$SHA'): commit first, or set ALLOW_DIRTY=1"
    exit 1
fi

# ------------------------------------------------------------ machine state
PKG_ZONE=""
for z in /sys/class/thermal/thermal_zone*; do
    [[ "$(cat "$z/type" 2>/dev/null)" == x86_pkg_temp ]] && PKG_ZONE="$z/temp"
done
pkg_c() { [[ -n "$PKG_ZONE" ]] && echo $(( $(cat "$PKG_ZONE") / 1000 )) || echo -1; }
NCPU=$(nproc --all)
{
    echo "date        : $(date -Iseconds)"
    echo "command     : $0 $([[ $SMOKE == 1 ]] && echo --smoke) --only $ONLY"
    echo "tree sha    : $TREE_SHA"
    echo "w04 build   : $SHA"
    echo "protocol    : warmup=$W iters=$I reps=$R rounds(GPU)=$ROUNDS cool-down<=${START_C}C (max ${WAIT_S}s)"
    echo "sizes       : W01/W02 N=$N_VEC  W03 N=$N_GEMM  W04 N=${W04_NS[*]} ${W04_SHAPE[*]}"
    echo "threads     : W01 $T_W01  W02 $T_W02  W03 $T_W03  W04 $T_W04  (OMP_PROC_BIND=close OMP_PLACES=cores)"
    echo "compiler    : $(g++ --version | head -n 1) / $(nvcc --version 2>/dev/null | tail -n 1)"
    echo "cpu         : $(lscpu | sed -n 's/^Model name: *//p')"
    echo "kernel      : $(uname -r)"
    echo "ac online   : $(cat /sys/class/power_supply/A*/online 2>/dev/null | head -n 1 || echo n/a)"
    echo "governor    : $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo n/a)"
    echo "epp         : $(cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference 2>/dev/null || echo n/a)"
    echo "pkg temp    : $(pkg_c) C"
    echo "gpu         : $(nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv,noheader 2>/dev/null || echo n/a)"
    echo "gpu limits  : $(nvidia-smi --query-gpu=enforced.power.limit,power.max_limit,clocks.max.sm --format=csv,noheader 2>/dev/null || echo n/a) (enforced, max power; max SM clock)"
    echo "gpu procs   : $(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | tr '\n' ';')"
} > "$RUN/run_info.txt"
echo "run folder: $RUN"

# ------------------------------------------------------------ correctness gate
if [[ "$SMOKE" != 1 ]] && want w04; then
    echo "== correctness gate (tests/test_w04_correctness.sh), log: $RUN/logs/gate.log"
    if ! bash tests/test_w04_correctness.sh > "$RUN/logs/gate.log" 2>&1; then
        tail -5 "$RUN/logs/gate.log"; echo "correctness gate failed: no results recorded"; exit 1
    fi
    tail -n 1 "$RUN/logs/gate.log"
fi

# ------------------------------------------------------------ stage commands
B=build
# stage_cmd <KEY> <csv> : sets CMD=(binary args...) and THR (threads, 0 = no OpenMP)
stage_cmd() {
    local key=$1 csv=$2 n
    THR=0
    case "$key" in
        # W01: CPU named flags; GPU positional N I R W [BLOCK] F
        W01_CPU_S1) CMD=($B/W01_VECTOR_ADD_CPU_S1_BASE --size $N_VEC --iterations $I --repetitions $R --warmup $W --output "$csv"); THR=1 ;;
        W01_CPU_S2) CMD=($B/W01_VECTOR_ADD_CPU_S2_SWOPT_OMP --size $N_VEC --iterations $I --repetitions $R --warmup $W --output "$csv"); THR=$T_W01 ;;
        W01_CPU_S3) CMD=($B/W01_VECTOR_ADD_CPU_S3_HWACC_AVX2 --size $N_VEC --iterations $I --repetitions $R --warmup $W --output "$csv"); THR=$T_W01 ;;
        W01_GPU_S1) CMD=($B/W01_VECTOR_ADD_GPU_S1_BASE_NAIVE $N_VEC $I $R $W "$csv") ;;
        W01_GPU_S2) CMD=($B/W01_VECTOR_ADD_GPU_S2_SWOPT_COALESCED $N_VEC $I $R $W 128 "$csv") ;;
        # W02: CPU positional N I R W [T] F; GPU S1 named flags, GPU S2 positional
        W02_CPU_S1) CMD=($B/W02_DOT_PRODUCT_CPU_S1_BASE $N_VEC $I $R $W "$csv"); THR=1 ;;
        W02_CPU_S2) CMD=($B/W02_DOT_PRODUCT_CPU_S2_SWOPT_OMP $N_VEC $I $R $W $T_W02 "$csv"); THR=$T_W02 ;;
        W02_CPU_S3) CMD=($B/W02_DOT_PRODUCT_CPU_S3_HWACC_AVX2_FMA $N_VEC $I $R $W $T_W02 "$csv"); THR=$T_W02 ;;
        W02_GPU_S1) CMD=($B/W02_DOT_PRODUCT_GPU_S1_BASE_NAIVE --size $N_VEC --iterations $I --repetitions $R --warmup $W --output "$csv") ;;
        W02_GPU_S2) CMD=($B/W02_DOT_PRODUCT_GPU_S2_SWOPT_BLOCK_REDUCE $N_VEC $I $R $W "$csv") ;;
        # W03: positional N I R W F (CPU default N is 2048, so N is always passed)
        W03_CPU_S1) CMD=($B/W03_GEMM_CPU_S1_BASE $N_GEMM $I $R $W "$csv"); THR=1 ;;
        W03_CPU_S2) CMD=($B/W03_GEMM_CPU_S2_SWOPT_CACHE_BLOCK $N_GEMM $I $R $W "$csv"); THR=$T_W03 ;;
        W03_CPU_S3) CMD=($B/W03_GEMM_CPU_S3_HWACC_AVX2_FMA $N_GEMM $I $R $W "$csv"); THR=$T_W03 ;;
        W03_GPU_S1) CMD=($B/W03_GEMM_GPU_S1_BASE_NAIVE $N_GEMM $I $R $W "$csv") ;;
        W03_GPU_S2) CMD=($B/W03_GEMM_GPU_S2_SWOPT_SHARED_TILE $N_GEMM $I $R $W "$csv") ;;
        W03_GPU_S3) CMD=($B/W03_GEMM_GPU_S3_HWACC_TENSOR_CORE $N_GEMM $I $R $W "$csv") ;;
        # W04: one named-flag CLI; key suffix _N<n>
        W04_*) n=${key##*_N}
            local p=(--n "$n" "${W04_SHAPE[@]}" --iters $I --reps $R --warmup $W --out "$csv")
            case "${key%_N*}" in
                W04_CPU_S1) CMD=($B/W04_ATTENTION_CPU_S1_BASE "${p[@]}"); THR=1 ;;
                W04_CPU_S2) CMD=($B/W04_ATTENTION_CPU_S2_SWOPT_BLOCKED_OMP "${p[@]}"); THR=$T_W04 ;;
                W04_CPU_S3) CMD=($B/W04_ATTENTION_CPU_S3_HWACC_AVX2 "${p[@]}"); THR=$T_W04 ;;
                W04_GPU_S1) CMD=($B/W04_ATTENTION_GPU_S1_BASE_NAIVE "${p[@]}") ;;
                W04_GPU_S2a) CMD=($B/W04_ATTENTION_GPU_S2a_SWOPT_UNFUSED "${p[@]}") ;;
                W04_GPU_S2b) CMD=($B/W04_ATTENTION_GPU_S2b_SWOPT_FUSED "${p[@]}") ;;
                W04_GPU_S3_FP16) CMD=($B/W04_ATTENTION_GPU_S3_HWACC_TENSOR_CORE_FP16 "${p[@]}") ;;
                W04_GPU_S3_TF32) CMD=($B/W04_ATTENTION_GPU_S3_HWACC_TENSOR_CORE_TF32 "${p[@]}") ;;
                W04_GPU_REF) CMD=($B/W04_ATTENTION_GPU_REF_CUBLAS "${p[@]}") ;;
                *) echo "unknown key $key"; return 1 ;;
            esac ;;
        *) echo "unknown key $key"; return 1 ;;
    esac
}

GPU_KEYS=(); CPU_KEYS=()
want w01 && { GPU_KEYS+=(W01_GPU_S1 W01_GPU_S2); CPU_KEYS+=(W01_CPU_S1 W01_CPU_S2 W01_CPU_S3); }
want w02 && { GPU_KEYS+=(W02_GPU_S1 W02_GPU_S2); CPU_KEYS+=(W02_CPU_S1 W02_CPU_S2 W02_CPU_S3); }
want w03 && { GPU_KEYS+=(W03_GPU_S1 W03_GPU_S2 W03_GPU_S3); CPU_KEYS+=(W03_CPU_S1 W03_CPU_S2 W03_CPU_S3); }
if want w04; then
    for n in "${W04_NS[@]}"; do
        for s in S1 S2a S2b S3_FP16 S3_TF32 REF; do GPU_KEYS+=("W04_GPU_${s}_N$n"); done
        for s in S1 S2 S3; do CPU_KEYS+=("W04_CPU_${s}_N$n"); done
    done
fi

# ------------------------------------------------------------ one run
# CPU sampler: "<temp mC> <mean P-core kHz> <mean E-core kHz>" every 0.2 s
sampler() {
    local f t p e c np=0 ne=0
    while :; do
        p=0; e=0; np=0; ne=0
        for c in $(seq 0 $((NCPU - 1))); do
            read -r f < /sys/devices/system/cpu/cpu$c/cpufreq/scaling_cur_freq 2>/dev/null || continue
            if (( c < 12 )); then p=$((p + f)); np=$((np + 1)); else e=$((e + f)); ne=$((ne + 1)); fi
        done
        t=0; [[ -n "$PKG_ZONE" ]] && read -r t < "$PKG_ZONE"
        echo "$t $((np ? p / np : 0)) $((ne ? e / ne : 0))"
        sleep 0.2
    done
}

echo "order,round,key,device,threads,start_c,waited_s,mean_c,max_c,mean_p_mhz,mean_e_mhz,exit_code,wall_s,correct_line,csv,log,t_start,t_end,primed" > "$RUN/runs.csv"
ORDER=0; FAILS=0
# run_one <device> <round> <KEY>
run_one() {
    local dev=$1 round=$2 key=$3 sfx="" waited=0 start
    (( ROUNDS > 1 )) && [[ $dev == GPU ]] && sfx="_r$round"
    local csv="$RUN/csv/$key$sfx.csv" log="$RUN/logs/$key$sfx.log"
    stage_cmd "$key" "$csv" || return
    if [[ ! -x "${CMD[0]}" ]]; then
        echo "$((++ORDER)),$round,$key,$dev,$THR,,,,,,,127,0,MISSING,$csv,$log,,,0" >> "$RUN/runs.csv"
        printf '  MISSING %-24s %s\n' "$key$sfx" "${CMD[0]}"; FAILS=$((FAILS + 1)); return
    fi
    if [[ $dev == CPU ]]; then
        while (( $(pkg_c) > START_C && waited < WAIT_S )); do sleep 2; waited=$((waited + 2)); done
    fi
    # GPU W01-W03: one untimed priming run of the same binary right before the
    # measured run. Their timed windows are 0.04-0.9 s, shorter than the time the
    # driver takes to raise the memory clock after a gap in GPU load (7001 ->
    # 8001 MHz), so without it a stage could run at either memory clock.
    local primed=0
    if [[ $dev == GPU && $key == W0[123]_* ]]; then
        local sI=$I sR=$R sW=$W
        R=1; W=1
        mkdir -p "$RUN/logs/prime"
        stage_cmd "$key" "$RUN/logs/prime/$key$sfx.csv"
        echo "[$(date +%FT%T.%3N)] PRIME ${CMD[*]}" >> "$RUN/commands.log"
        "${CMD[@]}" > "$RUN/logs/prime/$key$sfx.log" 2>&1 && primed=1
        I=$sI; R=$sR; W=$sW
        stage_cmd "$key" "$csv"
    fi
    start=$(pkg_c)
    local envs=(OMP_PROC_BIND=close OMP_PLACES=cores)
    (( THR > 0 )) && envs+=(OMP_NUM_THREADS="$THR")
    echo "[$(date +%FT%T.%3N)] ${envs[*]} ${CMD[*]}" >> "$RUN/commands.log"
    local samp="$RUN/logs/.samp_$$" t0 t1 rc spid
    sampler > "$samp" & spid=$!
    t0=$(date +%s.%N)
    env "${envs[@]}" "${CMD[@]}" > "$log" 2>&1; rc=$?
    t1=$(date +%s.%N)
    kill "$spid" 2>/dev/null; wait "$spid" 2>/dev/null
    local st; st=$(awk '{t+=$1; if ($1>m) m=$1; p+=$2; e+=$3; n++} END {if (n) printf "%.1f,%.1f,%.0f,%.0f", t/n/1000, m/1000, p/n/1000, e/n/1000; else printf ",,,"}' "$samp")
    rm -f "$samp"
    local cl; cl=$(grep -E '^Correct *: ' "$log" | tail -n 1 | sed 's/  */ /g; s/,/;/g')
    local wall; wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN {printf "%.1f", b - a}')
    echo "$((++ORDER)),$round,$key,$dev,$THR,$start,$waited,$st,$rc,$wall,${cl:-},$csv,$log,$t0,$t1,$primed" >> "$RUN/runs.csv"
    local tag="ok"; (( rc != 0 )) && { tag="FAIL"; FAILS=$((FAILS + 1)); }
    printf '  %-4s %-24s %6ss  exit=%d  %s\n' "$tag" "$key$sfx" "$wall" "$rc" "${cl:-}"
}

# ------------------------------------------------------------ GPU block
if (( ${#GPU_KEYS[@]} > 0 )); then
    echo "== GPU: ${#GPU_KEYS[@]} stage runs x $ROUNDS round(s), order rotated per round"
    nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,pstate,clocks_event_reasons.active \
        --format=csv -lms 250 > "$RUN/gpu_state.csv" 2>/dev/null & GPID=$!
    for round in $(seq 1 "$ROUNDS"); do
        echo "-- round $round"
        nk=${#GPU_KEYS[@]}; off=$(( (round - 1) * nk / ROUNDS ))
        for ((i = 0; i < nk; i++)); do run_one GPU "$round" "${GPU_KEYS[$(( (i + off) % nk ))]}"; done
    done
    kill "$GPID" 2>/dev/null; wait "$GPID" 2>/dev/null
fi

# ------------------------------------------------------------ CPU block
if (( ${#CPU_KEYS[@]} > 0 )); then
    echo "== CPU: ${#CPU_KEYS[@]} stage runs, cool-down to <= ${START_C} C before each"
    for key in "${CPU_KEYS[@]}"; do run_one CPU 1 "$key"; done
fi

echo "== done: $ORDER runs, $FAILS with a non-zero exit or missing binary"
echo "run folder: $RUN"
python3 tools/final_summary.py "$RUN"
