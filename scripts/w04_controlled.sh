#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# W04 ATTENTION — thermally controlled CPU rerun (P1 follow-up)
#
#   bash scripts/w04_controlled.sh
#
# Why: in the first P1 session the same configuration measured up to 40% apart
# in different experiments, while reps within one run agreed to 1–3%, and the
# package reached ~97 °C. This rerun separates configuration from machine
# state:
#   * every configuration runs in ROUNDS rounds; the order inside each round
#     is shuffled with a fixed seed, so drift is spread across configurations
#   * before each run it waits until x86_pkg_temp <= START_C (max WAIT_S seconds)
#     and records the temperature it actually started at
#   * during each run a sampler records package temperature and the mean
#     scaling_cur_freq of the P-cores (CPUs 0–11) and E-cores (12–19) every
#     0.2 s (whole process, including the untimed correctness check)
#
# Configurations:
#   tiles   S2, S3 x Br in {32,64,128} x Bc in {32,64,128} x N in {1024,4096}, T=20
#   threads S2 at T in {6,12,14,20} x N in {1024,2048} (the N=2048 plateau)
#
# Output: results/raw/W04_ATTENTION/<YYYYMMDD-HHMMSS>-controlled/
#   run_info.txt, commands.log, this script (copy), state.csv (one row per run:
#   order, round, config, start/mean/max temperature, mean P/E clock),
#   tiles/<STAGE_ID>.csv and threads/<STAGE_ID>.csv (normal stage CSVs, row
#   order matches state.csv), logs/
#
# Overrides: ROUNDS (3), START_C (70), WAIT_S (120), SEED (1), QUICK=1, RESULTS_ROOT,
# ALLOW_DIRTY=1. No root needed. Turbo state is recorded, not changed.
# -----------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."

ROUNDS=${ROUNDS:-3}
START_C=${START_C:-70}
WAIT_S=${WAIT_S:-120}
SEED=${SEED:-1}
QUICK=${QUICK:-0}
RESULTS_ROOT=${RESULTS_ROOT:-results/raw/W04_ATTENTION}
RUN="$RESULTS_ROOT/$(date +%Y%m%d-%H%M%S)-controlled"

S2=W04_ATTENTION_CPU_S2_SWOPT_BLOCKED_OMP
S3=W04_ATTENTION_CPU_S3_HWACC_AVX2
PROTO=(--bh 8 --d 64 --warmup 2 --iters 20 --reps 5)
TILE_NS=(1024 4096); TILE_B=(32 64 128); THR_NS=(1024 2048); THR_T=(6 12 14 20)
if [[ "$QUICK" == 1 ]]; then
    PROTO=(--bh 2 --d 64 --warmup 0 --iters 1 --reps 2)
    TILE_NS=(256); TILE_B=(32 64); THR_NS=(256); THR_T=(1 20); ROUNDS=2; START_C=200
fi

PKG_ZONE=""
for z in /sys/class/thermal/thermal_zone*; do
    [[ "$(cat "$z/type")" == x86_pkg_temp ]] && PKG_ZONE="$z/temp"
done
[[ -n "$PKG_ZONE" ]] || { echo "x86_pkg_temp thermal zone not found"; exit 1; }
pkg_c() { echo $(( $(cat "$PKG_ZONE") / 1000 )); }

make --no-print-directory w04 >/dev/null
SHA=$(sed -n 's/^#define W04_GIT_SHA "\(.*\)"/\1/p' build/w04_build_info.h)
if [[ "$SHA" == nogit || "$SHA" == *-dirty ]] && [[ "${ALLOW_DIRTY:-0}" != 1 ]]; then
    echo "refusing to record results from build '$SHA' (commit, rebuild, or set ALLOW_DIRTY=1)"; exit 1
fi
[[ -e "$RUN" ]] && { echo "output folder exists: $RUN"; exit 1; }
mkdir -p "$RUN/logs"
cp "$0" "$RUN/w04_controlled.sh"
{
    echo "date        : $(date -Iseconds)"
    echo "build sha   : $SHA"
    echo "settings    : ROUNDS=$ROUNDS START_C=$START_C WAIT_S=$WAIT_S SEED=$SEED QUICK=$QUICK"
    echo "ac online   : $(cat /sys/class/power_supply/A*/online 2>/dev/null | head -n 1 || echo n/a)"
    echo "governor    : $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo n/a)"
    echo "epp         : $(cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference 2>/dev/null || echo n/a)"
    echo "no_turbo    : $(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo n/a)"
    echo "pkg temp    : $(pkg_c) C"
    echo "gpu         : $(nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv,noheader 2>/dev/null || echo n/a)"
    echo "compiler    : $(g++ --version | head -n 1)"
} > "$RUN/run_info.txt"
echo "run folder: $RUN"

# Configurations: "<experiment> <stage> <threads> <args...>"
CONFIGS=()
for st in "$S2" "$S3"; do
    for n in "${TILE_NS[@]}"; do for br in "${TILE_B[@]}"; do for bc in "${TILE_B[@]}"; do
        CONFIGS+=("tiles $st 20 --n $n --br $br --bc $bc")
    done; done; done
done
for n in "${THR_NS[@]}"; do for t in "${THR_T[@]}"; do
    CONFIGS+=("threads $S2 $t --n $n")
done; done

# Background sampler: "<temp mC> <mean P-core kHz> <mean E-core kHz>" every 0.2 s.
# Uses only shell builtins (read), so it adds almost no load of its own.
sampler() {
    local f t p e c
    while :; do
        p=0; e=0
        for c in 0 1 2 3 4 5 6 7 8 9 10 11; do read -r f < /sys/devices/system/cpu/cpu$c/cpufreq/scaling_cur_freq; p=$((p + f)); done
        for c in 12 13 14 15 16 17 18 19; do read -r f < /sys/devices/system/cpu/cpu$c/cpufreq/scaling_cur_freq; e=$((e + f)); done
        read -r t < "$PKG_ZONE"
        echo "$t $((p / 12)) $((e / 8))"
        sleep 0.2
    done
}

echo "order,round,experiment,stage,threads,args,start_c,waited_s,mean_c,max_c,mean_p_mhz,mean_e_mhz,correct" > "$RUN/state.csv"
FAILS=0
order=0
for round in $(seq 1 "$ROUNDS"); do
    mapfile -t shuffled < <(printf '%s\n' "${CONFIGS[@]}" | \
        shuf --random-source=<(openssl enc -aes-256-ctr -pass pass:"$SEED-$round" -nosalt </dev/zero 2>/dev/null))
    echo "== round $round/$ROUNDS (${#shuffled[@]} configurations) =="
    for cfg in "${shuffled[@]}"; do
        read -r exp st thr args <<<"$cfg"
        order=$((order + 1))
        waited=0
        while (( $(pkg_c) > START_C && waited < WAIT_S )); do sleep 1; waited=$((waited + 1)); done
        start=$(pkg_c)
        tag=$(echo "$args" | tr -d ' -')
        log="$RUN/logs/$(printf '%04d' $order)_${exp}_${st}_t${thr}_${tag}.log"
        samples="$log.samples"
        csv="$RUN/$exp/$st.csv"
        mkdir -p "$RUN/$exp"
        echo "[$order] OMP_NUM_THREADS=$thr OMP_PROC_BIND=close OMP_PLACES=cores build/$st $args ${PROTO[*]} --out $csv" >> "$RUN/commands.log"
        sampler > "$samples" & spid=$!
        # shellcheck disable=SC2086
        if env OMP_NUM_THREADS="$thr" OMP_PROC_BIND=close OMP_PLACES=cores "build/$st" $args "${PROTO[@]}" \
               --out "$csv" > "$log" 2>&1 \
           && [[ "$(grep -E '^Correct : ' "$log" | tail -n 1)" == "Correct : YES" ]]; then ok=YES; else ok=NO; FAILS=$((FAILS + 1)); fi
        kill "$spid" 2>/dev/null; wait "$spid" 2>/dev/null || true
        read -r mean_c max_c pm em < <(awk '{t+=$1; if($1>m)m=$1; p+=$2; e+=$3; n++} END {if(n==0){print "nan nan nan nan"} else printf "%.1f %.1f %.0f %.0f\n", t/n/1000, m/1000, p/n/1000, e/n/1000}' "$samples")
        echo "$order,$round,$exp,$st,$thr,\"$args\",$start,$waited,$mean_c,$max_c,$pm,$em,$ok" >> "$RUN/state.csv"
        printf '  %-4s %-3s %-7s %-40s T=%-2s start=%sC mean=%sC P=%sMHz E=%sMHz | %s\n' "$order" "$ok" "$exp" "${st#W04_ATTENTION_CPU_}" "$thr" "$start" "$mean_c" "$pm" "$em" \
            "$(grep -oE '[0-9.]+ GFLOP/s' "$log" | head -n 1)"
    done
done
echo "== done: $FAILS failed runs; results in $RUN =="
(( FAILS == 0 ))
