#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# W04 ATTENTION — profiling (P6, decision C32)
#
#   bash scripts/w04_profile.sh e3            write the E3 ncu commands (DRAM bytes, H1 counters)
#   bash scripts/w04_profile.sh e4            write the E4 ncu commands (stalls, occupancy, source)
#   bash scripts/w04_profile.sh e4x           write the P6 follow-up ncu commands (issues 16, 27, 17)
#   bash scripts/w04_profile.sh e5            run Nsight Systems: launch gaps of S1/S2a/REF/S2b/S3
#   bash scripts/w04_profile.sh e8            run CPU counters: perf stat (differential) + VTune hotspots
#   bash scripts/w04_profile.sh import <dir>  export every ncu report in <dir> to CSV
#
# Output: profiling/W04_ATTENTION/<exp>/<YYYYMMDD-HHMMSS>/ (refused if it exists)
#
# e3 / e4: Nsight Compute needs root on this machine, so nothing is run. The
# script writes <dir>/ncu_commands.sh (one ncu command per line, absolute
# paths) and prints how to run it with sudo. Every command profiles one kernel
# (-k regex, -c 1, --launch-skip to pick a later match) of one run with --cache-control all --clock-control
# base, so the counters are taken at the base clock from a flushed cache, not
# under the 55 W power cap of the benchmarks. The binary's own CSV goes to
# <dir>/csv/. Afterwards: bash scripts/w04_profile.sh import <dir>.
#
# e5: nsys profile -t cuda; per run the cuda_gpu_trace CSV export is kept and
# tools/w04_nsys_gaps.py writes <dir>/e5_gaps.txt.
#
# e8: per (stage, N) two runs that differ only in the number of timed passes
# (A: P_A passes, B: P_B passes). Every run also does the untimed setup and
# the double-reference check; those are identical in A and B, so
# (count_B - count_A) / (P_B - P_A) is the count of one forward pass alone.
# Counters: perf stat (P-core and E-core events, two event groups so no
# group is multiplexed). VTune hotspots (CPU time per function) gives the
# share of the exp functions; the reference's functions are listed apart.
# tools/w04_e8_counters.py writes <dir>/e8_counters.txt.
#
# Binaries from a "-dirty" tree are refused unless ALLOW_DIRTY=1.
# Environment overrides (smoke tests): QUICK=1, PROFILE_ROOT, ALLOW_DIRTY=1.
# -----------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)

usage() { sed -n '4,11p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
[[ $# -ge 1 ]] || usage
CMD=$1

QUICK=${QUICK:-0}
PROFILE_ROOT=${PROFILE_ROOT:-profiling/W04_ATTENTION}
TS=$(date +%Y%m%d-%H%M%S)
NCU=${NCU:-/opt/nvidia/nsight-compute/2026.3.0/ncu}   # absolute path, one kernel per command
NSYS=${NSYS:-/usr/local/cuda-13.4/bin/nsys}
VTUNE=${VTUNE:-/opt/intel/oneapi/vtune/2026.6/bin64/vtune}

C1=W04_ATTENTION_CPU_S1_BASE
C2=W04_ATTENTION_CPU_S2_SWOPT_BLOCKED_OMP
C3=W04_ATTENTION_CPU_S3_HWACC_AVX2
G1=W04_ATTENTION_GPU_S1_BASE_NAIVE
G2A=W04_ATTENTION_GPU_S2a_SWOPT_UNFUSED
GREF=W04_ATTENTION_GPU_REF_CUBLAS
G2B=W04_ATTENTION_GPU_S2b_SWOPT_FUSED
G3H=W04_ATTENTION_GPU_S3_HWACC_TENSOR_CORE_FP16
G3T=W04_ATTENTION_GPU_S3_HWACC_TENSOR_CORE_TF32

SHAPE=(--bh 8 --d 64)
ONE=(--iters 1 --reps 1 --warmup 0)          # ncu: the counters come from one launch
NS_E3=(2048 4096)
N_E4=2048
NS_E5=(512 2048)
NS_E8=(2048 4096)
NTHREADS=20
if [[ "$QUICK" == 1 ]]; then
    SHAPE=(--bh 2 --d 64); NS_E3=(256); N_E4=256; NS_E5=(128 256); NS_E8=(256)
fi

# ------------------------------------------------------------------ helpers
check_sha() {
    local sha
    sha=$(sed -n 's/^#define W04_GIT_SHA "\(.*\)"/\1/p' build/w04_build_info.h 2>/dev/null || true)
    [[ -n "$sha" ]] || { echo "build/w04_build_info.h missing (run: make w04)"; exit 1; }
    if [[ "$sha" == nogit || "$sha" == *-dirty ]]; then
        [[ "${ALLOW_DIRTY:-0}" == 1 ]] || {
            echo "refusing to profile build '$sha' (commit, rebuild, or set ALLOW_DIRTY=1)"; exit 1; }
        echo "WARNING: profiling build '$sha' (ALLOW_DIRTY=1)"
    fi
    BUILD_SHA=$sha
}

new_dir() {   # <experiment> -> sets DIR
    DIR="$PROFILE_ROOT/$1/$TS"
    [[ -e "$DIR" ]] && { echo "output folder exists: $DIR"; exit 1; }
    mkdir -p "$DIR/csv"
    ADIR=$(realpath "$DIR")   # absolute, for the generated root-run commands
    {
        echo "date      : $(date -Iseconds)"
        echo "command   : $0 $CMD (QUICK=$QUICK)"
        echo "build sha : $BUILD_SHA"
        echo "gpu       : $(nvidia-smi --query-gpu=name,clocks.sm,power.draw,temperature.gpu,enforced.power.limit --format=csv,noheader 2>/dev/null || echo n/a)"
        echo "ncu       : $("$NCU" --version 2>/dev/null | tail -n 1 || echo n/a)"
        echo "nsys      : $("$NSYS" --version 2>/dev/null || echo n/a)"
    } > "$DIR/run_info.txt"
}

# ---------------------------------------------------------------- ncu (e3, e4)
NCU_LINES=()
# ncu_cmd <name> <kernel regex> <skip> "<ncu what>" <binary> [binary args...]
# Always -c 1: one kernel launch per command. <skip> = matching launches to
# skip first (0 = the first match).
ncu_cmd() {
    local name=$1 kre=$2 skip=$3 what=$4 bin=$5; shift 5
    NCU_LINES+=("env OMP_NUM_THREADS=$NTHREADS $NCU -f -c 1 --launch-skip $skip -k regex:'$kre' --cache-control all --clock-control base $what -o $ADIR/$name $REPO/build/$bin --n ${N_CUR} ${SHAPE[*]} ${ONE[*]} $* --out $ADIR/csv/$name.csv")
}

write_ncu_script() {
    local f="$DIR/ncu_commands.sh"
    {
        echo "#!/usr/bin/env bash"
        echo "# Run as: sudo bash $ADIR/ncu_commands.sh"
        echo "# Generated by scripts/w04_profile.sh $CMD at $TS from build $BUILD_SHA; ${#NCU_LINES[@]} ncu commands."
        echo "set -u"
        echo "cd $REPO"
        echo "fail=0"
        local l
        for l in "${NCU_LINES[@]}"; do
            echo "echo \"== \$(date +%T) ${l##* --out $ADIR/csv/}\""
            echo "$l || { echo FAILED; fail=\$((fail + 1)); }"
        done
        echo "chown -R $(id -u):$(id -g) $ADIR"
        echo "echo \"done: \$fail failed; next: bash scripts/w04_profile.sh import $DIR\""
    } > "$f"
    echo "${#NCU_LINES[@]} ncu commands written to $f"
    echo "Run them (needs root):"
    echo "    sudo bash $ADIR/ncu_commands.sh"
    echo "then: bash scripts/w04_profile.sh import $DIR"
}

e3() {
    new_dir e3
    local metrics="--metrics dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,gpu__time_duration.sum"
    # H1 (C33): the same commands for S1 also carry SpeedOfLight and the
    # global-load sectors per request (coalescing of the naive kernels).
    local h1="--section SpeedOfLight --metrics dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__t_requests_pipe_lsu_mem_global_op_st.sum"
    local n
    for n in "${NS_E3[@]}"; do
        N_CUR=$n
        ncu_cmd "S1_qk_n$n"      'qk_naive'      0 "$h1" "$G1"
        ncu_cmd "S1_softmax_n$n" 'softmax_naive' 0 "$h1" "$G1"
        ncu_cmd "S1_pv_n$n"      'pv_naive'      0 "$h1" "$G1"
        ncu_cmd "REF_qk_n$n"      'gemm'              0 "$metrics" "$GREF"   # 1st cuBLAS GEMM: S = Q·Kᵀ
        ncu_cmd "REF_pv_n$n"      'gemm'              1 "$metrics" "$GREF"   # 2nd: O = P·V
        ncu_cmd "REF_softmax_n$n" 'softmax_rows_warp' 0 "$metrics" "$GREF"
        ncu_cmd "S2b_64x32w4_n$n" 'flash_fwd_fp32' 0 "$metrics" "$G2B" --br 64 --bc 32 --warps 4
        ncu_cmd "S3FP16_tuned_n$n" 'flash_fwd_tc' 0 "$metrics" "$G3H"
        ncu_cmd "S3TF32_tuned_n$n" 'flash_fwd_tc' 0 "$metrics" "$G3T"
    done
    # H1 contrast: the tiled unfused kernels under the same H1 metrics.
    N_CUR=${NS_E3[0]}
    ncu_cmd "S2a_qk_h1_n$N_CUR"      'qk_tiled'          0 "$h1" "$G2A"
    ncu_cmd "S2a_softmax_h1_n$N_CUR" 'softmax_rows_warp' 0 "$h1" "$G2A"
    ncu_cmd "S2a_pv_h1_n$N_CUR"      'pv_tiled'          0 "$h1" "$G2A"
    write_ncu_script
}

e4() {
    new_dir e4
    N_CUR=$N_E4
    local wss="--section WarpStateStats --section Occupancy --section LaunchStats"
    local ld="--metrics smsp__inst_executed_op_shared_ld.sum,smsp__inst_executed_op_ldsm.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,smsp__inst_executed.sum,smsp__warps_eligible.avg.per_cycle_active,smsp__warps_active.avg.per_cycle_active"
    # P5 hypothesis: 64x32 W4 wins through 2 blocks/SM (fewer barrier stalls) or loads per FMA.
    ncu_cmd "S2b_64x32w4_n${N_CUR}_wss" 'flash_fwd_fp32' 0 "$wss" "$G2B" --br 64 --bc 32 --warps 4
    ncu_cmd "S2b_64x64w8_n${N_CUR}_wss" 'flash_fwd_fp32' 0 "$wss" "$G2B" --br 64 --bc 64 --warps 8
    ncu_cmd "S2b_64x32w4_n${N_CUR}_ld"  'flash_fwd_fp32' 0 "$ld"  "$G2B" --br 64 --bc 32 --warps 4
    ncu_cmd "S2b_64x64w8_n${N_CUR}_ld"  'flash_fwd_fp32' 0 "$ld"  "$G2B" --br 64 --bc 64 --warps 8
    # Issue 19: 12 vs 8 warps per SM give the same time.
    ncu_cmd "S3FP16_64x32w4_n${N_CUR}_wss" 'flash_fwd_tc' 0 "$wss" "$G3H" --br 64 --bc 32 --warps 4
    ncu_cmd "S3FP16_64x64w4_n${N_CUR}_wss" 'flash_fwd_tc' 0 "$wss" "$G3H" --br 64 --bc 64 --warps 4
    # Issue 20: which instructions carry the math-pipe-throttle stalls.
    ncu_cmd "S3FP16_64x64w4_n${N_CUR}_src" 'flash_fwd_tc' 0 "--section SourceCounters --section WarpStateStats" \
        "$G3H" --br 64 --bc 64 --warps 4
    write_ncu_script
}

# P6 follow-ups (open issues 16, 27, 17), each one kernel per command:
#   27  SourceCounters + WarpStateStats on S2b 64x32 W4 and 64x64 W8 (N = N_E4):
#       which instructions carry the short-scoreboard stalls of each variant
#   16  S2b reads above the compulsory Q, K, V bytes: L2 hit rate and L2 read
#       misses from L1 for S2b 64x32 W4 and 64x64 W8 at N = the last NS_E3
#   17  TF32 after the explicit shared-memory fragment loads: WarpStateStats +
#       Occupancy (was: generic LD.E, long scoreboard) and the E3 DRAM metrics
#       at every NS_E3 (the TF32 Roofline point moves with the new kernel)
e4x() {
    new_dir e4x
    local wss="--section WarpStateStats --section Occupancy --section LaunchStats"
    local src="--section SourceCounters --section WarpStateStats"
    local l2="--metrics dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,gpu__time_duration.sum,lts__t_sector_hit_rate.pct,lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum,lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum"
    local metrics="--metrics dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,gpu__time_duration.sum"
    local n nl=${NS_E3[${#NS_E3[@]}-1]}
    N_CUR=$N_E4
    ncu_cmd "S2b_64x32w4_n${N_CUR}_src" 'flash_fwd_fp32' 0 "$src" "$G2B" --br 64 --bc 32 --warps 4
    ncu_cmd "S2b_64x64w8_n${N_CUR}_src" 'flash_fwd_fp32' 0 "$src" "$G2B" --br 64 --bc 64 --warps 8
    N_CUR=$nl
    ncu_cmd "S2b_64x32w4_n${N_CUR}_l2" 'flash_fwd_fp32' 0 "$l2" "$G2B" --br 64 --bc 32 --warps 4
    ncu_cmd "S2b_64x64w8_n${N_CUR}_l2" 'flash_fwd_fp32' 0 "$l2" "$G2B" --br 64 --bc 64 --warps 8
    N_CUR=$N_E4
    ncu_cmd "S3TF32_64x32w4_n${N_CUR}_wss" 'flash_fwd_tc' 0 "$wss" "$G3T" --br 64 --bc 32 --warps 4
    for n in "${NS_E3[@]}"; do
        N_CUR=$n
        ncu_cmd "S3TF32_tuned_n$n" 'flash_fwd_tc' 0 "$metrics" "$G3T"
    done
    write_ncu_script
}

import_dir() {
    local d=$1 f base n=0
    [[ -d "$d" ]] || { echo "no such folder: $d"; exit 1; }
    for f in "$d"/*.ncu-rep "$d"/*.ncu-repz; do
        [[ -e "$f" ]] || continue
        base=${f%.ncu-rep*}
        "$NCU" --import "$f" --page raw --csv > "$base.raw.csv"
        "$NCU" --import "$f" --page details --csv > "$base.details.csv"
        if [[ "$base" == *_src ]]; then
            "$NCU" --import "$f" --page source --print-source sass --csv > "$base.source.csv"
        fi
        n=$((n + 1))
    done
    echo "exported $n reports in $d"
}

# ---------------------------------------------------------------- nsys (e5)
e5() {
    new_dir e5
    local -a entries=("S1|$G1|" "S2a|$G2A|" "REF|$GREF|" "S2b_tuned|$G2B|" "S3FP16_tuned|$G3H|")
    local e name bin extra n fails=0
    for n in "${NS_E5[@]}"; do
        for e in "${entries[@]}"; do
            IFS='|' read -r name bin extra <<<"$e"
            local out="$DIR/${name}_n$n" log="$DIR/${name}_n$n.log"
            local -a run=(env OMP_NUM_THREADS=$NTHREADS "build/$bin" --n "$n" "${SHAPE[@]}"
                          --warmup 2 --iters 20 --reps 1 --out "$DIR/csv/${name}_n$n.csv")
            echo "${run[*]}" >> "$DIR/commands.log"
            if "$NSYS" profile -t cuda --force-overwrite true -o "$out" "${run[@]}" > "$log" 2>&1 \
               && [[ "$(grep -E '^Correct : ' "$log" | tail -n 1)" == "Correct : YES" ]] \
               && "$NSYS" stats --report cuda_gpu_trace --format csv --force-export true \
                      -o "$out" "$out.nsys-rep" >> "$log" 2>&1; then
                echo "  ok    ${name}_n$n"
            else
                echo "  FAIL  ${name}_n$n (log: $log)"; fails=$((fails + 1))
            fi
            rm -f "$out.sqlite"
        done
    done
    python3 tools/w04_nsys_gaps.py "$DIR" | tee "$DIR/e5_gaps.txt"
    echo "== e5 done: $fails failed; $DIR =="
    (( fails == 0 ))
}

# ---------------------------------------------------------- perf + VTune (e8)
# Two groups so that no event is multiplexed: P-core cache levels with E-core
# totals, then P-core FP arithmetic by width.
PERF_G1="cpu_core/cycles/,cpu_core/instructions/,cpu_core/mem_load_retired.l1_miss/,cpu_core/mem_load_retired.l2_hit/,cpu_core/mem_load_retired.l2_miss/,cpu_core/mem_load_retired.l3_miss/,cpu_atom/cycles/,cpu_atom/instructions/,cpu_atom/mem_load_uops_retired.l2_hit/,cpu_atom/mem_load_uops_retired.l2_miss/"
PERF_G2="cpu_core/cycles/,cpu_core/instructions/,cpu_core/fp_arith_inst_retired.scalar_single/,cpu_core/fp_arith_inst_retired.128b_packed_single/,cpu_core/fp_arith_inst_retired.256b_packed_single/,cpu_atom/cycles/,cpu_atom/instructions/"

e8_passes() {   # <stage tag> <N> -> "P_A P_B threads" (timed passes per run)
    local st=$1 n=$2
    if [[ "$QUICK" == 1 ]]; then echo "1 3 $NTHREADS"; return; fi
    case "$st" in
        S1) (( n >= 4096 )) && echo "1 3 1" || echo "2 6 1" ;;      # S1 is single-threaded (P1 sweep)
        S2*) (( n >= 4096 )) && echo "5 25 $NTHREADS" || echo "10 50 $NTHREADS" ;;
        S3) (( n >= 4096 )) && echo "10 50 $NTHREADS" || echo "20 100 $NTHREADS" ;;
    esac
}

e8() {
    command -v perf >/dev/null || { echo "perf not found"; exit 1; }
    make --no-print-directory w04-flagstudy >/dev/null
    new_dir e8
    {
        echo "perf      : $(perf --version)"
        echo "paranoid  : $(cat /proc/sys/kernel/perf_event_paranoid)"
        echo "vtune     : $("$VTUNE" --version 2>/dev/null | head -n 1 || echo n/a)"
        echo "perf g1   : $PERF_G1"
        echo "perf g2   : $PERF_G2"
    } >> "$DIR/run_info.txt"
    # S2 is the default -O3 build (SSE, scalar libm expf); S2.O3nf the flag-study
    # build with libmvec vector expf; S3 the AVX2 intrinsics with exp256_ps.
    local -a entries=("S1|build/$C1" "S2|build/$C2" "S2_O3nf|build/flagstudy/$C2.O3nf" "S3|build/$C3")
    local e tag bin n g p fails=0
    for n in "${NS_E8[@]}"; do
        for e in "${entries[@]}"; do
            IFS='|' read -r tag bin <<<"$e"
            local pa pb th
            read -r pa pb th < <(e8_passes "$tag" "$n")
            for p in "$pa" "$pb"; do
                for g in 1 2; do
                    local ev=$PERF_G1; [[ $g == 2 ]] && ev=$PERF_G2
                    local name="${tag}_n${n}_p${p}_g$g"
                    local -a run=(env OMP_NUM_THREADS=$th OMP_PROC_BIND=close OMP_PLACES=cores "$bin"
                                  --n "$n" "${SHAPE[@]}" --warmup 0 --iters "$p" --reps 1 --out "$DIR/csv/$tag.csv")
                    echo "perf stat -x, -e <g$g> ${run[*]}" >> "$DIR/commands.log"
                    if perf stat -x, -o "$DIR/$name.perf.csv" -e "$ev" "${run[@]}" > "$DIR/$name.log" 2>&1 \
                       && [[ "$(grep -E '^Correct : ' "$DIR/$name.log" | tail -n 1)" == "Correct : YES" ]]; then
                        echo "  ok    perf $name"
                    else
                        echo "  FAIL  perf $name (log: $DIR/$name.log)"; fails=$((fails + 1))
                    fi
                done
            done
            # VTune hotspots: CPU time per function, run B only.
            local vr="$DIR/vtune_${tag}_n$n"
            local -a run=(env OMP_NUM_THREADS=$th OMP_PROC_BIND=close OMP_PLACES=cores "$bin"
                          --n "$n" "${SHAPE[@]}" --warmup 0 --iters "$pb" --reps 1 --out "$DIR/csv/${tag}_vtune.csv")
            echo "vtune -collect hotspots ${run[*]}" >> "$DIR/commands.log"
            if "$VTUNE" -collect hotspots -r "$vr" -- "${run[@]}" > "$vr.log" 2>&1 \
               && "$VTUNE" -report hotspots -r "$vr" -group-by function -format csv -csv-delimiter comma \
                      > "$DIR/vtune_${tag}_n$n.hotspots.csv" 2>> "$vr.log"; then
                echo "  ok    vtune ${tag}_n$n"
            else
                echo "  FAIL  vtune ${tag}_n$n (log: $vr.log)"; fails=$((fails + 1))
            fi
        done
    done
    python3 tools/w04_e8_counters.py "$DIR" | tee "$DIR/e8_counters.txt"
    echo "== e8 done: $fails failed; $DIR =="
    (( fails == 0 ))
}

# ------------------------------------------------------------------- main
case "$CMD" in
    import) [[ $# -eq 2 ]] || usage; import_dir "$2"; exit 0 ;;
    e3|e4|e4x|e5|e8) [[ $# -eq 1 ]] || usage ;;
    *) usage ;;
esac
make --no-print-directory w04 >/dev/null
check_sha
case "$CMD" in
    e3) e3 ;;
    e4) e4 ;;
    e4x) e4x ;;
    e5) e5 ;;
    e8) e8 ;;
esac
