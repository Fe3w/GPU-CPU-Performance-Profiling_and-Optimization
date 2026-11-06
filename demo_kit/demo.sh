#!/usr/bin/env bash
# demo.sh — live, on-stage demo for the CPU/GPU profiling project.
#
#   ./demo.sh            interactive menu
#   ./demo.sh check      preflight (run this before you walk on stage)
#   ./demo.sh race [all|gpu|w01|w02|w03|w04|w04cpu ...]
#                        live benchmark race + speedup scoreboard + charts (default: all)
#                        ROUNDS=3 ./demo.sh race w04   interleaved rounds, median over rounds
#   ./demo.sh fusion [--print|--backup]
#                        W04 fusion: ncu DRAM bytes S2a vs S2b (sudo), then the time it buys
#   ./demo.sh ncu        Nsight Compute live: W03 GEMM S3 vs S1
#   ./demo.sh ncu-gui    open Nsight Compute GUI with the reports
#   ./demo.sh vtune      VTune live: W03 CPU naive hotspots (+ compare to S3)
#   ./demo.sh vtune-gui  open VTune GUI with the results
#   ./demo.sh video      play the race animation
#   ./demo.sh wall       bonus: W01 memory-wall profile (DRAM ~90%)
#   ./demo.sh ncu-side   Nsight GUI: naive and Tensor Core reports tiled left/right
#   ./demo.sh vtune-side VTune: S1 vs S3 comparison report + two GUIs tiled
#   ./demo.sh hw         GPU facts from the driver (SMs, registers, L2, occupancy maths)
#   ./demo.sh plot [dir] charts + tables for the newest (or a given) race run
#
# Paths come from demo.conf: by default the repository that contains this kit
# (PROJECT=/other/checkout ./demo.sh ... to use another one).
# Nothing here overwrites earlier results: every run writes to
# results/live/<timestamp>/ and profiling/demo/live_<timestamp>_*.

set -uo pipefail
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=demo.conf
source "$KIT/demo.conf"
PY="python3"

C_T=$'\033[38;5;37m'; C_O=$'\033[38;5;208m'; C_G=$'\033[38;5;245m'
C_B=$'\033[1m'; C_R=$'\033[31m'; C_0=$'\033[0m'

banner() { printf '\n%s━━━ %s ━━━%s\n' "$C_T$C_B" "$1" "$C_0"; }
say()    { printf '%s▸ %s%s\n' "$C_O" "$1" "$C_0"; }
note()   { printf '%s  %s%s\n' "$C_G" "$1" "$C_0"; }
fail()   { printf '%s✗ %s%s\n' "$C_R" "$1" "$C_0"; }
ok()     { printf '%s✓%s %s\n' "$C_T" "$C_0" "$1"; }
# Show the command on screen before running it, so the audience sees it.
run()    { printf '%s$ %s%s\n' "$C_G" "$*" "$C_0" >&2; "$@"; }
pause()  { [[ "${NOPAUSE:-0}" == 1 ]] || read -r -p "${C_G}  [enter] to continue${C_0}" _; }

autodetect_cpu() {
  [[ -z "$W03_CPU_S2" ]] && W03_CPU_S2="$(ls "$BUILD"/W03_GEMM_CPU_S2* 2>/dev/null | head -1)"
  [[ -z "$W03_CPU_S3" ]] && W03_CPU_S3="$(ls "$BUILD"/W03_GEMM_CPU_S3* 2>/dev/null | head -1)"
}
autodetect_cpu

kfilter() { [[ -n "$NCU_KFILTER" ]] && echo "-k $NCU_KFILTER"; }
# ncu 2026.x writes compressed .ncu-repz; older versions write .ncu-rep. Resolve either.
rep() { local e; for e in ncu-repz ncu-rep; do [[ -f "$1.$e" ]] && { echo "$1.$e"; return 0; }; done; return 1; }
latest_rep() { ls -t "$@" 2>/dev/null | head -1; }

# ---------------------------------------------------------------- preflight --
cmd_check() {
  banner "PREFLIGHT"
  cd "$PROJECT" || { fail "PROJECT not found: $PROJECT"; return 1; }
  local bad=0
  for b in W01_GPU_S1 W01_GPU_S2 W02_GPU_S1 W02_GPU_S2 W03_GPU_S1 W03_GPU_S2 W03_GPU_S3 W03_CPU_S1 W03_CPU_S2 W03_CPU_S3; do
    if [[ -x "${!b}" ]]; then ok "$b  ${C_G}${!b#$PROJECT/}${C_0}"; else fail "$b missing: '${!b}' (run: make -C \"$PROJECT\" legacy)"; bad=1; fi
  done
  for b in W04_GPU_S1 W04_GPU_S2a W04_GPU_S2b W04_GPU_S3_FP16 W04_GPU_S3_TF32 W04_GPU_REF W04_CPU_S1 W04_CPU_S2 W04_CPU_S3; do
    if [[ -x "${!b}" ]]; then ok "$b  ${C_G}${!b#$W04_PROJECT/}${C_0}"; else fail "$b missing: '${!b}' (run: make -C $W04_PROJECT w04)"; bad=1; fi
  done
  [[ -n "$(fusion_backup_dir)" ]] && ok "fusion backup  $(fusion_backup_dir)" || note "no fusion backup (FUSION_BACKUP in demo.conf); ./demo.sh fusion runs live only"
  [[ -x "$NCU" ]] && ok "ncu  $($NCU --version | tail -1)" || { fail "ncu not at $NCU"; bad=1; }
  [[ -x "$NCU_UI" ]] && ok "ncu-ui present" || { fail "ncu-ui not at $NCU_UI"; bad=1; }
  command -v vtune >/dev/null && ok "vtune  $(vtune --version | head -1)" || { fail "vtune not on PATH (source /opt/intel/oneapi/setvars.sh)"; bad=1; }
  command -v vtune-gui >/dev/null && ok "vtune-gui present" || fail "vtune-gui not on PATH"
  nvidia-smi --query-gpu=name,pstate,temperature.gpu,clocks.sm --format=csv,noheader \
    && ok "GPU visible" || { fail "nvidia-smi failed"; bad=1; }
  [[ "$(cat /proc/sys/kernel/perf_event_paranoid)" -le 2 ]] \
    && ok "perf_event_paranoid=$(cat /proc/sys/kernel/perf_event_paranoid) (VTune user-mode sampling OK)" \
    || note "perf_event_paranoid>2: run  sudo sysctl kernel.perf_event_paranoid=1"
  [[ -f "$VIDEO" ]] && ok "video found" || note "video not at $VIDEO (set VIDEO in demo.conf)"

  say "Checking which kernel ncu will catch (first launch, or NCU_KFILTER)…"
  mkdir -p "$LIVE"
  if [[ -z "$NCU_SUDO" ]] || sudo -n true 2>/dev/null; then
    $NCU_SUDO "$NCU" $(kfilter) -c 1 --section LaunchStats --print-summary none \
       "$W03_GPU_S3" "$GEMM_N" $PROFILE_ARGS "$LIVE/pre_s3.csv" 2>&1 \
       | grep -E '^\s{2}[A-Za-z_].*\(|==PROF==|Correct' | head -5
  else
    note "skipped (needs sudo). Run 'sudo -v' and re-run check, or run it yourself:"
    note "  $NCU_SUDO $NCU $(kfilter) -c 1 --section LaunchStats --print-summary none $W03_GPU_S3 $GEMM_N $PROFILE_ARGS $LIVE/pre_s3.csv"
  fi
  note "That kernel name should be the Tensor Core GEMM. If it is an init/convert kernel, set NCU_KFILTER in demo.conf."

  say "Backup reports in $DEMO_REPORTS:"
  ls -1 "$DEMO_REPORTS" 2>/dev/null | sed 's/^/    /' || note "none yet — run ./prep.sh"
  [[ $bad == 0 ]] && ok "PREFLIGHT PASSED" || fail "Fix the items above before presenting."
}

# --------------------------------------------------------------------- race --
# Each workload's stages run back to back; with ROUNDS>1 the whole selection
# repeats and files get a _r<k> suffix (median over rounds in the scoreboard).
SFX=""
o() { echo "$LIVE/$1$SFX.csv"; }
# W04 binaries print several lines; keep them in a log and show one line per stage.
runw() {  # runw <label> <binary> args...
  local label="$1" bin="$2"; shift 2
  local log="$LIVE/logs/$label$SFX.log"
  mkdir -p "$LIVE/logs"
  printf '%s$ %s %s%s\n' "$C_G" "${bin#$W04_PROJECT/}" "$*" "$C_0" >&2
  if "$bin" "$@" >"$log" 2>&1 && grep -q '^Correct : YES' "$log"; then
    ok "$label  $(grep -m1 '^summary' "$log" | grep -oE '[0-9.]+ ± [0-9.]+ ms \| [0-9.]+ GFLOP/s')  Correct : YES"
  else
    fail "$label failed or not correct (log: $log)"
  fi
}
# W01-W03 GPU stages are short (0.04-0.9 s timed): one untimed priming run of
# the same binary first, so they do not start at a lower memory or SM clock
# (the same rule as scripts/run_all.sh). Output of the priming run is discarded.
prime() { mkdir -p "$LIVE/prime"; "$@" >/dev/null 2>&1; }
race_w01() {
  say "W01 Vector Add (memory-bound)"
  prime "$W01_GPU_S1" "$VEC_N" 20 1 1 "$LIVE/prime/W01_S1.csv"
  run "$W01_GPU_S1" "$VEC_N" 20 5 2     "$(o W01_S1)" >/dev/null
  prime "$W01_GPU_S2" "$VEC_N" 20 1 1 128 "$LIVE/prime/W01_S2.csv"
  run "$W01_GPU_S2" "$VEC_N" 20 5 2 128 "$(o W01_S2)" >/dev/null
}
race_w02() {
  say "W02 Dot Product (reduction)"
  prime "$W02_GPU_S1" --size "$VEC_N" --iterations 20 --repetitions 1 --warmup 1 --output "$LIVE/prime/W02_S1.csv"
  run "$W02_GPU_S1" --size "$VEC_N" --iterations 20 --repetitions 5 --warmup 2 --output "$(o W02_S1)" >/dev/null
  prime "$W02_GPU_S2" "$VEC_N" 20 1 1 "$LIVE/prime/W02_S2.csv"
  run "$W02_GPU_S2" "$VEC_N" 20 5 2 "$(o W02_S2)" >/dev/null
}
race_w03() {
  say "W03 GEMM (compute-bound)"
  local s; for s in S1 S2 S3; do
    local v="W03_GPU_$s"
    prime "${!v}" "$GEMM_N" 20 1 1 "$LIVE/prime/W03_$s.csv"
    run "${!v}" "$GEMM_N" 20 5 2 "$(o W03_$s)" >/dev/null
  done
}
race_w04() {
  say "W04 Attention, GPU, N=$W04_N (naive → unfused tiled → fused → Tensor Core; cuBLAS reference)"
  local a=(--n "$W04_N" $W04_SHAPE $W04_ARGS)
  runw W04_S1      "$W04_GPU_S1"      "${a[@]}" --out "$(o W04_S1)"
  runw W04_S2a     "$W04_GPU_S2a"     "${a[@]}" --out "$(o W04_S2a)"
  runw W04_S2b     "$W04_GPU_S2b"     "${a[@]}" --out "$(o W04_S2b)"
  runw W04_S3_FP16 "$W04_GPU_S3_FP16" "${a[@]}" --out "$(o W04_S3_FP16)"
  runw W04_S3_TF32 "$W04_GPU_S3_TF32" "${a[@]}" --out "$(o W04_S3_TF32)"
  runw W04_REF     "$W04_GPU_REF"     "${a[@]}" --out "$(o W04_REF)"
}
race_w04cpu() {
  say "W04 Attention, CPU, N=$W04_CPU_N (scalar → blocked+OpenMP → AVX2/FMA)"
  local e=(env OMP_PROC_BIND=close OMP_PLACES=cores)
  [[ -n "$W04_CPU_THREADS" ]] && e+=(OMP_NUM_THREADS="$W04_CPU_THREADS")
  local s=(--n "$W04_CPU_N" $W04_SHAPE)
  runw W04CPU_S1 "${e[@]}" "$W04_CPU_S1" "${s[@]}" $W04_CPU_S1_ARGS --out "$(o W04CPU_S1)"
  runw W04CPU_S2 "${e[@]}" "$W04_CPU_S2" "${s[@]}" $W04_CPU_ARGS    --out "$(o W04CPU_S2)"
  runw W04CPU_S3 "${e[@]}" "$W04_CPU_S3" "${s[@]}" $W04_CPU_ARGS    --out "$(o W04CPU_S3)"
}

cmd_race() {
  local sel=() w
  [[ $# -eq 0 ]] && set -- all
  for w in "$@"; do
    case "$w" in
      all) sel+=($RACE_ALL);; gpu) sel+=($RACE_GPU);;
      w01|w02|w03|w04|w04cpu) sel+=("$w");;
      *) fail "unknown workload '$w' (use all, gpu, w01, w02, w03, w04, w04cpu)"; return 1;;
    esac
  done
  banner "LIVE RACE — same machine, same inputs, only the code changes (${sel[*]}, rounds=$ROUNDS)"
  cd "$PROJECT" || return 1
  mkdir -p "$LIVE"
  nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv,noheader | sed 's/^/  GPU: /'
  nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv > "$LIVE/gpu_state_start.csv" 2>/dev/null
  local r
  for ((r = 1; r <= ROUNDS; r++)); do
    if (( ROUNDS > 1 )); then SFX="_r$r"; say "Round $r of $ROUNDS"; else SFX=""; fi
    for w in "${sel[@]}"; do "race_$w"; done
  done
  SFX=""
  local args=() has=" ${sel[*]} "
  [[ $has == *" w01 "* ]] && args+=(--group "W01 Vector Add · GPU · OI 0.083 FLOP/B" --id W01 S1="$LIVE/W01_S1" S2="$LIVE/W01_S2")
  [[ $has == *" w02 "* ]] && args+=(--group "W02 Dot Product · GPU · OI 0.25 FLOP/B" --id W02 S1="$LIVE/W02_S1" S2="$LIVE/W02_S2")
  [[ $has == *" w03 "* ]] && args+=(--group "W03 GEMM N=$GEMM_N · GPU · OI 170.7 FLOP/B" --id W03 S1="$LIVE/W03_S1" S2="$LIVE/W03_S2" S3="$LIVE/W03_S3")
  [[ $has == *" w04 "* ]] && args+=(--group "W04 Attention N=$W04_N BH=8 d=64 · GPU · OI 15 unfused → ~360 fused (DRAM)" --id W04
      S1="$LIVE/W04_S1" S2a="$LIVE/W04_S2a" S2b="$LIVE/W04_S2b" S3_FP16="$LIVE/W04_S3_FP16" S3_TF32="$LIVE/W04_S3_TF32" REF="$LIVE/W04_REF")
  [[ $has == *" w04cpu "* ]] && args+=(--group "W04 Attention N=$W04_CPU_N BH=8 d=64 · CPU (S1: 1 iter × 3 reps)" --id W04C
      S1="$LIVE/W04CPU_S1" S2="$LIVE/W04CPU_S2" S3="$LIVE/W04CPU_S3")
  if [[ $has == *" w04 "* && $has == *" w04cpu "* ]]; then
    args+=(--pair "W04 CPU S1 naive → GPU S1 naive" W04C:S1 W04:S1
           --pair "W04 CPU S2 blocked+OMP → GPU S2b fused" W04C:S2 W04:S2b
           --pair "W04 CPU S3 AVX2 → GPU S3 FP16 Tensor Core" W04C:S3 W04:S3_FP16
           --pair "W04 CPU S1 → GPU S3 FP16 (end to end)" W04C:S1 W04:S3_FP16)
  fi
  "$PY" "$KIT/tools/summarize.py" "${args[@]}"
  [[ $has == *" w04 "* ]] && note "W04 caveat: the GPU runs at its 55 W cap, each stage at its own clock; ×REF is a power-cap ratio."
  if [[ "${NOPLOT:-0}" != 1 ]]; then
    "$PY" "$KIT/tools/plot_race.py" "$LIVE" | sed 's/^/ /'
  fi
  note "CSVs, logs and charts saved in $LIVE"
}

# ------------------------------------------------------------ nsight compute --
ncu_profile() {  # $1=label $2=binary, rest = args ; prints the report path (.ncu-repz or .ncu-rep)
  local label="$1" bin="$2"; shift 2
  local out="$DEMO_REPORTS/live_${STAMP}_${label}"
  mkdir -p "$DEMO_REPORTS"
  run $NCU_SUDO "$NCU" $(kfilter) -c 1 $NCU_SECTIONS --metrics "$NCU_METRICS" \
      -f -o "$out" "$bin" "$@" >/dev/null
  [[ -n "$NCU_SUDO" ]] && sudo chown "$USER" "$out".ncu-rep* 2>/dev/null
  rep "$out" || { fail "no report written for $label" >&2; echo "$out.missing"; }
}

cmd_ncu() {
  banner "NSIGHT COMPUTE — why is S3 faster? (W03 GEMM, N=$GEMM_N)"
  cd "$PROJECT" || return 1
  mkdir -p "$LIVE"
  say "1/3  Prove the hardware path exists: Tensor Core instructions in the binary"
  run bash -c "cuobjdump -sass '$W03_GPU_S3' | grep -oE 'HMMA[.A-Z0-9]*' | sort | uniq -c | head -5"
  pause
  say "2/3  Profile ONE launch of the Tensor Core kernel, live (~10 s; ncu replays it to read counters)"
  local s3; s3="$(ncu_profile W03_S3 "$W03_GPU_S3" "$GEMM_N" $PROFILE_ARGS "$LIVE/ncu_s3.csv" | tail -1)"
  run "$NCU" --import "$s3" --page details --section SpeedOfLight --section Occupancy 2>/dev/null \
     | grep -E 'Duration|Compute \(SM\) Throughput|Memory Throughput|DRAM Throughput|Registers Per Thread|Theoretical Occupancy|Achieved Occupancy|  [a-z_].*\(' | head -14
  pause
  say "3/3  Same counters, naive kernel vs Tensor Core kernel"
  local s1; s1="$(rep "$DEMO_REPORTS/W03_S1")"
  if [[ -z "$s1" ]]; then
    note "No backup S1 report — profiling S1 live too"
    s1="$(ncu_profile W03_S1 "$W03_GPU_S1" "$GEMM_N" $PROFILE_ARGS "$LIVE/ncu_s1.csv" | tail -1)"
  fi
  local s2; s2="$(rep "$DEMO_REPORTS/W03_S2")"; local args=("S1 naive=$s1")
  [[ -n "$s2" ]] && args+=("S2 shared tile=$s2")
  args+=("S3 Tensor Core (live)=$s3")
  "$PY" "$KIT/tools/ncu_compare.py" "$NCU" "${args[@]}"
  note "Talking point: S3 does the same FLOPs in fewer cycles on the tensor pipe;"
  note "56 regs/thread caps theoretical occupancy at 75% — the next tuning lever."
  LAST_NCU="$s3"
}

cmd_wall() {
  banner "NSIGHT COMPUTE — the memory wall (W01 Vector Add)"
  cd "$PROJECT" || return 1
  mkdir -p "$LIVE"
  local r; r="$(ncu_profile W01_S1 "$W01_GPU_S1" "$VEC_N" 1 1 1 "$LIVE/ncu_w01.csv" | tail -1)"
  "$PY" "$KIT/tools/ncu_compare.py" "$NCU" "W01 S1 (live)=$r"
  note "DRAM near its peak and SM nearly idle: no code change can make this much faster."
}

cmd_ncu_gui() {
  banner "NSIGHT COMPUTE GUI"
  local reps=()
  local s1; s1="$(rep "$DEMO_REPORTS/W03_S1")" && reps+=("$s1")
  local latest; latest="$(latest_rep "$DEMO_REPORTS"/live_*_W03_S3.ncu-rep* "$DEMO_REPORTS"/W03_S3.ncu-rep*)"
  [[ -n "$latest" ]] && reps+=("$latest")
  [[ ${#reps[@]} -eq 0 ]] && { fail "No reports. Run ./demo.sh ncu or ./prep.sh first."; return 1; }
  note "In the GUI: S1 tab active → Add Baseline (toolbar) → switch to the S3 tab to see the differences."
  mkdir -p "$LIVE"
  note "Log: $LIVE/ncu-ui.log   (if it closes instantly, run ./gui_doctor.sh)"
  printf '%s$ QT_QPA_PLATFORM=xcb %s %s &%s\n' "$C_G" "$NCU_UI" "${reps[*]}" "$C_0"
  QT_QPA_PLATFORM=xcb nohup "$NCU_UI" "${reps[@]}" >"$LIVE/ncu-ui.log" 2>&1 &
  sleep 4
  if kill -0 $! 2>/dev/null; then ok "ncu-ui running (pid $!)"; else fail "ncu-ui exited:"; tail -20 "$LIVE/ncu-ui.log"; fi
}

# -------------------------------------------------------------------- vtune --
cmd_vtune() {
  banner "VTUNE — where does the CPU spend its time? (W03 GEMM naive, N=$CPU_GEMM_N)"
  cd "$PROJECT" || return 1
  mkdir -p "$LIVE"
  local r="$DEMO_REPORTS/vtune_live_${STAMP}_W03_S1"
  say "1/3  Hotspots collection, live (~10–20 s including finalisation)"
  run env OMP_NUM_THREADS="$CPU_THREADS" OMP_PROC_BIND=close OMP_PLACES=cores \
      "$VTUNE" -collect hotspots -result-dir "$r" -- \
      "$W03_CPU_S1" "$CPU_GEMM_N" $CPU_GEMM_ARGS "$LIVE/vtune_s1.csv" 2>&1 \
      | grep -vE 'kallsyms|kernel modules|^vtune: (Executing|Collection|Analyzing|Start|Using)' | tail -8
  pause
  say "2/3  Summary: how much of the CPU did the naive code actually use?"
  run "$VTUNE" -report summary -result-dir "$r" -format text 2>/dev/null \
     | grep -iE 'Elapsed Time|CPU Time|Effective (Physical )?Core Utilization|Effective CPU Utilization|Logical Core|Physical Core|Total Thread Count|Paused' | head -10
  say "     Top functions"
  run "$VTUNE" -report hotspots -result-dir "$r" -format text 2>/dev/null | head -8
  pause
  say "3/3  Compare with the optimised AVX2/FMA + OpenMP build (S3)"
  local s3="$DEMO_REPORTS/vtune_W03_S3"
  if [[ ! -d "$s3" ]]; then
    note "No backup S3 result — collecting S3 live (fast)"
    s3="$DEMO_REPORTS/vtune_live_${STAMP}_W03_S3"
    env OMP_NUM_THREADS="$CPU_THREADS" OMP_PROC_BIND=close OMP_PLACES=cores \
      "$VTUNE" -collect hotspots -result-dir "$s3" -- \
      "$W03_CPU_S3" "$CPU_GEMM_S3_N" $CPU_GEMM_S3_ARGS "$LIVE/vtune_s3.csv" >/dev/null 2>&1
  fi
  run "$VTUNE" -report summary -result-dir "$s3" -format text 2>/dev/null \
     | grep -iE 'Elapsed Time|CPU Time|Effective (Physical )?Core Utilization|Effective CPU Utilization|Total Thread Count' | head -6
  run "$VTUNE" -report hotspots -result-dir "$s3" -format text 2>/dev/null | head -8
  note "Talking point: S1 = one thread in one triple loop; S3 = OpenMP workers running vfmadd213ps."
  LAST_VT="$r"
}

cmd_vtune_gui() {
  banner "VTUNE GUI"
  local r; r="$(ls -td "$DEMO_REPORTS"/vtune_live_*_W03_S1 "$DEMO_REPORTS"/vtune_W03_S1 2>/dev/null | head -1)"
  [[ -z "$r" ]] && { fail "No VTune result. Run ./demo.sh vtune or ./prep.sh first."; return 1; }
  note "In the GUI: Summary → Bottom-up (group by Function/Thread) → double-click the hotspot for source."
  printf '%s$ %s %s &%s\n' "$C_G" "$VTUNE_GUI" "$r" "$C_0"
  mkdir -p "$LIVE"
  nohup "$VTUNE_GUI" "$r" >"$LIVE/vtune-gui.log" 2>&1 &
  sleep 4; kill -0 $! 2>/dev/null && ok "vtune-gui running" || { fail "vtune-gui exited"; tail -20 "$LIVE/vtune-gui.log"; }
}

# ------------------------------------------------------------------ fusion --
# W04: what fusion removes (DRAM bytes, ncu) and what it buys (time, benchmark).
fusion_backup_dir() {
  if [[ -d "$FUSION_BACKUP" ]]; then echo "$FUSION_BACKUP"
  elif [[ -d "$DEMO_REPORTS/fusion_backup" ]]; then echo "$DEMO_REPORTS/fusion_backup"; fi
}
cmd_fusion() {
  local mode=live
  case "${1:-}" in --print) mode=print;; --backup) mode=backup;; "") ;; *) fail "use: fusion [--print|--backup]"; return 1;; esac
  local N="$FUSION_N" dir="$DEMO_REPORTS/live_${STAMP}_fusion"
  banner "FUSION — W04 N=$N: DRAM bytes and time, unfused S2a vs fused S2b"
  cd "$W04_PROJECT" || return 1
  local a="--n $N $W04_SHAPE --iters 1 --reps 1 --warmup 0"
  local names=(S2a_qk S2a_softmax S2a_pv S2b_tuned) kres=(qk_tiled softmax_rows_warp pv_tiled flash_fwd_fp32)
  local bins=("$W04_GPU_S2a" "$W04_GPU_S2a" "$W04_GPU_S2a" "$W04_GPU_S2b") extra=("" "" "" "")
  if [[ "$FUSION_TILE16" == 1 ]]; then
    names+=(S2b_16x16w8); kres+=(flash_fwd_fp32); bins+=("$W04_GPU_S2b"); extra+=("--br 16 --bc 16 --warps 8")
  fi
  local cmds=() i
  for i in "${!names[@]}"; do
    cmds+=("$NCU_SUDO $NCU -f -c 1 --launch-skip 0 -k regex:${kres[i]} --cache-control all --clock-control base --metrics $FUSION_METRICS -o $dir/${names[i]}_n$N ${bins[i]} $a ${extra[i]} --out $dir/csv/${names[i]}_n$N.csv")
  done
  say "1/3  ncu commands: one kernel each (-c 1), counters need root${NCU_SUDO:+ (sudo)}"
  for i in "${!cmds[@]}"; do printf '%s  %d) %s%s\n' "$C_G" $((i + 1)) "${cmds[i]}" "$C_0"; done
  [[ $mode == print ]] && return 0
  if [[ $mode == live ]]; then
    pause
    mkdir -p "$dir/csv"
    if [[ -n "$NCU_SUDO" ]] && ! sudo -v; then fail "sudo refused: showing the recorded run instead"; mode=backup; fi
  fi
  local tables=()
  if [[ $mode == live ]]; then
    for i in "${!cmds[@]}"; do
      say "     $((i + 1))/${#cmds[@]}  ${names[i]}"
      ${cmds[i]} 2>&1 | grep -E '==ERROR==|Correct' | sed 's/^/       /'
    done
    [[ -n "$NCU_SUDO" ]] && sudo chown -R "$USER" "$dir" 2>/dev/null
    local f
    for f in "$dir"/*.ncu-rep*; do
      [[ -f "$f" ]] && "$NCU" --import "$f" --page raw --csv > "${f%.ncu-rep*}.raw.csv" 2>/dev/null
    done
    if (( $(ls "$dir"/*.raw.csv 2>/dev/null | wc -l) >= 4 )); then
      tables+=("LIVE (this run)=$dir")
    else
      fail "live profile incomplete: showing the recorded run"
    fi
  fi
  local b; b="$(fusion_backup_dir)"
  if [[ -n "$b" ]]; then tables+=("RECORDED (P6 ncu E3 run)=$b"); else note "no recorded backup found (FUSION_BACKUP in demo.conf)"; fi
  say "2/3  What fusion removes: DRAM bytes per forward pass"
  [[ ${#tables[@]} -gt 0 ]] && "$PY" "$KIT/tools/fusion_bytes.py" --n "$N" "${tables[@]}"
  note "Talking point: S2a writes S and reads it back, writes P and reads it back (4 passes over N×N);"
  note "S2b keeps the tiles on chip and touches DRAM for Q, K, V and O only."
  pause
  say "3/3  What fusion buys: benchmark time (no sudo, protocol $W04_ARGS)"
  local t="$LIVE/fusion" ta=(--n "$N" $W04_SHAPE $W04_ARGS)
  mkdir -p "$t"
  runw fusion_S2a       "$W04_GPU_S2a" "${ta[@]}" --out "$t/S2a.csv"
  runw fusion_S2b_16x16 "$W04_GPU_S2b" "${ta[@]}" --br 16 --bc 16 --warps 8 --out "$t/S2b_16x16.csv"
  runw fusion_S2b_tuned "$W04_GPU_S2b" "${ta[@]}" --out "$t/S2b_tuned.csv"
  "$PY" "$KIT/tools/summarize.py" --group "W04 N=$N: unfused → fused, same per-thread work (16×16 W8) → fused, register-tiled (tuned)" \
      --id FUS --ordered S2a="$t/S2a" S2b_16x16="$t/S2b_16x16" S2b_tuned="$t/S2b_tuned"
  note "S2a→S2b_16x16 = fusion alone (recorded: 1.15×); S2b_16x16→S2b_tuned = register tiling + tile shape (2.74×)."
  note "Bytes fall ~24–29×, time only ~15% from fusion: S2a's GEMMs are not DRAM-bound (only its softmax is)."
  note "ncu durations above are at base clock with flushed caches; speedups come from the benchmark runs only."
}

# ------------------------------------------------------------ side by side --
# Tiles two GUI windows left/right. Uses wmctrl if installed (sudo apt install wmctrl);
# otherwise opens both and you press Super+Left / Super+Right.
screen_wh() { xrandr 2>/dev/null | awk '/\*/{print $1; exit}'; }
open_tiled() {  # $1=left|right, rest = command
  local side="$1"; shift
  local before; before="$(wmctrl -l 2>/dev/null | awk '{print $1}' | sort)"
  QT_QPA_PLATFORM=xcb nohup "$@" >/dev/null 2>&1 &
  command -v wmctrl >/dev/null || { note "wmctrl not installed: press Super+$( [[ $side == left ]] && echo Left || echo Right ) on the new window"; return; }
  local wid="" i
  for i in $(seq 1 40); do
    sleep 0.5
    wid="$(comm -13 <(echo "$before") <(wmctrl -l | awk '{print $1}' | sort) | tail -1)"
    [[ -n "$wid" ]] && break
  done
  [[ -z "$wid" ]] && { note "window not found yet; tile it by hand (Super+Left/Right)"; return; }
  sleep 1.5
  local wh w h x; wh="$(screen_wh)"; w="${wh%x*}"; h="${wh#*x}"; w=$((w/2)); x=0; [[ $side == right ]] && x=$w
  wmctrl -i -r "$wid" -b remove,maximized_vert,maximized_horz
  wmctrl -i -r "$wid" -e "0,$x,0,$w,$((h-40))"
  ok "$side window placed"
}

cmd_ncu_side() {
  banner "NSIGHT COMPUTE — naive (left) vs Tensor Core (right)"
  local s1 s3
  s1="$(rep "$DEMO_REPORTS/W03_S1")" || { fail "run ./prep.sh first"; return 1; }
  s3="$(latest_rep "$DEMO_REPORTS"/live_*_W03_S3.ncu-rep* "$DEMO_REPORTS"/W03_S3.ncu-rep*)"
  open_tiled left  "$NCU_UI" "$s1"
  open_tiled right "$NCU_UI" "$s3"
  note "Both windows: Page = Details, expand GPU Speed Of Light and Compute Workload Analysis."
  note "For one combined view instead: in the right window, File > Open the S1 report, Add Baseline."
}

cmd_vtune_side() {
  banner "VTUNE — naive S1 vs AVX2/FMA S3, side by side"
  local a b
  a="$(ls -td "$DEMO_REPORTS"/vtune_W03_S1 "$DEMO_REPORTS"/vtune_live_*_W03_S1 2>/dev/null | head -1)"
  b="$(ls -td "$DEMO_REPORTS"/vtune_W03_S3 "$DEMO_REPORTS"/vtune_live_*_W03_S3 2>/dev/null | head -1)"
  [[ -z "$a" || -z "$b" ]] && { fail "run ./prep.sh first"; return 1; }
  say "Terminal comparison (two -r flags = VTune comparison report)"
  run "$VTUNE" -report summary -r "$a" -r "$b" -format text 2>/dev/null \
     | grep -iE 'Elapsed Time|CPU Time|Effective (Physical )?Core Utilization|Effective CPU Utilization|Total Thread Count|Result' | head -12
  run "$VTUNE" -report hotspots -r "$a" -r "$b" -format text 2>/dev/null | head -10
  say "GUIs"
  open_tiled left  "$VTUNE_GUI" "$a"
  open_tiled right "$VTUNE_GUI" "$b"
  note "Both windows: Summary tab first, then Bottom-up. Same N; S3 repeats the GEMM more times,"
  note "so compare utilisation, hotspots and time per GEMM, not total elapsed."
}

cmd_hw() {
  banner "HARDWARE FACTS (from the CUDA driver)"
  local bin="$KIT/tools/devinfo"
  [[ -x "$bin" ]] || run nvcc -O2 -o "$bin" "$KIT/tools/devinfo.cu" || { fail "nvcc build failed"; return 1; }
  "$bin"
  lscpu | grep -E 'Model name|^CPU\(s\)|Thread|L1d|L2|L3' | sed 's/^/  /'
}

cmd_video() {
  [[ -f "$VIDEO" ]] || { fail "Video not at $VIDEO"; return 1; }
  for p in mpv vlc xdg-open; do
    command -v $p >/dev/null && { nohup $p "$VIDEO" >/dev/null 2>&1 & ok "playing with $p"; return; }
  done
  fail "no video player found"
}

menu() {
  while true; do
    printf '\n%s CPU/GPU PROFILING — LIVE DEMO %s\n' "$C_T$C_B" "$C_0"
    cat <<EOF
  1) Live race, all workloads (W01-W04 GPU + W04 CPU) ~45 s
  r) Live race, W04 only (GPU)                       ~10 s
  f) Fusion: W04 DRAM bytes S2a vs S2b (ncu, sudo)  ~60 s
  2) Nsight Compute live: W03 S3 vs S1               ~40 s
  3) Nsight Compute GUI                              opens window
  4) VTune live: W03 CPU naive vs S3                 ~40 s
  5) VTune GUI                                       opens window
  6) Play race animation                             38 s
  7) Bonus: W01 memory wall (Nsight)                 ~15 s
  8) Hardware facts (SMs, registers, L2, caches)     ~2 s
  9) Nsight side by side: naive | Tensor Core       two windows
  0) VTune side by side: S1 | S3 (+ terminal diff)   two windows
  c) Preflight check          q) Quit
EOF
    read -r -p "> " c
    case "$c" in
      1) cmd_race all;; r) cmd_race w04;; f) cmd_fusion;; 2) cmd_ncu;; 3) cmd_ncu_gui;; 4) cmd_vtune;; 5) cmd_vtune_gui;;
      6) cmd_video;; 7) cmd_wall;; 8) cmd_hw;; 9) cmd_ncu_side;; 0) cmd_vtune_side;; c) cmd_check;; q) exit 0;; *) ;;
    esac
  done
}

case "${1:-menu}" in
  check) cmd_check;; race) shift; cmd_race "$@";; fusion) shift; cmd_fusion "$@";;
  plot) shift; "$PY" "$KIT/tools/plot_race.py" "$@";; ncu) cmd_ncu;; ncu-gui) cmd_ncu_gui;;
  vtune) cmd_vtune;; vtune-gui) cmd_vtune_gui;; video) cmd_video;; wall) cmd_wall;; hw) cmd_hw;; ncu-side) cmd_ncu_side;; vtune-side) cmd_vtune_side;;
  menu) menu;; *) sed -n '2,24p' "$0";;
esac
