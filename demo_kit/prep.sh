#!/usr/bin/env bash
# prep.sh — run the day before (and again 30 min before) the talk.
# Captures BACKUP profiler results so every demo step has a fallback if a
# live collection fails on stage. Never overwrites: backups go to
# profiling/demo/ and old ones are moved to profiling/demo/old_<stamp>/.

set -uo pipefail
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$KIT/demo.conf"
cd "$PROJECT" || { echo "PROJECT not found: $PROJECT"; exit 1; }

W03_CPU_S3="${W03_CPU_S3:-$(ls "$BUILD"/W03_GEMM_CPU_S3* 2>/dev/null | head -1)}"
mkdir -p "$DEMO_REPORTS" "$LIVE"
if ls "$DEMO_REPORTS"/W03_S1.ncu-rep* >/dev/null 2>&1; then
  mkdir -p "$DEMO_REPORTS/old_$STAMP"
  mv "$DEMO_REPORTS"/W0*.ncu-rep* "$DEMO_REPORTS"/vtune_W03_S* "$DEMO_REPORTS/old_$STAMP/" 2>/dev/null
  echo "Previous backups moved to $DEMO_REPORTS/old_$STAMP/"
fi

K=""; [[ -n "$NCU_KFILTER" ]] && K="-k $NCU_KFILTER"
prof() {  # label binary args...
  local l="$1" b="$2"; shift 2
  echo "== ncu $l"
  $NCU_SUDO "$NCU" $K -c 1 $NCU_SECTIONS --metrics "$NCU_METRICS" -f \
     -o "$DEMO_REPORTS/$l" "$b" "$@" 2>&1 | grep -E '==PROF==|==ERROR==|Correct' | tail -3
  [[ -n "$NCU_SUDO" ]] && sudo chown "$USER" "$DEMO_REPORTS/$l".ncu-rep* 2>/dev/null
}
prof W03_S1 "$W03_GPU_S1" "$GEMM_N" $PROFILE_ARGS "$LIVE/p_w03s1.csv"
prof W03_S2 "$W03_GPU_S2" "$GEMM_N" $PROFILE_ARGS "$LIVE/p_w03s2.csv"
prof W03_S3 "$W03_GPU_S3" "$GEMM_N" $PROFILE_ARGS "$LIVE/p_w03s3.csv"
prof W01_S1 "$W01_GPU_S1" "$VEC_N" 1 1 1 "$LIVE/p_w01s1.csv"

for s in S1 S3; do
  bin_var="W03_CPU_$s"; bin="${!bin_var}"
  if [[ $s == S3 ]]; then n="$CPU_GEMM_S3_N"; a="$CPU_GEMM_S3_ARGS"; else n="$CPU_GEMM_N"; a="$CPU_GEMM_ARGS"; fi
  echo "== vtune W03 CPU $s ($bin)"
  OMP_NUM_THREADS="$CPU_THREADS" OMP_PROC_BIND=close OMP_PLACES=cores \
    "$VTUNE" -collect hotspots -result-dir "$DEMO_REPORTS/vtune_W03_$s" -- \
    "$bin" "$n" $a "$LIVE/p_vt_$s.csv" 2>&1 | grep -iE 'error|Elapsed' | tail -2
done

# W04 fusion cue: keep a copy of the recorded ncu exports next to the other backups
if [[ -d "$FUSION_BACKUP" ]]; then
  mkdir -p "$DEMO_REPORTS/fusion_backup"
  cp -n "$FUSION_BACKUP"/S2a_*_n"$FUSION_N".raw.csv "$FUSION_BACKUP"/S2b_*_n"$FUSION_N".raw.csv "$DEMO_REPORTS/fusion_backup/" 2>/dev/null
  echo "== fusion backup: $(ls "$DEMO_REPORTS/fusion_backup" | wc -l) ncu exports in $DEMO_REPORTS/fusion_backup"
fi

echo
echo "== Comparison table that will appear on stage:"
r() { ls "$DEMO_REPORTS/$1".ncu-rep* 2>/dev/null | head -1; }
python3 "$KIT/tools/ncu_compare.py" "$NCU" \
  "S1 naive=$(r W03_S1)" "S2 tiled=$(r W03_S2)" "S3 tensor=$(r W03_S3)" "W01 S1=$(r W01_S1)"

cat <<EOF

Backups ready in $DEMO_REPORTS.
NOW, while the GUIs work, take screenshots for the slides' fallback:
  1. ./demo.sh ncu-gui   → S3 Details page with S1 as baseline   (save as ncu_s3_vs_s1.png)
  2. ./demo.sh vtune-gui → Summary page + Bottom-up page        (save as vtune_s1_summary.png, vtune_s1_bottomup.png)
Put the PNGs in $PROJECT/demo/screens/ — the deck has placeholders for them.
EOF
