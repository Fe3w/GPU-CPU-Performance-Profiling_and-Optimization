#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# w03_s3_ab.sh — paired A/B of two W03 GPU S3 binaries (before / after a change)
#
#   bash scripts/w03_s3_ab.sh <binary A> <binary B> [N ...]      (default N: 1024 2048)
#
# For each N: ROUNDS (3) rounds, order A,B in odd rounds and B,A in even rounds;
# each measured run is preceded by one untimed priming run of the same binary
# (memory/SM clock, as in run_all.sh). Protocol 2 warm-ups, 20 iterations,
# 5 repetitions. Output: results/raw/W03_GEMM/<YYYYMMDD-HHMMSS>-s3ab/
#   <label>_n<N>_r<k>.csv (the binary's own CSV), logs/, gpu_state.csv,
#   run_info.txt, summary.txt (median of round means, ratio A/B).
# -----------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$0")/.."
[[ $# -ge 2 ]] || { sed -n '4,13p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
A=$1; B=$2; shift 2
NS=("$@"); [[ ${#NS[@]} -gt 0 ]] || NS=(1024 2048)
ROUNDS=${ROUNDS:-3}
RUN="results/raw/W03_GEMM/$(date +%Y%m%d-%H%M%S)-s3ab"
mkdir -p "$RUN/logs"
{
    echo "date     : $(date -Iseconds)"
    echo "commit   : $(git rev-parse --short=12 HEAD 2>/dev/null)$([[ -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" ]] && echo -dirty)"
    echo "A        : $A"
    echo "B        : $B"
    echo "N        : ${NS[*]}  rounds: $ROUNDS  protocol: N 20 5 2"
    echo "gpu      : $(nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv,noheader)"
} > "$RUN/run_info.txt"
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv -lms 250 > "$RUN/gpu_state.csv" 2>/dev/null & GP=$!
one() {  # label binary N round
    "$2" "$3" 20 1 1 "$RUN/logs/prime_$1_n$3_r$4.csv" > /dev/null 2>&1
    local t0 t1; t0=$(date +%s.%N)
    "$2" "$3" 20 5 2 "$RUN/$1_n$3_r$4.csv" > "$RUN/logs/$1_n$3_r$4.log" 2>&1
    t1=$(date +%s.%N); echo "$1,$3,$4,$t0,$t1" >> "$RUN/times.csv"
    printf '  %-6s N=%-5s r%s  %s\n' "$1" "$3" "$4" "$(grep -E 'Mean time|Correct' "$RUN/logs/$1_n$3_r$4.log" | tr -s ' ' | tr '\n' ' ')"
}
for n in "${NS[@]}"; do
    for r in $(seq 1 "$ROUNDS"); do
        if (( r % 2 )); then one A "$A" "$n" "$r"; one B "$B" "$n" "$r"; else one B "$B" "$n" "$r"; one A "$A" "$n" "$r"; fi
    done
done
kill "$GP" 2>/dev/null; wait "$GP" 2>/dev/null
python3 - "$RUN" "${NS[@]}" <<'PY' | tee "$RUN/summary.txt"
import csv, datetime as dt, glob, statistics as st, sys
run, ns = sys.argv[1], sys.argv[2:]
samples = []
for r in csv.reader(open(f"{run}/gpu_state.csv")):
    try:
        samples.append((dt.datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp(),
                        float(r[1].split()[0]), float(r[2].split()[0])))
    except (ValueError, IndexError):
        pass
clk = {}
for lab, n, rnd, t0, t1 in csv.reader(open(f"{run}/times.csv")):
    w = [x for x in samples if float(t0) <= x[0] <= float(t1) and x[1] > 1000]
    clk[(lab, n, rnd)] = (st.median(x[1] for x in w), st.median(x[2] for x in w)) if w else (None, None)
for n in ns:
    res = {}
    for lab in ("A", "B"):
        rounds = []
        for f in sorted(glob.glob(f"{run}/{lab}_n{n}_r*.csv")):
            rnd = f.rsplit("_r", 1)[1][:-4]
            sm, mem = clk.get((lab, n, rnd), (None, None))
            print(f"  {lab} N={n} r{rnd}: SM {sm} MHz, memory {mem} MHz")
            r = list(csv.DictReader(open(f)))[-1]
            rounds.append((float(r["runtime_mean_ms"]), float(r["runtime_std_ms"]), r["correct"], float(r["max_rel_error_pct"]), float(r["gflops"])))
        res[lab] = rounds
        m = [x[0] for x in rounds]
        print(f"N={n} {lab}: round means {', '.join(f'{x:.4f}' for x in m)} ms; median {st.median(m):.4f} ms; "
              f"spread {(max(m) - min(m)) / st.median(m) * 100:.1f}%; GFLOP/s {st.median(x[4] for x in rounds):.0f}; "
              f"correct {'/'.join(sorted({x[2] for x in rounds}))}; max rel err {max(x[3] for x in rounds):.4f}%")
    a = st.median(x[0] for x in res["A"]); b = st.median(x[0] for x in res["B"])
    print(f"N={n} A/B time ratio {a / b:.3f} (B is {a / b:.2f}x faster than A)")
PY
echo "run folder: $RUN"
