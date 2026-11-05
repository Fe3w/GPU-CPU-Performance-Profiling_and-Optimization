#!/usr/bin/env python3
"""W04 E7 autotune: finalists, ranking report and the tuned-defaults header.

    python3 tools/w04_e7_select.py report <run_dir> [--write-header]
    python3 tools/w04_e7_select.py finalists <run_dir> <n> <stage_id> [--top 3]
    python3 tools/w04_e7_select.py resources <binary>

report     Reads <run_dir>/e7/W04_E7_AUTOTUNE.csv (screen and final rows) and
           <run_dir>/e7/skipped.csv, and, if present, the per-instance
           resources written by the sweep (the "e7 resources" line of
           run_info.txt, else profiling/W04_ATTENTION/codegen/<run TS>/
           e7_resources.csv).
           Per stage and N: the screen ranking (median of the screen reps) and
           the finals (median over rounds of the run means, min-max spread of
           the run means in %), GFLOP/s from the model FLOPs 4·BH·N²·d. Winner = lowest final median. If the runner-up
           is closer than the larger of the two spreads, the two are a tie, and
           among the tied configurations the one that also wins or ties at the
           other tuned N is preferred (a stable choice across N).
           --write-header rewrites the table between the BEGIN/END markers of
           src/common/w04_tuned.hpp, with the run folder, git SHA and medians.
finalists  The top <top> configurations of the screen at <n> for <stage_id>,
           plus the incumbent (the P3/P4 default) if it is not among them;
           one "br bc warps" line each (used by scripts/w04_sweep.sh).
resources  Registers, stack and local memory per flash kernel instance of a
           binary (cuobjdump --dump-resource-usage), as CSV lines
           br,bc,warps,regs,stack_B,local_B.

Screen and final rows are told apart by (iters, reps), read from the
"e7 screen" / "e7 final" lines of <run_dir>/run_info.txt (default: screen
iters 10 reps 3, final iters 20 reps 5, the protocol).
Theoretical occupancy uses the documented sm_89 limits: 65,536 registers and
48 warps per SM, 24 blocks per SM, registers allocated per warp in units of
256 (8 per thread), 102,400 B shared memory per SM with 1,024 B reserved per
block. It is computed, not measured.

Standard library only. Never writes into the run folder.
"""
import csv
import os
import re
import statistics as st
import subprocess
import sys
from collections import defaultdict

E7_CSV = os.path.join("e7", "W04_E7_AUTOTUNE.csv")
SKIPPED_CSV = os.path.join("e7", "skipped.csv")
HEADER = os.path.join("src", "common", "w04_tuned.hpp")
SCREEN = (10, 3)
FINAL = (20, 5)
# The defaults before E7 (C18, C24); always a finalist.
INCUMBENT = {
    "W04_ATTENTION_GPU_S2b_SWOPT_FUSED": (32, 32, 8),
    "W04_ATTENTION_GPU_S3_HWACC_TENSOR_CORE_FP16": (64, 64, 4),
    "W04_ATTENTION_GPU_S3_HWACC_TENSOR_CORE_TF32": (64, 64, 4),
}


def stage_id(row):
    return f"W04_ATTENTION_{row['device']}_{row['stage']}_{row['variant']}"


def protocols(run):
    """(screen (iters, reps), final (iters, reps)) from run_info.txt."""
    found = {"screen": SCREEN, "final": FINAL}
    path = os.path.join(run, "run_info.txt")
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                m = re.match(r"e7 (screen|final)\s*: (.*)", line)
                if m:
                    it = re.search(r"--iters (\d+)", m.group(2))
                    rp = re.search(r"--reps (\d+)", m.group(2))
                    found[m.group(1)] = (int(it.group(1)), int(rp.group(1)))
    if found["screen"] == found["final"]:
        raise SystemExit(f"{path}: screen and final protocols are identical, phases cannot be told apart")
    return found["screen"], found["final"]


def load_runs(run):
    """{(phase, sid, n, (br, bc, w)): {timestamp: [ms per rep]}} and per-key metadata."""
    screen, final = protocols(run)
    path = os.path.join(run, E7_CSV)
    per_run = defaultdict(lambda: defaultdict(list))  # key -> timestamp -> [ms]
    meta = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["correct"] != "YES":
                raise SystemExit(f"{path}: a row is not correct: {r}")
            it_reps = (int(r["iters"]), int(r["reps"]))
            phase = "screen" if it_reps == screen else "final" if it_reps == final else None
            if phase is None:
                raise SystemExit(f"{path}: unexpected iters/reps {it_reps}")
            cfg = (int(r["br"]), int(r["bc"]), int(r["warps"]))
            key = (phase, stage_id(r), int(r["n"]), cfg)
            per_run[key][r["timestamp"]].append(float(r["runtime_ms"]))
            meta[key] = (r["git_sha"], 4.0 * int(r["bh"]) * int(r["n"]) ** 2 * int(r["d"]))
    return per_run, meta


def screen_table(per_run, sid, n):
    """[(median ms, cfg)] sorted, from the screen phase."""
    out = []
    for (phase, s, nn, cfg), runs in per_run.items():
        if phase == "screen" and s == sid and nn == n:
            reps = [ms for v in runs.values() for ms in v]
            out.append((st.median(reps), cfg))
    return sorted(out)


def final_table(per_run, sid, n):
    """[(median of run means, spread %, rounds, cfg)] sorted."""
    out = []
    for (phase, s, nn, cfg), runs in per_run.items():
        if phase == "final" and s == sid and nn == n:
            means = [st.mean(v) for v in runs.values()]
            med = st.median(means)
            out.append((med, 100 * (max(means) - min(means)) / med, len(means), cfg))
    return sorted(out)


def load_resources(run):
    ts = os.path.basename(os.path.normpath(run))
    path = os.path.join("profiling", "W04_ATTENTION", "codegen", ts, "e7_resources.csv")
    info = os.path.join(run, "run_info.txt")
    if os.path.exists(info):
        with open(info) as f:
            for line in f:
                m = re.match(r"e7 resources\s*: (\S+)", line)
                if m:
                    path = m.group(1)
    res = {}
    if os.path.exists(path):
        with open(path) as f:
            for r in csv.DictReader(f):
                res[(r["stage_id"], int(r["br"]), int(r["bc"]), int(r["warps"]))] = r
    return res, path


def occupancy(regs, warps, smem):
    regs_per_warp = -(-regs // 8) * 8 * 32
    by_regs = (65536 // regs_per_warp) // warps
    by_smem = 102400 // (smem + 1024)
    by_warps = 48 // warps
    blocks = min(by_regs, by_smem, by_warps, 24)
    limit = min((by_regs, "regs"), (by_smem, "smem"), (by_warps, "warps"), (24, "blocks"))[1]
    return blocks, blocks * warps, limit


def cfg_str(cfg):
    return f"{cfg[0]}x{cfg[1]} W{cfg[2]}"


def resource_str(res, sid, cfg):
    r = res.get((sid, *cfg))
    if not r:
        return ""
    regs, smem = int(r["regs"]), int(r["smem_dynamic_B"])
    blocks, wps, limit = occupancy(regs, cfg[2], smem)
    spill = f" stack {r['stack_B']} B" if int(r["stack_B"]) else ""
    return f"{regs:>3} regs{spill:<14} {smem:>6} B smem  {blocks} blk/SM {wps:>2} warps/SM ({100 * wps / 48:.0f}%, {limit})"


def choose(per_run, sid, ns):
    """{n: (cfg, note)} with the tie rule."""
    finals = {n: final_table(per_run, sid, n) for n in ns}

    def tie_set(n):
        t = finals[n]
        if not t:
            return []
        best = t[0]
        return [x[3] for x in t if 100 * (x[0] - best[0]) / best[0] < max(best[1], x[1])]

    out = {}
    for n in ns:
        t = finals[n]
        if not t:
            continue
        ties = tie_set(n)
        fastest = t[0][3]
        winner, note = fastest, "fastest final median"
        if len(ties) > 1:
            others = [m for m in ns if m != n]
            stable = [c for c in ties if all(c in tie_set(m) for m in others)]
            tied = ", ".join(cfg_str(c) for c in ties)
            where = ",".join(map(str, others))
            if fastest in stable:
                note = f"fastest; tied within the spread: {tied}; it also wins or ties at N={where}"
            elif stable:
                winner = stable[0]
                note = (f"tied within the spread: {tied}; fastest {cfg_str(fastest)} does not win or tie at "
                        f"N={where}, {cfg_str(winner)} does")
            else:
                note = f"tied within the spread: {tied}; none wins or ties at N={where}, fastest kept"
        out[n] = (winner, note)
    return out


def cmd_report(run, write_header):
    per_run, meta = load_runs(run)
    res, res_path = load_resources(run)
    sids = sorted({k[1] for k in per_run})
    ns = sorted({k[2] for k in per_run})
    shas = sorted({m[0] for m in meta.values()})
    print(f"E7 autotune report: {run}  (git_sha {', '.join(shas)})")
    print(f"resources: {res_path if res else 'not found (' + res_path + ')'}")
    skipped = defaultdict(list)
    sp = os.path.join(run, SKIPPED_CSV)
    if os.path.exists(sp):
        with open(sp) as f:
            for r in csv.DictReader(f):
                skipped[(r["stage_id"], int(r["n"]))].append(r)
    selection = {}
    for sid in sids:
        sel = choose(per_run, sid, ns)
        for n in ns:
            flops = next(m[1] for k, m in meta.items() if k[1] == sid and k[2] == n)
            print(f"\n== {sid}  N={n} ==")
            scr = screen_table(per_run, sid, n)
            print(f"  screen ({len(scr)} configs, median of 3 reps; skipped {len(skipped[(sid, n)])}):")
            for rank, (ms, cfg) in enumerate(scr, 1):
                print(f"   {rank:>2}. {cfg_str(cfg):<11} {ms:>9.4f} ms {flops / ms / 1e6:>8.0f} GFLOP/s  "
                      f"{resource_str(res, sid, cfg)}")
            reasons = defaultdict(list)
            for r in skipped[(sid, n)]:
                reasons[r["reason"]].append(f"{r['br']}x{r['bc']} W{r['warps']}")
            for why, cells in sorted(reasons.items()):
                print(f"   skipped ({len(cells)}): {why}: {' '.join(cells)}")
            fin = final_table(per_run, sid, n)
            print(f"  finals (median over rounds of the run means):")
            for med, spread, rounds, cfg in fin:
                print(f"       {cfg_str(cfg):<11} {med:>9.4f} ms {flops / med / 1e6:>8.0f} GFLOP/s  "
                      f"spread {spread:4.1f}%  rounds {rounds}"
                      f"{'  (incumbent)' if cfg == INCUMBENT.get(sid) else ''}")
            if n in sel:
                cfg, note = sel[n]
                inc = next((x for x in fin if x[3] == INCUMBENT.get(sid)), None)
                win = next(x for x in fin if x[3] == cfg)
                gain = f", {inc[0] / win[0]:.3f}x vs incumbent {cfg_str(inc[3])}" if inc else ""
                print(f"  selected: {cfg_str(cfg)} ({note}{gain})")
                selection[(sid, n)] = (cfg, win[0])
    if write_header:
        write_tuned_header(run, shas, selection)
    return 0


def write_tuned_header(run, shas, selection):
    with open(HEADER) as f:
        text = f.read()
    begin = text.index("// BEGIN TUNED TABLE")
    end = text.index("// END TUNED TABLE")
    lines = ["// BEGIN TUNED TABLE (generated by tools/w04_e7_select.py --write-header)",
             f"// Provenance: E7 autotune, run {os.path.normpath(run)}, git_sha {', '.join(shas)};",
             "//   final median runtime (BH=8, d=64, normal input) of each selected config:"]
    for (sid, n), (cfg, ms) in sorted(selection.items()):
        lines.append(f"//   {sid.replace('W04_ATTENTION_', '')} N={n}: {cfg_str(cfg)} {ms:.4f} ms")
    lines.append("inline constexpr TunedConfig kTunedConfigs[] = {")
    for (sid, n), (cfg, _) in sorted(selection.items()):
        lines.append(f'    {{"{sid}", {n}, {cfg[0]}, {cfg[1]}, {cfg[2]}}},')
    lines.append("};")
    with open(HEADER, "w") as f:
        f.write(text[:begin] + "\n".join(lines) + "\n" + text[end:])
    print(f"\nwrote {HEADER}")


def cmd_finalists(run, n, sid, top):
    per_run, _ = load_runs(run)
    scr = screen_table(per_run, sid, n)
    if not scr:
        raise SystemExit(f"no screen rows for {sid} at N={n}")
    cfgs = [c for _, c in scr[:top]]
    inc = INCUMBENT.get(sid)
    if inc and inc not in cfgs:
        cfgs.append(inc)
    for c in cfgs:
        print(*c)
    return 0


def cmd_resources(binary):
    out = subprocess.run(["cuobjdump", "--dump-resource-usage", binary], check=True, capture_output=True,
                         text=True).stdout
    func = None
    for line in out.splitlines():
        m = re.search(r"Function (\S+):", line)
        if m:
            func = m.group(1)
            continue
        m = re.search(r"REG:(\d+) STACK:(\d+) SHARED:\d+ LOCAL:(\d+)", line)
        if m and func and ("flash_fwd" in func):
            t = re.search(r"Li(\d+)ELi(\d+)ELi(\d+)E", func)
            if t:
                print(f"{t.group(1)},{t.group(2)},{t.group(3)},{m.group(1)},{m.group(2)},{m.group(3)}")
            func = None
    return 0


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    cmd = argv[1]
    if cmd == "report":
        return cmd_report(argv[2], "--write-header" in argv[3:])
    if cmd == "finalists" and len(argv) >= 5:
        top = int(argv[argv.index("--top") + 1]) if "--top" in argv else 3
        return cmd_finalists(argv[2], int(argv[3]), argv[4], top)
    if cmd == "resources":
        return cmd_resources(argv[2])
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
