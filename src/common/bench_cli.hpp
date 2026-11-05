// -----------------------------------------------------------------------------
// W04 ATTENTION — the one command-line interface for every W04 binary (P0)
//
//   --n --bh --d --iters --reps --warmup --out --seed --input   (+ --br --bc if tunable)
//                                                               (+ --warps if the stage has a warps axis, P5)
//
// Accepts "--flag value" and "--flag=value". Anything unexpected is an error,
// never silently ignored: unknown flags, missing or non-numeric values, values
// out of range, --br/--bc on a binary that is not tunable and --warps on a
// binary without a warps axis.
// W01–W03 mixed positional and named CLIs; W04 does not.
// -----------------------------------------------------------------------------
#pragma once

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <string>

#include "attention_ref.hpp"

namespace w04 {

struct BenchArgs {
    std::size_t n = 1024;
    int bh = 8;
    int d = 64;
    int iters = 20;
    int reps = 5;
    int warmup = 2;
    std::string out;              // empty -> default_out_path(stage_id)
    std::uint64_t seed = 42;
    int br = 0;                   // 0 = stage default (only settable if tunable)
    int bc = 0;
    int warps = 0;                // 0 = stage default (only settable if the stage has a warps axis)
    InputCase input = InputCase::Normal;

    AttnShape shape() const { return AttnShape{bh, n, d}; }
};

struct CliSpec {
    const char* stage_id;   // e.g. "W04_ATTENTION_GPU_S2b_SWOPT_FUSED"
    bool tunable_br_bc;
    bool tunable_warps = false;   // P5 (decision C28): GPU S2b and S3
};

struct CliResult {
    bool ok = false;
    bool help = false;
    std::string error;
    BenchArgs args;
};

inline void print_usage(const CliSpec& spec, std::FILE* f = stdout) {
    std::fprintf(f,
                 "usage: %s [--n N] [--bh BH] [--d D] [--iters I] [--reps R] [--warmup W]\n"
                 "          [--seed S] [--input normal|large_logits] [--out FILE.csv]%s%s\n"
                 "defaults: --n 1024 --bh 8 --d 64 --iters 20 --reps 5 --warmup 2 --seed 42 --input normal\n"
                 "          --out results/raw/W04_ATTENTION/<YYYYMMDD-HHMMSS>/%s.csv (appended, never overwritten)\n",
                 spec.stage_id, spec.tunable_br_bc ? " [--br BR] [--bc BC]" : "", spec.tunable_warps ? " [--warps W]" : "",
                 spec.stage_id);
}

namespace detail {
// Strict unsigned parse: digits only (strtoull would accept "-1" and "12abc").
inline bool parse_u64(const std::string& s, std::uint64_t& out) {
    if (s.empty()) return false;
    for (char ch : s)
        if (ch < '0' || ch > '9') return false;
    errno = 0;
    char* end = nullptr;
    const unsigned long long v = std::strtoull(s.c_str(), &end, 10);
    if (errno == ERANGE || end == nullptr || *end != '\0') return false;
    out = static_cast<std::uint64_t>(v);
    return true;
}
inline bool parse_int(const std::string& s, int& out) {
    std::uint64_t v = 0;
    if (!parse_u64(s, v) || v > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) return false;
    out = static_cast<int>(v);
    return true;
}
}  // namespace detail

// Non-exiting parser (used by the self-test and by parse_bench_args).
inline CliResult try_parse_bench_args(int argc, const char* const* argv, const CliSpec& spec) {
    CliResult r;
    BenchArgs& a = r.args;
    auto fail = [&](const std::string& msg) { r.ok = false; r.error = msg; return r; };

    for (int i = 1; i < argc; ++i) {
        std::string tok = argv[i];
        if (tok == "--help" || tok == "-h") { r.help = true; r.ok = true; return r; }
        if (tok.rfind("--", 0) != 0) return fail("unexpected positional argument '" + tok + "'");
        std::string key = tok.substr(2), val;
        const std::size_t eq = key.find('=');
        if (eq != std::string::npos) {
            val = key.substr(eq + 1);
            key = key.substr(0, eq);
        } else {
            if (i + 1 >= argc) return fail("missing value for --" + key);
            val = argv[++i];
        }

        std::uint64_t u = 0;
        int iv = 0;
        if (key == "n") {
            if (!detail::parse_u64(val, u) || u == 0) return fail("--n must be a positive integer");
            a.n = static_cast<std::size_t>(u);
        } else if (key == "bh" || key == "d" || key == "iters" || key == "reps") {
            if (!detail::parse_int(val, iv) || iv <= 0) return fail("--" + key + " must be a positive integer");
            (key == "bh" ? a.bh : key == "d" ? a.d : key == "iters" ? a.iters : a.reps) = iv;
        } else if (key == "warmup") {
            if (!detail::parse_int(val, iv)) return fail("--warmup must be a non-negative integer");
            a.warmup = iv;
        } else if (key == "seed") {
            if (!detail::parse_u64(val, u)) return fail("--seed must be a non-negative integer");
            a.seed = u;
        } else if (key == "out") {
            if (val.empty()) return fail("--out needs a file path");
            a.out = val;
        } else if (key == "input") {
            if (val == "normal") a.input = InputCase::Normal;
            else if (val == "large_logits") a.input = InputCase::LargeLogits;
            else return fail("--input must be 'normal' or 'large_logits'");
        } else if (key == "br" || key == "bc") {
            if (!spec.tunable_br_bc) return fail("--" + key + " is not supported by this binary (not tunable)");
            if (!detail::parse_int(val, iv) || iv <= 0) return fail("--" + key + " must be a positive integer");
            (key == "br" ? a.br : a.bc) = iv;
        } else if (key == "warps") {
            if (!spec.tunable_warps) return fail("--warps is not supported by this binary (no warps axis)");
            if (!detail::parse_int(val, iv) || iv <= 0) return fail("--warps must be a positive integer");
            a.warps = iv;
        } else {
            return fail("unknown flag --" + key);
        }
    }
    r.ok = true;
    return r;
}

// Exits with status 0 on --help and 2 on any error (after printing usage).
inline BenchArgs parse_bench_args(int argc, char** argv, const CliSpec& spec) {
    CliResult r = try_parse_bench_args(argc, argv, spec);
    if (r.help) { print_usage(spec); std::exit(0); }
    if (!r.ok) {
        std::fprintf(stderr, "%s: error: %s\n", spec.stage_id, r.error.c_str());
        print_usage(spec, stderr);
        std::exit(2);
    }
    return r.args;
}

inline std::string args_to_string(const BenchArgs& a) {
    std::ostringstream os;
    os << "n=" << a.n << " bh=" << a.bh << " d=" << a.d << " iters=" << a.iters << " reps=" << a.reps
       << " warmup=" << a.warmup << " seed=" << a.seed << " input=" << to_string(a.input);
    if (a.br || a.bc) os << " br=" << a.br << " bc=" << a.bc;
    if (a.warps) os << " warps=" << a.warps;
    return os.str();
}

}  // namespace w04
