// -----------------------------------------------------------------------------
// W04 ATTENTION — append-only CSV writer and result reporting (P0)
//
// * One row per repetition. Schema = plan §7 plus five columns approved in P0
//   (seed, input_case, warmup, reps, threads) and two added in P4 (decision
//   C23), appended at the end:
//     prep_ms          time of the stage's untimed preparation step (FP16
//                      conversion of Q, K, V in GPU S3), same protocol, per
//                      rep; empty for stages without one
//     max_rel_err_main the error that decides `correct` (large_logits with a
//                      reduced-precision stage: reference on the rounded
//                      inputs, decision C2); equals max_rel_err otherwise
//   and one added in P5 (decision C28):
//     warps            warps per block of a stage with a warps axis (GPU S2b,
//                      S3); 0 for every other stage
// * The file is opened in append mode and never truncated (W03 used
//   std::ofstream(path), which truncates). The header is written only into an
//   empty file; a non-empty file with a different header is rejected.
// * git_sha and compiler_flags are compiled in by the Makefile
//   (build/w04_build_info.h and -DW04_COMPILER_FLAGS), never typed by hand.
// -----------------------------------------------------------------------------
#pragma once

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__has_include)
#if __has_include("w04_build_info.h")
#include "w04_build_info.h"
#endif
#endif
#ifndef W04_GIT_SHA
#define W04_GIT_SHA "unknown"
#endif
#ifndef W04_COMPILER_FLAGS
#define W04_COMPILER_FLAGS "unknown"
#endif

#include "attention_ref.hpp"
#include "bench_cli.hpp"

namespace w04 {

struct W04Row {
    std::string workload = "W04_ATTENTION";
    std::string stage;       // e.g. "S2b_SWOPT"
    std::string variant;     // e.g. "FUSED"
    std::string device;      // "CPU" | "GPU"
    std::string precision;   // "FP32" | "TF32" | "FP16"
    std::size_t n = 0;
    int bh = 0, d = 0, iters = 0, rep = 0;
    double runtime_ms = 0, gflops = 0, alg_bytes = 0, max_abs_err = 0, max_rel_err = 0;
    bool correct = false;
    std::uint64_t peak_mem_bytes = 0;
    int br = 0, bc = 0;
    std::string compiler_flags = W04_COMPILER_FLAGS;
    std::string timestamp;
    std::string git_sha = W04_GIT_SHA;
    // P0 additions
    std::uint64_t seed = 0;
    std::string input_case;
    int warmup = 0, reps = 0;
    int threads = 0;          // host threads used by the timed stage; 0 = n/a (GPU)
    // P4 additions (C23)
    bool has_prep = false;    // false -> prep_ms is written as an empty field
    double prep_ms = 0;
    double max_rel_err_main = 0;
    // P5 addition (C28)
    int warps = 0;            // 0 = the stage has no warps axis
};

inline const char* w04_csv_header() {
    return "workload,stage,variant,device,precision,n,bh,d,iters,rep,runtime_ms,gflops,alg_bytes,"
           "max_abs_err,max_rel_err,correct,peak_mem_bytes,br,bc,compiler_flags,timestamp,git_sha,"
           "seed,input_case,warmup,reps,threads,prep_ms,max_rel_err_main,warps";
}

inline std::string csv_field(const std::string& s) {
    if (s.find_first_of(",\"\n") == std::string::npos) return s;
    std::string q = "\"";
    for (char c : s) { if (c == '"') q += '"'; q += c; }
    return q + "\"";
}

inline std::string csv_num(double v) {
    std::ostringstream os;
    os << std::setprecision(9) << v;
    return os.str();
}

class W04CsvWriter {
public:
    explicit W04CsvWriter(const std::string& path) : path_(path) {
        namespace fs = std::filesystem;
        const fs::path p(path);
        if (p.has_parent_path()) fs::create_directories(p.parent_path());
        bool empty = true;
        {
            std::ifstream in(path);
            std::string first;
            if (in && std::getline(in, first)) {
                empty = false;
                if (first != w04_csv_header())
                    throw std::runtime_error("CSV header mismatch in existing file " + path +
                                             " (refusing to append to a different schema)");
            }
        }
        out_.open(path, std::ios::out | std::ios::app);
        if (!out_) throw std::runtime_error("cannot open CSV for append: " + path);
        if (empty) out_ << w04_csv_header() << '\n';
    }

    void append(const W04Row& r) {
        // Traceability columns must never be empty.
        if (r.timestamp.empty() || r.git_sha.empty() || r.compiler_flags.empty())
            throw std::runtime_error("W04Row has an empty timestamp, git_sha or compiler_flags: " + path_);
        out_ << csv_field(r.workload) << ',' << csv_field(r.stage) << ',' << csv_field(r.variant) << ','
             << csv_field(r.device) << ',' << csv_field(r.precision) << ',' << r.n << ',' << r.bh << ','
             << r.d << ',' << r.iters << ',' << r.rep << ',' << csv_num(r.runtime_ms) << ','
             << csv_num(r.gflops) << ',' << csv_num(r.alg_bytes) << ',' << csv_num(r.max_abs_err) << ','
             << csv_num(r.max_rel_err) << ',' << (r.correct ? "YES" : "NO") << ',' << r.peak_mem_bytes
             << ',' << r.br << ',' << r.bc << ',' << csv_field(r.compiler_flags) << ','
             << csv_field(r.timestamp) << ',' << csv_field(r.git_sha) << ',' << r.seed << ','
             << csv_field(r.input_case) << ',' << r.warmup << ',' << r.reps << ',' << r.threads << ','
             << (r.has_prep ? csv_num(r.prep_ms) : std::string()) << ',' << csv_num(r.max_rel_err_main)
             << ',' << r.warps << '\n';
        out_.flush();
        if (!out_) throw std::runtime_error("write failed: " + path_);
    }

private:
    std::string path_;
    std::ofstream out_;
};

// ------------------------------------------------------------- timestamps
inline std::string format_local_time(const char* fmt) {
    const std::time_t t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm tm{};
    localtime_r(&t, &tm);
    char buf[64];
    std::strftime(buf, sizeof buf, fmt, &tm);
    return buf;
}
inline std::string run_timestamp() { return format_local_time("%Y%m%d-%H%M%S"); }
inline std::string iso_timestamp() { return format_local_time("%Y-%m-%dT%H:%M:%S%z"); }

inline std::string default_out_path(const std::string& stage_id) {
    return "results/raw/W04_ATTENTION/" + run_timestamp() + "/" + stage_id + ".csv";
}

// ---------------------------------------------------------- stage reporting
// Static identity of a stage; one per binary.
struct StageMeta {
    const char* stage_id;   // full name, e.g. "W04_ATTENTION_GPU_S2b_SWOPT_FUSED"
    const char* stage;      // "S2b_SWOPT"
    const char* variant;    // "FUSED"
    const char* device;     // "CPU" | "GPU"
    Precision precision;
};

// Prints the correctness details, the one-line summary and the exact
// "Correct : YES|NO" line the test greps for; appends one CSV row per rep.
// Returns the process exit code (0 only if correct).
// prep_ms: per-rep times of an untimed preparation step (nullptr = none); must
// have one entry per rep if given. warps: 0 for stages without a warps axis.
inline int report_and_write(const StageMeta& m, const BenchArgs& a, const std::vector<double>& rep_ms,
                            const CheckResult& chk, std::uint64_t peak_mem_bytes, int threads,
                            int br = 0, int bc = 0, const std::vector<double>* prep_ms = nullptr,
                            int warps = 0) {
    if (prep_ms != nullptr && prep_ms->size() != rep_ms.size())
        throw std::runtime_error("report_and_write: prep_ms needs one entry per rep");
    const AttnShape s = a.shape();
    double mean = 0, sd = 0;
    for (double t : rep_ms) mean += t;
    mean /= static_cast<double>(rep_ms.size());
    if (rep_ms.size() > 1) {
        for (double t : rep_ms) sd += (t - mean) * (t - mean);
        sd = std::sqrt(sd / static_cast<double>(rep_ms.size() - 1));
    }
    const double gflops = model_flops(s) / (mean * 1e-3) / 1e9;

    std::printf("check   : main max_abs=%.3e max_rel=%.3e nonfinite=%zu | plain max_rel=%.3e | "
                "rowsum max_abs=%.3e nonfinite=%zu | elem_rel(diag)=%.3e | tol=%.0e\n",
                chk.main.max_abs_err, chk.main.max_rel_err, chk.main.n_nonfinite, chk.plain.max_rel_err,
                chk.rowsum.max_abs_err, chk.rowsum.n_nonfinite, chk.main.max_elem_rel, tolerance_for(m.precision));
    std::printf("summary : %s %s | %.4f ± %.4f ms | %.1f GFLOP/s | %s\n", m.stage_id, args_to_string(a).c_str(),
                mean, sd, gflops, to_string(m.precision));

    {
        const std::string sha = W04_GIT_SHA;
        if (sha == "nogit" || (sha.size() > 6 && sha.compare(sha.size() - 6, 6, "-dirty") == 0))
            std::printf("warning : git_sha=%s (results not reproducible from a clean commit)\n", sha.c_str());
    }
    const std::string path = a.out.empty() ? default_out_path(m.stage_id) : a.out;
    W04CsvWriter w(path);
    const std::string ts = iso_timestamp();
    for (std::size_t r = 0; r < rep_ms.size(); ++r) {
        W04Row row;
        row.stage = m.stage; row.variant = m.variant; row.device = m.device;
        row.precision = to_string(m.precision);
        row.n = s.n; row.bh = s.bh; row.d = s.d; row.iters = a.iters; row.rep = static_cast<int>(r + 1);
        row.runtime_ms = rep_ms[r];
        row.gflops = model_flops(s) / (rep_ms[r] * 1e-3) / 1e9;
        row.alg_bytes = alg_bytes(s);
        row.max_abs_err = chk.plain.max_abs_err;   // plain FP32-input reference (decision C2)
        row.max_rel_err = chk.plain.max_rel_err;
        row.correct = chk.correct;
        row.peak_mem_bytes = peak_mem_bytes;
        row.br = br; row.bc = bc;
        row.timestamp = ts;
        row.seed = a.seed; row.input_case = to_string(a.input);
        row.warmup = a.warmup; row.reps = a.reps; row.threads = threads;
        row.has_prep = prep_ms != nullptr;
        if (row.has_prep) row.prep_ms = (*prep_ms)[r];
        row.max_rel_err_main = chk.main.max_rel_err;
        row.warps = warps;
        w.append(row);
    }
    std::printf("csv     : %s\n", path.c_str());
    std::printf("Correct : %s\n", chk.correct ? "YES" : "NO");
    return chk.correct ? 0 : 1;
}

}  // namespace w04
