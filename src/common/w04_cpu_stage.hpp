// -----------------------------------------------------------------------------
// W04 ATTENTION — shared main() flow for the CPU stages
//
//   parse CLI -> resolve Br/Bc -> stage validation -> inputs
//   -> timed protocol (bench_timing.hpp) -> peak RSS (before the reference runs)
//   -> check_stage (double reference, V=ones row-sum) -> report_and_write
//
// A stage provides only its forward function:
//   fwd(Q, K, V, O, shape, Br, Bc)   pointers into [BH][N][d] FP32 arrays
// -----------------------------------------------------------------------------
#pragma once

#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "attention_ref.hpp"
#include "bench_cli.hpp"
#include "bench_timing.hpp"
#include "w04_csv.hpp"

namespace w04 {

using CpuFwd = std::function<void(const float*, const float*, const float*, float*, const AttnShape&, int, int)>;

struct CpuStageSpec {
    StageMeta meta;
    bool tunable = false;   // accepts --br/--bc
    int default_br = 0;     // used when tunable and the flag is absent
    int default_bc = 0;
    int threads = 1;        // host threads the forward pass uses (CSV column)
    // Returns an empty string if the shape/tiles are supported, else the reason.
    std::function<std::string(const BenchArgs&)> validate;
};

inline int run_cpu_stage(int argc, char** argv, const CpuStageSpec& spec, const CpuFwd& fwd) {
    BenchArgs a = parse_bench_args(argc, argv, CliSpec{spec.meta.stage_id, spec.tunable});
    if (spec.tunable) {
        if (a.br == 0) a.br = spec.default_br;
        if (a.bc == 0) a.bc = spec.default_bc;
    }
    if (spec.validate) {
        const std::string why = spec.validate(a);
        if (!why.empty()) {
            std::fprintf(stderr, "%s: error: %s\n", spec.meta.stage_id, why.c_str());
            return 2;
        }
    }
    const AttnShape s = a.shape();
    std::printf("%s  git=%s  threads=%d\nflags   : %s\n", spec.meta.stage_id, W04_GIT_SHA, spec.threads,
                W04_COMPILER_FLAGS);

    std::vector<double> rep_ms;
    {
        const AttnInputs in = make_inputs(s, a.seed, a.input);
        std::vector<float> O(s.elems(), 0.0f);
        rep_ms = time_protocol_cpu(a, [&] { fwd(in.Q.data(), in.K.data(), in.V.data(), O.data(), s, a.br, a.bc); });
    }
    const std::uint64_t rss = cpu_peak_rss_bytes();  // before the reference allocates anything

    const CheckResult chk = check_stage(s, a.seed, a.input, spec.meta.precision,
                                        [&](const AttnInputs& in, std::vector<float>& O) {
                                            fwd(in.Q.data(), in.K.data(), in.V.data(), O.data(), s, a.br, a.bc);
                                        });
    return report_and_write(spec.meta, a, rep_ms, chk, rss, spec.threads, a.br, a.bc);
}

}  // namespace w04
