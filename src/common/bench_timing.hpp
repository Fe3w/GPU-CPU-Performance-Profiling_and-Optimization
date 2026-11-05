// -----------------------------------------------------------------------------
// W04 ATTENTION — CPU measurement protocol (P0)
//
// Protocol: `warmup` untimed calls, then `reps` repetitions
// of `iters` timed calls each, timed with std::chrono::steady_clock.
// Returns one value per repetition: mean milliseconds per call.
// The GPU equivalent (CUDA events) is in cuda_utils.cuh.
// -----------------------------------------------------------------------------
#pragma once

#include <chrono>
#include <cmath>
#include <cstdint>
#include <vector>

#include <sys/resource.h>

#include "bench_cli.hpp"

namespace w04 {

template <class F>
std::vector<double> time_protocol_cpu(const BenchArgs& a, F&& fn) {
    for (int w = 0; w < a.warmup; ++w) fn();
    std::vector<double> rep_ms;
    rep_ms.reserve(static_cast<std::size_t>(a.reps));
    for (int r = 0; r < a.reps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < a.iters; ++it) fn();
        const auto t1 = std::chrono::steady_clock::now();
        rep_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count() / a.iters);
    }
    return rep_ms;
}

struct MeanSd {
    double mean = 0.0, sd = 0.0;  // sd = sample standard deviation (n-1)
};

inline MeanSd mean_sd(const std::vector<double>& v) {
    MeanSd m;
    if (v.empty()) return m;
    for (double x : v) m.mean += x;
    m.mean /= static_cast<double>(v.size());
    if (v.size() > 1) {
        double sq = 0.0;
        for (double x : v) sq += (x - m.mean) * (x - m.mean);
        m.sd = std::sqrt(sq / static_cast<double>(v.size() - 1));
    }
    return m;
}

// Peak resident set size of this process so far (Linux reports ru_maxrss in KiB).
// Call it after the timed runs and before the reference, so the reference's
// buffers are not counted.
inline std::uint64_t cpu_peak_rss_bytes() {
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    return static_cast<std::uint64_t>(ru.ru_maxrss) * 1024ull;
}

}  // namespace w04
