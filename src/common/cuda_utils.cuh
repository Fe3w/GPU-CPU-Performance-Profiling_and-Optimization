// -----------------------------------------------------------------------------
// W04 ATTENTION — CUDA helpers shared by every GPU stage (P0)
//
// * W04_CUDA_CHECK: prints file:line and exits. Unlike the W03 macro
//   (`return EXIT_FAILURE`) it works in any function, not only main().
// * time_protocol_gpu: same protocol as the CPU, timed with CUDA events.
//   ONE event pair encloses all `iters` launches of a repetition (W03 synced
//   after every launch). For the 3-kernel stages, inter-kernel gaps are thus
//   included in exactly the same way for every stage (relevant to E5).
// * GpuMemTracker: device memory in use above a baseline, via cudaMemGetInfo
//   (E6: O(N^2) vs O(N) footprint).
// -----------------------------------------------------------------------------
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "bench_cli.hpp"

#define W04_CUDA_CHECK(call)                                                               \
    do {                                                                                   \
        const cudaError_t err__ = (call);                                                  \
        if (err__ != cudaSuccess) {                                                        \
            std::fprintf(stderr, "CUDA error: %s (%s) at %s:%d\n", cudaGetErrorString(err__), \
                         #call, __FILE__, __LINE__);                                       \
            std::exit(EXIT_FAILURE);                                                       \
        }                                                                                  \
    } while (0)

namespace w04 {

// fn() must only enqueue work on the default stream (kernel launches); it is not synchronised per call.
template <class F>
std::vector<double> time_protocol_gpu(const BenchArgs& a, F&& fn) {
    for (int w = 0; w < a.warmup; ++w) fn();
    W04_CUDA_CHECK(cudaGetLastError());
    W04_CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    W04_CUDA_CHECK(cudaEventCreate(&start));
    W04_CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<double> rep_ms;
    rep_ms.reserve(static_cast<std::size_t>(a.reps));
    for (int r = 0; r < a.reps; ++r) {
        W04_CUDA_CHECK(cudaEventRecord(start));
        for (int it = 0; it < a.iters; ++it) fn();
        W04_CUDA_CHECK(cudaEventRecord(stop));
        W04_CUDA_CHECK(cudaEventSynchronize(stop));
        W04_CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        W04_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        rep_ms.push_back(static_cast<double>(ms) / a.iters);
    }
    W04_CUDA_CHECK(cudaEventDestroy(start));
    W04_CUDA_CHECK(cudaEventDestroy(stop));
    return rep_ms;
}

// Construct BEFORE the stage allocates; peak_bytes() after all allocations.
// Measures device memory in use above the baseline (includes allocator rounding).
class GpuMemTracker {
public:
    GpuMemTracker() { base_used_ = used_now(); }
    std::uint64_t bytes_above_base() const {
        const std::uint64_t u = used_now();
        return u > base_used_ ? u - base_used_ : 0;
    }

private:
    static std::uint64_t used_now() {
        std::size_t free_b = 0, total_b = 0;
        W04_CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
        return static_cast<std::uint64_t>(total_b - free_b);
    }
    std::uint64_t base_used_ = 0;
};

}  // namespace w04
