// -----------------------------------------------------------------------------
// W04 ATTENTION — shared main() flow for the GPU stages (P2)
//
//   parse CLI -> resolve Br/Bc -> stage validation -> device buffers + stage
//   -> upload inputs -> timed protocol (CUDA events, cuda_utils.cuh)
//   -> peak device memory -> per-kernel breakdown (separate pass)
//   -> check_stage (double reference, V=ones row-sum) -> report_and_write
//
// A stage is a class with
//   Stage(const AttnShape&, const BenchArgs&)   allocates its workspace (e.g. S)
//   int phases() const                          number of kernels per forward pass
//   const char* phase_name(int) const
//   void launch(int phase, const DeviceAttn&)   enqueues one kernel on the default stream
// and optionally (P4, decision C23)
//   void prepare(const DeviceAttn&)             enqueues untimed preparation work that
//                                               depends only on the inputs (GPU S3 FP16:
//                                               Q, K, V -> half). It runs after every
//                                               upload, outside the timed forward pass, and
//                                               is timed on its own with the same protocol
//                                               (CSV column prep_ms).
//
// Notes
//   * Peak memory = device memory in use above the baseline taken before any
//     allocation, read after the timed run (so lazily allocated library
//     workspace is included). It includes allocator rounding.
//   * The breakdown pass records an event between every two kernels. It is
//     printed for diagnosis only; runtime_ms in the CSV comes from the timed
//     protocol, which has no events between kernels.
//   * Before each correctness run O is filled with 0xFF bytes (NaN), so an
//     output the stage never writes is reported as non-finite.
//   * If the environment variable W04_DUMP_O names a file, the O of the first
//     correctness run (the requested input, not the V=ones run) is written there
//     as raw float32 [BH][N][d], for direct stage-vs-stage comparisons
//     (tools/w04_compare_o.py). The CLI is unchanged.
// -----------------------------------------------------------------------------
#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "attention_ref.hpp"
#include "bench_cli.hpp"
#include "cuda_utils.cuh"
#include "w04_csv.hpp"

namespace w04 {

// Device views of one attention problem, all [BH][N][d] FP32.
struct DeviceAttn {
    const float* Q;
    const float* K;
    const float* V;
    float* O;
    AttnShape s;
    int br, bc;
};

struct GpuStageSpec {
    StageMeta meta;
    bool tunable = false;
    int default_br = 0;
    int default_bc = 0;
    std::function<std::string(const BenchArgs&)> validate;  // "" if supported, else the reason
};

// True if Stage has a member prepare(const DeviceAttn&).
template <class S, class = void>
struct has_prepare : std::false_type {};
template <class S>
struct has_prepare<S, std::void_t<decltype(std::declval<S&>().prepare(std::declval<const DeviceAttn&>()))>>
    : std::true_type {};

template <class Stage>
void print_phase_breakdown(const BenchArgs& a, Stage& st, const DeviceAttn& io) {
    const int np = st.phases();
    std::vector<cudaEvent_t> ev(static_cast<std::size_t>(np) + 1);
    for (auto& e : ev) W04_CUDA_CHECK(cudaEventCreate(&e));
    std::vector<double> sum(static_cast<std::size_t>(np), 0.0);
    const int iters = a.iters > 0 ? a.iters : 1;
    for (int it = 0; it < iters; ++it) {
        for (int p = 0; p < np; ++p) {
            W04_CUDA_CHECK(cudaEventRecord(ev[p]));
            st.launch(p, io);
        }
        W04_CUDA_CHECK(cudaEventRecord(ev[np]));
        W04_CUDA_CHECK(cudaEventSynchronize(ev[np]));
        W04_CUDA_CHECK(cudaGetLastError());
        for (int p = 0; p < np; ++p) {
            float ms = 0.0f;
            W04_CUDA_CHECK(cudaEventElapsedTime(&ms, ev[p], ev[p + 1]));
            sum[p] += ms;
        }
    }
    double total = 0.0;
    for (double t : sum) total += t;
    std::printf("phases  :");
    for (int p = 0; p < np; ++p)
        std::printf(" %s %.4f ms (%.1f%%)%s", st.phase_name(p), sum[p] / iters,
                    total > 0 ? 100.0 * sum[p] / total : 0.0, p + 1 < np ? " |" : "");
    std::printf("  [separate pass, events between kernels, not the benchmark time]\n");
    for (auto& e : ev) W04_CUDA_CHECK(cudaEventDestroy(e));
}

// Writes O to the file named by W04_DUMP_O (if set); exits 1 if the write fails.
inline void dump_output(const std::vector<float>& O) {
    const char* path = std::getenv("W04_DUMP_O");
    if (path == nullptr || *path == '\0') return;
    std::FILE* f = std::fopen(path, "wb");
    const bool ok = f != nullptr && std::fwrite(O.data(), sizeof(float), O.size(), f) == O.size();
    if (f != nullptr && std::fclose(f) != 0) {
        std::fprintf(stderr, "error: closing %s failed\n", path);
        std::exit(1);
    }
    if (!ok) {
        std::fprintf(stderr, "error: cannot write O to %s\n", path);
        std::exit(1);
    }
    std::printf("dump    : O (%zu floats, first correctness run) -> %s\n", O.size(), path);
}

template <class Stage>
int run_gpu_stage(int argc, char** argv, const GpuStageSpec& spec) {
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
    cudaDeviceProp prop{};
    W04_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("%s  git=%s  device=%s (sm_%d%d)\nflags   : %s\n", spec.meta.stage_id, W04_GIT_SHA, prop.name,
                prop.major, prop.minor, W04_COMPILER_FLAGS);

    GpuMemTracker mem;  // baseline before any allocation
    const std::size_t bytes = s.elems() * sizeof(float);
    float *dQ = nullptr, *dK = nullptr, *dV = nullptr, *dO = nullptr;
    W04_CUDA_CHECK(cudaMalloc(&dQ, bytes));
    W04_CUDA_CHECK(cudaMalloc(&dK, bytes));
    W04_CUDA_CHECK(cudaMalloc(&dV, bytes));
    W04_CUDA_CHECK(cudaMalloc(&dO, bytes));

    std::vector<double> rep_ms, prep_ms;
    std::uint64_t peak = 0;
    CheckResult chk;
    {
        Stage st(s, a);
        const DeviceAttn io{dQ, dK, dV, dO, s, a.br, a.bc};
        auto upload = [&](const AttnInputs& in) {
            W04_CUDA_CHECK(cudaMemcpy(dQ, in.Q.data(), bytes, cudaMemcpyHostToDevice));
            W04_CUDA_CHECK(cudaMemcpy(dK, in.K.data(), bytes, cudaMemcpyHostToDevice));
            W04_CUDA_CHECK(cudaMemcpy(dV, in.V.data(), bytes, cudaMemcpyHostToDevice));
            if constexpr (has_prepare<Stage>::value) {
                st.prepare(io);
                W04_CUDA_CHECK(cudaGetLastError());
                W04_CUDA_CHECK(cudaDeviceSynchronize());
            }
        };
        auto forward = [&] {
            for (int p = 0; p < st.phases(); ++p) st.launch(p, io);
        };

        bool dumped = false;
        upload(make_inputs(s, a.seed, a.input));
        rep_ms = time_protocol_gpu(a, forward);
        peak = mem.bytes_above_base();
        print_phase_breakdown(a, st, io);
        if constexpr (has_prepare<Stage>::value) {
            // Same protocol as the forward pass, after it; re-preparing the same inputs is idempotent.
            prep_ms = time_protocol_gpu(a, [&] { st.prepare(io); });
            double mean = 0, sd = 0;
            for (double t : prep_ms) mean += t;
            mean /= static_cast<double>(prep_ms.size());
            for (double t : prep_ms) sd += (t - mean) * (t - mean);
            if (prep_ms.size() > 1) sd = std::sqrt(sd / static_cast<double>(prep_ms.size() - 1));
            std::printf("prepare : %s %.4f ± %.4f ms per call  [untimed in runtime_ms; CSV prep_ms]\n",
                        st.prepare_name(), mean, sd);
        }

        chk = check_stage(s, a.seed, a.input, spec.meta.precision,
                          [&](const AttnInputs& in, std::vector<float>& O) {
                              upload(in);
                              W04_CUDA_CHECK(cudaMemset(dO, 0xFF, bytes));
                              forward();
                              W04_CUDA_CHECK(cudaGetLastError());
                              W04_CUDA_CHECK(cudaDeviceSynchronize());
                              W04_CUDA_CHECK(cudaMemcpy(O.data(), dO, bytes, cudaMemcpyDeviceToHost));
                              if (!dumped) {
                                  dumped = true;
                                  dump_output(O);
                              }
                          });
    }
    W04_CUDA_CHECK(cudaFree(dQ));
    W04_CUDA_CHECK(cudaFree(dK));
    W04_CUDA_CHECK(cudaFree(dV));
    W04_CUDA_CHECK(cudaFree(dO));
    return report_and_write(spec.meta, a, rep_ms, chk, peak, /*threads=*/0, a.br, a.bc,
                            has_prepare<Stage>::value ? &prep_ms : nullptr);
}

}  // namespace w04
