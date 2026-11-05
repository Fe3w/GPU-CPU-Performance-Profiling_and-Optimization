// -----------------------------------------------------------------------------
// W04 ATTENTION — P0 GPU self-test
//
// 1. Proves all shared headers compile under nvcc (host code with -fopenmp).
// 2. Prints and checks the device facts the W04 plan relies on:
//    sm_89, 24 SMs, shared memory per block (default / opt-in) and per SM,
//    L2 size (risk C6: S/P may stay L2-resident at small N), bus width.
// 3. Checks the host rounding helpers used for the large-logits reference
//    against the CUDA conversions: round_to_fp16 vs __float2half_rn (host),
//    round_to_tf32 vs nvcuda::wmma::__float_to_tf32 (device).
// 4. Exercises time_protocol_gpu and GpuMemTracker on a trivial kernel.
// -----------------------------------------------------------------------------
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "attention_ref.hpp"
#include "bench_cli.hpp"
#include "bench_timing.hpp"
#include "cuda_utils.cuh"
#include "w04_csv.hpp"

using namespace w04;

static int g_fail = 0, g_pass = 0;
#define CHECK(cond, ...)                                                    \
    do {                                                                    \
        if (cond) { ++g_pass; }                                             \
        else { ++g_fail; std::printf("  FAIL %s:%d: ", __FILE__, __LINE__);  \
               std::printf(__VA_ARGS__); std::printf("\n"); }               \
    } while (0)

__global__ void to_tf32_kernel(const float* in, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = nvcuda::wmma::__float_to_tf32(in[i]);
}

__global__ void scale_kernel(float* x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] * 0.5f + 1.0f;
}

// Test values: random magnitudes over the FP16 range plus exact ties and edge cases.
static std::vector<float> rounding_test_values() {
    std::vector<float> v = {0.0f, -0.0f, 1.0f, 1.0f + 0x1p-11f, 1.0f + 0x1p-12f, 1.0f + 3 * 0x1p-11f,
                            65504.0f, 65519.0f, 65520.0f, 0x1p-14f, 0x1p-24f, 0x1p-25f, 3 * 0x1p-26f,
                            -30.0f, 29.999f, 1e-5f, -1e-7f};
    std::uint64_t st = 99;
    for (int i = 0; i < 200000; ++i) {
        const float u = uniform_pm1(st);
        const float e = std::ldexp(1.0f, static_cast<int>(splitmix64(st) % 40) - 24);  // 2^-24 .. 2^15
        v.push_back(u * e);
    }
    return v;
}

int main() {
    std::printf("W04_ATTENTION_SELFTEST_GPU  git=%s\nflags: %s\n", W04_GIT_SHA, W04_COMPILER_FLAGS);

    // ------------------------------------------------------------ device facts
    int dev = 0;
    W04_CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp p{};
    W04_CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    int bus_width = 0, l2 = 0, smem_sm = 0, smem_optin = 0;
    W04_CUDA_CHECK(cudaDeviceGetAttribute(&bus_width, cudaDevAttrGlobalMemoryBusWidth, dev));
    W04_CUDA_CHECK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev));
    W04_CUDA_CHECK(cudaDeviceGetAttribute(&smem_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev));
    W04_CUDA_CHECK(cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    std::printf("[device]\n");
    std::printf("  name                     : %s\n", p.name);
    std::printf("  compute capability       : %d.%d\n", p.major, p.minor);
    std::printf("  SMs                      : %d\n", p.multiProcessorCount);
    std::printf("  global memory            : %.0f MiB\n", p.totalGlobalMem / 1048576.0);
    std::printf("  memory bus width         : %d bit\n", bus_width);
    std::printf("  L2 cache                 : %d bytes (%.1f MiB)\n", l2, l2 / 1048576.0);
    std::printf("  smem per block (default) : %zu bytes\n", p.sharedMemPerBlock);
    std::printf("  smem per block (opt-in)  : %d bytes\n", smem_optin);
    std::printf("  smem per SM              : %d bytes\n", smem_sm);
    std::printf("  regs per SM              : %d\n", p.regsPerMultiprocessor);
    std::printf("  max threads per SM       : %d\n", p.maxThreadsPerMultiProcessor);
    CHECK(p.major == 8 && p.minor == 9, "expected sm_89, got %d.%d", p.major, p.minor);
    CHECK(p.multiProcessorCount == 24, "expected 24 SMs, got %d", p.multiProcessorCount);
    CHECK(bus_width == 128, "expected 128-bit bus, got %d", bus_width);
    CHECK(p.sharedMemPerBlock == 48 * 1024, "default smem per block %zu", p.sharedMemPerBlock);

    // ------------------------------------------------------- rounding helpers
    std::printf("[rounding helpers vs CUDA]\n");
    const std::vector<float> vals = rounding_test_values();
    std::size_t fp16_mismatch = 0;
    for (float x : vals) {
        const float want = __half2float(__float2half_rn(x));
        const float got = round_to_fp16(x);
        if (float_bits(want) != float_bits(got)) {
            if (fp16_mismatch < 5) std::printf("  fp16 mismatch x=%a want=%a got=%a\n", x, want, got);
            ++fp16_mismatch;
        }
    }
    CHECK(fp16_mismatch == 0, "round_to_fp16 differs from __float2half_rn for %zu of %zu values", fp16_mismatch,
          vals.size());

    const int nv = static_cast<int>(vals.size());
    float *d_in = nullptr, *d_out = nullptr;
    W04_CUDA_CHECK(cudaMalloc(&d_in, nv * sizeof(float)));
    W04_CUDA_CHECK(cudaMalloc(&d_out, nv * sizeof(float)));
    W04_CUDA_CHECK(cudaMemcpy(d_in, vals.data(), nv * sizeof(float), cudaMemcpyHostToDevice));
    to_tf32_kernel<<<(nv + 255) / 256, 256>>>(d_in, d_out, nv);
    W04_CUDA_CHECK(cudaGetLastError());
    std::vector<float> tf(vals.size());
    W04_CUDA_CHECK(cudaMemcpy(tf.data(), d_out, nv * sizeof(float), cudaMemcpyDeviceToHost));
    std::size_t tf32_mismatch = 0;
    for (std::size_t i = 0; i < vals.size(); ++i) {
        const float got = round_to_tf32(vals[i]);
        if (float_bits(tf[i]) != float_bits(got)) {
            if (tf32_mismatch < 5) std::printf("  tf32 mismatch x=%a device=%a host=%a\n", vals[i], tf[i], got);
            ++tf32_mismatch;
        }
    }
    CHECK(tf32_mismatch == 0, "round_to_tf32 differs from wmma::__float_to_tf32 for %zu of %zu values", tf32_mismatch,
          vals.size());
    W04_CUDA_CHECK(cudaFree(d_in));
    W04_CUDA_CHECK(cudaFree(d_out));

    // -------------------------------------------------- timing + memory tracker
    std::printf("[timing protocol + memory tracker]\n");
    GpuMemTracker mem;
    const int n = 1 << 24;  // 64 MiB
    float* d_x = nullptr;
    W04_CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    W04_CUDA_CHECK(cudaMemset(d_x, 0, n * sizeof(float)));
    const std::uint64_t used = mem.bytes_above_base();
    std::printf("  tracker: %llu bytes above baseline after a 64 MiB allocation\n",
                static_cast<unsigned long long>(used));
    CHECK(used >= static_cast<std::uint64_t>(n) * sizeof(float), "memory tracker must see the allocation");

    BenchArgs a;
    a.warmup = 2; a.iters = 10; a.reps = 3;
    int launches = 0;
    const auto t = time_protocol_gpu(a, [&] { scale_kernel<<<(n + 255) / 256, 256>>>(d_x, n); ++launches; });
    const MeanSd m = mean_sd(t);
    std::printf("  scale kernel (64 MiB read + write): %.4f ± %.4f ms -> %.1f GB/s\n", m.mean, m.sd,
                2.0 * n * sizeof(float) / (m.mean * 1e-3) / 1e9);
    CHECK(launches == 2 + 10 * 3 && t.size() == 3 && m.mean > 0.0, "launches=%d reps=%zu", launches, t.size());
    W04_CUDA_CHECK(cudaFree(d_x));

    // ------------------------------------------- host reference under nvcc
    std::printf("[reference compiled by nvcc host compiler]\n");
    {
        AttnShape s{1, 64, 64};
        AttnInputs in = make_inputs(s, 1, InputCase::LargeLogits, true);
        std::vector<double> O;
        attention_reference(in.Q.data(), in.K.data(), in.V.data(), s, select_check_rows(s, 1), O);
        double worst = 0.0;
        for (double o : O) worst = std::max(worst, std::fabs(o - 1.0));
        CHECK(worst < 1e-12, "V=ones large_logits under nvcc: max |O-1| = %.3e", worst);
    }

    std::printf("selftest_gpu: %d passed, %d failed\n", g_pass, g_fail);
    std::printf("Correct : %s\n", g_fail == 0 ? "YES" : "NO");
    return g_fail == 0 ? 0 : 1;
}
