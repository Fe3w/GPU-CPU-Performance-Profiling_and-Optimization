#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace nvcuda;

#define CUDA_CHECK(call)                                                   \
    do {                                                                    \
        cudaError_t err__ = (call);                                         \
        if (err__ != cudaSuccess) {                                        \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)        \
                      << " at " << __FILE__ << ":" << __LINE__ << '\n';     \
            return EXIT_FAILURE;                                            \
        }                                                                   \
    } while (0)

// -----------------------------------------------------------------------------
// W03 GEMM — GPU S3 TENSOR CORE TF32
//
// Tensor Core GEMM using the WMMA API with TF32 inputs and FP32 accumulation:
//   4 warps per block, each warp computes a 16x16 output tile, so a block
//   covers a 32x32 output tile
//   A and B are staged through shared memory in k-tiles of BLOCK_K = 32
//   (4 m16n16k8 MMAs per warp between two barriers)
//   no cuBLAS
//
// Changes against the first version (k-tile 8, load_matrix_sync), made after
// the final re-test of 2026-11-06:
//   * fragment loads name the .shared state space (inline PTX, as in W04 S3
//     TF32): wmma::load_matrix_sync for tf32 emits the generic-address form,
//     which ptxas compiles to LD.E (the global load path) although the data
//     is in shared memory
//   * k-tile 32 instead of 8: four MMAs per pair of __syncthreads instead of one
//   * shared tiles padded by 4 floats (ldm 36: a multiple of 4 as WMMA tf32
//     requires; fragment addresses stay 32-byte aligned)
//   * inputs rounded to TF32 once, when stored to shared memory
//     (__float_to_tf32); the first version passed FP32 bits to the tf32 MMA
//   * edge tiles are stored with bounds checks, so any N works (the first
//     version required N to be a multiple of 16)
//
// Row-major:
//   A[i*N + k]
//   B[k*N + j]
//   C[i*N + j]
//
// TF32 has a 10-bit mantissa, so the relative error vs. an fp64 reference is
// larger than for the FP32 S1/S2 kernels (~0.07%, inside the 0.1% check).
// -----------------------------------------------------------------------------

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 8

#define BLOCK_M 32
#define BLOCK_N 32
#define BLOCK_K 32
#define SMEM_PAD 4
#define BLOCK_THREADS 128

static void initialize_matrices(std::vector<float>& A,
                                std::vector<float>& B) {
    for (std::size_t idx = 0; idx < A.size(); ++idx) {
        A[idx] = static_cast<float>(idx % 1000) * 0.001f;
        B[idx] = static_cast<float>((idx * 3) % 1000) * 0.001f;
    }
}

// Independent double-precision reference for the top-left REF_N x REF_N
// output region, while using the full K=N dimension.
static void reference_gemm_subset(const std::vector<float>& A,
                                  const std::vector<float>& B,
                                  std::vector<double>& C,
                                  std::size_t N,
                                  std::size_t REF_N) {
    for (std::size_t i = 0; i < REF_N; ++i) {
        for (std::size_t j = 0; j < REF_N; ++j) {
            double sum = 0.0;

            for (std::size_t k = 0; k < N; ++k) {
                sum += static_cast<double>(A[i * N + k]) *
                       static_cast<double>(B[k * N + j]);
            }

            C[i * REF_N + j] = sum;
        }
    }
}

static double relative_error_percent(double result,
                                     double reference) {
    const double denominator =
        std::max(std::abs(reference),
                 std::numeric_limits<double>::min());

    return std::abs(result - reference) / denominator * 100.0;
}

// -----------------------------------------------------------------------------
// Tensor Core GEMM kernel (TF32 WMMA)
// -----------------------------------------------------------------------------
// Fragment loads from shared memory with the .shared state space named in the
// PTX (same instruction and fragment layout as wmma::load_matrix_sync).
using frag_a_t = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major>;
using frag_b_t = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major>;

#define W03_TF32_LDS(AB)                                                                               \
    unsigned u0, u1, u2, u3;                                                                           \
    const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));                             \
    asm volatile("wmma.load." AB ".sync.aligned.row.m16n16k8.shared.tf32 {%0, %1, %2, %3}, [%4], %5;" \
                 : "=r"(u0), "=r"(u1), "=r"(u2), "=r"(u3)                                               \
                 : "r"(a), "r"(ld)                                                                     \
                 : "memory");                                                                          \
    f.x[0] = __uint_as_float(u0); f.x[1] = __uint_as_float(u1);                                        \
    f.x[2] = __uint_as_float(u2); f.x[3] = __uint_as_float(u3);

__device__ inline void load_shared(frag_a_t& f, const float* p, unsigned ld) { W03_TF32_LDS("a") }
__device__ inline void load_shared(frag_b_t& f, const float* p, unsigned ld) { W03_TF32_LDS("b") }
#undef W03_TF32_LDS

__global__ void gemm_tensor_core(const float* A,
                                 const float* B,
                                 float* C,
                                 std::size_t N) {

    // ldm = BLOCK_K + 4 and BLOCK_N + 4 = 36 floats; 32-byte aligned rows of 16
    __shared__ __align__(32) float As[BLOCK_M][BLOCK_K + SMEM_PAD];
    __shared__ __align__(32) float Bs[BLOCK_K][BLOCK_N + SMEM_PAD];

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;

    const int warp_row = warp_id / 2;
    const int warp_col = warp_id % 2;

    const std::size_t block_row =
        static_cast<std::size_t>(blockIdx.y) * BLOCK_M;

    const std::size_t block_col =
        static_cast<std::size_t>(blockIdx.x) * BLOCK_N;

    const std::size_t row =
        block_row +
        static_cast<std::size_t>(warp_row) * WMMA_M;

    const std::size_t col =
        block_col +
        static_cast<std::size_t>(warp_col) * WMMA_N;

    frag_a_t a_frag;
    frag_b_t b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (std::size_t kt = 0; kt < N; kt += BLOCK_K) {

        // Cooperatively load A: 32 x 32 floats, consecutive threads read
        // consecutive k (coalesced); rounded to TF32 once here.
        for (int idx = tid; idx < BLOCK_M * BLOCK_K; idx += BLOCK_THREADS) {
            const int r = idx / BLOCK_K;
            const int k = idx % BLOCK_K;
            const std::size_t gr = block_row + r;
            const std::size_t gk = kt + k;
            As[r][k] = wmma::__float_to_tf32((gr < N && gk < N) ? A[gr * N + gk] : 0.0f);
        }

        // Cooperatively load B: 32 x 32 floats, consecutive threads read
        // consecutive columns (coalesced).
        for (int idx = tid; idx < BLOCK_K * BLOCK_N; idx += BLOCK_THREADS) {
            const int k = idx / BLOCK_N;
            const int c = idx % BLOCK_N;
            const std::size_t gk = kt + k;
            const std::size_t gc = block_col + c;
            Bs[k][c] = wmma::__float_to_tf32((gk < N && gc < N) ? B[gk * N + gc] : 0.0f);
        }

        __syncthreads();

        // Each warp: four m16n16k8 MMAs on its 16x16 C tile.
#pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk += WMMA_K) {
            load_shared(a_frag, &As[warp_row * WMMA_M][kk], BLOCK_K + SMEM_PAD);
            load_shared(b_frag, &Bs[kk][warp_col * WMMA_N], BLOCK_N + SMEM_PAD);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        __syncthreads();
    }

    // Full 16x16 tiles are stored directly. A tile that crosses the matrix
    // edge (N not a multiple of 16), or an ldm that is not a multiple of 4, goes
    // through a per-warp shared buffer and is copied with bounds checks; the
    // first version stored whole tiles past the edge and failed for such N.
    __shared__ __align__(32) float Cs[BLOCK_THREADS / 32][WMMA_M * WMMA_N];
    const bool full = row + WMMA_M <= N && col + WMMA_N <= N && N % 4 == 0;
    if (full) {
        wmma::store_matrix_sync(&C[row * N + col], c_frag, N, wmma::mem_row_major);
    } else if (row < N && col < N) {
        wmma::store_matrix_sync(Cs[warp_id], c_frag, WMMA_N, wmma::mem_row_major);
        __syncwarp();
        const int lane = tid % 32;
        for (int e = lane; e < WMMA_M * WMMA_N; e += 32) {
            const std::size_t r = row + e / WMMA_N;
            const std::size_t c = col + e % WMMA_N;
            if (r < N && c < N) C[r * N + c] = Cs[warp_id][e];
        }
    }
}

int main(int argc, char* argv[]) {

    std::size_t N = 1024;
    int iterations = 20;
    int repetitions = 5;
    int warmup = 2;

    std::string output_path =
        "results/raw/W03_GEMM/"
        "W03_GEMM_GPU_S3_HWACC_TENSOR_CORE.csv";

    if (argc > 1) N = std::stoull(argv[1]);
    if (argc > 2) iterations = std::stoi(argv[2]);
    if (argc > 3) repetitions = std::stoi(argv[3]);
    if (argc > 4) warmup = std::stoi(argv[4]);
    if (argc > 5) output_path = argv[5];

    if (N == 0 || iterations <= 0 ||
        repetitions <= 0 || warmup < 0) {
        std::cerr << "Invalid benchmark parameters.\n";
        return EXIT_FAILURE;
    }

    const double total_flops =
        2.0 * static_cast<double>(N) *
        static_cast<double>(N) *
        static_cast<double>(N);

    const double algorithmic_bytes =
        3.0 * static_cast<double>(N) *
        static_cast<double>(N) * 4.0;

    const double operational_intensity =
        total_flops / algorithmic_bytes;

    std::cout << "========================================\n";
    std::cout << "W03 GEMM GPU S3 TENSOR CORE TF32\n";
    std::cout << "========================================\n";
    std::cout << "N                     : " << N << '\n';
    std::cout << "Iterations            : " << iterations << '\n';
    std::cout << "Repetitions           : " << repetitions << '\n';
    std::cout << "Warmup                : " << warmup << '\n';
    std::cout << "Block tile            : "
              << BLOCK_M << " x " << BLOCK_N << '\n';
    std::cout << "WMMA shape            : "
              << WMMA_M << " x " << WMMA_N << " x " << WMMA_K << '\n';
    std::cout << "Threads per block     : " << BLOCK_THREADS << '\n';
    std::cout << "FLOPs                 : " << total_flops << '\n';
    std::cout << "Algorithmic bytes     : " << algorithmic_bytes << '\n';
    std::cout << "Operational Intensity : "
              << operational_intensity << " FLOP/Byte\n";

    const std::size_t elems = N * N;
    const std::size_t bytes = elems * sizeof(float);

    std::vector<float> hA(elems);
    std::vector<float> hB(elems);
    std::vector<float> hC(elems, 0.0f);

    initialize_matrices(hA, hB);

    float* dA = nullptr;
    float* dB = nullptr;
    float* dC = nullptr;

    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));

    CUDA_CHECK(cudaMemcpy(
        dA, hA.data(), bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        dB, hB.data(), bytes, cudaMemcpyHostToDevice));

    const dim3 block(BLOCK_THREADS);
    const dim3 grid(
        static_cast<unsigned>((N + BLOCK_N - 1) / BLOCK_N),
        static_cast<unsigned>((N + BLOCK_M - 1) / BLOCK_M));

    // Warmup
    for (int w = 0; w < warmup; ++w) {
        gemm_tensor_core<<<grid, block>>>(dA, dB, dC, N);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> times_ms;
    times_ms.reserve(repetitions);

    cudaEvent_t start_event;
    cudaEvent_t stop_event;

    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));

    for (int rep = 1; rep <= repetitions; ++rep) {

        float total_ms = 0.0f;

        for (int it = 0; it < iterations; ++it) {

            CUDA_CHECK(cudaEventRecord(start_event));

            gemm_tensor_core<<<grid, block>>>(dA, dB, dC, N);

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(stop_event));
            CUDA_CHECK(cudaEventSynchronize(stop_event));

            float elapsed_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(
                &elapsed_ms, start_event, stop_event));

            total_ms += elapsed_ms;
        }

        const double avg_ms =
            static_cast<double>(total_ms) /
            static_cast<double>(iterations);

        const double avg_s = avg_ms * 1e-3;

        const double gflops =
            (total_flops / avg_s) / 1e9;

        const double bandwidth_gbs =
            (algorithmic_bytes / avg_s) / 1e9;

        times_ms.push_back(avg_ms);

        std::cout << "\nRepetition "
                  << rep << "/" << repetitions << '\n';

        std::cout << "  Average time (ms) : "
                  << std::fixed << std::setprecision(4)
                  << avg_ms << '\n';

        std::cout << "  GFLOP/s           : "
                  << gflops << '\n';

        std::cout << "  Bandwidth (GB/s)  : "
                  << bandwidth_gbs << '\n';

        std::cout << "  Operational Intensity: "
                  << operational_intensity
                  << " FLOP/Byte\n";
    }

    // Copy output back
    CUDA_CHECK(cudaMemcpy(
        hC.data(), dC, bytes, cudaMemcpyDeviceToHost));

    auto mean = [](const std::vector<double>& v) {
        double s = 0.0;
        for (double x : v)
            s += x;
        return s / static_cast<double>(v.size());
    };

    auto stddev = [](const std::vector<double>& v, double avg) {
        if (v.size() < 2)
            return 0.0;

        double sq = 0.0;

        for (double x : v) {
            const double d = x - avg;
            sq += d * d;
        }

        return std::sqrt(
            sq / static_cast<double>(v.size() - 1));
    };

    const double mean_time = mean(times_ms);
    const double std_time = stddev(times_ms, mean_time);

    // Correctness
    constexpr std::size_t REF_N = 256;

    bool correct = true;
    double ref_checksum = 0.0;
    double out_checksum = 0.0;
    double rel_error = 0.0;

    if (REF_N <= N) {

        std::vector<double> refC(
            REF_N * REF_N, 0.0);

        reference_gemm_subset(
            hA, hB, refC, N, REF_N);

        for (std::size_t i = 0; i < REF_N; ++i) {
            for (std::size_t j = 0; j < REF_N; ++j) {

                const double ref =
                    refC[i * REF_N + j];

                const double got =
                    static_cast<double>(
                        hC[i * N + j]);

                const double err =
                    relative_error_percent(got, ref);

                if (err > rel_error)
                    rel_error = err;

                ref_checksum += ref;
                out_checksum += got;
            }
        }

        // TF32 has a 10-bit mantissa; a looser threshold is appropriate.
        correct = rel_error < 0.1;
    }

    std::cout << "\n========================================\n";
    std::cout << "SUMMARY\n";
    std::cout << "========================================\n";

    std::cout << "Mean time (ms)      : "
              << mean_time << " ± "
              << std_time << '\n';

    std::cout << "Mean GFLOP/s        : "
              << (total_flops /
                  (mean_time * 1e-3)) / 1e9
              << '\n';

    std::cout << "Reference checksum  : "
              << ref_checksum << '\n';

    std::cout << "Output checksum     : "
              << out_checksum << '\n';

    std::cout << "Max rel error (%)   : "
              << rel_error << '\n';

    std::cout << "Correct             : "
              << (correct ? "YES" : "NO") << '\n';

    std::ofstream file(output_path);

    if (!file) {
        std::cerr << "Could not open output file: "
                  << output_path << '\n';

        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);

        return EXIT_FAILURE;
    }

    file << std::fixed << std::setprecision(8);

    file << "experiment_id,workload,platform,stage,variant,"
         << "N,iterations,repetitions,warmup,"
         << "runtime_mean_ms,runtime_std_ms,gflops,"
         << "bandwidth_gbs,flops,algorithmic_bytes,"
         << "operational_intensity,ref_checksum,"
         << "out_checksum,max_rel_error_pct,correct,"
         << "compiler,compiler_flags\n";

    file << "W03_GEMM_GPU_S3_HWACC_TENSOR_CORE,"
         << "W03_GEMM,GPU,S3_HWACC,TENSOR_CORE_TF32_K32_LDS,"
         << N << ","
         << iterations << ","
         << repetitions << ","
         << warmup << ","
         << mean_time << ","
         << std_time << ","
         << (total_flops /
             (mean_time * 1e-3)) / 1e9 << ","
         << (algorithmic_bytes /
             (mean_time * 1e-3)) / 1e9 << ","
         << total_flops << ","
         << algorithmic_bytes << ","
         << operational_intensity << ","
         << ref_checksum << ","
         << out_checksum << ","
         << rel_error << ","
         << (correct ? "YES" : "NO") << ","
         << "\"NVCC\","
         << "\"-O3 -std=c++17 -g "
            "-gencode arch=compute_89,code=sm_89\"\n";

    file.close();

    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}