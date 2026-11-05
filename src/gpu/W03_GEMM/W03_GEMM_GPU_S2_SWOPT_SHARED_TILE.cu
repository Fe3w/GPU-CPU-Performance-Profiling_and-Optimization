#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

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
// W03 GEMM — GPU S2 SHARED-MEMORY TILED
//
// Shared-memory tiled CUDA GEMM:
//   one CUDA thread computes one C[i,j]
//   16x16 tile loaded into shared memory per k-step
//   no register tiling
//   no Tensor Cores
//   no cuBLAS
//
// Row-major:
//   A[i*N + k]
//   B[k*N + j]
//   C[i*N + j]
// -----------------------------------------------------------------------------

#define TILE_SIZE 16

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
// Shared-memory tiled GEMM kernel
// -----------------------------------------------------------------------------
__global__ void gemm_shared_tiled(const float* A,
                                  const float* B,
                                  float* C,
                                  std::size_t N) {

    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const std::size_t row =
        static_cast<std::size_t>(blockIdx.y) * TILE_SIZE + ty;

    const std::size_t col =
        static_cast<std::size_t>(blockIdx.x) * TILE_SIZE + tx;

    float sum = 0.0f;

    for (std::size_t tile = 0;
         tile < N;
         tile += TILE_SIZE) {

        const std::size_t a_col = tile + tx;
        const std::size_t b_row = tile + ty;

        if (row < N && a_col < N)
            tileA[ty][tx] = A[row * N + a_col];
        else
            tileA[ty][tx] = 0.0f;

        if (b_row < N && col < N)
            tileB[ty][tx] = B[b_row * N + col];
        else
            tileB[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[ty][k] * tileB[k][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}

int main(int argc, char* argv[]) {

    std::size_t N = 1024;
    int iterations = 20;
    int repetitions = 5;
    int warmup = 2;

    std::string output_path =
        "results/raw/W03_GEMM/"
        "W03_GEMM_GPU_S2_SWOPT_SHARED_TILE.csv";

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
    std::cout << "W03 GEMM GPU S2 SHARED-MEMORY TILED\n";
    std::cout << "========================================\n";
    std::cout << "N                     : " << N << '\n';
    std::cout << "Iterations            : " << iterations << '\n';
    std::cout << "Repetitions           : " << repetitions << '\n';
    std::cout << "Warmup                : " << warmup << '\n';
    std::cout << "Block size            : " << TILE_SIZE
              << " x " << TILE_SIZE << '\n';
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

    const dim3 block(TILE_SIZE, TILE_SIZE);
    const dim3 grid(
        static_cast<unsigned>((N + block.x - 1) / block.x),
        static_cast<unsigned>((N + block.y - 1) / block.y));

    // Warmup
    for (int w = 0; w < warmup; ++w) {
        gemm_shared_tiled<<<grid, block>>>(dA, dB, dC, N);
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

            gemm_shared_tiled<<<grid, block>>>(dA, dB, dC, N);

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

    file << "W03_GEMM_GPU_S2_SWOPT_SHARED_TILE,"
         << "W03_GEMM,GPU,S2_SWOPT,SHARED_MEMORY_TILED,"
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