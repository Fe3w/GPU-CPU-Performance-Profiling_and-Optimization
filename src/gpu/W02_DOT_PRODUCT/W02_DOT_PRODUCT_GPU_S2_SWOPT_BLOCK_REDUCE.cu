#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)         \
                      << " at " << __FILE__ << ":" << __LINE__ << '\n';      \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

static constexpr int BLOCK_SIZE = 256;

// -----------------------------------------------------------------------------
// S2: block-level software reduction.
//
// Each thread computes multiple products using a grid-stride loop.
// Threads within a block reduce their partial sums in shared memory.
// Only one thread per block writes the block result.
//
// This removes the per-element global atomicAdd() used by S1.
// -----------------------------------------------------------------------------
__global__ void dot_product_block_reduce(const float* A,
                                         const float* B,
                                         float* partial,
                                         std::size_t N) {
    __shared__ float shared_sum[BLOCK_SIZE];

    const unsigned int tid = threadIdx.x;

    float local_sum = 0.0f;

    for (std::size_t i =
             static_cast<std::size_t>(blockIdx.x) * blockDim.x + tid;
         i < N;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        local_sum += A[i] * B[i];
    }

    shared_sum[tid] = local_sum;

    __syncthreads();

    // Tree reduction in shared memory.
    for (unsigned int stride = BLOCK_SIZE / 2;
         stride > 0;
         stride >>= 1) {

        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) {
        partial[blockIdx.x] = shared_sum[0];
    }
}

__global__ void reduce_partial_sums(const float* partial,
                                    float* result,
                                    std::size_t count) {
    __shared__ float shared_sum[BLOCK_SIZE];

    const unsigned int tid = threadIdx.x;

    float local_sum = 0.0f;

    for (std::size_t i = tid; i < count; i += BLOCK_SIZE) {
        local_sum += partial[i];
    }

    shared_sum[tid] = local_sum;

    __syncthreads();

    for (unsigned int stride = BLOCK_SIZE / 2;
         stride > 0;
         stride >>= 1) {

        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) {
        *result = shared_sum[0];
    }
}

// Same workload as W02 CPU.
static void initialize_inputs(std::vector<float>& A,
                              std::vector<float>& B) {
    const std::size_t N = A.size();

    for (std::size_t i = 0; i < N; ++i) {
        A[i] = static_cast<float>(i % 1000) * 0.001f;
        B[i] = static_cast<float>((i * 3) % 1000) * 0.001f;
    }
}

static double compute_reference(const std::vector<float>& A,
                                const std::vector<float>& B) {
    double sum = 0.0;

    for (std::size_t i = 0; i < A.size(); ++i) {
        sum += static_cast<double>(A[i]) *
               static_cast<double>(B[i]);
    }

    return sum;
}

static float compute_fp32_reference(const std::vector<float>& A,
                                    const std::vector<float>& B) {
    float sum = 0.0f;

    for (std::size_t i = 0; i < A.size(); ++i) {
        sum += A[i] * B[i];
    }

    return sum;
}

static double relative_error_percent(float result,
                                     double reference) {
    const double denominator =
        std::max(std::abs(reference),
                 std::numeric_limits<double>::min());

    return std::abs(static_cast<double>(result) - reference) /
           denominator * 100.0;
}

static double relative_error_fp32_percent(float result,
                                          float reference) {
    const double denominator =
        std::max(std::abs(static_cast<double>(reference)),
                 std::numeric_limits<double>::min());

    return std::abs(static_cast<double>(result) -
                    static_cast<double>(reference)) /
           denominator * 100.0;
}

int main(int argc, char* argv[]) {
    std::size_t N = 33554432;
    int iterations = 20;
    int repetitions = 5;
    int warmup = 2;

    std::string output_path =
        "results/raw/W02_DOT_PRODUCT/"
        "W02_DOT_PRODUCT_GPU_S2_SWOPT_BLOCK_REDUCE.csv";

    if (argc > 1) N = std::stoull(argv[1]);
    if (argc > 2) iterations = std::stoi(argv[2]);
    if (argc > 3) repetitions = std::stoi(argv[3]);
    if (argc > 4) warmup = std::stoi(argv[4]);
    if (argc > 5) output_path = argv[5];

    if (N == 0 || iterations <= 0 || repetitions <= 0 || warmup < 0) {
        std::cerr << "Invalid benchmark parameters.\n";
        return EXIT_FAILURE;
    }

    constexpr double FLOPS_PER_ELEMENT = 2.0;
    constexpr double BYTES_PER_ELEMENT = 8.0;
    constexpr double OPERATIONAL_INTENSITY =
        FLOPS_PER_ELEMENT / BYTES_PER_ELEMENT;

    const double total_flops =
        FLOPS_PER_ELEMENT * static_cast<double>(N);

    const double total_bytes =
        BYTES_PER_ELEMENT * static_cast<double>(N);

    std::cout << "========================================\n";
    std::cout << "W02 DOT PRODUCT GPU S2 BLOCK REDUCTION\n";
    std::cout << "========================================\n";
    std::cout << "N                    : " << N << '\n';
    std::cout << "Iterations           : " << iterations << '\n';
    std::cout << "Repetitions          : " << repetitions << '\n';
    std::cout << "Warmup               : " << warmup << '\n';
    std::cout << "Block size           : " << BLOCK_SIZE << '\n';
    std::cout << "Operational Intensity: "
              << OPERATIONAL_INTENSITY
              << " FLOP/Byte\n";

    std::vector<float> h_A(N);
    std::vector<float> h_B(N);

    initialize_inputs(h_A, h_B);

    const double reference_result =
        compute_reference(h_A, h_B);

    const float fp32_reference_result =
        compute_fp32_reference(h_A, h_B);

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_partial = nullptr;
    float* d_result = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(
        &d_result,
        sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
        d_A,
        h_A.data(),
        N * sizeof(float),
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_B,
        h_B.data(),
        N * sizeof(float),
        cudaMemcpyHostToDevice));

    // One block per 256 input elements.
    const int num_blocks =
        static_cast<int>(
            (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    CUDA_CHECK(cudaMalloc(
        &d_partial,
        static_cast<std::size_t>(num_blocks) *
        sizeof(float)));

    std::cout << "Grid size             : "
              << num_blocks << '\n';

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // -------------------------------------------------------------------------
    // Warmup
    // -------------------------------------------------------------------------
    for (int w = 0; w < warmup; ++w) {

        dot_product_block_reduce<<<
            num_blocks,
            BLOCK_SIZE>>>(
                d_A,
                d_B,
                d_partial,
                N);

        CUDA_CHECK(cudaGetLastError());

        reduce_partial_sums<<<1, BLOCK_SIZE>>>(
            d_partial,
            d_result,
            num_blocks);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::vector<double> times_ms;
    std::vector<double> gflops;
    std::vector<double> bandwidth_gbs;

    for (int rep = 0; rep < repetitions; ++rep) {

        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(start));

        for (int it = 0; it < iterations; ++it) {

            dot_product_block_reduce<<<
                num_blocks,
                BLOCK_SIZE>>>(
                    d_A,
                    d_B,
                    d_partial,
                    N);

            CUDA_CHECK(cudaGetLastError());

            reduce_partial_sums<<<1, BLOCK_SIZE>>>(
                d_partial,
                d_result,
                num_blocks);

            CUDA_CHECK(cudaGetLastError());
        }

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float total_time_ms = 0.0f;

        CUDA_CHECK(cudaEventElapsedTime(
            &total_time_ms,
            start,
            stop));

        const double average_time_ms =
            static_cast<double>(total_time_ms) /
            static_cast<double>(iterations);

        const double average_time_s =
            average_time_ms * 1.0e-3;

        const double achieved_gflops =
            (total_flops / average_time_s) / 1.0e9;

        const double achieved_bandwidth =
            (total_bytes / average_time_s) / 1.0e9;

        times_ms.push_back(average_time_ms);
        gflops.push_back(achieved_gflops);
        bandwidth_gbs.push_back(achieved_bandwidth);

        std::cout << "\nRepetition "
                  << (rep + 1) << '/'
                  << repetitions << '\n';

        std::cout << "  Average time (ms) : "
                  << std::fixed
                  << std::setprecision(4)
                  << average_time_ms << '\n';

        std::cout << "  GFLOP/s            : "
                  << achieved_gflops << '\n';

        std::cout << "  Bandwidth (GB/s)   : "
                  << achieved_bandwidth << '\n';

        std::cout << "  Operational Intensity: "
                  << OPERATIONAL_INTENSITY
                  << " FLOP/Byte\n";
    }

    float final_result = 0.0f;

    CUDA_CHECK(cudaMemcpy(
        &final_result,
        d_result,
        sizeof(float),
        cudaMemcpyDeviceToHost));

    auto mean = [](const std::vector<double>& values) {
        double total = 0.0;

        for (double v : values) {
            total += v;
        }

        return total / values.size();
    };

    auto stddev = [](const std::vector<double>& values,
                     double average) {
        if (values.size() < 2) {
            return 0.0;
        }

        double sum_sq = 0.0;

        for (double v : values) {
            const double diff = v - average;
            sum_sq += diff * diff;
        }

        return std::sqrt(
            sum_sq /
            static_cast<double>(values.size() - 1));
    };

    const double mean_time = mean(times_ms);
    const double std_time =
        stddev(times_ms, mean_time);

    const double mean_gflops = mean(gflops);
    const double std_gflops =
        stddev(gflops, mean_gflops);

    const double mean_bandwidth =
        mean(bandwidth_gbs);

    const double std_bandwidth =
        stddev(bandwidth_gbs, mean_bandwidth);

    const double error_vs_fp32_pct =
        relative_error_fp32_percent(
            final_result,
            fp32_reference_result);

    const double error_vs_double_pct =
        relative_error_percent(
            final_result,
            reference_result);

    // Validation is against the double-precision mathematical reference
    // (same check for every W02 stage: relative error <= 0.1 %).
    const bool correct =
        std::isfinite(static_cast<double>(final_result)) && error_vs_double_pct <= 0.1;

    std::ofstream file(output_path);

    if (!file) {
        std::cerr << "Could not open output file: "
                  << output_path << '\n';
        return EXIT_FAILURE;
    }

    file << std::fixed << std::setprecision(8);

    file << "experiment_id,workload,platform,stage,variant,"
         << "N,threads,iterations,repetitions,warmup,"
         << "block_size,runtime_ms,runtime_sd_ms,gflops,bandwidth_gbs,"
         << "operational_intensity,result,fp32_reference,"
         << "double_reference,error_vs_fp32_pct,"
         << "error_vs_double_pct,execution_valid,"
         << "compiler,compiler_flags\n";

    file << "W02_DOT_PRODUCT_GPU_S2_SWOPT_BLOCK_REDUCE,"
         << "W02_DOT_PRODUCT,GPU,S2_SWOPT,"
         << "BLOCK_REDUCE,"
         << N << ","
         << 1 << ","
         << iterations << ","
         << repetitions << ","
         << warmup << ","
         << BLOCK_SIZE << ","
         << mean_time << ","
         << std_time << ","
         << mean_gflops << ","
         << mean_bandwidth << ","
         << OPERATIONAL_INTENSITY << ","
         << final_result << ","
         << fp32_reference_result << ","
         << reference_result << ","
         << error_vs_fp32_pct << ","
         << error_vs_double_pct << ","
         << (correct ? "YES" : "NO") << ","
         << "nvcc,"
         << "\"-O3 -std=c++17 -g -gencode arch=compute_89,code=sm_89\"\n";

    file.close();

    std::cout << "\n========================================\n";
    std::cout << "SUMMARY\n";
    std::cout << "========================================\n";

    std::cout << "Mean time (ms)      : "
              << mean_time
              << " ± " << std_time << '\n';

    std::cout << "Mean GFLOP/s        : "
              << mean_gflops
              << " ± " << std_gflops << '\n';

    std::cout << "Mean bandwidth GB/s : "
              << mean_bandwidth
              << " ± " << std_bandwidth << '\n';

    std::cout << "Result              : "
              << final_result << '\n';

    std::cout << "FP32 reference      : "
              << fp32_reference_result << '\n';

    std::cout << "Double reference    : "
              << reference_result << '\n';

    std::cout << "Error vs FP32 (%)   : "
              << error_vs_fp32_pct << '\n';

    std::cout << "Error vs double (%) : "
              << error_vs_double_pct << '\n';

    std::cout << "Correct             : "
              << (correct ? "YES" : "NO") << '\n';

    std::cout << "\nCSV saved to:\n"
              << output_path << '\n';

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_partial));
    CUDA_CHECK(cudaFree(d_result));

    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}