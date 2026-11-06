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
            std::exit(EXIT_FAILURE);                                        \
        }                                                                    \
    } while (0)

static constexpr int BLOCK_SIZE = 256;

// -----------------------------------------------------------------------------
// S1 BASELINE: intentionally simple GPU dot product.
// Each CUDA thread computes one element product and performs a global atomic
// accumulation into a single result. The accumulator is double (FP32 inputs):
// with an FP32 accumulator the result was 6.8 % off the double sum at
// N = 33.5 M and passed only against an FP32 serial reference with the same
// error. Correct = |s - s_double| / s_double <= 0.1 % (same check for every W02 stage).
// -----------------------------------------------------------------------------
// D = sum(A[i] * B[i])
// FLOPs = 2 per element
// Algorithmic bytes = 8 per element
// Operational Intensity = 0.25 FLOP/Byte
// -----------------------------------------------------------------------------
__global__ void dot_product_naive(const float* A,
                                  const float* B,
                                  double* result,
                                  std::size_t N) {
    std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < N) {
        double product = static_cast<double>(A[i]) * static_cast<double>(B[i]);
        atomicAdd(result, product);
    }
}

// -----------------------------------------------------------------------------
// Deterministic input generation.
//
// IMPORTANT:
// For the final CPU-vs-GPU comparison, these initialization expressions must
// match the W02 CPU benchmark's initialization exactly.
// -----------------------------------------------------------------------------
static void initialize_inputs(std::vector<float>& A,
                              std::vector<float>& B) {
    const std::size_t N = A.size();

    for (std::size_t i = 0; i < N; ++i) {
        A[i] = static_cast<float>(i % 1000) * 0.001f;
        B[i] = static_cast<float>((i * 3) % 1000) * 0.001f;
    }
}

// -----------------------------------------------------------------------------
// Double-precision CPU reference (accumulation-error baseline).
// -----------------------------------------------------------------------------
static double compute_reference(const std::vector<float>& A,
                                const std::vector<float>& B) {
    double sum = 0.0;

    for (std::size_t i = 0; i < A.size(); ++i) {
        sum += static_cast<double>(A[i]) *
               static_cast<double>(B[i]);
    }

    return sum;
}

// -----------------------------------------------------------------------------
// Serial FP32 CPU reference (the correctness target).
// Matches the numerical class the GPU is actually producing.
// -----------------------------------------------------------------------------
static float compute_fp32_reference(const std::vector<float>& A,
                                    const std::vector<float>& B) {
    float sum = 0.0f;

    for (std::size_t i = 0; i < A.size(); ++i) {
        sum += A[i] * B[i];
    }

    return sum;
}

// -----------------------------------------------------------------------------
// Relative error, percent, against the double reference.
// -----------------------------------------------------------------------------
static double relative_error_percent(double result, double reference) {
    const double denominator = std::max(std::abs(reference),
                                        std::numeric_limits<double>::min());

    return std::abs(static_cast<double>(result) - reference) /
           denominator * 100.0;
}

// -----------------------------------------------------------------------------
// Relative error, percent, against the serial FP32 reference.
// -----------------------------------------------------------------------------
static double relative_error_fp32_percent(double result,
                                          float reference) {
    const double denominator =
        std::max(std::abs(static_cast<double>(reference)),
                 std::numeric_limits<double>::min());

    return std::abs(static_cast<double>(result) -
                    static_cast<double>(reference)) /
           denominator * 100.0;
}

// -----------------------------------------------------------------------------
// CSV writer
// -----------------------------------------------------------------------------
static void write_csv(const std::string& output_path,
                      std::size_t N,
                      int iterations,
                      int repetitions,
                      int warmup,
                      double operational_intensity,
                      double result,
                      double reference_result,
                      double relative_error,
                      double mean_ms,
                      double stddev_ms,
                      double mean_gflops,
                      double stddev_gflops,
                      double mean_bandwidth_gbs,
                      double stddev_bandwidth_gbs,
                      bool correct) {
    std::ofstream file(output_path);

    if (!file) {
        std::cerr << "Failed to open output file: "
                  << output_path << '\n';
        std::exit(EXIT_FAILURE);
    }

    file << std::fixed << std::setprecision(8);

    file << "N,iterations,repetitions,warmup,"
         << "block_size,operational_intensity,"
         << "result,reference_result,relative_error_pct,"
         << "mean_time_ms,stddev_time_ms,"
         << "mean_gflops,stddev_gflops,"
         << "mean_bandwidth_gbs,stddev_bandwidth_gbs,"
         << "correct\n";

    file << N << ','
         << iterations << ','
         << repetitions << ','
         << warmup << ','
         << BLOCK_SIZE << ','
         << operational_intensity << ','
         << result << ','
         << reference_result << ','
         << relative_error << ','
         << mean_ms << ','
         << stddev_ms << ','
         << mean_gflops << ','
         << stddev_gflops << ','
         << mean_bandwidth_gbs << ','
         << stddev_bandwidth_gbs << ','
         << (correct ? "YES" : "NO")
         << '\n';
}

int main(int argc, char* argv[]) {
    // -------------------------------------------------------------------------
    // Defaults
    // -------------------------------------------------------------------------
    std::size_t N = 33554432;
    int iterations = 20;
    int repetitions = 5;
    int warmup = 2;
    std::string output_path =
        "results/raw/W02_DOT_PRODUCT/"
        "W02_DOT_PRODUCT_GPU_S1_BASE_NAIVE.csv";

    // -------------------------------------------------------------------------
    // Command-line arguments
    // -------------------------------------------------------------------------
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        auto require_value = [&](int index) {
            if (index + 1 >= argc) {
                std::cerr << "Missing value for argument: "
                          << argv[index] << '\n';
                std::exit(EXIT_FAILURE);
            }
        };

        if (arg == "--size") {
            require_value(i);
            N = static_cast<std::size_t>(std::stoull(argv[++i]));
        } else if (arg == "--iterations") {
            require_value(i);
            iterations = std::stoi(argv[++i]);
        } else if (arg == "--repetitions") {
            require_value(i);
            repetitions = std::stoi(argv[++i]);
        } else if (arg == "--warmup") {
            require_value(i);
            warmup = std::stoi(argv[++i]);
        } else if (arg == "--output") {
            require_value(i);
            output_path = argv[++i];
        } else if (arg == "--help") {
            std::cout
                << "Usage:\n"
                << "  " << argv[0]
                << " [--size N]"
                << " [--iterations I]"
                << " [--repetitions R]"
                << " [--warmup W]"
                << " [--output FILE]\n";
            return EXIT_SUCCESS;
        } else {
            std::cerr << "Unknown argument: " << arg << '\n';
            return EXIT_FAILURE;
        }
    }

    if (N == 0 || iterations <= 0 || repetitions <= 0 || warmup < 0) {
        std::cerr << "Invalid benchmark parameters.\n";
        return EXIT_FAILURE;
    }

    // -------------------------------------------------------------------------
    // Benchmark accounting
    // -------------------------------------------------------------------------
    constexpr double FLOPS_PER_ELEMENT = 2.0;
    constexpr double BYTES_PER_ELEMENT = 8.0;
    constexpr double OPERATIONAL_INTENSITY =
        FLOPS_PER_ELEMENT / BYTES_PER_ELEMENT;

    const double total_flops =
        FLOPS_PER_ELEMENT * static_cast<double>(N);

    const double total_bytes =
        BYTES_PER_ELEMENT * static_cast<double>(N);

    std::cout << "========================================\n";
    std::cout << "W02 DOT PRODUCT GPU S1 BASELINE\n";
    std::cout << "========================================\n";
    std::cout << "N                    : " << N << '\n';
    std::cout << "Iterations            : " << iterations << '\n';
    std::cout << "Repetitions           : " << repetitions << '\n';
    std::cout << "Warmup                : " << warmup << '\n';
    std::cout << "Block size            : " << BLOCK_SIZE << '\n';
    std::cout << "FLOPs / element       : "
              << FLOPS_PER_ELEMENT << '\n';
    std::cout << "Bytes / element       : "
              << BYTES_PER_ELEMENT << '\n';
    std::cout << "Operational Intensity : "
              << OPERATIONAL_INTENSITY
              << " FLOP/Byte\n";

    // -------------------------------------------------------------------------
    // Host allocation and initialization
    // -------------------------------------------------------------------------
    std::vector<float> h_A(N);
    std::vector<float> h_B(N);

    initialize_inputs(h_A, h_B);

    const double reference_result =
        compute_reference(h_A, h_B);

    const float fp32_reference_result =
        compute_fp32_reference(h_A, h_B);

    // -------------------------------------------------------------------------
    // Device allocation
    // -------------------------------------------------------------------------
    float* d_A = nullptr;
    float* d_B = nullptr;
    double* d_result = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(double)));

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

    const int grid_size =
        static_cast<int>((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    std::cout << "Grid size             : "
              << grid_size << '\n';

    // -------------------------------------------------------------------------
    // CUDA events for kernel timing
    // -------------------------------------------------------------------------
    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // -------------------------------------------------------------------------
    // Warmup
    // -------------------------------------------------------------------------
    for (int w = 0; w < warmup; ++w) {
        CUDA_CHECK(cudaMemset(d_result, 0, sizeof(double)));

        dot_product_naive<<<grid_size, BLOCK_SIZE>>>(
            d_A, d_B, d_result, N);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // -------------------------------------------------------------------------
    // Repetitions
    // -------------------------------------------------------------------------
    std::vector<double> times_ms;
    std::vector<double> gflops;
    std::vector<double> bandwidth_gbs;

    times_ms.reserve(repetitions);
    gflops.reserve(repetitions);
    bandwidth_gbs.reserve(repetitions);

    double final_result = 0.0;

    for (int rep = 0; rep < repetitions; ++rep) {
        CUDA_CHECK(cudaMemset(d_result, 0, sizeof(double)));

        CUDA_CHECK(cudaEventRecord(start));

        for (int iter = 0; iter < iterations; ++iter) {
            // Reset the accumulation before every timed iteration.
            CUDA_CHECK(cudaMemsetAsync(
                d_result,
                0,
                sizeof(double)));

            dot_product_naive<<<grid_size, BLOCK_SIZE>>>(
                d_A, d_B, d_result, N);

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
            (total_flops / average_time_s) /
            1.0e9;

        const double achieved_bandwidth_gbs =
            (total_bytes / average_time_s) /
            1.0e9;

        // Retrieve the result from the final iteration.
        CUDA_CHECK(cudaMemcpy(
            &final_result,
            d_result,
            sizeof(double),
            cudaMemcpyDeviceToHost));

        times_ms.push_back(average_time_ms);
        gflops.push_back(achieved_gflops);
        bandwidth_gbs.push_back(achieved_bandwidth_gbs);

        std::cout << "\nRepetition "
                  << (rep + 1) << '/' << repetitions << '\n';

        std::cout << "  Average time (ms) : "
                  << std::fixed << std::setprecision(4)
                  << average_time_ms << '\n';

        std::cout << "  GFLOP/s            : "
                  << achieved_gflops << '\n';

        std::cout << "  Bandwidth (GB/s)   : "
                  << achieved_bandwidth_gbs << '\n';

        std::cout << "  Operational Intensity: "
                  << OPERATIONAL_INTENSITY
                  << " FLOP/Byte\n";
    }

    // -------------------------------------------------------------------------
    // Statistics
    // -------------------------------------------------------------------------
    auto mean = [](const std::vector<double>& values) {
        double sum = 0.0;

        for (double v : values) {
            sum += v;
        }

        return sum / static_cast<double>(values.size());
    };

    auto stddev = [&](const std::vector<double>& values,
                      double avg) {
        if (values.size() < 2) {
            return 0.0;
        }

        double sum_sq = 0.0;

        for (double v : values) {
            const double diff = v - avg;
            sum_sq += diff * diff;
        }

        return std::sqrt(
            sum_sq / static_cast<double>(values.size() - 1));
    };

    const double mean_time = mean(times_ms);
    const double std_time = stddev(times_ms, mean_time);

    const double mean_gflops = mean(gflops);
    const double std_gflops = stddev(gflops, mean_gflops);

    const double mean_bandwidth = mean(bandwidth_gbs);
    const double std_bandwidth =
        stddev(bandwidth_gbs, mean_bandwidth);

    const double relative_error =
        relative_error_percent(
            final_result,
            reference_result);

    const double relative_error_vs_fp32 =
        relative_error_fp32_percent(
            final_result,
            fp32_reference_result);

    // Validation is against the double-precision sum (same check for every
    // W02 stage); the serial FP32 sum is printed for comparison only.
    const bool correct =
        std::isfinite(final_result) && relative_error <= 0.1;

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
              << std::setprecision(10)
              << final_result << '\n';

    std::cout << "FP32 reference      : "
              << fp32_reference_result << '\n';

    std::cout << "Double reference    : "
              << reference_result << '\n';

    std::cout << "Error vs FP32 (%)   : "
              << relative_error_vs_fp32 << '\n';

    std::cout << "Error vs double (%) : "
              << relative_error << '\n';

    std::cout << "Correct             : "
              << (correct ? "YES" : "NO") << '\n';

    // -------------------------------------------------------------------------
    // CSV output
    // -------------------------------------------------------------------------
    write_csv(
        output_path,
        N,
        iterations,
        repetitions,
        warmup,
        OPERATIONAL_INTENSITY,
        final_result,
        reference_result,
        relative_error,
        mean_time,
        std_time,
        mean_gflops,
        std_gflops,
        mean_bandwidth,
        std_bandwidth,
        correct);

    std::cout << "\nCSV saved to:\n"
              << output_path << '\n';

    // -------------------------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------------------------
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_result));

    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}