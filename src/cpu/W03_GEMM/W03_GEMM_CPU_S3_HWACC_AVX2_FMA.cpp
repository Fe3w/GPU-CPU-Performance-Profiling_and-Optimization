#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <immintrin.h>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

// -----------------------------------------------------------------------------
// W03 GEMM — CPU S3 AVX2/FMA (OpenMP + Cache Blocking)
//
// Cache-blocked triple-loop GEMM with OpenMP parallelization on the outer
// tile index, and an 8-wide AVX2 FMA kernel on the innermost j loop:
//
//   for ii in [0, N) step BLOCK_SIZE          (parallel)
//     for kk in [0, N) step BLOCK_SIZE
//       for jj in [0, N) step BLOCK_SIZE
//         for i in tile(ii)
//           for k in tile(kk)
//             a = A[i*N + k]
//             for j in tile(jj)  (vectorized, 8-wide FMA)
//               C[i*N + j] += a * B[k*N + j]
//
// Because the k-tiles accumulate into C, C must be zeroed once per full
// GEMM.  That zeroing is inside the timed region so it is accounted for
// honestly against S1, which overwrote C and needed no reset.
//
// The tile size is configurable at compile time:
//   -DGEMM_BLOCK_SIZE=<value>     (defaults to 64)
//
// Row-major layout, flat 1D storage:
//   A[i * N + k],  B[k * N + j],  C[i * N + j]
//
// Workload accounting (FP32):
//   FLOPs         = 2 * N^3
//   Bytes         = 3 * N^2 * 4   (read A, read B, write C)
//   OI            = 2N^3 / (12N^2) = N / 6  FLOP/Byte
// -----------------------------------------------------------------------------

#ifndef GEMM_BLOCK_SIZE
#define GEMM_BLOCK_SIZE 64
#endif

static constexpr std::size_t BLOCK_SIZE = GEMM_BLOCK_SIZE;

static void initialize_matrices(std::vector<float>& A,
                                std::vector<float>& B) {
    const std::size_t N = A.size();

    for (std::size_t idx = 0; idx < N; ++idx) {
        A[idx] = static_cast<float>(idx % 1000) * 0.001f;
        B[idx] = static_cast<float>((idx * 3) % 1000) * 0.001f;
    }
}

// -----------------------------------------------------------------------------
// Independent reference for the top-left REF_N x REF_N corner of C.
//
// IMPORTANT: the K dimension runs over the FULL N, not REF_N.  We are
// computing a subset of the *output* of the full N x N GEMM, not a
// smaller GEMM.
// -----------------------------------------------------------------------------
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

int main(int argc, char* argv[]) {

    // -------------------------------------------------------------------------
    // Defaults
    // -------------------------------------------------------------------------
    std::size_t N = 2048;
    int iterations = 20;
    int repetitions = 5;
    int warmup = 2;

    std::string output_path =
        "results/raw/W03_GEMM/"
        "W03_GEMM_CPU_S3_HWACC_AVX2_FMA.csv";

    // -------------------------------------------------------------------------
    // Command-line arguments (positional, matching W01/W02 style)
    // -------------------------------------------------------------------------
    if (argc > 1) N = std::stoull(argv[1]);
    if (argc > 2) iterations = std::stoi(argv[2]);
    if (argc > 3) repetitions = std::stoi(argv[3]);
    if (argc > 4) warmup = std::stoi(argv[4]);
    if (argc > 5) output_path = argv[5];

    if (N == 0 || iterations <= 0 || repetitions <= 0 || warmup < 0) {
        std::cerr << "Invalid benchmark parameters.\n";
        return EXIT_FAILURE;
    }

    // -------------------------------------------------------------------------
    // Accounting
    // -------------------------------------------------------------------------
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
    std::cout << "W03 GEMM CPU S3 AVX2/FMA\n";
    std::cout << "========================================\n";
    std::cout << "N                     : " << N << '\n';
    std::cout << "Iterations            : " << iterations << '\n';
    std::cout << "Repetitions           : " << repetitions << '\n';
    std::cout << "Warmup                : " << warmup << '\n';
    std::cout << "Block size            : " << BLOCK_SIZE << '\n';
    std::cout << "FLOPs                 : " << total_flops << '\n';
    std::cout << "Algorithmic bytes     : " << algorithmic_bytes << '\n';
    std::cout << "Operational Intensity : "
              << operational_intensity << " FLOP/Byte\n";

    // -------------------------------------------------------------------------
    // Allocate and initialize
    // -------------------------------------------------------------------------
    const std::size_t elems = N * N;

    std::vector<float> A(elems);
    std::vector<float> B(elems);
    std::vector<float> C(elems, 0.0f);

    initialize_matrices(A, B);

    // -------------------------------------------------------------------------
    // Warmup (cache-blocked, OpenMP, AVX2/FMA inner loop)
    // -------------------------------------------------------------------------
    for (int w = 0; w < warmup; ++w) {

        std::fill(C.begin(), C.end(), 0.0f);

        #pragma omp parallel for schedule(static)
        for (std::size_t ii = 0; ii < N; ii += BLOCK_SIZE) {

            const std::size_t i_end = std::min(ii + BLOCK_SIZE, N);

            for (std::size_t kk = 0; kk < N; kk += BLOCK_SIZE) {

                const std::size_t k_end = std::min(kk + BLOCK_SIZE, N);

                for (std::size_t jj = 0; jj < N; jj += BLOCK_SIZE) {

                    const std::size_t j_end = std::min(jj + BLOCK_SIZE, N);

                    for (std::size_t i = ii; i < i_end; ++i) {
                        for (std::size_t k = kk; k < k_end; ++k) {

                            const float a = A[i * N + k];
                            const __m256 a_vec = _mm256_set1_ps(a);

                            std::size_t j = jj;

                            for (; j + 8 <= j_end; j += 8) {
                                __m256 c_vec =
                                    _mm256_loadu_ps(&C[i * N + j]);

                                __m256 b_vec =
                                    _mm256_loadu_ps(&B[k * N + j]);

                                c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);

                                _mm256_storeu_ps(&C[i * N + j], c_vec);
                            }

                            // Handle remaining elements for non-multiple-of-8 sizes.
                            for (; j < j_end; ++j) {
                                C[i * N + j] += a * B[k * N + j];
                            }
                        }
                    }
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Benchmark repetitions
    // -------------------------------------------------------------------------
    std::vector<double> times_ms;
    times_ms.reserve(repetitions);

    for (int rep = 1; rep <= repetitions; ++rep) {

        double total_ms = 0.0;

        for (int it = 0; it < iterations; ++it) {

            auto start =
                std::chrono::steady_clock::now();

            // Zero C once per full GEMM because the kernel accumulates.
            std::fill(C.begin(), C.end(), 0.0f);

            #pragma omp parallel for schedule(static)
            for (std::size_t ii = 0; ii < N; ii += BLOCK_SIZE) {

                const std::size_t i_end = std::min(ii + BLOCK_SIZE, N);

                for (std::size_t kk = 0; kk < N; kk += BLOCK_SIZE) {

                    const std::size_t k_end = std::min(kk + BLOCK_SIZE, N);

                    for (std::size_t jj = 0; jj < N; jj += BLOCK_SIZE) {

                        const std::size_t j_end = std::min(jj + BLOCK_SIZE, N);

                        for (std::size_t i = ii; i < i_end; ++i) {
                            for (std::size_t k = kk; k < k_end; ++k) {

                                const float a = A[i * N + k];
                                const __m256 a_vec = _mm256_set1_ps(a);

                                std::size_t j = jj;

                                for (; j + 8 <= j_end; j += 8) {
                                    __m256 c_vec =
                                        _mm256_loadu_ps(&C[i * N + j]);

                                    __m256 b_vec =
                                        _mm256_loadu_ps(&B[k * N + j]);

                                    c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);

                                    _mm256_storeu_ps(&C[i * N + j], c_vec);
                                }

                                // Handle remaining elements for non-multiple-of-8 sizes.
                                for (; j < j_end; ++j) {
                                    C[i * N + j] += a * B[k * N + j];
                                }
                            }
                        }
                    }
                }
            }

            auto stop =
                std::chrono::steady_clock::now();

            const double elapsed_ms =
                std::chrono::duration<double, std::milli>(
                    stop - start).count();

            total_ms += elapsed_ms;
        }

        const double avg_ms =
            total_ms / static_cast<double>(iterations);

        const double avg_s = avg_ms * 1e-3;

        const double gflops =
            (total_flops / avg_s) / 1e9;

        const double bandwidth_gbs =
            (algorithmic_bytes / avg_s) / 1e9;

        times_ms.push_back(avg_ms);

        std::cout << "\nRepetition " << rep << "/" << repetitions
                  << "\n"
                  << "  Average time (ms) : "
                  << std::fixed << std::setprecision(4)
                  << avg_ms << '\n'
                  << "  GFLOP/s           : " << gflops << '\n'
                  << "  Bandwidth (GB/s)  : " << bandwidth_gbs << '\n'
                  << "  Operational Intensity: "
                  << operational_intensity << " FLOP/Byte\n";
    }

    // -------------------------------------------------------------------------
    // Statistics
    // -------------------------------------------------------------------------
    auto mean = [](const std::vector<double>& v) {
        double s = 0.0;
        for (double x : v) s += x;
        return s / static_cast<double>(v.size());
    };

    auto stddev = [](const std::vector<double>& v, double avg) {
        if (v.size() < 2) return 0.0;
        double sq = 0.0;
        for (double x : v) {
            const double d = x - avg;
            sq += d * d;
        }
        return std::sqrt(sq / static_cast<double>(v.size() - 1));
    };

    const double mean_time = mean(times_ms);
    const double std_time = stddev(times_ms, mean_time);

    // -------------------------------------------------------------------------
    // Correctness check
    //
    // Compute the top-left REF_N x REF_N corner of C independently, using the
    // full K = N dimension.  Compare against the same corner of the kernel's
    // output.  Set REF_N = 0 to skip.
    // -------------------------------------------------------------------------
    constexpr std::size_t REF_N = 256;

    bool correct = true;
    double ref_checksum = 0.0;
    double out_checksum = 0.0;
    double rel_error = 0.0;

    if (REF_N > 0 && REF_N <= N) {

        const std::size_t r_elems = REF_N * REF_N;

        std::vector<double> rC(r_elems, 0.0);

        reference_gemm_subset(A, B, rC, N, REF_N);

        for (std::size_t i = 0; i < REF_N; ++i) {
            for (std::size_t j = 0; j < REF_N; ++j) {
                const double ref = rC[i * REF_N + j];
                const double got =
                    static_cast<double>(C[i * N + j]);

                const double e = relative_error_percent(got, ref);

                if (e > rel_error)
                    rel_error = e;

                ref_checksum += ref;
                out_checksum += got;
            }
        }

        // Tolerate FP32 accumulation error; tighten if desired.
        correct = rel_error < 0.1;  // percent
    }

    // -------------------------------------------------------------------------
    // CSV
    // -------------------------------------------------------------------------
    std::ofstream file(output_path);

    if (!file) {
        std::cerr << "Could not open output file: "
                  << output_path << '\n';
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

    file << "W03_GEMM_CPU_S3_HWACC_AVX2_FMA,"
         << "W03_GEMM,CPU,S3_HWACC,AVX2_FMA_CACHE_BLOCKED,"
         << N << ","
         << iterations << ","
         << repetitions << ","
         << warmup << ","
         << mean_time << ","
         << std_time << ","
         << (total_flops / (mean_time * 1e-3)) / 1e9 << ","
         << (algorithmic_bytes / (mean_time * 1e-3)) / 1e9 << ","
         << total_flops << ","
         << algorithmic_bytes << ","
         << operational_intensity << ","
         << ref_checksum << ","
         << out_checksum << ","
         << rel_error << ","
         << (correct ? "YES" : "NO") << ","
         << "\"G++ " << __VERSION__ << "\","
         << "\"-O3 -fopenmp -mavx2 -mfma -fno-tree-vectorize -fno-tree-slp-vectorize -std=c++17 -g\"\n";

    file.close();

    std::cout << "\n========================================\n";
    std::cout << "SUMMARY\n";
    std::cout << "========================================\n";
    std::cout << "Mean time (ms)      : "
              << mean_time << " ± " << std_time << '\n';
    std::cout << "Mean GFLOP/s        : "
              << (total_flops / (mean_time * 1e-3)) / 1e9 << '\n';
    std::cout << "Reference checksum  : " << ref_checksum << '\n';
    std::cout << "Output checksum     : " << out_checksum << '\n';
    std::cout << "Max rel error (%)   : " << rel_error << '\n';
    std::cout << "Correct             : "
              << (correct ? "YES" : "NO") << '\n';
    std::cout << "\nCSV saved to:\n" << output_path << '\n';

    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}