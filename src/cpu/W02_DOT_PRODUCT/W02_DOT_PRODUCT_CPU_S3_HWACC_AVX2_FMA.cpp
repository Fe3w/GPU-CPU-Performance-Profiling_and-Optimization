#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <immintrin.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    std::size_t N = 33554432;
    int iterations = 20;
    int repetitions = 5;
    int warmup = 2;
    int threads = 1;

    std::string output =
        "results/raw/W02_DOT_PRODUCT/"
        "W02_DOT_PRODUCT_CPU_S3_HWACC_AVX2_FMA_T1.csv";

    if (argc > 1) N = std::stoull(argv[1]);
    if (argc > 2) iterations = std::stoi(argv[2]);
    if (argc > 3) repetitions = std::stoi(argv[3]);
    if (argc > 4) warmup = std::stoi(argv[4]);
    if (argc > 5) threads = std::stoi(argv[5]);
    if (argc > 6) output = argv[6];

    omp_set_num_threads(threads);

    std::vector<float> A(N);
    std::vector<float> B(N);
    std::vector<float> partial(threads, 0.0f);

    for (std::size_t i = 0; i < N; ++i) {
        A[i] = static_cast<float>(i % 1000) * 0.001f;
        B[i] = static_cast<float>((i * 3) % 1000) * 0.001f;
    }

    // Independent double-precision reference.
    double reference = 0.0;

    for (std::size_t i = 0; i < N; ++i) {
        reference +=
            static_cast<double>(A[i]) *
            static_cast<double>(B[i]);
    }

    // N is 2^25 and therefore divisible by 8.
    const std::size_t vector_end = N - (N % 8);

    auto dot_product = [&]() -> float {
        std::fill(partial.begin(), partial.end(), 0.0f);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();

            __m256 vsum = _mm256_setzero_ps();

            #pragma omp for schedule(static)
            for (std::size_t i = 0; i < vector_end; i += 8) {
                __m256 va = _mm256_loadu_ps(&A[i]);
                __m256 vb = _mm256_loadu_ps(&B[i]);

                // Explicit hardware FMA:
                // vsum = va * vb + vsum
                vsum = _mm256_fmadd_ps(va, vb, vsum);
            }

            alignas(32) float tmp[8];
            _mm256_store_ps(tmp, vsum);

            float local = 0.0f;
            for (int k = 0; k < 8; ++k) {
                local += tmp[k];
            }

            partial[tid] = local;
        }

        float total = 0.0f;

        for (int t = 0; t < threads; ++t) {
            total += partial[t];
        }

        // Tail handling, although N is divisible by 8 here.
        for (std::size_t i = vector_end; i < N; ++i) {
            total += A[i] * B[i];
        }

        return total;
    };

    // Warm-up
    for (int w = 0; w < warmup; ++w) {
        volatile float warmup_result = dot_product();
        (void)warmup_result;
    }

    const double flops =
        2.0 * static_cast<double>(N);

    const double algorithmic_bytes =
        8.0 * static_cast<double>(N);

    const double operational_intensity =
        flops / algorithmic_bytes;

    std::ofstream file(output);

    if (!file) {
        std::cerr << "Could not open output file: " << output << '\n';
        return EXIT_FAILURE;
    }

    file << "experiment_id,workload,platform,stage,variant,"
         << "N,threads,iterations,repetition,"
         << "runtime_ms,gflops,bandwidth_gbs,flops,"
         << "algorithmic_bytes,operational_intensity,result,"
         << "reference_result,relative_error_pct,execution_valid,"
         << "compiler,compiler_flags\n";

    float final_result = 0.0f;

    for (int rep = 1; rep <= repetitions; ++rep) {

        auto start = std::chrono::steady_clock::now();

        for (int it = 0; it < iterations; ++it) {
            final_result = dot_product();
        }

        auto stop = std::chrono::steady_clock::now();

        const double total_ms =
            std::chrono::duration<double, std::milli>(
                stop - start).count();

        const double runtime_ms =
            total_ms / iterations;

        const double gflops =
            (flops / (runtime_ms * 1e-3)) / 1e9;

        const double bandwidth_gbs =
            (algorithmic_bytes /
             (runtime_ms * 1e-3)) / 1e9;

        const double relative_error_pct =
            (reference != 0.0)
            ? std::fabs(
                  (static_cast<double>(final_result) - reference)
                  / reference) * 100.0
            : 0.0;

        const bool execution_valid =
            std::isfinite(static_cast<double>(final_result));

        file << "W02_DOT_PRODUCT_CPU_S3_HWACC_AVX2_FMA_T"
             << threads
             << "_R"
             << std::setfill('0') << std::setw(2) << rep
             << std::setfill(' ')
             << ",W02_DOT_PRODUCT,CPU,S3_HWACC,OMP_STATIC_AVX2_FMA_F32,"
             << N << ","
             << threads << ","
             << iterations << ","
             << rep << ","
             << std::fixed << std::setprecision(6)
             << runtime_ms << ","
             << gflops << ","
             << bandwidth_gbs << ","
             << flops << ","
             << algorithmic_bytes << ","
             << operational_intensity << ","
             << final_result << ","
             << reference << ","
             << relative_error_pct << ","
             << (execution_valid ? "YES" : "NO") << ","
             << "g++,"
             << "\"-O3 -mavx2 -mfma -fopenmp "
                "-fno-tree-vectorize "
                "-fno-tree-slp-vectorize "
                "-std=c++17 -g\"\n";
    }

    file.close();

    std::cout << "\nW02 CPU S3 HWACC AVX2 FMA\n";
    std::cout << "N                   : " << N << '\n';
    std::cout << "Threads             : " << threads << '\n';
    std::cout << "Operational Intensity: "
              << operational_intensity << " FLOP/Byte\n";
    std::cout << "Final result        : " << final_result << '\n';

    return 0;
}