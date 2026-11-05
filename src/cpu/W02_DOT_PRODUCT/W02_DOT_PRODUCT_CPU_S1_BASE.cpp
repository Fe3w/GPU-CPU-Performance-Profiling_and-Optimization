#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>



int main(int argc, char** argv)
{
    std::size_t N = 33554432;   // 2^25
    int iterations = 20;
    int repetitions = 5;
    int warmup = 2;

    std::string output =
        "results/raw/W02_DOT_PRODUCT/"
        "W02_DOT_PRODUCT_CPU_S1_BASE.csv";

    if (argc > 1) N = std::stoull(argv[1]);
    if (argc > 2) iterations = std::stoi(argv[2]);
    if (argc > 3) repetitions = std::stoi(argv[3]);
    if (argc > 4) warmup = std::stoi(argv[4]);
    if (argc > 5) output = argv[5];

    std::vector<float> A(N);
    std::vector<float> B(N);

    for (std::size_t i = 0; i < N; ++i) {
        A[i] = static_cast<float>(i % 1000) * 0.001f;
        B[i] = static_cast<float>((i * 3) % 1000) * 0.001f;
    }

    volatile float sink = 0.0f;

    // Warm-up
    for (int w = 0; w < warmup; ++w) {
        float sum = 0.0f;

        for (std::size_t i = 0; i < N; ++i) {
            sum += A[i] * B[i];
        }

        sink = sum;
    }

    // FLOPs:
    // multiply + add = 2 FLOPs/element
    const double flops =
        2.0 * static_cast<double>(N);

    // Algorithmic traffic:
    // read A + read B = 8 bytes/element
    const double algorithmic_bytes =
        8.0 * static_cast<double>(N);

    const double operational_intensity =
        flops / algorithmic_bytes;

    std::ofstream file(output);

    if (!file) {
        std::cerr << "Could not open output file: "
                  << output << '\n';
        return EXIT_FAILURE;
    }

    file << "experiment_id,workload,platform,stage,variant,"
        << "N,threads,iterations,repetition,"
        << "runtime_ms,gflops,bandwidth_gbs,flops,"
        << "algorithmic_bytes,operational_intensity,result,"
        << "reference_result,relative_error_pct,execution_valid,"
        << "compiler,compiler_flags\n";

    for (int rep = 1; rep <= repetitions; ++rep) {

        float final_result = 0.0f;

        auto start =
            std::chrono::steady_clock::now();

        for (int it = 0; it < iterations; ++it) {

            float sum = 0.0f;

            for (std::size_t i = 0; i < N; ++i) {
                sum += A[i] * B[i];
            }

            final_result = sum;
            sink = sum;
        }

        auto stop =
            std::chrono::steady_clock::now();

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

        // Recompute using double precision only for validation.
        double reference = 0.0;

        for (std::size_t i = 0; i < N; ++i) {
            reference +=
                static_cast<double>(A[i]) *
                static_cast<double>(B[i]);
        }

        // Float accumulation and double reference will differ slightly.
        const bool correct =
            std::fabs(static_cast<double>(final_result) -
                      reference) <=
            std::max(1.0, std::fabs(reference)) * 1e-4;

        file << "W02_DOT_PRODUCT_CPU_S1_BASE_R"
             << std::setfill('0') << std::setw(2) << rep
             << std::setfill(' ')
             << ",W02_DOT_PRODUCT,CPU,S1_BASE,O0_SCALAR,"
             << N << ","
             << 1 << ","
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
             << (correct ? "YES" : "NO") << ","
             << "g++,"
             << "\"-O0 -std=c++17 -g\"\n";
    }

    file.close();

    std::cout << "\nW02 CPU S1 BASE\n";
    std::cout << "N                  : " << N << '\n';
    std::cout << "Threads            : 1\n";
    std::cout << "Runtime/iter       : measured per repetition\n";
    std::cout << "Operational Intensity: "
              << operational_intensity
              << " FLOP/Byte\n";
    std::cout << "Final result       : " << sink << '\n';

    return EXIT_SUCCESS;
}
