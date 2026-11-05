#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
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
        "W02_DOT_PRODUCT_CPU_S2_SWOPT_OMP_T1.csv";

    if (argc > 1) N = std::stoull(argv[1]);
    if (argc > 2) iterations = std::stoi(argv[2]);
    if (argc > 3) repetitions = std::stoi(argv[3]);
    if (argc > 4) warmup = std::stoi(argv[4]);
    if (argc > 5) threads = std::stoi(argv[5]);
    if (argc > 6) output = argv[6];

    if (threads < 1) {
        std::cerr << "Invalid thread count\n";
        return EXIT_FAILURE;
    }

    omp_set_num_threads(threads);

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

        #pragma omp parallel for schedule(static) reduction(+:sum)
        for (std::size_t i = 0; i < N; ++i) {
            sum += A[i] * B[i];
        }

        sink = sum;
    }

    // Dot product:
    // multiply + add = 2 FLOPs/element
    const double flops =
        2.0 * static_cast<double>(N);

    // Read A + read B = 8 bytes/element
    const double algorithmic_bytes =
        8.0 * static_cast<double>(N);

    const double operational_intensity =
        flops / algorithmic_bytes;

    // Reference result for correctness
    double reference = 0.0;

    for (std::size_t i = 0; i < N; ++i) {
        reference +=
            static_cast<double>(A[i]) *
            static_cast<double>(B[i]);
    }

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
         << "correct,compiler,compiler_flags\n";

    for (int rep = 1; rep <= repetitions; ++rep) {

        float final_result = 0.0f;

        auto start = std::chrono::steady_clock::now();

        for (int it = 0; it < iterations; ++it) {

            float sum = 0.0f;

            #pragma omp parallel for schedule(static) reduction(+:sum)
            for (std::size_t i = 0; i < N; ++i) {
                sum += A[i] * B[i];
            }

            final_result = sum;
            sink = sum;
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

        // Parallel floating-point reduction can have a slightly
        // different rounding order from the serial reference.
        const bool correct =
            std::fabs(static_cast<double>(final_result) - reference)
            <= std::max(1.0, std::fabs(reference)) * 1e-4;

        file << "W02_DOT_PRODUCT_CPU_S2_SWOPT_OMP_T"
             << threads
             << "_R"
             << std::setfill('0') << std::setw(2) << rep
             << std::setfill(' ')
             << ",W02_DOT_PRODUCT,CPU,S2_SWOPT,OMP_STATIC_SCALAR,"
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
             << (correct ? "YES" : "NO") << ","
             << "g++,"
             << "\"-O3 -fopenmp "
                "-fno-tree-vectorize "
                "-fno-tree-slp-vectorize "
                "-std=c++17 -g\"\n";
    }

    file.close();

    std::cout << "\nW02 CPU S2 SWOPT OMP\n";
    std::cout << "N                   : " << N << '\n';
    std::cout << "Threads             : " << threads << '\n';
    std::cout << "Iterations          : " << iterations << '\n';
    std::cout << "Repetitions         : " << repetitions << '\n';
    std::cout << "Operational Intensity: "
              << operational_intensity
              << " FLOP/Byte\n";
    std::cout << "Correct             : YES/NO recorded in CSV\n";
    std::cout << "Final result        : " << sink << '\n';

    return EXIT_SUCCESS;
}
