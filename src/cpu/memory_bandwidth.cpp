#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <omp.h>

int main(int argc, char** argv) {
    // 256 MiB per array by default.
    // 3 arrays -> ~768 MiB total.
    const std::size_t N =
        (argc > 1) ? std::stoull(argv[1]) : (1ULL << 26);

    const int iterations =
        (argc > 2) ? std::atoi(argv[2]) : 5;

    const float scalar = 3.0f;

    std::vector<float> A(N);
    std::vector<float> B(N);
    std::vector<float> C(N);

    // Initialization is intentionally outside the timed region.
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < N; ++i) {
        A[i] = 1.0f;
        B[i] = 2.0f;
        C[i] = 0.0f;
    }

    // Warm-up.
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < N; ++i) {
        C[i] = A[i] + scalar * B[i];
    }

    // Benchmark.
    double total_ms = 0.0;

    for (int it = 0; it < iterations; ++it) {
        auto start = std::chrono::steady_clock::now();

        #pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < N; ++i) {
            C[i] = A[i] + scalar * B[i];
        }

        auto end = std::chrono::steady_clock::now();

        total_ms +=
            std::chrono::duration<double, std::milli>(
                end - start).count();
    }

    const double avg_ms = total_ms / iterations;

    // Triad:
    // A read  = 4 bytes
    // B read  = 4 bytes
    // C write = 4 bytes
    //
    // The conventional STREAM accounting uses 24 bytes/element
    // for write-allocate behavior on typical cache-based CPUs.
    const double bytes_per_element = 24.0;
    const double total_bytes =
        static_cast<double>(N) * bytes_per_element;

    const double bandwidth_gbs =
        (total_bytes / (avg_ms / 1000.0)) / 1e9;

    const double flops =
        static_cast<double>(N) * 2.0;

    const double gflops =
        (flops / (avg_ms / 1000.0)) / 1e9;

    // Prevent the optimizer from treating the output as unused.
    double checksum = 0.0;

    #pragma omp parallel for reduction(+:checksum)
    for (std::size_t i = 0; i < N; ++i) {
        checksum += C[i];
    }

    const bool correct =
        std::abs(checksum -
                 static_cast<double>(N) * 7.0) < 1e-2;

    std::cout << std::fixed << std::setprecision(4);

    std::cout << "N                  : " << N << '\n';
    std::cout << "Data per array (MiB): "
              << (static_cast<double>(N) * sizeof(float)) /
                     (1024.0 * 1024.0)
              << '\n';
    std::cout << "OpenMP threads      : "
              << omp_get_max_threads() << '\n';
    std::cout << "Iterations          : " << iterations << '\n';
    std::cout << "Average time (ms)   : " << avg_ms << '\n';
    std::cout << "GFLOP/s             : " << gflops << '\n';
    std::cout << "Bandwidth (GB/s)     : " << bandwidth_gbs << '\n';
    std::cout << "Operational Intensity: "
              << flops / total_bytes << " FLOP/Byte\n";
    std::cout << "Checksum            : " << checksum << '\n';
    std::cout << "Correct             : "
              << (correct ? "YES" : "NO") << '\n';

    return correct ? 0 : 1;
}
