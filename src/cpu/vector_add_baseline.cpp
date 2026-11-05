#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    // Number of elements.
    // Default: 16 million.
    const std::size_t N =
        (argc > 1) ? std::stoull(argv[1]) : (1ULL << 24);

    const int iterations =
        (argc > 2) ? std::atoi(argv[2]) : 10;

    std::vector<float> A(N);
    std::vector<float> B(N);
    std::vector<float> C(N);

    // Initialization
    for (std::size_t i = 0; i < N; ++i) {
        A[i] = 1.0f;
        B[i] = 2.0f;
        C[i] = 0.0f;
    }

    // Warm-up
    for (std::size_t i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
    }

    // Benchmark
    double total_time = 0.0;

    for (int it = 0; it < iterations; ++it) {
        auto start = std::chrono::steady_clock::now();

        for (std::size_t i = 0; i < N; ++i) {
            C[i] = A[i] + B[i];
        }

        auto end = std::chrono::steady_clock::now();

        total_time +=
            std::chrono::duration<double, std::milli>(
                end - start).count();
    }

    const double avg_ms = total_time / iterations;

    // Vector addition:
    // 1 addition = 1 FLOP
    const double flops = static_cast<double>(N);

    // Read A + read B + write C = 12 bytes/element
    const double bytes =
        static_cast<double>(N) * 12.0;

    const double gflops =
        (flops / (avg_ms / 1000.0)) / 1e9;

    const double bandwidth_gbs =
        (bytes / (avg_ms / 1000.0)) / 1e9;

    // Correctness check
    bool correct = true;

    for (std::size_t i = 0; i < N; ++i) {
        if (std::fabs(C[i] - 3.0f) > 1e-6f) {
            correct = false;
            break;
        }
    }

    std::cout << std::fixed << std::setprecision(4);

    std::cout << "N                 : " << N << '\n';
    std::cout << "Iterations        : " << iterations << '\n';
    std::cout << "Average time (ms) : " << avg_ms << '\n';
    std::cout << "GFLOP/s            : " << gflops << '\n';
    std::cout << "Bandwidth (GB/s)   : " << bandwidth_gbs << '\n';
    std::cout << "Operational Intensity (FLOP/Byte): "
              << flops / bytes << '\n';
    std::cout << "Correct            : "
              << (correct ? "YES" : "NO") << '\n';

    return correct ? 0 : 1;
}
