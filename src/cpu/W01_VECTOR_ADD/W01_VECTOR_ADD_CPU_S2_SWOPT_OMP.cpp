#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

static void print_usage(const char* prog) {
    std::cout
        << "Usage:\n"
        << "  " << prog
        << " --size N"
        << " --iterations I"
        << " --repetitions R"
        << " --warmup W"
        << " --output FILE\n\n"
        << "Example:\n"
        << "  " << prog
        << " --size 33554432"
        << " --iterations 20"
        << " --repetitions 5"
        << " --warmup 2"
        << " --output results/raw/W01_VECTOR_ADD/"
           "W01_VECTOR_ADD_CPU_S2_SWOPT_OMP.csv\n";
}

int main(int argc, char** argv) {

    // ------------------------------------------------------------
    // Experiment defaults
    // ------------------------------------------------------------
    std::size_t N = 33554432;
    int iterations = 20;
    int repetitions = 5;
    int warmup = 2;

    std::string output_file =
        "results/raw/W01_VECTOR_ADD/"
        "W01_VECTOR_ADD_CPU_S2_SWOPT_OMP.csv";

    // ------------------------------------------------------------
    // Parse command-line arguments
    // ------------------------------------------------------------
    for (int i = 1; i < argc; ++i) {

        std::string arg = argv[i];

        if (arg == "--size" && i + 1 < argc) {
            N = std::stoull(argv[++i]);
        }
        else if (arg == "--iterations" && i + 1 < argc) {
            iterations = std::stoi(argv[++i]);
        }
        else if (arg == "--repetitions" && i + 1 < argc) {
            repetitions = std::stoi(argv[++i]);
        }
        else if (arg == "--warmup" && i + 1 < argc) {
            warmup = std::stoi(argv[++i]);
        }
        else if (arg == "--output" && i + 1 < argc) {
            output_file = argv[++i];
        }
        else if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
        else {
            std::cerr << "Unknown or incomplete argument: "
                      << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    if (N == 0 || iterations <= 0 ||
        repetitions <= 0 || warmup < 0) {
        std::cerr << "Invalid benchmark parameters.\n";
        return 1;
    }

    // ------------------------------------------------------------
    // Allocate data
    // ------------------------------------------------------------
    std::vector<float> A(N);
    std::vector<float> B(N);
    std::vector<float> C(N);

    // ------------------------------------------------------------
    // Initialize data
    // ------------------------------------------------------------
    for (std::size_t i = 0; i < N; ++i) {
        A[i] = 1.0f;
        B[i] = 2.0f;
        C[i] = 0.0f;
    }

    // ------------------------------------------------------------
    // Warm-up runs
    // ------------------------------------------------------------
    for (int w = 0; w < warmup; ++w) {
        #pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < N; ++i) {
            C[i] = A[i] + B[i];
        }
    }

    // ------------------------------------------------------------
    // Fixed workload characteristics
    //
    // Vector addition:
    //   C[i] = A[i] + B[i]
    //
    // FLOPs:
    //   1 floating-point addition / element
    //
    // Algorithmic traffic:
    //   A read = 4 bytes
    //   B read = 4 bytes
    //   C write = 4 bytes
    //   total = 12 bytes / element
    // ------------------------------------------------------------
    const double total_flops =
        static_cast<double>(N);

    const double algorithmic_bytes =
        static_cast<double>(N) * 12.0;

    const double operational_intensity =
        total_flops / algorithmic_bytes;

    // ------------------------------------------------------------
    // Create output file
    // ------------------------------------------------------------
    std::ofstream out(output_file);

    if (!out) {
        std::cerr << "Could not open output file:\n"
                  << output_file << "\n";
        return 1;
    }

    out << "experiment_id,"
        << "workload,"
        << "platform,"
        << "stage,"
        << "variant,"
        << "N,"
        << "iterations,"
        << "repetition,"
        << "runtime_ms,"
        << "gflops,"
        << "bandwidth_gbs,"
        << "flops,"
        << "algorithmic_bytes,"
        << "operational_intensity,"
        << "correct,"
        << "compiler,"
        << "compiler_flags\n";

    // ------------------------------------------------------------
    // Benchmark repetitions
    // ------------------------------------------------------------
    for (int rep = 1; rep <= repetitions; ++rep) {

        double total_ms = 0.0;

        for (int it = 0; it < iterations; ++it) {

            auto start =
                std::chrono::steady_clock::now();

            // ----------------------------------------------------
            // S2 SWOPT OMP:
            // scalar arithmetic, OpenMP parallel for, static schedule
            // ----------------------------------------------------
            #pragma omp parallel for schedule(static)
            for (std::size_t i = 0; i < N; ++i) {
                C[i] = A[i] + B[i];
            }

            auto end =
                std::chrono::steady_clock::now();

            const double elapsed_ms =
                std::chrono::duration<double, std::milli>(
                    end - start).count();

            total_ms += elapsed_ms;
        }

        const double avg_ms =
            total_ms / static_cast<double>(iterations);

        const double gflops =
            (total_flops / (avg_ms / 1000.0)) / 1e9;

        const double bandwidth_gbs =
            (algorithmic_bytes / (avg_ms / 1000.0)) / 1e9;

        // --------------------------------------------------------
        // Correctness
        // (single-threaded to avoid contaminating timed region)
        // --------------------------------------------------------
        bool correct = true;

        for (std::size_t i = 0; i < N; ++i) {
            if (std::fabs(C[i] - 3.0f) > 1e-6f) {
                correct = false;
                break;
            }
        }

        // --------------------------------------------------------
        // CSV row
        // --------------------------------------------------------
        out << std::fixed << std::setprecision(6);

        out << "W01_VECTOR_ADD_CPU_S2_SWOPT_OMP,"
            << "VECTOR_ADD,"
            << "CPU,"
            << "S2_SWOPT,"
            << "OMP_STATIC_SCALAR,"
            << N << ","
            << iterations << ","
            << rep << ","
            << avg_ms << ","
            << gflops << ","
            << bandwidth_gbs << ","
            << total_flops << ","
            << algorithmic_bytes << ","
            << operational_intensity << ","
            << (correct ? 1 : 0) << ","
            << "\"G++ " << __VERSION__ << "\","
            << "\"-O0 -std=c++17 -g -fopenmp\""
            << "\n";

        // --------------------------------------------------------
        // Human-readable output
        // --------------------------------------------------------
        std::cout << std::fixed << std::setprecision(4);

        std::cout
            << "Repetition " << rep << "/" << repetitions
            << "\n"
            << "  N                 : " << N << "\n"
            << "  Average time (ms) : " << avg_ms << "\n"
            << "  GFLOP/s            : " << gflops << "\n"
            << "  Bandwidth (GB/s)   : " << bandwidth_gbs << "\n"
            << "  Operational Intensity: "
            << operational_intensity << " FLOP/Byte\n"
            << "  Correct            : "
            << (correct ? "YES" : "NO") << "\n\n";

        if (!correct) {
            std::cerr << "Correctness check failed.\n";
            return 1;
        }
    }

    out.close();

    std::cout << "CSV saved to:\n"
              << output_file << "\n";

    return 0;
}