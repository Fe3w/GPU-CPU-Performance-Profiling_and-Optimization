#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)           \
                      << " at " << __FILE__ << ":" << __LINE__ << '\n';      \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

__global__ void vector_add_naive(const float* A,
                                 const float* B,
                                 float* C,
                                 std::size_t N)
{
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x
                  + threadIdx.x;

    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char** argv)
{
    std::size_t N = 33554432;   // 2^25 = 128 MiB per array
    int iterations = 20;
    int repetitions = 5;
    int warmup = 2;
    std::string output = "results/raw/W01_VECTOR_ADD/W01_VECTOR_ADD_GPU_S1_BASE_NAIVE.csv";

    if (argc > 1) N = std::stoull(argv[1]);
    if (argc > 2) iterations = std::stoi(argv[2]);
    if (argc > 3) repetitions = std::stoi(argv[3]);
    if (argc > 4) warmup = std::stoi(argv[4]);
    if (argc > 5) output = argv[5];

    const std::size_t bytes = N * sizeof(float);

    float* h_A = static_cast<float*>(std::malloc(bytes));
    float* h_B = static_cast<float*>(std::malloc(bytes));
    float* h_C = static_cast<float*>(std::malloc(bytes));

    if (!h_A || !h_B || !h_C) {
        std::cerr << "Host allocation failed\n";
        return EXIT_FAILURE;
    }

    for (std::size_t i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i % 1000) * 0.001f;
        h_B[i] = static_cast<float>((i * 3) % 1000) * 0.001f;
        h_C[i] = 0.0f;
    }

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    constexpr int threads_per_block = 256;
    const int blocks = static_cast<int>(
        (N + threads_per_block - 1) / threads_per_block
    );

    // Warm-up
    for (int w = 0; w < warmup; ++w) {
        vector_add_naive<<<blocks, threads_per_block>>>(d_A, d_B, d_C, N);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int it = 0; it < iterations; ++it) {
        vector_add_naive<<<blocks, threads_per_block>>>(d_A, d_B, d_C, N);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Verify
    bool correct = true;
    for (std::size_t i = 0; i < N; ++i) {
        float expected = h_A[i] + h_B[i];
        if (std::fabs(h_C[i] - expected) > 1e-6f) {
            correct = false;
            std::cerr << "Mismatch at " << i
                      << ": got " << h_C[i]
                      << ", expected " << expected << '\n';
            break;
        }
    }

    // Per-iteration kernel time
    double runtime_ms = static_cast<double>(elapsed_ms) / iterations;

    // Vector add = 1 FLOP/element
    double flops = 1.0 * static_cast<double>(N);

    // Algorithmic traffic = read A + read B + write C = 12 B/element
    double algorithmic_bytes = 12.0 * static_cast<double>(N);

    double gflops =
        (flops / (runtime_ms * 1e-3)) / 1e9;

    double bandwidth_gbs =
        (algorithmic_bytes / (runtime_ms * 1e-3)) / 1e9;

    double operational_intensity =
        flops / algorithmic_bytes;

    std::ofstream file(output);
    if (!file) {
        std::cerr << "Could not open output file: " << output << '\n';
        return EXIT_FAILURE;
    }

    file << "experiment_id,workload,platform,stage,variant,"
         << "N,threads,block_size,iterations,repetition,"
         << "runtime_ms,gflops,bandwidth_gbs,flops,"
         << "algorithmic_bytes,operational_intensity,correct,"
         << "compiler,compiler_flags\n";

    for (int rep = 1; rep <= repetitions; ++rep) {
        // For the first implementation, repeat the measured execution.
        CUDA_CHECK(cudaEventRecord(start));

        for (int it = 0; it < iterations; ++it) {
            vector_add_naive<<<blocks, threads_per_block>>>(d_A, d_B, d_C, N);
        }

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());

        float rep_elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&rep_elapsed_ms, start, stop));

        double rep_runtime_ms =
            static_cast<double>(rep_elapsed_ms) / iterations;

        double rep_gflops =
            (flops / (rep_runtime_ms * 1e-3)) / 1e9;

        double rep_bandwidth_gbs =
            (algorithmic_bytes / (rep_runtime_ms * 1e-3)) / 1e9;

        file << "W01_VECTOR_ADD_GPU_S1_BASE_NAIVE_R"
             << std::setfill('0') << std::setw(2) << rep
             << std::setfill(' ') << ","
             << "W01_VECTOR_ADD,GPU,S1_BASE,NAIVE_GLOBAL,"
             << N << ","
             << 1 << ","
             << threads_per_block << ","
             << iterations << ","
             << rep << ","
             << std::fixed << std::setprecision(6)
             << rep_runtime_ms << ","
             << rep_gflops << ","
             << rep_bandwidth_gbs << ","
             << flops << ","
             << algorithmic_bytes << ","
             << operational_intensity << ","
             << (correct ? "YES" : "NO") << ","
             << "nvcc,"
             << "\"-O3 -std=c++17 -lineinfo\"\n";
    }

    file.close();

    std::cout << "\nW01 GPU S1 BASE NAIVE\n";
    std::cout << "N              : " << N << '\n';
    std::cout << "Blocks         : " << blocks << '\n';
    std::cout << "Threads/block  : " << threads_per_block << '\n';
    std::cout << "Runtime/iter   : " << runtime_ms << " ms\n";
    std::cout << "GFLOP/s        : " << gflops << '\n';
    std::cout << "Bandwidth      : " << bandwidth_gbs << " GB/s\n";
    std::cout << "Operational Intensity: " << operational_intensity
              << " FLOP/Byte\n";
    std::cout << "Correct        : " << (correct ? "YES" : "NO") << '\n';

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    std::free(h_A);
    std::free(h_B);
    std::free(h_C);

    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
