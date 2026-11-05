// -----------------------------------------------------------------------------
// W04 ATTENTION — GPU S1 BASE NAIVE: three kernels, S materialized in DRAM
//
//   O = softmax(Q·Kᵀ / √d) · V      layout [BH][N][d], FP32, S/P buffer [BH][N][N]
//
//   1. qk_naive      one thread per S[h][i][j]: dot product over d from global
//                    memory, times 1/√d (16×16 blocks, threadIdx.x → j)
//   2. softmax_naive one thread per row: 3 passes over the row in global memory
//                    (max; exp and sum, writing exp; divide), numerically stable
//   3. pv_naive      one thread per O[h][i][c]: Σ_j P[i][j]·V[j][c] from global
//                    memory (threadIdx.x → c)
//
// The GPU counterpart of CPU S1 and of W03 GPU S1 (naive GEMM): simplest
// correct code, no shared memory, no fusion.
//
// Known limitations (deliberate; S2a removes the first three, S2b the last):
//   * qk_naive: the 32 threads of a warp read K rows that are d floats apart,
//     so each K load is uncoalesced; nothing is reused through shared memory
//   * softmax_naive: neighbouring threads own neighbouring rows, so every load
//     and store is strided by N floats (uncoalesced), and the row is read 3
//     times and written twice
//   * pv_naive: every thread re-reads its row of P from global memory
//   * S/P is BH·N²·4 B (512 MiB at BH=8, N=4096) and goes through DRAM
//     between the kernels; at N ≤ 1024 much of it can stay in the 32 MiB L2
// -----------------------------------------------------------------------------
#include <cmath>
#include <cstddef>

#include "w04_gpu_stage.cuh"

namespace {

constexpr int kTile = 16;

__global__ void qk_naive(const float* __restrict__ Q, const float* __restrict__ K, float* __restrict__ S,
                         int N, int d, float scale) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int h = blockIdx.z;
    if (i >= N || j >= N) return;
    const float* q = Q + (static_cast<std::size_t>(h) * N + i) * d;
    const float* k = K + (static_cast<std::size_t>(h) * N + j) * d;
    float acc = 0.0f;
    for (int c = 0; c < d; ++c) acc += q[c] * k[c];
    S[(static_cast<std::size_t>(h) * N + i) * N + j] = acc * scale;
}

__global__ void softmax_naive(float* __restrict__ S, std::size_t rows, int N) {
    const std::size_t r = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    float* row = S + r * static_cast<std::size_t>(N);
    float m = row[0];
    for (int j = 1; j < N; ++j) m = fmaxf(m, row[j]);
    float l = 0.0f;
    for (int j = 0; j < N; ++j) {
        const float e = expf(row[j] - m);
        row[j] = e;
        l += e;
    }
    for (int j = 0; j < N; ++j) row[j] /= l;
}

__global__ void pv_naive(const float* __restrict__ P, const float* __restrict__ V, float* __restrict__ O,
                         int N, int d) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int h = blockIdx.z;
    if (i >= N || c >= d) return;
    const float* p = P + (static_cast<std::size_t>(h) * N + i) * N;
    const float* v = V + static_cast<std::size_t>(h) * N * d;
    float acc = 0.0f;
    for (int j = 0; j < N; ++j) acc += p[j] * v[static_cast<std::size_t>(j) * d + c];
    O[(static_cast<std::size_t>(h) * N + i) * d + c] = acc;
}

class StageS1 {
public:
    StageS1(const w04::AttnShape& s, const w04::BenchArgs&) {
        W04_CUDA_CHECK(cudaMalloc(&S_, static_cast<std::size_t>(s.bh) * s.n * s.n * sizeof(float)));
    }
    ~StageS1() { cudaFree(S_); }
    StageS1(const StageS1&) = delete;
    StageS1& operator=(const StageS1&) = delete;

    int phases() const { return 3; }
    const char* phase_name(int p) const { return p == 0 ? "qk" : p == 1 ? "softmax" : "pv"; }

    void launch(int p, const w04::DeviceAttn& io) {
        const int N = static_cast<int>(io.s.n), d = io.s.d, bh = io.s.bh;
        const dim3 block(kTile, kTile);
        if (p == 0) {
            const dim3 grid((N + kTile - 1) / kTile, (N + kTile - 1) / kTile, bh);
            qk_naive<<<grid, block>>>(io.Q, io.K, S_, N, d, 1.0f / std::sqrt(static_cast<float>(d)));
        } else if (p == 1) {
            const std::size_t rows = static_cast<std::size_t>(bh) * N;
            softmax_naive<<<static_cast<unsigned>((rows + 255) / 256), 256>>>(S_, rows, N);
        } else {
            const dim3 grid((d + kTile - 1) / kTile, (N + kTile - 1) / kTile, bh);
            pv_naive<<<grid, block>>>(S_, io.V, io.O, N, d);
        }
    }

private:
    float* S_ = nullptr;
};

}  // namespace

int main(int argc, char** argv) {
    w04::GpuStageSpec spec;
    spec.meta = {"W04_ATTENTION_GPU_S1_BASE_NAIVE", "S1_BASE", "NAIVE", "GPU", w04::Precision::FP32};
    return w04::run_gpu_stage<StageS1>(argc, argv, spec);
}
