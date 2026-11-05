// -----------------------------------------------------------------------------
// W04 ATTENTION — GPU S2a SWOPT UNFUSED: better kernels, same 3-kernel structure
//
// Changes versus S1 (the S/P matrix is still materialized in DRAM):
//   1. qk_tiled   16×16 shared-memory tiled GEMM, as in W03 GPU S2: each block
//                 computes a 16×16 tile of S; Q and K tiles of 16×16 floats are
//                 loaded coalesced (row-wise) and reused 16 times from shared
//                 memory. The K tile is read column-wise (Kt[j][c] for fixed c
//                 across the warp), so it is padded to 17 columns to avoid
//                 shared-memory bank conflicts.
//   2. softmax    one warp per row, coalesced, warp-shuffle reductions with an
//                 online (m, l) pass: 2 reads + 1 write per element
//                 (w04_softmax_warp.cuh, shared with REF_CUBLAS)
//   3. pv_tiled   16×16 shared-memory tiled GEMM over j: P and V tiles reused
//                 16 times from shared memory
//
// What it isolates: S1 → S2a is "kernel quality" with the unfused data flow
// unchanged; S2a → S2b (P3) is then fusion alone.
//
// Known limitations:
//   * one output per thread, no register tiling: each FMA needs two shared-
//     memory loads, so the GEMMs are limited by shared-memory bandwidth, not by
//     the FP32 pipes (the same limit W03 S2 had)
//   * for the PV GEMM the output is only d = 64 columns wide, so a block covers
//     a 16-column slice and P is re-read from global memory d/16 = 4 times
//   * S/P still makes 5 passes over BH·N² floats (S write, S read twice in
//     the softmax, P write, P read) ≈ 20·BH·N² bytes if none of it stays in
//     L2; this is what S2b removes
// -----------------------------------------------------------------------------
#include <cmath>
#include <cstddef>

#include "w04_gpu_stage.cuh"
#include "w04_softmax_warp.cuh"

namespace {

constexpr int kTile = 16;

// S[h][i][j] = scale · Σ_c Q[h][i][c] · K[h][j][c]
__global__ void qk_tiled(const float* __restrict__ Q, const float* __restrict__ K, float* __restrict__ S,
                         int N, int d, float scale) {
    __shared__ float Qs[kTile][kTile];
    __shared__ float Ks[kTile][kTile + 1];
    const int tx = threadIdx.x, ty = threadIdx.y, h = blockIdx.z;
    const int i0 = blockIdx.y * kTile, j0 = blockIdx.x * kTile;
    const float* Qh = Q + static_cast<std::size_t>(h) * N * d;
    const float* Kh = K + static_cast<std::size_t>(h) * N * d;

    float acc = 0.0f;
    for (int c0 = 0; c0 < d; c0 += kTile) {
        const int qi = i0 + ty, kj = j0 + ty, c = c0 + tx;
        Qs[ty][tx] = (qi < N && c < d) ? Qh[static_cast<std::size_t>(qi) * d + c] : 0.0f;
        Ks[ty][tx] = (kj < N && c < d) ? Kh[static_cast<std::size_t>(kj) * d + c] : 0.0f;
        __syncthreads();
#pragma unroll
        for (int cc = 0; cc < kTile; ++cc) acc += Qs[ty][cc] * Ks[tx][cc];
        __syncthreads();
    }
    const int i = i0 + ty, j = j0 + tx;
    if (i < N && j < N) S[(static_cast<std::size_t>(h) * N + i) * N + j] = acc * scale;
}

// O[h][i][c] = Σ_j P[h][i][j] · V[h][j][c]
__global__ void pv_tiled(const float* __restrict__ P, const float* __restrict__ V, float* __restrict__ O,
                         int N, int d) {
    __shared__ float Ps[kTile][kTile];
    __shared__ float Vs[kTile][kTile];
    const int tx = threadIdx.x, ty = threadIdx.y, h = blockIdx.z;
    const int i0 = blockIdx.y * kTile, c0 = blockIdx.x * kTile;
    const float* Ph = P + static_cast<std::size_t>(h) * N * N;
    const float* Vh = V + static_cast<std::size_t>(h) * N * d;

    float acc = 0.0f;
    for (int j0 = 0; j0 < N; j0 += kTile) {
        const int pi = i0 + ty, pj = j0 + tx, vj = j0 + ty, vc = c0 + tx;
        Ps[ty][tx] = (pi < N && pj < N) ? Ph[static_cast<std::size_t>(pi) * N + pj] : 0.0f;
        Vs[ty][tx] = (vj < N && vc < d) ? Vh[static_cast<std::size_t>(vj) * d + vc] : 0.0f;
        __syncthreads();
#pragma unroll
        for (int jj = 0; jj < kTile; ++jj) acc += Ps[ty][jj] * Vs[jj][tx];
        __syncthreads();
    }
    const int i = i0 + ty, c = c0 + tx;
    if (i < N && c < d) O[(static_cast<std::size_t>(h) * N + i) * d + c] = acc;
}

class StageS2a {
public:
    StageS2a(const w04::AttnShape& s, const w04::BenchArgs&) {
        W04_CUDA_CHECK(cudaMalloc(&S_, static_cast<std::size_t>(s.bh) * s.n * s.n * sizeof(float)));
    }
    ~StageS2a() { cudaFree(S_); }
    StageS2a(const StageS2a&) = delete;
    StageS2a& operator=(const StageS2a&) = delete;

    int phases() const { return 3; }
    const char* phase_name(int p) const { return p == 0 ? "qk" : p == 1 ? "softmax" : "pv"; }

    void launch(int p, const w04::DeviceAttn& io) {
        const int N = static_cast<int>(io.s.n), d = io.s.d, bh = io.s.bh;
        const dim3 block(kTile, kTile);
        if (p == 0) {
            const dim3 grid((N + kTile - 1) / kTile, (N + kTile - 1) / kTile, bh);
            qk_tiled<<<grid, block>>>(io.Q, io.K, S_, N, d, 1.0f / std::sqrt(static_cast<float>(d)));
        } else if (p == 1) {
            w04::launch_softmax_rows_warp(S_, static_cast<std::size_t>(bh) * N, N);
        } else {
            const dim3 grid((d + kTile - 1) / kTile, (N + kTile - 1) / kTile, bh);
            pv_tiled<<<grid, block>>>(S_, io.V, io.O, N, d);
        }
    }

private:
    float* S_ = nullptr;
};

}  // namespace

int main(int argc, char** argv) {
    w04::GpuStageSpec spec;
    spec.meta = {"W04_ATTENTION_GPU_S2a_SWOPT_UNFUSED", "S2a_SWOPT", "UNFUSED", "GPU", w04::Precision::FP32};
    return w04::run_gpu_stage<StageS2a>(argc, argv, spec);
}
