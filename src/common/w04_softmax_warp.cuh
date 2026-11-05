// -----------------------------------------------------------------------------
// W04 ATTENTION — warp-per-row softmax over a materialized S matrix (P2)
//
// Used by GPU S2a_SWOPT_UNFUSED and by REF_CUBLAS, so both unfused pipelines
// share exactly the same softmax and differ only in their GEMMs.
//
//   P[r][j] = exp(S[r][j] − m_r) / l_r,   m_r = max_j S[r][j],  l_r = Σ_j exp(S[r][j] − m_r)
//
// One warp per row; lane t handles columns t, t+32, t+64, ... (coalesced).
//   pass 1: each lane keeps an online (m, l) pair over its columns; the 32
//           pairs are merged with warp shuffles:
//             m = max(m_a, m_b),  l = l_a·exp(m_a − m) + l_b·exp(m_b − m)
//   pass 2: read S again, write P (in place).
// Traffic per row: 2 reads + 1 write of N floats (S1's one-thread-per-row
// softmax does 3 reads + 2 writes, and its accesses are strided by N).
//
// m starts at −FLT_MAX instead of −inf so that an empty lane (N < 32) gives
// exp(−FLT_MAX − m) = 0 rather than exp(−inf + inf) = NaN.
// -----------------------------------------------------------------------------
#pragma once

#include <cuda_runtime.h>

#include <cfloat>
#include <cstddef>

namespace w04 {

constexpr int kSoftmaxWarpsPerBlock = 8;

__global__ void softmax_rows_warp(float* S, std::size_t rows, int N) {
    const int lane = threadIdx.x & 31;
    const std::size_t r = static_cast<std::size_t>(blockIdx.x) * kSoftmaxWarpsPerBlock + (threadIdx.x >> 5);
    if (r >= rows) return;  // whole warp exits together (r is uniform per warp)
    float* row = S + r * static_cast<std::size_t>(N);

    float m = -FLT_MAX, l = 0.0f;
    for (int j = lane; j < N; j += 32) {
        const float x = row[j];
        if (x > m) {
            l = l * expf(m - x) + 1.0f;
            m = x;
        } else {
            l += expf(x - m);
        }
    }
    for (int off = 16; off > 0; off >>= 1) {
        const float m_o = __shfl_xor_sync(0xffffffffu, m, off);
        const float l_o = __shfl_xor_sync(0xffffffffu, l, off);
        const float m_new = fmaxf(m, m_o);
        l = l * expf(m - m_new) + l_o * expf(m_o - m_new);
        m = m_new;
    }
    for (int j = lane; j < N; j += 32) row[j] = expf(row[j] - m) / l;
}

inline void launch_softmax_rows_warp(float* S, std::size_t rows, int N) {
    const unsigned blocks = static_cast<unsigned>((rows + kSoftmaxWarpsPerBlock - 1) / kSoftmaxWarpsPerBlock);
    softmax_rows_warp<<<blocks, 32 * kSoftmaxWarpsPerBlock>>>(S, rows, N);
}

}  // namespace w04
