// -----------------------------------------------------------------------------
// W04 ATTENTION — CPU S2 SWOPT: fused, blocked, online softmax, OpenMP
//
// Changes versus S1:
//   * fused: no N×N buffer. Each work item (head h, query block of Br rows)
//     streams K/V in tiles of Bc rows and keeps only an S tile [Br][Bc]
//   * online softmax (FlashAttention): running row max m and denominator l;
//     when a tile raises the max from m to m_new, the accumulated l and O are
//     rescaled by alpha = exp(m − m_new) before the tile is added. O / l at the
//     end equals softmax(S)·V. Every exponent is <= 0, so nothing overflows
//   * OpenMP over work items, numbered head-major, schedule(dynamic, 1): P- and
//     E-cores run at different speeds, so a static split would idle the P-cores
//   * the K tile is transposed (Kt[d][Bc]) so the S-tile loop is an axpy over j
//     and the P·V loop an axpy over c: both vectorizable without -ffast-math,
//     because neither is a floating-point reduction
//   * 1/√d is folded into the Q block (exact for d = 64)
//
// Cache sizing (decision C7, sysfs on this machine): P-core L1d 48 KiB and L2
// 1.25 MiB; E-core L1d 32 KiB and L2 2 MiB per 4-core cluster; L3 24 MiB.
//   * L1: Kt and V tiles = 2·Bc·d·4 B (Bc = 64 -> 32 KiB) plus the rows in use
//   * L2: K_h and V_h (512·N B for d = 64) are re-read once per query block;
//     they fit a P-core L2 up to N ≈ 2048 and spill to L3 beyond
//   * default Br = Bc = 64, chosen from a thermally controlled Br x Bc grid
//     (Bc = 64 best at every Br; Br = 64 ahead of 32 and 128 at N = 1024)
//
// Known limitations (S3 addresses the first two):
//   * the vector width and FMA use are whatever the compiler chooses; the -O3
//     default build targets SSE2 only
//   * exp is the scalar library expf, and the row max/sum are scalar reductions
//   * no infinities are used (a first-tile flag replaces m = −inf) so that the
//     -ffast-math builds of the compiler-flag study stay correct
// -----------------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <vector>

#include <omp.h>

#include "w04_cpu_stage.hpp"

namespace {

void attention_s2(const float* Q, const float* K, const float* V, float* O, const w04::AttnShape& s, int Br,
                  int Bc) {
    const std::size_t N = s.n;
    const int d = s.d;
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));
    const long nqb = static_cast<long>((N + Br - 1) / Br);
    const long items = nqb * s.bh;

#pragma omp parallel
    {
        std::vector<float> Qb(static_cast<std::size_t>(Br) * d), Ob(static_cast<std::size_t>(Br) * d);
        std::vector<float> Kt(static_cast<std::size_t>(d) * Bc), S(static_cast<std::size_t>(Br) * Bc);
        std::vector<float> m(Br), l(Br);

#pragma omp for schedule(dynamic, 1)
        for (long item = 0; item < items; ++item) {
            const int h = static_cast<int>(item / nqb);
            const std::size_t i0 = static_cast<std::size_t>(item % nqb) * Br;
            const int rows = static_cast<int>(std::min<std::size_t>(Br, N - i0));
            const float* Kh = K + s.index(h, 0, 0);
            const float* Vh = V + s.index(h, 0, 0);

            for (int i = 0; i < rows; ++i)
                for (int c = 0; c < d; ++c) Qb[i * d + c] = Q[s.index(h, i0 + i, c)] * scale;
            std::fill(Ob.begin(), Ob.end(), 0.0f);

            bool first = true;
            for (std::size_t j0 = 0; j0 < N; j0 += Bc) {
                const int cols = static_cast<int>(std::min<std::size_t>(Bc, N - j0));

                // Kt[c][j] = K[j0 + j][c]
                for (int j = 0; j < cols; ++j)
                    for (int c = 0; c < d; ++c) Kt[c * Bc + j] = Kh[(j0 + j) * d + c];

                // S tile = Qb · Ktᵀ  (axpy over j)
                for (int i = 0; i < rows; ++i) {
                    float* Si = &S[i * Bc];
                    for (int j = 0; j < cols; ++j) Si[j] = 0.0f;
                    for (int c = 0; c < d; ++c) {
                        const float q = Qb[i * d + c];
                        const float* kt = &Kt[c * Bc];
                        for (int j = 0; j < cols; ++j) Si[j] += q * kt[j];
                    }
                }

                // online softmax + O += P·V  (axpy over c)
                for (int i = 0; i < rows; ++i) {
                    float* Si = &S[i * Bc];
                    float tile_max = Si[0];
                    for (int j = 1; j < cols; ++j) tile_max = std::max(tile_max, Si[j]);
                    const float m_new = first ? tile_max : std::max(m[i], tile_max);
                    const float alpha = first ? 0.0f : std::exp(m[i] - m_new);
                    float rowsum = 0.0f;
                    for (int j = 0; j < cols; ++j) {
                        Si[j] = std::exp(Si[j] - m_new);
                        rowsum += Si[j];
                    }
                    l[i] = (first ? 0.0f : alpha * l[i]) + rowsum;
                    float* o = &Ob[i * d];
                    if (!first)
                        for (int c = 0; c < d; ++c) o[c] *= alpha;
                    for (int j = 0; j < cols; ++j) {
                        const float p = Si[j];
                        const float* v = Vh + (j0 + j) * d;
                        for (int c = 0; c < d; ++c) o[c] += p * v[c];
                    }
                    m[i] = m_new;
                }
                first = false;
            }

            for (int i = 0; i < rows; ++i) {
                const float inv_l = 1.0f / l[i];
                for (int c = 0; c < d; ++c) O[s.index(h, i0 + i, c)] = Ob[i * d + c] * inv_l;
            }
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    w04::CpuStageSpec spec;
    spec.meta = {"W04_ATTENTION_CPU_S2_SWOPT_BLOCKED_OMP", "S2_SWOPT", "BLOCKED_OMP", "CPU", w04::Precision::FP32};
    spec.tunable = true;
    spec.default_br = 64;   // controlled Br x Bc rerun (P1): best at N = 1024, +2.8% over Br = 32
    spec.default_bc = 64;
    spec.threads = omp_get_max_threads();
    return w04::run_cpu_stage(argc, argv, spec, attention_s2);
}
