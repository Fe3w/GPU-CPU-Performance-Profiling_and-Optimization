// -----------------------------------------------------------------------------
// W04 ATTENTION — CPU S3 HWACC: S2 with AVX2/FMA intrinsics and a vector exp
//
// Same algorithm and work split as S2 (fused, online softmax, OpenMP over
// head-major (h, query block) items, dynamic schedule, transposed K tile).
// Changes versus S2:
//   * S tile = Qb·Ktᵀ with a register block of 4 rows × 16 columns
//     (8 ymm accumulators); per c: 2 loads of Kt, 4 broadcasts of Qb, 8 FMAs
//   * row max and row sum: 8-lane max/add, then one horizontal reduction
//   * P = exp(S − m) with exp256_ps (src/common/avx2_exp.hpp, ≤ 2 ulp measured
//     on the softmax range), 8 lanes at a time; alpha uses the same function
//   * O += P·V: for d = 64 the output row lives in exactly 8 ymm registers
//     for the whole tile: per key j, one broadcast of P and 8 FMAs
//   * ragged last tile: the S tile is padded to a multiple of 8 columns with
//     −FLT_MAX, whose exp is exactly 0, so the vector loops need no tail code
//
// Requirements: d % 8 == 0 and Bc % 8 == 0 (checked; error otherwise).
// Known limitations:
//   * FP32 only; no software prefetch; the Kt transpose is scalar
//   * d != 64 takes a slower P·V path that keeps O in memory
// -----------------------------------------------------------------------------
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <string>
#include <vector>

#include <immintrin.h>
#include <omp.h>

#include "avx2_exp.hpp"
#include "w04_cpu_stage.hpp"

namespace {

inline float hmax8(__m256 v) {
    __m128 x = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    x = _mm_max_ps(x, _mm_movehl_ps(x, x));
    x = _mm_max_ss(x, _mm_shuffle_ps(x, x, 1));
    return _mm_cvtss_f32(x);
}

inline float hsum8(__m256 v) {
    __m128 x = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    x = _mm_add_ps(x, _mm_movehl_ps(x, x));
    x = _mm_add_ss(x, _mm_shuffle_ps(x, x, 1));
    return _mm_cvtss_f32(x);
}

// S[i..i+R)[j..j+8·NV) = Qb[i..i+R) · Kt[:, j..j+8·NV)
template <int R, int NV>
inline void s_block(const float* Qb, const float* Kt, float* S, int d, int Bc, int i, int j) {
    __m256 acc[R][NV];
    for (int r = 0; r < R; ++r)
        for (int v = 0; v < NV; ++v) acc[r][v] = _mm256_setzero_ps();
    for (int c = 0; c < d; ++c) {
        __m256 kt[NV];
        for (int v = 0; v < NV; ++v) kt[v] = _mm256_loadu_ps(Kt + c * Bc + j + 8 * v);
        for (int r = 0; r < R; ++r) {
            const __m256 q = _mm256_broadcast_ss(Qb + (i + r) * d + c);
            for (int v = 0; v < NV; ++v) acc[r][v] = _mm256_fmadd_ps(q, kt[v], acc[r][v]);
        }
    }
    for (int r = 0; r < R; ++r)
        for (int v = 0; v < NV; ++v) _mm256_storeu_ps(S + (i + r) * Bc + j + 8 * v, acc[r][v]);
}

template <int R>
inline void s_rows(const float* Qb, const float* Kt, float* S, int d, int Bc, int i, int colsp) {
    int j = 0;
    for (; j + 16 <= colsp; j += 16) s_block<R, 2>(Qb, Kt, S, d, Bc, i, j);
    if (j < colsp) s_block<R, 1>(Qb, Kt, S, d, Bc, i, j);  // colsp is a multiple of 8
}

void attention_s3(const float* Q, const float* K, const float* V, float* O, const w04::AttnShape& s, int Br,
                  int Bc) {
    const std::size_t N = s.n;
    const int d = s.d;
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));
    const long nqb = static_cast<long>((N + Br - 1) / Br);
    const long items = nqb * s.bh;

#pragma omp parallel
    {
        std::vector<float> Qb(static_cast<std::size_t>(Br) * d), Ob(static_cast<std::size_t>(Br) * d);
        std::vector<float> Kt(static_cast<std::size_t>(d) * Bc, 0.0f), S(static_cast<std::size_t>(Br) * Bc);
        std::vector<float> m(Br), l(Br);

#pragma omp for schedule(dynamic, 1)
        for (long item = 0; item < items; ++item) {
            const int h = static_cast<int>(item / nqb);
            const std::size_t i0 = static_cast<std::size_t>(item % nqb) * Br;
            const int rows = static_cast<int>(std::min<std::size_t>(Br, N - i0));
            const float* Kh = K + s.index(h, 0, 0);
            const float* Vh = V + s.index(h, 0, 0);

            const __m256 vscale = _mm256_set1_ps(scale);
            for (int i = 0; i < rows; ++i)
                for (int c = 0; c < d; c += 8)
                    _mm256_storeu_ps(&Qb[i * d + c], _mm256_mul_ps(_mm256_loadu_ps(Q + s.index(h, i0 + i, c)), vscale));
            std::fill(Ob.begin(), Ob.end(), 0.0f);

            bool first = true;
            for (std::size_t j0 = 0; j0 < N; j0 += Bc) {
                const int cols = static_cast<int>(std::min<std::size_t>(Bc, N - j0));
                const int colsp = (cols + 7) & ~7;

                for (int j = 0; j < cols; ++j)
                    for (int c = 0; c < d; ++c) Kt[c * Bc + j] = Kh[(j0 + j) * d + c];
                for (int j = cols; j < colsp; ++j)
                    for (int c = 0; c < d; ++c) Kt[c * Bc + j] = 0.0f;

                int i = 0;
                for (; i + 4 <= rows; i += 4) s_rows<4>(Qb.data(), Kt.data(), S.data(), d, Bc, i, colsp);
                for (; i < rows; ++i) s_rows<1>(Qb.data(), Kt.data(), S.data(), d, Bc, i, colsp);

                for (int r = 0; r < rows; ++r) {
                    float* Si = &S[r * Bc];
                    for (int j = cols; j < colsp; ++j) Si[j] = -FLT_MAX;  // exp(−FLT_MAX − m) == 0

                    __m256 vmax = _mm256_loadu_ps(Si);
                    for (int j = 8; j < colsp; j += 8) vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(Si + j));
                    const float tile_max = hmax8(vmax);
                    const float m_new = first ? tile_max : std::max(m[r], tile_max);
                    const __m256 vm = _mm256_set1_ps(m_new);

                    __m256 vsum = _mm256_setzero_ps();
                    for (int j = 0; j < colsp; j += 8) {
                        const __m256 p = w04::exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(Si + j), vm));
                        _mm256_storeu_ps(Si + j, p);
                        vsum = _mm256_add_ps(vsum, p);
                    }
                    const float rowsum = hsum8(vsum);

                    float* o = &Ob[r * d];
                    if (first) {
                        l[r] = rowsum;
                    } else {
                        const float alpha = _mm256_cvtss_f32(w04::exp256_ps(_mm256_set1_ps(m[r] - m_new)));
                        l[r] = alpha * l[r] + rowsum;
                        const __m256 va = _mm256_set1_ps(alpha);
                        for (int c = 0; c < d; c += 8) _mm256_storeu_ps(o + c, _mm256_mul_ps(va, _mm256_loadu_ps(o + c)));
                    }

                    if (d == 64) {
                        __m256 acc[8];
                        for (int k = 0; k < 8; ++k) acc[k] = _mm256_loadu_ps(o + 8 * k);
                        for (int j = 0; j < cols; ++j) {
                            const __m256 p = _mm256_broadcast_ss(Si + j);
                            const float* v = Vh + (j0 + j) * 64;
                            for (int k = 0; k < 8; ++k) acc[k] = _mm256_fmadd_ps(p, _mm256_loadu_ps(v + 8 * k), acc[k]);
                        }
                        for (int k = 0; k < 8; ++k) _mm256_storeu_ps(o + 8 * k, acc[k]);
                    } else {
                        for (int j = 0; j < cols; ++j) {
                            const __m256 p = _mm256_broadcast_ss(Si + j);
                            const float* v = Vh + (j0 + j) * d;
                            for (int c = 0; c < d; c += 8)
                                _mm256_storeu_ps(o + c, _mm256_fmadd_ps(p, _mm256_loadu_ps(v + c), _mm256_loadu_ps(o + c)));
                        }
                    }
                    m[r] = m_new;
                }
                first = false;
            }

            for (int r = 0; r < rows; ++r) {
                const __m256 inv_l = _mm256_set1_ps(1.0f / l[r]);
                for (int c = 0; c < d; c += 8)
                    _mm256_storeu_ps(O + s.index(h, i0 + r, c), _mm256_mul_ps(_mm256_loadu_ps(&Ob[r * d + c]), inv_l));
            }
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    w04::CpuStageSpec spec;
    spec.meta = {"W04_ATTENTION_CPU_S3_HWACC_AVX2", "S3_HWACC", "AVX2_FMA", "CPU", w04::Precision::FP32};
    spec.tunable = true;
    spec.default_br = 32;   // controlled Br x Bc rerun (P1): Bc = 64 best; Br = 32/64/128 tie at N = 1024
    spec.default_bc = 64;
    spec.threads = omp_get_max_threads();
    spec.validate = [](const w04::BenchArgs& a) -> std::string {
        if (a.d % 8 != 0) return "d must be a multiple of 8 (AVX2 lanes), got " + std::to_string(a.d);
        if (a.bc % 8 != 0) return "Bc must be a multiple of 8 (AVX2 lanes), got " + std::to_string(a.bc);
        return "";
    };
    return w04::run_cpu_stage(argc, argv, spec, attention_s3);
}
