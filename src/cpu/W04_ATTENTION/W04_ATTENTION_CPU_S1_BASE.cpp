// -----------------------------------------------------------------------------
// W04 ATTENTION — CPU S1 BASE: scalar, unfused, one thread
//
//   O = softmax(Q·Kᵀ / √d) · V      layout [BH][N][d], FP32
//
// Per head, three separate passes over a full N×N buffer S:
//   1. S = Q·Kᵀ·scale                       (triple loop, dot product over d)
//   2. softmax per row, numerically stable: m = max_j S, P = exp(S − m) / Σ exp(S − m)
//      (P overwrites S)
//   3. O = P·V                              (triple loop, axpy over d)
//
// This is the baseline: simplest correct code, built with -O2 (decision C5),
// no OpenMP in the forward pass (the reference check is multithreaded, untimed).
//
// Known limitations (deliberate; S2 removes them):
//   * the N×N buffer is O(N²) memory per head (64 MiB at N = 4096) and is
//     written and re-read in every pass instead of staying in cache
//   * the dot product in pass 1 is one serial dependency chain per (i, j)
//   * the buffer is allocated inside the timed call (page faults included)
// -----------------------------------------------------------------------------
#include <cmath>
#include <vector>

#include "w04_cpu_stage.hpp"

namespace {

void attention_s1(const float* Q, const float* K, const float* V, float* O, const w04::AttnShape& s, int, int) {
    const std::size_t N = s.n;
    const int d = s.d;
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));
    std::vector<float> S(N * N);

    for (int h = 0; h < s.bh; ++h) {
        const float* Qh = Q + s.index(h, 0, 0);
        const float* Kh = K + s.index(h, 0, 0);
        const float* Vh = V + s.index(h, 0, 0);
        float* Oh = O + s.index(h, 0, 0);

        // 1. S = Q·Kᵀ·scale
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t j = 0; j < N; ++j) {
                float acc = 0.0f;
                for (int c = 0; c < d; ++c) acc += Qh[i * d + c] * Kh[j * d + c];
                S[i * N + j] = acc * scale;
            }

        // 2. stable softmax per row (P overwrites S)
        for (std::size_t i = 0; i < N; ++i) {
            float* row = &S[i * N];
            float m = row[0];
            for (std::size_t j = 1; j < N; ++j) m = std::fmax(m, row[j]);
            float l = 0.0f;
            for (std::size_t j = 0; j < N; ++j) {
                row[j] = std::exp(row[j] - m);
                l += row[j];
            }
            for (std::size_t j = 0; j < N; ++j) row[j] /= l;
        }

        // 3. O = P·V
        for (std::size_t i = 0; i < N; ++i) {
            float* o = Oh + i * d;
            for (int c = 0; c < d; ++c) o[c] = 0.0f;
            for (std::size_t j = 0; j < N; ++j) {
                const float p = S[i * N + j];
                const float* v = Vh + j * d;
                for (int c = 0; c < d; ++c) o[c] += p * v[c];
            }
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    w04::CpuStageSpec spec;
    spec.meta = {"W04_ATTENTION_CPU_S1_BASE", "S1_BASE", "SCALAR_UNFUSED", "CPU", w04::Precision::FP32};
    spec.threads = 1;
    return w04::run_cpu_stage(argc, argv, spec, attention_s1);
}
