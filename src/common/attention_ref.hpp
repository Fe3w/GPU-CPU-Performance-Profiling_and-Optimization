// -----------------------------------------------------------------------------
// W04 ATTENTION — shared reference, input generator and error metrics (P0)
//
// Everything every W04 stage needs to prove it is correct:
//   * AttnShape / layout   : row-major [BH][N][d], FP32
//   * make_inputs()        : bit-reproducible inputs (own splitmix64 RNG),
//                            Normal and LargeLogits cases, optional V = ones
//   * attention_reference(): double-precision, numerically stable softmax,
//                            OpenMP over rows, full or sampled rows
//   * compare_rows()       : max_abs_err, normwise max_rel_err, non-finite count
//   * round_to_fp16/tf32() : host emulation of the input rounding that S3 sees,
//                            used for the large-logits reference (decision C2)
//
// Decisions recorded in docs/W04_PROGRESS.md (P0):
//   C2  large-logits pass rule = finite + V=ones row-sum + stage tolerance against
//       a double reference computed from the SAME rounded inputs the stage used.
//   C3  max_rel_err = max|O - O_ref| / max|O_ref|  (normwise, infinity norm).
//       Element-wise relative error (floor 1e-3) is only a printed diagnostic.
//
// Header-only; compiles with g++ and as nvcc host code.
// -----------------------------------------------------------------------------
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace w04 {

// ---------------------------------------------------------------- shape/types
struct AttnShape {
    int bh = 8;            // batch x heads
    std::size_t n = 1024;  // sequence length
    int d = 64;            // head dimension
    std::size_t elems() const { return static_cast<std::size_t>(bh) * n * static_cast<std::size_t>(d); }
    std::size_t index(int h, std::size_t i, int c) const {
        return (static_cast<std::size_t>(h) * n + i) * static_cast<std::size_t>(d) + static_cast<std::size_t>(c);
    }
};

enum class InputCase { Normal, LargeLogits };
enum class Precision { FP32, TF32, FP16 };

inline const char* to_string(InputCase c) { return c == InputCase::Normal ? "normal" : "large_logits"; }
inline const char* to_string(Precision p) {
    switch (p) {
        case Precision::FP32: return "FP32";
        case Precision::TF32: return "TF32";
        case Precision::FP16: return "FP16";
    }
    return "?";
}

// Q and K are each multiplied by this in the LargeLogits case (plan §5).
// Logit std ≈ 0.33·12² ≈ 48, so the row max is far above 88 and exp() without
// max-subtraction overflows FP32 (verified: the unstable control is non-finite).
// The plan's original ×30 was changed to ×12 in P0 (user decision): at ×30
// (logits up to ~±1000) the FP32 rounding of Q·K alone gives ~1.8e-4 output
// error even for a correct stable softmax, above the 1e-4 FP32 tolerance.
// Measured worst case at ×12 (stable FP32 control, n ≤ 4096, 3 seeds): 4.0e-5.
constexpr float kLargeLogitScale = 12.0f;

// Plan §5 tolerances. These only decide the YES/NO flag; actual values are recorded.
inline double tolerance_for(Precision p) {
    switch (p) {
        case Precision::FP32: return 1e-4;
        case Precision::TF32: return 5e-3;
        case Precision::FP16: return 1e-2;
    }
    return 0.0;
}

// ---------------------------------------------------------------------- RNG
// splitmix64: fully specified, so inputs are identical across compilers and
// library versions (std::uniform_real_distribution is implementation-defined).
inline std::uint64_t splitmix64(std::uint64_t& state) {
    std::uint64_t z = (state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// Uniform in [-1, 1) on a grid of 2^-23; every value is exactly representable.
inline float uniform_pm1(std::uint64_t& state) {
    const std::uint32_t m = static_cast<std::uint32_t>(splitmix64(state) >> 40);  // 24 bits
    return static_cast<float>(m) * (2.0f / 16777216.0f) - 1.0f;
}

// ------------------------------------------------------------------- inputs
struct AttnInputs {
    std::vector<float> Q, K, V;
};

inline AttnInputs make_inputs(const AttnShape& s, std::uint64_t seed, InputCase c, bool v_ones = false) {
    AttnInputs in;
    const std::size_t e = s.elems();
    in.Q.resize(e);
    in.K.resize(e);
    in.V.resize(e);
    // Independent stream per tensor, so Q/K/V do not depend on each other's sizes.
    std::uint64_t sq = seed ^ 0x5151515151515151ull;
    std::uint64_t sk = seed ^ 0x4B4B4B4B4B4B4B4Bull;
    std::uint64_t sv = seed ^ 0x5656565656565656ull;
    const float qk_scale = (c == InputCase::LargeLogits) ? kLargeLogitScale : 1.0f;
    for (std::size_t i = 0; i < e; ++i) in.Q[i] = uniform_pm1(sq) * qk_scale;
    for (std::size_t i = 0; i < e; ++i) in.K[i] = uniform_pm1(sk) * qk_scale;
    for (std::size_t i = 0; i < e; ++i) {
        const float v = uniform_pm1(sv);  // always drawn, so V is identical with/without v_ones elsewhere
        in.V[i] = v_ones ? 1.0f : v;
    }
    return in;
}

// --------------------------------------------------------- input rounding
// Bit helpers that survive -ffast-math (std::isfinite may be folded to true there).
inline std::uint32_t float_bits(float x) { std::uint32_t u; std::memcpy(&u, &x, sizeof u); return u; }
inline float bits_float(std::uint32_t u) { float x; std::memcpy(&x, &u, sizeof x); return x; }
inline bool is_finite_bits(float x) { return (float_bits(x) & 0x7F800000u) != 0x7F800000u; }
inline bool is_finite_bits(double x) {
    std::uint64_t u; std::memcpy(&u, &x, sizeof u);
    return (u & 0x7FF0000000000000ull) != 0x7FF0000000000000ull;
}

// FP32 -> FP16 -> FP32, round-to-nearest-even (matches __float2half_rn).
inline float round_to_fp16(float x) {
    if (!is_finite_bits(x)) return x;
    const float ax = std::fabs(x);
    if (ax >= 65520.0f) return std::copysign(INFINITY, x);  // rounds past max half (65504)
    if (ax < 6.103515625e-05f) {                             // half subnormal range: quantum 2^-24
        return static_cast<float>(std::nearbyint(static_cast<double>(x) * 16777216.0) / 16777216.0);
    }
    std::uint32_t u = float_bits(x);
    u += 0x0FFFu + ((u >> 13) & 1u);  // RNE on the 13 dropped mantissa bits
    u &= ~0x1FFFu;
    return bits_float(u);
}

// FP32 -> TF32 (10-bit mantissa), round-to-nearest, ties away from zero:
// matches nvcuda::wmma::__float_to_tf32 (cvt.rna.tf32.f32). Checked on the GPU in the self-test.
inline float round_to_tf32(float x) {
    if (!is_finite_bits(x)) return x;
    std::uint32_t u = float_bits(x);
    u += 0x1000u;
    u &= ~0x1FFFu;
    return bits_float(u);
}

inline void round_inputs(AttnInputs& in, Precision p) {
    if (p == Precision::FP32) return;
    auto f = (p == Precision::FP16) ? round_to_fp16 : round_to_tf32;
    for (auto* v : {&in.Q, &in.K, &in.V})
        for (float& x : *v) x = f(x);
}

// ------------------------------------------------------------ row selection
struct RowId {
    int h;
    std::size_t i;
};

// N <= full_up_to_n: every row of every head.
// Otherwise per head: rows 0 and N-1 (tile edges) plus distinct random rows, sorted.
inline std::vector<RowId> select_check_rows(const AttnShape& s, std::uint64_t seed,
                                            std::size_t full_up_to_n = 2048,
                                            std::size_t per_head = 64) {
    std::vector<RowId> rows;
    if (s.n <= full_up_to_n) {
        rows.reserve(static_cast<std::size_t>(s.bh) * s.n);
        for (int h = 0; h < s.bh; ++h)
            for (std::size_t i = 0; i < s.n; ++i) rows.push_back({h, i});
        return rows;
    }
    std::uint64_t st = seed ^ 0x526F777352617773ull;
    const std::size_t want = std::min(per_head, s.n);
    for (int h = 0; h < s.bh; ++h) {
        std::vector<std::size_t> picked = {0, s.n - 1};
        while (picked.size() < want) {
            const std::size_t r = static_cast<std::size_t>(splitmix64(st) % s.n);
            if (std::find(picked.begin(), picked.end(), r) == picked.end()) picked.push_back(r);
        }
        std::sort(picked.begin(), picked.end());
        for (std::size_t r : picked) rows.push_back({h, r});
    }
    return rows;
}

// --------------------------------------------------------------- reference
// O_ref[r*d + c] for each requested row r. Double precision, stable softmax:
//   S_j = (q · k_j) / sqrt(d);  m = max_j S_j;  P_j = exp(S_j - m) / sum_j exp(S_j - m)
// Memory: one N-length double buffer per thread, never N x N.
inline void attention_reference(const float* Q, const float* K, const float* V, const AttnShape& s,
                                const std::vector<RowId>& rows, std::vector<double>& O_ref) {
    const int d = s.d;
    const std::size_t n = s.n;
    O_ref.assign(rows.size() * static_cast<std::size_t>(d), 0.0);
    const double scale = 1.0 / std::sqrt(static_cast<double>(d));
    const long nrows = static_cast<long>(rows.size());

#pragma omp parallel
    {
        std::vector<double> S(n);
        std::vector<double> acc(static_cast<std::size_t>(d));
#pragma omp for schedule(dynamic, 8)
        for (long r = 0; r < nrows; ++r) {
            const int h = rows[static_cast<std::size_t>(r)].h;
            const std::size_t i = rows[static_cast<std::size_t>(r)].i;
            const float* q = Q + s.index(h, i, 0);
            double m = -INFINITY;
            for (std::size_t j = 0; j < n; ++j) {
                const float* k = K + s.index(h, j, 0);
                double dot = 0.0;
                for (int c = 0; c < d; ++c) dot += static_cast<double>(q[c]) * static_cast<double>(k[c]);
                S[j] = dot * scale;
                m = std::max(m, S[j]);
            }
            double l = 0.0;
            std::fill(acc.begin(), acc.end(), 0.0);
            for (std::size_t j = 0; j < n; ++j) {
                const double p = std::exp(S[j] - m);
                l += p;
                const float* v = V + s.index(h, j, 0);
                for (int c = 0; c < d; ++c) acc[static_cast<std::size_t>(c)] += p * static_cast<double>(v[c]);
            }
            double* o = O_ref.data() + static_cast<std::size_t>(r) * static_cast<std::size_t>(d);
            for (int c = 0; c < d; ++c) o[c] = acc[static_cast<std::size_t>(c)] / l;
        }
    }
}

// ------------------------------------------------------------------ metrics
struct ErrorStats {
    // If ANY checked output is NaN/Inf, all three error values are +inf: a
    // non-finite output is an unbounded error, never a skipped one.
    double max_abs_err = 0.0;   // max |O - O_ref|
    double max_rel_err = 0.0;   // max_abs_err / max |O_ref|   (normwise, C3)
    double max_elem_rel = 0.0;  // diagnostic: max |O - O_ref| / max(|O_ref|, 1e-3)
    double max_abs_ref = 0.0;
    std::size_t n_nonfinite = 0;
    std::size_t n_checked = 0;
    RowId worst{0, 0};
    int worst_col = 0;
};

// O is the stage's full output [bh][n][d]; O_ref holds only the requested rows.
// NaN/Inf are detected by bit pattern (robust under -ffast-math) and counted:
// `if (err > max)` is false for NaN, so a naive max-loop would report zero
// error for an overflowed output. Any non-finite output makes every error +inf.
inline ErrorStats compare_rows(const float* O, const AttnShape& s, const std::vector<RowId>& rows,
                               const std::vector<double>& O_ref) {
    ErrorStats e;
    const int d = s.d;
    for (std::size_t r = 0; r < rows.size(); ++r) {
        for (int c = 0; c < d; ++c) {
            const float o = O[s.index(rows[r].h, rows[r].i, c)];
            const double ref = O_ref[r * static_cast<std::size_t>(d) + static_cast<std::size_t>(c)];
            ++e.n_checked;
            e.max_abs_ref = std::max(e.max_abs_ref, std::fabs(ref));
            if (!is_finite_bits(o)) {
                if (e.n_nonfinite == 0) { e.worst = rows[r]; e.worst_col = c; }
                ++e.n_nonfinite;
                continue;
            }
            const double abs_err = std::fabs(static_cast<double>(o) - ref);
            if (abs_err > e.max_abs_err && e.n_nonfinite == 0) { e.worst = rows[r]; e.worst_col = c; }
            e.max_abs_err = std::max(e.max_abs_err, abs_err);
            e.max_elem_rel = std::max(e.max_elem_rel, abs_err / std::max(std::fabs(ref), 1e-3));
        }
    }
    if (e.n_nonfinite > 0) {
        e.max_abs_err = INFINITY;
        e.max_rel_err = INFINITY;
        e.max_elem_rel = INFINITY;
    } else {
        e.max_rel_err = (e.max_abs_ref > 0.0) ? e.max_abs_err / e.max_abs_ref : e.max_abs_err;
    }
    return e;
}

inline bool passes(const ErrorStats& e, Precision p) {
    return e.n_checked > 0 && e.n_nonfinite == 0 && e.max_rel_err <= tolerance_for(p);
}

// ------------------------------------------------------- full stage check
// One correctness evaluation for a stage, shared by every W04 binary.
//   main   : output on the benchmark inputs vs the double reference
//            (Normal: reference on the FP32 inputs;
//             LargeLogits: reference on the inputs rounded to the stage precision, C2)
//   plain  : always vs the reference on the unrounded FP32 inputs (this goes in the CSV)
//   rowsum : second, untimed run with V = ones; every output must equal 1 (C4)
struct CheckResult {
    ErrorStats main, plain, rowsum;
    bool correct = false;
};

// fwd(const AttnInputs&, std::vector<float>& O) runs the stage once and fills O (size elems()).
template <class Fwd>
CheckResult check_stage(const AttnShape& s, std::uint64_t seed, InputCase c, Precision p, Fwd&& fwd) {
    CheckResult res;
    const std::vector<RowId> rows = select_check_rows(s, seed);

    AttnInputs in = make_inputs(s, seed, c);
    std::vector<float> O(s.elems(), 0.0f);
    fwd(in, O);

    std::vector<double> ref;
    attention_reference(in.Q.data(), in.K.data(), in.V.data(), s, rows, ref);
    res.plain = compare_rows(O.data(), s, rows, ref);
    if (c == InputCase::LargeLogits && p != Precision::FP32) {
        AttnInputs rounded = in;
        round_inputs(rounded, p);
        attention_reference(rounded.Q.data(), rounded.K.data(), rounded.V.data(), s, rows, ref);
        res.main = compare_rows(O.data(), s, rows, ref);
    } else {
        res.main = res.plain;
    }

    AttnInputs ones = make_inputs(s, seed, c, /*v_ones=*/true);
    std::vector<float> O1(s.elems(), 0.0f);
    fwd(ones, O1);
    std::vector<double> ref1(rows.size() * static_cast<std::size_t>(s.d), 1.0);
    res.rowsum = compare_rows(O1.data(), s, rows, ref1);

    res.correct = passes(res.main, p) && passes(res.rowsum, p);
    return res;
}

// --------------------------------------------------------------- model cost
inline double model_flops(const AttnShape& s) {  // two GEMMs: 4·BH·N²·d (softmax not counted)
    return 4.0 * s.bh * static_cast<double>(s.n) * static_cast<double>(s.n) * s.d;
}
inline double alg_bytes(const AttnShape& s) {    // Q, K, V, O once each: 16·BH·N·d
    return 4.0 * 4.0 * s.bh * static_cast<double>(s.n) * s.d;
}

}  // namespace w04
