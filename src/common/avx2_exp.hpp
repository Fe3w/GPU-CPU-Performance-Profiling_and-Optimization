// -----------------------------------------------------------------------------
// W04 ATTENTION — 8-lane single-precision exp for AVX2/FMA (used by CPU S3)
//
// exp256_ps(x) ≈ e^x for 8 floats, Cephes-style:
//   1. clamp x to [kExpLo, kExpHi]; NaN is kept (operand order of min/max, below)
//   2. n = round(x · log2 e)
//   3. r = x − n·C1 − n·C2         Cody–Waite: C1 + C2 = ln 2, C1 has few bits, so
//                                  n·C1 is exact and |r| ≤ ln2/2 is accurate
//   4. e^r ≈ 1 + r + r²·P(r)       P: degree-5 Cephes minimax coefficients
//   5. multiply by 2^n, built directly as the float bit pattern (n + 127) << 23
//   Lanes with x < kExpLo return exactly 0 (true value < 1.2e-38; no denormals).
//   Lanes with x > kExpHi return +inf. kExpHi (≈88.376) is below ln(FLT_MAX)
//   (≈88.723) because 2^n is built from bits and n = 128 is not a normal
//   exponent; e^x on (88.376, 88.723] is therefore reported as +inf. Softmax
//   never gets there (all its arguments are ≤ 0).
//
// Accuracy target: ≤ 2 ulp relative error on [kExpLo, 0], the range softmax uses
// (all arguments are s − m ≤ 0). Checked exhaustively by tests/w04_exp_avx2_test.cpp.
// Requires -mavx2 -mfma.
// -----------------------------------------------------------------------------
#pragma once

#include <immintrin.h>

namespace w04 {

constexpr float kExpHi = 88.3762626647949f;   // largest x with 2^n still a normal float
constexpr float kExpLo = -87.3365447504019f;  // ln(FLT_MIN): below this the result is 0

inline __m256 exp256_ps(__m256 x) {
    const __m256 hi = _mm256_set1_ps(kExpHi);
    const __m256 lo = _mm256_set1_ps(kExpLo);
    // _mm256_min_ps(a, b) returns b when either operand is NaN, so with x as the
    // second operand a NaN input stays NaN instead of being clamped to a number.
    const __m256 underflow = _mm256_cmp_ps(x, lo, _CMP_LT_OQ);  // false for NaN
    const __m256 overflow = _mm256_cmp_ps(x, hi, _CMP_GT_OQ);   // false for NaN
    x = _mm256_min_ps(hi, x);
    x = _mm256_max_ps(lo, x);

    const __m256 n = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f)),
                                     _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693359375f), x);   // x − n·C1
    r = _mm256_fnmadd_ps(n, _mm256_set1_ps(-2.12194440e-4f), r);       // − n·C2

    __m256 p = _mm256_set1_ps(1.9875691500e-4f);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.3981999507e-3f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(8.3334519073e-3f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.1665795894e-2f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.6666665459e-1f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(5.0000001201e-1f));
    const __m256 r2 = _mm256_mul_ps(r, r);
    __m256 y = _mm256_fmadd_ps(p, r2, _mm256_add_ps(r, _mm256_set1_ps(1.0f)));

    const __m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
    y = _mm256_mul_ps(y, _mm256_castsi256_ps(e));
    y = _mm256_blendv_ps(y, _mm256_set1_ps(__builtin_huge_valf()), overflow);
    return _mm256_andnot_ps(underflow, y);  // underflow lanes -> +0
}

}  // namespace w04
