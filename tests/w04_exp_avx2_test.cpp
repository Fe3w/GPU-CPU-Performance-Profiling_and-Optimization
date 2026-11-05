// -----------------------------------------------------------------------------
// W04 ATTENTION — accuracy test for exp256_ps (src/common/avx2_exp.hpp)
//
//   * exhaustive: every float in [kExpLo, 0] (the softmax range, ~1.1e9 values)
//     against std::exp in double; error in ulps of the correctly rounded result
//   * sampled:    every 64th float in (0, kExpHi]
//   * special values: exp(±0) = 1 exactly, NaN -> NaN, x < kExpLo -> +0,
//     -inf -> +0, x > kExpHi -> +inf, +inf -> +inf
// Pass: max error <= 2 ulp on both ranges and every special value exact.
// Prints the max error and its argument, then "Correct : YES|NO".
// -----------------------------------------------------------------------------
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

#include "avx2_exp.hpp"

namespace {

float bits_to_float(std::uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
std::uint32_t float_to_bits(float f) { std::uint32_t u; std::memcpy(&u, &f, 4); return u; }

// Size of one ulp of the float nearest to `ref` (ref >= 0); subnormals: 2^-149.
double ulp_of(double ref) {
    const float rf = static_cast<float>(ref);
    if (rf < std::numeric_limits<float>::min()) return std::ldexp(1.0, -149);
    int e = 0;
    std::frexp(rf, &e);            // rf = m·2^e, m in [0.5, 1)
    return std::ldexp(1.0, e - 24);
}

struct Worst { double ulp = 0.0; float x = 0.0f; float got = 0.0f; double ref = 0.0; std::uint64_t count = 0; bool nonfinite = false; };

// Runs exp256_ps over the float bit patterns [b0, b1] (inclusive) with stride `step`.
Worst scan(std::uint32_t b0, std::uint32_t b1, std::uint32_t step) {
    Worst w;
    const std::int64_t first = b0, last = b1;
    const std::int64_t nblocks = ((last - first) / step) / 8 + 1;
#pragma omp parallel
    {
        Worst lw;
#pragma omp for schedule(static)
        for (std::int64_t blk = 0; blk < nblocks; ++blk) {
            alignas(32) float xs[8], ys[8];
            int lanes = 0;
            for (int k = 0; k < 8; ++k) {
                const std::int64_t b = first + (blk * 8 + k) * static_cast<std::int64_t>(step);
                if (b > last) break;
                xs[k] = bits_to_float(static_cast<std::uint32_t>(b));
                ++lanes;
            }
            for (int k = lanes; k < 8; ++k) xs[k] = 0.0f;
            _mm256_store_ps(ys, w04::exp256_ps(_mm256_load_ps(xs)));
            for (int k = 0; k < lanes; ++k) {
                const double ref = std::exp(static_cast<double>(xs[k]));
                const double err = std::fabs(static_cast<double>(ys[k]) - ref) / ulp_of(ref);
                ++lw.count;
                if (!std::isfinite(ys[k])) lw.nonfinite = true;
                if (err > lw.ulp || !std::isfinite(ys[k])) { lw.ulp = std::isfinite(ys[k]) ? err : INFINITY; lw.x = xs[k]; lw.got = ys[k]; lw.ref = ref; }
            }
        }
#pragma omp critical
        {
            w.count += lw.count;
            w.nonfinite = w.nonfinite || lw.nonfinite;
            if (lw.ulp > w.ulp) { w.ulp = lw.ulp; w.x = lw.x; w.got = lw.got; w.ref = lw.ref; }
        }
    }
    return w;
}

float exp1(float x) {
    alignas(32) float v[8];
    _mm256_store_ps(v, w04::exp256_ps(_mm256_set1_ps(x)));
    return v[0];
}

}  // namespace

int main() {
    const auto t0 = std::chrono::steady_clock::now();
    int fails = 0;
    auto check = [&](bool ok, const char* what) {
        std::printf("  %s  %s\n", ok ? "PASS" : "FAIL", what);
        if (!ok) ++fails;
    };
    constexpr double kMaxUlp = 2.0;

    // Negative floats: bit patterns 0x80000000 (-0) .. bits(kExpLo), increasing magnitude.
    const Worst neg = scan(0x80000000u, float_to_bits(w04::kExpLo), 1);
    std::printf("[kExpLo, 0] exhaustive: %llu values, max %.3f ulp at x=%.9g (got %.9g, ref %.9g)\n",
                static_cast<unsigned long long>(neg.count), neg.ulp, neg.x, neg.got, neg.ref);
    check(!neg.nonfinite && neg.ulp <= kMaxUlp, "max error <= 2 ulp on [kExpLo, 0]");

    const Worst pos = scan(float_to_bits(std::numeric_limits<float>::denorm_min()), float_to_bits(w04::kExpHi), 64);
    std::printf("(0, kExpHi] every 64th: %llu values, max %.3f ulp at x=%.9g (got %.9g, ref %.9g)\n",
                static_cast<unsigned long long>(pos.count), pos.ulp, pos.x, pos.got, pos.ref);
    check(!pos.nonfinite && pos.ulp <= kMaxUlp, "max error <= 2 ulp on (0, kExpHi]");

    const float inf = std::numeric_limits<float>::infinity();
    check(exp1(0.0f) == 1.0f && exp1(-0.0f) == 1.0f, "exp(+-0) == 1 exactly");
    check(std::isnan(exp1(std::numeric_limits<float>::quiet_NaN())), "NaN -> NaN");
    check(float_to_bits(exp1(std::nextafter(w04::kExpLo, -inf))) == 0u && exp1(-1000.0f) == 0.0f, "x < kExpLo -> +0");
    check(float_to_bits(exp1(-inf)) == 0u, "-inf -> +0");
    check(exp1(std::nextafter(w04::kExpHi, inf)) == inf && exp1(inf) == inf, "x > kExpHi, +inf -> +inf");

    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("exp256_ps test: %d failed (%.2f s)\n", fails, secs);
    std::printf("Correct : %s\n", fails == 0 ? "YES" : "NO");
    return fails == 0 ? 0 : 1;
}
