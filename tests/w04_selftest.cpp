// -----------------------------------------------------------------------------
// W04 ATTENTION — P0 self-test of the shared harness (CPU only)
//
// Checks the double reference against analytic cases, the input generator,
// row selection, error metrics (including NaN/Inf handling), the CLI parser
// and the append-only CSV writer. It also runs two float32 CONTROL
// implementations that exist only here:
//   * stable   (max-subtracted) softmax -> must PASS normal and large_logits
//   * unstable (no max subtraction)     -> must PASS normal, FAIL large_logits
// This proves the correctness test discriminates, and calibrates whether the
// FP32 tolerance (1e-4) is reachable on large logits (risk C2 in the P0 plan).
//
// usage: W04_ATTENTION_SELFTEST [work_dir]   (default build/test_out/selftest)
// -----------------------------------------------------------------------------
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "attention_ref.hpp"
#include "bench_cli.hpp"
#include "bench_timing.hpp"
#include "w04_csv.hpp"

using namespace w04;

static int g_fail = 0, g_pass = 0;
#define CHECK(cond, ...)                                                   \
    do {                                                                   \
        if (cond) { ++g_pass; }                                            \
        else { ++g_fail; std::printf("  FAIL %s:%d: ", __FILE__, __LINE__); \
               std::printf(__VA_ARGS__); std::printf("\n"); }              \
    } while (0)

static std::vector<RowId> all_rows(const AttnShape& s) { return select_check_rows(s, 0, s.n + 1); }

// ------------------------------------------------------------- controls
// float32 attention, one row at a time.
//   Stable   : max-subtracted softmax (correct)
//   Unstable : no max subtraction      (negative control: overflows on large logits)
//   BadNorm  : stable, but l omits the last key (negative control for the V=ones
//              row-sum check: rows sum to slightly more than 1)
enum class Control { Stable, Unstable, BadNorm };
static const char* control_name(Control k) {
    return k == Control::Stable ? "stable" : k == Control::Unstable ? "unstable" : "bad_norm";
}
static void control_fp32(const AttnInputs& in, const AttnShape& s, Control kind, std::vector<float>& O) {
    const bool stable = kind != Control::Unstable;
    const float scale = 1.0f / std::sqrt(static_cast<float>(s.d));
    O.assign(s.elems(), 0.0f);
#pragma omp parallel
    {
        std::vector<float> S(s.n);
#pragma omp for collapse(2)
        for (int h = 0; h < s.bh; ++h)
            for (long i = 0; i < static_cast<long>(s.n); ++i) {
                const float* q = &in.Q[s.index(h, static_cast<std::size_t>(i), 0)];
                float m = -INFINITY;
                for (std::size_t j = 0; j < s.n; ++j) {
                    const float* k = &in.K[s.index(h, j, 0)];
                    float dot = 0.0f;
                    for (int c = 0; c < s.d; ++c) dot += q[c] * k[c];
                    S[j] = dot * scale;
                    m = std::max(m, S[j]);
                }
                if (!stable) m = 0.0f;
                float l = 0.0f;
                float* o = &O[s.index(h, static_cast<std::size_t>(i), 0)];
                for (std::size_t j = 0; j < s.n; ++j) {
                    const float p = std::exp(S[j] - m);
                    if (!(kind == Control::BadNorm && j + 1 == s.n)) l += p;
                    const float* v = &in.V[s.index(h, j, 0)];
                    for (int c = 0; c < s.d; ++c) o[c] += p * v[c];
                }
                for (int c = 0; c < s.d; ++c) o[c] /= l;
            }
    }
}

// ------------------------------------------------------------- sections
static void test_analytic() {
    std::printf("[reference: analytic cases]\n");
    // 1. Hand-computed: d=1, q=1, k=[0,1] -> logits [0,1]; v=[0,1] -> O = e/(1+e)
    {
        AttnShape s{1, 2, 1};
        std::vector<float> Q = {1.0f, 1.0f}, K = {0.0f, 1.0f}, V = {0.0f, 1.0f};
        std::vector<double> O;
        attention_reference(Q.data(), K.data(), V.data(), s, all_rows(s), O);
        const double want = std::exp(1.0) / (1.0 + std::exp(1.0));
        CHECK(std::fabs(O[0] - want) < 1e-15 && std::fabs(O[1] - want) < 1e-15,
              "hand case: got %.17g, want %.17g", O[0], want);
    }
    // 2. Q = 0 -> uniform softmax -> O = mean of V rows (per head)
    // 3. all K rows identical -> uniform softmax -> O = mean of V rows
    for (int variant = 0; variant < 2; ++variant) {
        AttnShape s{2, 96, 16};
        AttnInputs in = make_inputs(s, 7, InputCase::Normal);
        if (variant == 0) std::fill(in.Q.begin(), in.Q.end(), 0.0f);
        else
            for (int h = 0; h < s.bh; ++h)
                for (std::size_t j = 1; j < s.n; ++j)
                    for (int c = 0; c < s.d; ++c) in.K[s.index(h, j, c)] = in.K[s.index(h, 0, c)];
        const auto rows = all_rows(s);
        std::vector<double> O;
        attention_reference(in.Q.data(), in.K.data(), in.V.data(), s, rows, O);
        double worst = 0.0;
        for (std::size_t r = 0; r < rows.size(); ++r)
            for (int c = 0; c < s.d; ++c) {
                double mean = 0.0;
                for (std::size_t j = 0; j < s.n; ++j) mean += in.V[s.index(rows[r].h, j, c)];
                mean /= static_cast<double>(s.n);
                worst = std::max(worst, std::fabs(O[r * s.d + c] - mean));
            }
        CHECK(worst < 1e-14, "%s: O != mean(V), max diff %.3e", variant == 0 ? "Q=0" : "K identical", worst);
    }
    // 4. One dominant key (logit gap 1000) -> O = V[j*] exactly (exp(-1000) underflows)
    {
        AttnShape s{1, 8, 4};
        AttnInputs in = make_inputs(s, 3, InputCase::Normal);
        std::fill(in.Q.begin(), in.Q.end(), 0.0f);
        std::fill(in.K.begin(), in.K.end(), 0.0f);
        const std::size_t jstar = 5;
        for (std::size_t i = 0; i < s.n; ++i) in.Q[s.index(0, i, 0)] = 1.0f;
        in.K[s.index(0, jstar, 0)] = 2000.0f;  // logit = 2000 / sqrt(4) = 1000
        std::vector<double> O;
        attention_reference(in.Q.data(), in.K.data(), in.V.data(), s, all_rows(s), O);
        double worst = 0.0;
        for (std::size_t i = 0; i < s.n; ++i)
            for (int c = 0; c < s.d; ++c)
                worst = std::max(worst, std::fabs(O[i * s.d + c] - in.V[s.index(0, jstar, c)]));
        CHECK(worst == 0.0, "dominant key: O != V[j*], max diff %.3e", worst);
    }
    // 5. V = ones -> O = 1 (row sums of P are 1), also on large logits
    for (InputCase c : {InputCase::Normal, InputCase::LargeLogits}) {
        AttnShape s{2, 128, 64};
        AttnInputs in = make_inputs(s, 11, c, true);
        std::vector<double> O;
        attention_reference(in.Q.data(), in.K.data(), in.V.data(), s, all_rows(s), O);
        double worst = 0.0;
        for (double o : O) worst = std::max(worst, std::fabs(o - 1.0));
        CHECK(worst < 1e-12, "V=ones (%s): max |O-1| = %.3e", to_string(c), worst);
    }
}

static void test_generator() {
    std::printf("[input generator]\n");
    AttnShape s{2, 64, 64};
    const AttnInputs a = make_inputs(s, 42, InputCase::Normal);
    const AttnInputs b = make_inputs(s, 42, InputCase::Normal);
    const AttnInputs c = make_inputs(s, 43, InputCase::Normal);
    const AttnInputs L = make_inputs(s, 42, InputCase::LargeLogits);
    const AttnInputs one = make_inputs(s, 42, InputCase::Normal, true);
    CHECK(a.Q == b.Q && a.K == b.K && a.V == b.V, "same seed must give identical inputs");
    CHECK(a.Q != c.Q && a.K != c.K && a.V != c.V, "different seed must give different inputs");
    CHECK(a.Q != a.K && a.K != a.V, "Q, K, V streams must differ");
    bool scaled = true, in_range = true;
    for (std::size_t i = 0; i < s.elems(); ++i) {
        scaled &= (L.Q[i] == kLargeLogitScale * a.Q[i]) && (L.K[i] == kLargeLogitScale * a.K[i]);
        in_range &= a.Q[i] >= -1.0f && a.Q[i] < 1.0f && a.V[i] >= -1.0f && a.V[i] < 1.0f;
    }
    CHECK(scaled, "large_logits Q,K must equal kLargeLogitScale x normal Q,K exactly");
    CHECK(L.V == a.V, "large_logits must not change V");
    CHECK(in_range, "normal inputs must lie in [-1, 1)");
    bool ones = true;
    for (float v : one.V) ones &= (v == 1.0f);
    CHECK(ones && one.Q == a.Q && one.K == a.K, "v_ones: V = 1, Q and K unchanged");
    // A fixed known value guards against accidental RNG changes (which would silently change all inputs).
    std::uint64_t st = 42;
    const std::uint64_t first = splitmix64(st);
    CHECK(first == 0xBDD732262FEB6E95ull, "splitmix64(42) changed: %016llx", static_cast<unsigned long long>(first));
}

static void test_rows() {
    std::printf("[row selection]\n");
    AttnShape small{3, 2048, 64};
    CHECK(select_check_rows(small, 1).size() == 3u * 2048u, "N<=2048 must check every row");
    AttnShape big{4, 4096, 64};
    const auto rows = select_check_rows(big, 1);
    CHECK(rows.size() == 4u * 64u, "N>2048: 64 rows per head, got %zu", rows.size());
    bool ok = true;
    for (int h = 0; h < big.bh; ++h) {
        std::vector<std::size_t> r;
        for (const auto& x : rows)
            if (x.h == h) r.push_back(x.i);
        ok &= !r.empty() && r.front() == 0 && r.back() == big.n - 1;
        for (std::size_t k = 1; k < r.size(); ++k) ok &= r[k] > r[k - 1] && r[k] < big.n;
    }
    CHECK(ok, "sampled rows: 0 and N-1 per head, sorted, distinct, in range");
}

static void test_metrics() {
    std::printf("[error metrics]\n");
    AttnShape s{1, 1, 3};
    const std::vector<RowId> rows = {{0, 0}};
    const std::vector<double> ref = {1.0, 2.5, -3.0};
    {
        const std::vector<float> O = {1.0f, 2.0f, -3.0f};
        const ErrorStats e = compare_rows(O.data(), s, rows, ref);
        CHECK(e.max_abs_err == 0.5 && std::fabs(e.max_rel_err - 0.5 / 3.0) < 1e-15 && e.n_nonfinite == 0 &&
                  e.n_checked == 3 && e.worst_col == 1,
              "abs=%.3g rel=%.6g worst_col=%d", e.max_abs_err, e.max_rel_err, e.worst_col);
        CHECK(!passes(e, Precision::FP16), "rel 0.167 must fail FP16 tolerance");
    }
    for (float bad : {NAN, INFINITY, -INFINITY}) {
        const std::vector<float> O = {1.0f, bad, -3.0f};
        const ErrorStats e = compare_rows(O.data(), s, rows, ref);
        CHECK(e.n_nonfinite == 1 && std::isinf(e.max_rel_err) && !passes(e, Precision::FP16),
              "non-finite (%g) must be counted and fail", static_cast<double>(bad));
    }
    {
        const std::vector<float> O = {1.0f, 2.5f, -3.0f};
        CHECK(passes(compare_rows(O.data(), s, rows, ref), Precision::FP32), "exact output must pass");
    }
    CHECK(tolerance_for(Precision::FP32) == 1e-4 && tolerance_for(Precision::TF32) == 5e-3 &&
              tolerance_for(Precision::FP16) == 1e-2,
          "tolerances must match plan section 5");
    // Rounding helpers (host side; GPU self-test compares them with the CUDA intrinsics).
    CHECK(round_to_fp16(1.0f) == 1.0f && round_to_fp16(1.0f + 0x1p-11f) == 1.0f &&
              round_to_fp16(1.0f + 0x1p-11f + 0x1p-20f) == 1.0f + 0x1p-10f && round_to_fp16(70000.0f) == INFINITY &&
              round_to_fp16(0x1p-25f) == 0.0f,
          "round_to_fp16 basic cases");
    CHECK(round_to_tf32(1.0f + 0x1p-11f) == 1.0f + 0x1p-10f && round_to_tf32(1.0f + 0x1p-12f) == 1.0f,
          "round_to_tf32 basic cases (ties away from zero)");
}

static void test_controls() {
    std::printf("[float32 controls: discrimination + FP32 calibration]\n");
    int neg_total = 0, neg_failed = 0;
    for (std::size_t n : {std::size_t(128), std::size_t(256), std::size_t(1024)}) {
        const AttnShape s{2, n, 64};
        for (InputCase c : {InputCase::Normal, InputCase::LargeLogits}) {
            for (Control k : {Control::Stable, Control::Unstable, Control::BadNorm}) {
                const CheckResult r = check_stage(s, 42, c, Precision::FP32,
                                                  [&](const AttnInputs& in, std::vector<float>& O) {
                                                      control_fp32(in, s, k, O);
                                                  });
                // Negative controls: the unstable softmax on large logits, and bad_norm everywhere.
                const bool negative = (k == Control::Unstable && c == InputCase::LargeLogits) || k == Control::BadNorm;
                const char* verdict = negative ? (r.correct ? "UNEXPECTED PASS (negative control)"
                                                            : "EXPECTED FAIL (negative control)")
                                               : (r.correct ? "PASS" : "FAIL");
                std::printf("  n=%-5zu %-12s %-8s max_rel=%.3e max_abs=%.3e nonfinite=%-6zu "
                            "rowsum max_abs=%.3e nonfinite=%-6zu -> %s\n",
                            n, to_string(c), control_name(k), r.main.max_rel_err, r.main.max_abs_err,
                            r.main.n_nonfinite, r.rowsum.max_abs_err, r.rowsum.n_nonfinite, verdict);
                if (negative) {
                    ++neg_total;
                    neg_failed += r.correct ? 0 : 1;
                    CHECK(!r.correct, "negative control %s on %s n=%zu PASSED: the test does not discriminate",
                          control_name(k), to_string(c), n);
                } else {
                    CHECK(r.correct, "control %s on %s n=%zu: expected PASS", control_name(k), to_string(c), n);
                }
                if (k == Control::Unstable && c == InputCase::LargeLogits) {
                    // Overflow must be reported as unbounded error, in the main AND the row-sum check.
                    CHECK(r.main.n_nonfinite > 0 && std::isinf(r.main.max_abs_err) && std::isinf(r.main.max_rel_err),
                          "unstable/large_logits: non-finite outputs must give max_abs = max_rel = inf");
                    CHECK(r.rowsum.n_nonfinite > 0 && std::isinf(r.rowsum.max_abs_err) &&
                              !passes(r.rowsum, Precision::FP32),
                          "unstable/large_logits: NaN in the V=ones run must fail the row-sum check");
                }
                if (k == Control::BadNorm)
                    CHECK(!passes(r.rowsum, Precision::FP32) && r.rowsum.n_nonfinite == 0,
                          "bad_norm: the row-sum check by itself must fail (got max_abs=%.3e)", r.rowsum.max_abs_err);
                if (k == Control::Stable)
                    CHECK(passes(r.rowsum, Precision::FP32), "stable: row-sum check must pass");
            }
        }
    }
    std::printf("  negative controls: %d of %d failed as expected\n", neg_failed, neg_total);

    // A stage that returns NaN only in the V=ones run must be caught by the row-sum check alone,
    // which proves the row-sum check reads the output of the stage under test.
    {
        const AttnShape s{1, 64, 64};
        const CheckResult r = check_stage(s, 5, InputCase::Normal, Precision::FP32,
                                          [&](const AttnInputs& in, std::vector<float>& O) {
                                              control_fp32(in, s, Control::Stable, O);
                                              if (in.V[0] == 1.0f && in.V[1] == 1.0f) O[s.index(0, 7, 3)] = NAN;
                                          });
        CHECK(passes(r.main, Precision::FP32) && !passes(r.rowsum, Precision::FP32) && r.rowsum.n_nonfinite == 1 &&
                  std::isinf(r.rowsum.max_abs_err) && !r.correct,
              "NaN injected only into the V=ones run must fail the row-sum check");
    }
}

static void test_cli() {
    std::printf("[CLI]\n");
    const CliSpec fixed{"W04_TEST", false}, tunable{"W04_TEST", true};
    auto parse = [](const CliSpec& sp, std::vector<const char*> v) {
        v.insert(v.begin(), "prog");
        return try_parse_bench_args(static_cast<int>(v.size()), v.data(), sp);
    };
    {
        const CliResult r = parse(fixed, {});
        CHECK(r.ok && r.args.n == 1024 && r.args.bh == 8 && r.args.d == 64 && r.args.iters == 20 &&
                  r.args.reps == 5 && r.args.warmup == 2 && r.args.seed == 42 && r.args.out.empty() &&
                  r.args.input == InputCase::Normal,
              "defaults must match the plan");
    }
    {
        const CliResult r = parse(tunable, {"--n", "2048", "--bh=4", "--d", "64", "--iters", "3", "--reps=2",
                                            "--warmup", "0", "--seed", "7", "--out", "x/y.csv", "--input",
                                            "large_logits", "--br", "32", "--bc=64"});
        CHECK(r.ok && r.args.n == 2048 && r.args.bh == 4 && r.args.iters == 3 && r.args.reps == 2 &&
                  r.args.warmup == 0 && r.args.seed == 7 && r.args.out == "x/y.csv" &&
                  r.args.input == InputCase::LargeLogits && r.args.br == 32 && r.args.bc == 64,
              "full valid argv: %s", r.error.c_str());
    }
    CHECK(parse(fixed, {"--help"}).help, "--help");
    const std::vector<std::vector<const char*>> bad = {
        {"--nn", "5"}, {"--n"}, {"--n", "0"}, {"--n", "-1"}, {"--n", "12abc"}, {"--iters", "0"},
        {"--reps", "x"}, {"--warmup", "-1"}, {"--input", "huge"}, {"--br", "32"}, {"1024"}, {"--out="}};
    for (const auto& v : bad) {
        const CliResult r = parse(fixed, v);
        CHECK(!r.ok && !r.error.empty(), "must reject: %s %s", v[0], v.size() > 1 ? v[1] : "");
    }
    CHECK(!parse(tunable, {"--bc", "0"}).ok, "--bc 0 must be rejected");
}

// Minimal RFC-4180 split (quoted fields, doubled quotes) for checking our own output.
static std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> out(1);
    bool q = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char ch = line[i];
        if (q) {
            if (ch == '"' && i + 1 < line.size() && line[i + 1] == '"') { out.back() += '"'; ++i; }
            else if (ch == '"') q = false;
            else out.back() += ch;
        } else if (ch == '"') q = true;
        else if (ch == ',') out.emplace_back();
        else out.back() += ch;
    }
    return out;
}
static std::size_t col(const std::vector<std::string>& header, const char* name) {
    for (std::size_t i = 0; i < header.size(); ++i)
        if (header[i] == name) return i;
    return header.size();
}
// Bounds-checked field access: a missing column or short row yields "<missing>".
static std::string fld(const std::vector<std::string>& row, const std::vector<std::string>& header, const char* name) {
    const std::size_t i = col(header, name);
    return i < row.size() ? row[i] : std::string("<missing>");
}
// YYYY-MM-DDTHH:MM:SS+hhmm (from iso_timestamp)
static bool is_iso_timestamp(const std::string& t) {
    if (t.size() != 24) return false;
    const char* pat = "dddd-dd-ddTdd:dd:dd+dddd";
    for (std::size_t i = 0; i < t.size(); ++i) {
        if (pat[i] == 'd') { if (t[i] < '0' || t[i] > '9') return false; }
        else if (pat[i] == '+') { if (t[i] != '+' && t[i] != '-') return false; }
        else if (t[i] != pat[i]) return false;
    }
    return true;
}

static void test_csv(const std::string& dir) {
    std::printf("[CSV writer] in %s\n", dir.c_str());
    namespace fs = std::filesystem;
    const std::string path = dir + "/nested/new_dir/test.csv";
    fs::remove_all(dir + "/nested");
    W04Row row;
    row.stage = "S0_TEST"; row.variant = "A,B"; row.device = "CPU"; row.precision = "FP32";
    row.n = 128; row.rep = 1; row.timestamp = iso_timestamp();
    {
        W04CsvWriter w(path);
        w.append(row);
    }
    {
        W04CsvWriter w(path);  // re-open: must append, not truncate, and not repeat the header
        row.rep = 2;
        w.append(row);
    }
    std::ifstream in(path);
    std::vector<std::string> lines;
    for (std::string l; std::getline(in, l);) lines.push_back(l);
    CHECK(lines.size() == 3 && lines[0] == w04_csv_header(), "expected header + 2 rows, got %zu lines", lines.size());
    if (lines.size() == 3) {
        CHECK(lines[1].find(",\"A,B\",") != std::string::npos, "fields with commas must be quoted");
        CHECK(lines[2].find("W04_ATTENTION,S0_TEST,") == 0, "row content");
    }
    const std::vector<std::string> header = split_csv(w04_csv_header());
    CHECK(header.size() == 27, "schema has %zu columns, expected 22 (plan) + 5 (P0)", header.size());
    CHECK(col(header, "timestamp") == 20 && col(header, "git_sha") == 21 && col(header, "compiler_flags") == 19,
          "timestamp/git_sha/compiler_flags column positions");
    if (lines.size() == 3) {
        const std::vector<std::string> f = split_csv(lines[1]);
        CHECK(f.size() == header.size(), "row has %zu fields, header %zu", f.size(), header.size());
        if (f.size() == header.size()) {
            CHECK(fld(f, header, "variant") == "A,B", "quoted field must round-trip");
            CHECK(is_iso_timestamp(fld(f, header, "timestamp")), "timestamp column '%s' is not ISO-8601",
                  fld(f, header, "timestamp").c_str());
            CHECK(fld(f, header, "git_sha") == W04_GIT_SHA, "git_sha column '%s' != compiled-in '%s'",
                  fld(f, header, "git_sha").c_str(), W04_GIT_SHA);
            CHECK(fld(f, header, "compiler_flags") == W04_COMPILER_FLAGS, "compiler_flags column");
        }
    }
    {   // Empty traceability fields must be refused.
        W04CsvWriter w(dir + "/nested/empty_fields.csv");
        W04Row r2 = row;
        r2.timestamp.clear();
        bool threw2 = false;
        try { w.append(r2); } catch (const std::exception&) { threw2 = true; }
        CHECK(threw2, "a row with an empty timestamp must be refused");
    }

    // End to end: the exact path every stage uses (report_and_write).
    {
        BenchArgs a;
        a.n = 64; a.bh = 1; a.d = 64; a.iters = 4; a.reps = 3; a.warmup = 1; a.seed = 9;
        a.input = InputCase::LargeLogits;
        a.out = dir + "/nested/report.csv";
        const AttnShape s = a.shape();
        const CheckResult chk = check_stage(s, a.seed, a.input, Precision::FP32,
                                            [&](const AttnInputs& in, std::vector<float>& O) {
                                                control_fp32(in, s, Control::Stable, O);
                                            });
        const StageMeta meta{"W04_ATTENTION_CPU_S0_TEST_SELFTEST", "S0_TEST", "SELFTEST", "CPU", Precision::FP32};
        std::printf("  -- report_and_write output (sub-check; its Correct line is not the self-test verdict) --\n");
        const int rc = report_and_write(meta, a, {1.0, 1.5, 2.0}, chk, 12345, 7, 32, 64);
        std::ifstream rin(a.out);
        std::vector<std::vector<std::string>> rows;
        std::string l;
        std::getline(rin, l);
        CHECK(l == w04_csv_header(), "report: header");
        while (std::getline(rin, l)) rows.push_back(split_csv(l));
        CHECK(rc == 0 && rows.size() == 3, "report: exit %d, %zu rows (expected 0, 3)", rc, rows.size());
        bool ok = rows.size() == 3;
        for (std::size_t r = 0; ok && r < rows.size(); ++r) {
            const auto& f = rows[r];
            ok &= f.size() == 27 && fld(f, header, "rep") == std::to_string(r + 1) &&
                  fld(f, header, "stage") == "S0_TEST" && fld(f, header, "device") == "CPU" &&
                  fld(f, header, "precision") == "FP32" && fld(f, header, "n") == "64" &&
                  fld(f, header, "iters") == "4" && fld(f, header, "reps") == "3" && fld(f, header, "warmup") == "1" &&
                  fld(f, header, "seed") == "9" && fld(f, header, "input_case") == "large_logits" &&
                  fld(f, header, "threads") == "7" && fld(f, header, "peak_mem_bytes") == "12345" &&
                  fld(f, header, "br") == "32" && fld(f, header, "bc") == "64" && fld(f, header, "correct") == "YES" &&
                  is_iso_timestamp(fld(f, header, "timestamp")) && fld(f, header, "git_sha") == W04_GIT_SHA &&
                  fld(f, header, "compiler_flags") == W04_COMPILER_FLAGS && !fld(f, header, "git_sha").empty();
        }
        CHECK(ok, "report: per-rep rows carry the right values incl. timestamp and git_sha");
        if (rows.size() == 3)
            CHECK(std::stod(fld(rows[0], header, "runtime_ms")) == 1.0 &&
                      std::stod(fld(rows[2], header, "runtime_ms")) == 2.0,
                  "report: runtime_ms per rep");
    }

    const std::string other = dir + "/nested/other.csv";
    { std::ofstream o(other); o << "some,other,header\n"; }
    bool threw = false;
    try { W04CsvWriter w(other); } catch (const std::exception&) { threw = true; }
    CHECK(threw, "appending to a file with a different header must be refused");

    CHECK(run_timestamp().size() == 15 && run_timestamp()[8] == '-', "run timestamp format YYYYMMDD-HHMMSS");
    CHECK(default_out_path("X").rfind("results/raw/W04_ATTENTION/", 0) == 0, "default output folder");
    CHECK(std::string(W04_GIT_SHA) != "unknown", "git SHA must be compiled in (got '%s')", W04_GIT_SHA);
    CHECK(std::string(W04_COMPILER_FLAGS) != "unknown", "compiler flags must be compiled in");
}

static void test_timing() {
    std::printf("[timing protocol]\n");
    BenchArgs a;
    a.warmup = 2; a.iters = 5; a.reps = 3;
    int calls = 0;
    const auto t = time_protocol_cpu(a, [&] { ++calls; });
    CHECK(calls == 2 + 5 * 3 && t.size() == 3, "calls=%d reps=%zu", calls, t.size());
    const MeanSd m = mean_sd({1.0, 2.0, 3.0});
    CHECK(m.mean == 2.0 && m.sd == 1.0, "mean_sd({1,2,3}) = %.3g ± %.3g", m.mean, m.sd);
    CHECK(cpu_peak_rss_bytes() > 0, "peak RSS");
}

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "build/test_out/selftest";
    std::printf("W04_ATTENTION_SELFTEST  git=%s\nflags: %s\n", W04_GIT_SHA, W04_COMPILER_FLAGS);
    const auto t0 = std::chrono::steady_clock::now();
    test_analytic();
    test_generator();
    test_rows();
    test_metrics();
    test_cli();
    test_csv(dir);
    test_timing();
    test_controls();
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("selftest: %d passed, %d failed (%.2f s)\n", g_pass, g_fail, secs);
    std::printf("Correct : %s\n", g_fail == 0 ? "YES" : "NO");
    return g_fail == 0 ? 0 : 1;
}
