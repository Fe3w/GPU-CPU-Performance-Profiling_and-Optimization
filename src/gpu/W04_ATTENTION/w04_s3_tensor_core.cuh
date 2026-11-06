// -----------------------------------------------------------------------------
// W04 ATTENTION — GPU S3 HWACC TENSOR CORE: shared kernel template (P4)
//
// Change versus S2b: both inner GEMMs (S = Q·Kᵀ and O += P̃·V) run on the
// Tensor Cores through WMMA, with FP32 accumulators. The online softmax and O
// stay in FP32. Two precisions share this one kernel (decision C22):
//
//   PrecFP16  Q, K, V converted to half once by an untimed prepare step
//             (convert_qkv_half, timed separately: CSV prep_ms); WMMA
//             m16n16k16; P̃ converted to half before P̃·V
//   PrecTF32  FP32 inputs; WMMA m16n16k8 with precision::tf32; every A/B
//             fragment rounded with __float_to_tf32 (round to nearest, the
//             same rounding as the host reference, decision C25)
//
// Warp layout (FlashAttention-2 split of Q): W warps per block (P5, decision
// C27: W divides Br/16, so W ≤ Br/16). Warp w owns SPW = Br/(16W) strips of 16
// query rows, rows 16·(w·SPW + t) … +15 for t < SPW, for the whole kernel, so
// the softmax of a row never leaves its warp and the steps between Q·Kᵀ and
// P̃·V need only __syncwarp(). With W = Br/16 (SPW = 1) this is the P4 kernel
// unchanged. With SPW > 1 the warp runs the per-tile steps below once per
// strip, reusing one Sw/Pw strip; Q fragments, m, l and O are kept in
// registers for every strip (more registers, less shared memory per row).
// W > Br/16 would need warps to split Bc as well (a cross-warp max and sum),
// which this kernel does not do.
//
//   prologue : Q_i staged through shared memory into register fragments
//              (SPW·d/WK matrix_a fragments per warp), kept for the whole loop
//   per K/V tile j:
//     load K_j, V_j                                           __syncthreads (A)
//     for each strip of the warp:
//     S strip (16×Bc) = Q_w·K_jᵀ   Bc/16 accumulators, K as col_major matrix_b
//     store S strip -> Sw (FP32, shared)                       __syncwarp
//     online softmax in FP32: lane -> row lane%16, half lane/16 (Bc/2 columns,
//       float4 reads), max and sum combined over the lane pair by one shuffle;
//       P̃ -> Pw (half, FP16) or in place in Sw (TF32)            __syncwarp
//     PV (16×d) = P̃_w·V_j          4 accumulators, V as row_major matrix_b
//     store PV -> Sw (scratch)                                 __syncwarp
//     O = α·O + PV, per lane in registers (32 floats: row lane%16, columns
//       32·(lane/16) …), so O never depends on the WMMA accumulator layout,
//       which is not specified                       (SPW > 1: __syncwarp)
//                                                              __syncthreads (B)
//   epilogue : O / l -> global memory (the only write)
//
// Shared memory: K_j, V_j (Bc × (64 + pad) elements each; pad 8 halves / 4
// floats keeps ldm legal and spreads banks), Sw (16 × 68 floats per warp, wide
// enough for a 16×64 PV tile), Pw (16 × (Bc + 8) halves per warp, FP16 only).
// Q is staged through the same area before the loop. 64×64: FP16 45,056 B
// (default 48 KiB limit), TF32 52,224 B (opt-in).
//
// Known limitations:
//   * S and P̃ make a round trip through shared memory every tile, and so does
//     the PV result (WMMA fragments are opaque)
//   * no double buffering or prefetch of the next K/V tile; 2 barriers per tile
//   * TF32 at 64×64 needs the opt-in limit and fits 1 block (4 warps) per SM
//   * l is summed from the FP32 p, while P̃·V uses the rounded p (FP16/TF32)
//   * accurate expf (no fast-math), as in S2b
//   * d ≤ 64 and d % 8 == 0 (16-byte vector loads); shorter d zero-padded to 64
//   * Br, Bc ∈ {16, 32, 64} with (Br, W) ∈ {(16, 1), (32, 2), (64, 4)} (the
//     P4 tiles, one strip per warp) and (64, 2) (two strips per warp): 12
//     compile-time instantiations per precision
// -----------------------------------------------------------------------------
#pragma once

#include <cuda_fp16.h>
#include <mma.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <type_traits>

#include "w04_gpu_stage.cuh"

namespace w04s3 {

using namespace nvcuda;

constexpr int kDmax = 64;             // head dimension the tiles are sized for
constexpr int kSld = kDmax + 4;       // FP32 row stride of the per-warp S / PV strip
constexpr unsigned kFull = 0xffffffffu;

struct PrecFP16 {
    using elem = __half;                       // element type of Q, K, V in global and shared memory
    using frag_t = __half;                     // WMMA fragment type
    static constexpr int WK = 16;              // m16n16k16
    static constexpr int kPad = 8;             // elements of row padding
    static constexpr bool kSeparateP = true;   // P̃ in its own half buffer
    template <class Frag>
    __device__ static void round(Frag&) {}
    // nvcc infers the shared state space for the FP16 WMMA loads (LDS / LDSM in SASS).
    template <class Frag>
    __device__ static void load(Frag& f, const elem* p, unsigned ld) { wmma::load_matrix_sync(f, p, ld); }
};

struct PrecTF32 {
    using elem = float;
    using frag_t = wmma::precision::tf32;
    static constexpr int WK = 8;               // m16n16k8
    static constexpr int kPad = 4;
    static constexpr bool kSeparateP = false;  // P̃ overwrites S in place
    template <class Frag>
    __device__ static void round(Frag& f) {
#pragma unroll
        for (int t = 0; t < f.num_elements; ++t) f.x[t] = wmma::__float_to_tf32(f.x[t]);
    }
    // wmma::load_matrix_sync for precision::tf32 emits the generic-address PTX
    // form (wmma.load.*.tf32 without .shared), which ptxas compiles to LD.E: the
    // fragment loads from shared memory then run through the global load path.
    // The loads below name the .shared state space explicitly; the loaded bits
    // and the fragment layout are those of load_matrix_sync (same PTX
    // instruction, same operands), so results are unchanged. p must point to
    // shared memory.
    using frag_a_row = wmma::fragment<wmma::matrix_a, 16, 16, 8, frag_t, wmma::row_major>;
    using frag_b_col = wmma::fragment<wmma::matrix_b, 16, 16, 8, frag_t, wmma::col_major>;
    using frag_b_row = wmma::fragment<wmma::matrix_b, 16, 16, 8, frag_t, wmma::row_major>;
#define W04_TF32_LDS(AB, LAY)                                                                        \
    static_assert(sizeof(f.x) / sizeof(f.x[0]) == 4, "m16n16k8 tf32 A/B fragments hold 4 elements"); \
    unsigned u0, u1, u2, u3;                                                                         \
    const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));                           \
    asm volatile("wmma.load." AB ".sync.aligned." LAY ".m16n16k8.shared.tf32 {%0, %1, %2, %3}, [%4], %5;" \
                 : "=r"(u0), "=r"(u1), "=r"(u2), "=r"(u3)                                             \
                 : "r"(a), "r"(ld)                                                                   \
                 : "memory");                                                                        \
    f.x[0] = __uint_as_float(u0); f.x[1] = __uint_as_float(u1);                                      \
    f.x[2] = __uint_as_float(u2); f.x[3] = __uint_as_float(u3);
    __device__ static void load(frag_a_row& f, const float* p, unsigned ld) { W04_TF32_LDS("a", "row") }
    __device__ static void load(frag_b_col& f, const float* p, unsigned ld) { W04_TF32_LDS("b", "col") }
    __device__ static void load(frag_b_row& f, const float* p, unsigned ld) { W04_TF32_LDS("b", "row") }
#undef W04_TF32_LDS
};

template <class P, int Br, int Bc, int W>
struct Layout {
    static_assert(Br % 16 == 0 && Bc % 16 == 0, "Br and Bc must be multiples of 16");
    static_assert(W >= 1 && (Br / 16) % W == 0, "W must divide Br/16 (one or more 16-row strips per warp)");
    using elem = typename P::elem;
    static constexpr int kWarps = W;
    static constexpr int kStrips = Br / (16 * W);  // 16-row strips per warp (SPW)
    static constexpr int kThreads = 32 * kWarps;
    static constexpr int kKVld = kDmax + P::kPad;  // row stride of K, V (and staged Q), in elements
    static constexpr int kPld = Bc + 8;            // row stride of the half P̃ strip
    static constexpr std::size_t kv_bytes = 2ull * Bc * kKVld * sizeof(elem);
    static constexpr std::size_t s_bytes = static_cast<std::size_t>(kWarps) * 16 * kSld * sizeof(float);
    static constexpr std::size_t p_bytes =
        P::kSeparateP ? static_cast<std::size_t>(kWarps) * 16 * kPld * sizeof(__half) : 0;
    static constexpr std::size_t q_bytes = static_cast<std::size_t>(Br) * kKVld * sizeof(elem);
    static constexpr std::size_t loop_bytes = kv_bytes + s_bytes + p_bytes;
    static constexpr std::size_t bytes = loop_bytes > q_bytes ? loop_bytes : q_bytes;
};

// R rows × 64 columns of a [N][d] head into shared memory (row stride ld);
// rows ≥ N and columns ≥ d are zero. 16-byte chunks: d % 8 == 0 keeps every
// chunk either fully inside or fully outside [0, d) and 16-byte aligned.
template <class elem, int R, int ld>
__device__ __forceinline__ void load_rows(elem* dst, const elem* __restrict__ src, int row0, int N, int d,
                                          int tid, int nthreads) {
    constexpr int kVec = 16 / static_cast<int>(sizeof(elem));
    constexpr int kChunks = kDmax / kVec;
    for (int e = tid; e < R * kChunks; e += nthreads) {
        const int r = e / kChunks, c = (e % kChunks) * kVec, i = row0 + r;
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (i < N && c < d) v = *reinterpret_cast<const uint4*>(src + static_cast<std::size_t>(i) * d + c);
        *reinterpret_cast<uint4*>(dst + r * ld + c) = v;
    }
}

template <class P, int Br, int Bc, int W>
__global__ void __launch_bounds__(Layout<P, Br, Bc, W>::kThreads)
flash_fwd_tc(const typename P::elem* __restrict__ Q, const typename P::elem* __restrict__ K,
             const typename P::elem* __restrict__ V, float* __restrict__ O, int N, int d, float scale) {
    using L = Layout<P, Br, Bc, W>;
    constexpr int SPW = L::kStrips;
    using elem = typename P::elem;
    using frag_t = typename P::frag_t;
    constexpr int WK = P::WK, kKVld = L::kKVld, kPld = L::kPld;
    constexpr int kHalfCols = Bc / 2;  // softmax columns per lane

    extern __shared__ __align__(128) unsigned char smem_raw[];
    elem* Ks = reinterpret_cast<elem*>(smem_raw);
    elem* Vs = Ks + Bc * kKVld;
    float* Sall = reinterpret_cast<float*>(smem_raw + L::kv_bytes);
    __half* Pall = reinterpret_cast<__half*>(smem_raw + L::kv_bytes + L::s_bytes);
    elem* Qstage = reinterpret_cast<elem*>(smem_raw);  // before the loop only

    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int r = lane & 15, hsel = lane >> 4;  // softmax / O row and column half of this lane
    const int h = blockIdx.y, i0 = blockIdx.x * Br;
    const std::size_t head = static_cast<std::size_t>(h) * N * d;
    float* Sw = Sall + warp * 16 * kSld;
    __half* Pw = Pall + warp * 16 * kPld;

    // Q_i -> registers (SPW 16×64 strips per warp), staged through shared memory.
    load_rows<elem, Br, kKVld>(Qstage, Q + head, i0, N, d, tid, L::kThreads);
    __syncthreads();
    wmma::fragment<wmma::matrix_a, 16, 16, WK, frag_t, wmma::row_major> qf[SPW][kDmax / WK];
#pragma unroll
    for (int st = 0; st < SPW; ++st)
#pragma unroll
        for (int k = 0; k < kDmax / WK; ++k) {
            P::load(qf[st][k], Qstage + (warp * SPW + st) * 16 * kKVld + k * WK, kKVld);
            P::round(qf[st][k]);
        }
    __syncthreads();  // staging area free for K/V

    float m[SPW], l[SPW];
    float o[SPW][32];
#pragma unroll
    for (int st = 0; st < SPW; ++st) {
        m[st] = -FLT_MAX;
        l[st] = 0.0f;
#pragma unroll
        for (int q = 0; q < 32; ++q) o[st][q] = 0.0f;
    }

    for (int j0 = 0; j0 < N; j0 += Bc) {
        load_rows<elem, Bc, kKVld>(Ks, K + head, j0, N, d, tid, L::kThreads);
        load_rows<elem, Bc, kKVld>(Vs, V + head, j0, N, d, tid, L::kThreads);
        __syncthreads();  // (A) K_j, V_j visible
#pragma unroll
        for (int st = 0; st < SPW; ++st) {  // the warp's strips, one after another
            // S strip = Q_w·K_jᵀ (K row-major in shared memory = Kᵀ column-major).
            {
                wmma::fragment<wmma::accumulator, 16, 16, WK, float> acc[Bc / 16];
#pragma unroll
                for (int n = 0; n < Bc / 16; ++n) wmma::fill_fragment(acc[n], 0.0f);
#pragma unroll
                for (int k = 0; k < kDmax / WK; ++k) {
#pragma unroll
                    for (int n = 0; n < Bc / 16; ++n) {
                        wmma::fragment<wmma::matrix_b, 16, 16, WK, frag_t, wmma::col_major> kf;
                        P::load(kf, Ks + n * 16 * kKVld + k * WK, kKVld);
                        P::round(kf);
                        wmma::mma_sync(acc[n], qf[st][k], kf, acc[n]);
                    }
                }
#pragma unroll
                for (int n = 0; n < Bc / 16; ++n)
                    wmma::store_matrix_sync(Sw + n * 16, acc[n], kSld, wmma::mem_row_major);
            }
            __syncwarp();

            // Online softmax in FP32 on this lane's half row.
            float s[kHalfCols];
            {
                const float* srow = Sw + r * kSld + hsel * kHalfCols;
#pragma unroll
                for (int q = 0; q < kHalfCols; q += 4) {
                    const float4 v = *reinterpret_cast<const float4*>(srow + q);
                    s[q] = v.x; s[q + 1] = v.y; s[q + 2] = v.z; s[q + 3] = v.w;
                }
            }
            const int jbase = j0 + hsel * kHalfCols;
            float mt = -FLT_MAX;
#pragma unroll
            for (int q = 0; q < kHalfCols; ++q) {
                s[q] *= scale;
                if (jbase + q < N) mt = fmaxf(mt, s[q]);
            }
            mt = fmaxf(mt, __shfl_xor_sync(kFull, mt, 16));
            const float m_new = fmaxf(m[st], mt);
            float rs = 0.0f;
#pragma unroll
            for (int q = 0; q < kHalfCols; ++q) {
                const float p = (jbase + q < N) ? expf(s[q] - m_new) : 0.0f;
                s[q] = p;
                rs += p;
            }
            rs += __shfl_xor_sync(kFull, rs, 16);
            const float alpha = expf(m[st] - m_new);  // 0 on the first tile (m = −FLT_MAX)
            l[st] = alpha * l[st] + rs;
            m[st] = m_new;
            if constexpr (P::kSeparateP) {
                __half* prow = Pw + r * kPld + hsel * kHalfCols;
#pragma unroll
                for (int q = 0; q < kHalfCols; q += 2)
                    *reinterpret_cast<__half2*>(prow + q) = __floats2half2_rn(s[q], s[q + 1]);
            } else {
                float* prow = Sw + r * kSld + hsel * kHalfCols;
#pragma unroll
                for (int q = 0; q < kHalfCols; q += 4)
                    *reinterpret_cast<float4*>(prow + q) = make_float4(s[q], s[q + 1], s[q + 2], s[q + 3]);
            }
            __syncwarp();

            // PV (16×64) = P̃_w·V_j; all P̃ loads finish before the result reuses Sw.
            {
                wmma::fragment<wmma::accumulator, 16, 16, WK, float> acc[kDmax / 16];
#pragma unroll
                for (int t = 0; t < kDmax / 16; ++t) wmma::fill_fragment(acc[t], 0.0f);
#pragma unroll
                for (int k = 0; k < Bc; k += WK) {
                    wmma::fragment<wmma::matrix_a, 16, 16, WK, frag_t, wmma::row_major> pf;
                    if constexpr (P::kSeparateP)
                        P::load(pf, Pw + k, kPld);
                    else
                        P::load(pf, Sw + k, kSld);
                    P::round(pf);
#pragma unroll
                    for (int t = 0; t < kDmax / 16; ++t) {
                        wmma::fragment<wmma::matrix_b, 16, 16, WK, frag_t, wmma::row_major> vf;
                        P::load(vf, Vs + k * kKVld + t * 16, kKVld);
                        P::round(vf);
                        wmma::mma_sync(acc[t], pf, vf, acc[t]);
                    }
                }
                __syncwarp();
#pragma unroll
                for (int t = 0; t < kDmax / 16; ++t)
                    wmma::store_matrix_sync(Sw + t * 16, acc[t], kSld, wmma::mem_row_major);
            }
            __syncwarp();

            // O = α·O + PV for this lane's 32 values.
            {
                const float* prow = Sw + r * kSld + hsel * 32;
#pragma unroll
                for (int q = 0; q < 32; q += 4) {
                    const float4 v = *reinterpret_cast<const float4*>(prow + q);
                    o[st][q] = alpha * o[st][q] + v.x;
                    o[st][q + 1] = alpha * o[st][q + 1] + v.y;
                    o[st][q + 2] = alpha * o[st][q + 2] + v.z;
                    o[st][q + 3] = alpha * o[st][q + 3] + v.w;
                }
            }
            if constexpr (SPW > 1) __syncwarp();  // O update read Sw before the next strip writes it
        }
        __syncthreads();  // (B) K_j, V_j and Sw free for the next tile
    }

    // O_i = O_i / l_i (the only global write).
#pragma unroll
    for (int st = 0; st < SPW; ++st) {
        const int i = i0 + (warp * SPW + st) * 16 + r;
        if (i < N) {
            float* Orow = O + head + static_cast<std::size_t>(i) * d;
#pragma unroll
            for (int q = 0; q < 32; ++q) {
                const int c = hsel * 32 + q;
                if (c < d) Orow[c] = o[st][q] / l[st];
            }
        }
    }
}

// FP32 -> half for Q, K and V (n elements each, n % 4 == 0). Untimed prepare step.
// A template (H = __half) so that it is only compiled into the binary that uses it.
template <class H>
__global__ void convert_qkv_half(const float* __restrict__ Q, const float* __restrict__ K,
                                 const float* __restrict__ V, H* __restrict__ Qh, H* __restrict__ Kh,
                                 H* __restrict__ Vh, std::size_t n4) {
    for (std::size_t e = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; e < n4;
         e += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const float* src[3] = {Q, K, V};
        H* dst[3] = {Qh, Kh, Vh};
#pragma unroll
        for (int t = 0; t < 3; ++t) {
            const float4 v = reinterpret_cast<const float4*>(src[t])[e];
            __half2* out = reinterpret_cast<__half2*>(dst[t]) + 2 * e;  // H = __half
            out[0] = __floats2half2_rn(v.x, v.y);
            out[1] = __floats2half2_rn(v.z, v.w);
        }
    }
}

// ------------------------------------------------------------- host side
template <class P>
struct Instance {
    using Fn = void (*)(const typename P::elem*, const typename P::elem*, const typename P::elem*, float*, int,
                        int, float);
    int br, bc, warps;
    Fn fn;
    std::size_t smem;
    int threads;
};

template <class P, int Br, int Bc, int W>
constexpr Instance<P> make_instance() {
    using L = Layout<P, Br, Bc, W>;
    return Instance<P>{Br, Bc, W, &flash_fwd_tc<P, Br, Bc, W>, L::bytes, L::kThreads};
}

// The P4 tiles (W = Br/16) plus Br = 64 with 2 warps (two strips per warp).
template <class P>
const Instance<P>* find_instance(int br, int bc, int warps) {
    static constexpr Instance<P> kInstances[] = {
        make_instance<P, 16, 16, 1>(), make_instance<P, 16, 32, 1>(), make_instance<P, 16, 64, 1>(),
        make_instance<P, 32, 16, 2>(), make_instance<P, 32, 32, 2>(), make_instance<P, 32, 64, 2>(),
        make_instance<P, 64, 16, 4>(), make_instance<P, 64, 32, 4>(), make_instance<P, 64, 64, 4>(),
        make_instance<P, 64, 16, 2>(), make_instance<P, 64, 32, 2>(), make_instance<P, 64, 64, 2>(),
    };
    for (const Instance<P>& in : kInstances)
        if (in.br == br && in.bc == bc && in.warps == warps) return &in;
    return nullptr;
}

// Why a (Br, Bc, W) cannot run; "" if it can. The sweep records this text for
// skipped configurations.
template <class P>
std::string unsupported_reason(int br, int bc, int warps) {
    auto in_set = [](int v) { return v == 16 || v == 32 || v == 64; };
    if (!in_set(br) || !in_set(bc))
        return "Br=" + std::to_string(br) + " Bc=" + std::to_string(bc) +
               " is not instantiated (Br, Bc in {16, 32, 64})";
    if (warps <= 0 || warps > br / 16)
        return "warps=" + std::to_string(warps) + " > Br/16=" + std::to_string(br / 16) +
               ": the FA2 split needs at least 16 query rows per warp";
    if ((br / 16) % warps != 0)
        return "warps=" + std::to_string(warps) + " does not divide Br/16=" + std::to_string(br / 16);
    const Instance<P>* in = find_instance<P>(br, bc, warps);
    if (in == nullptr) return "warps=" + std::to_string(warps) + " is not instantiated for Br=" + std::to_string(br);
    cudaDeviceProp prop{};
    W04_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (in->smem > prop.sharedMemPerBlockOptin)
        return "shared memory " + std::to_string(in->smem) + " B exceeds the opt-in limit " +
               std::to_string(prop.sharedMemPerBlockOptin) + " B";
    return "";
}

// TF32 stage; the FP16 stage below adds the half buffers and the prepare step.
template <class P>
class StageS3 {
public:
    // The shared-memory limit was checked by validate (unsupported_reason).
    StageS3(const w04::AttnShape&, const w04::BenchArgs& a) : inst_(find_instance<P>(a.br, a.bc, a.warps)) {
        W04_CUDA_CHECK(cudaFuncSetAttribute(inst_->fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(inst_->smem)));
        std::printf("tile    : Br=%d Bc=%d, %d threads (%d warps), %zu B dynamic shared memory per block\n",
                    inst_->br, inst_->bc, inst_->threads, inst_->threads / 32, inst_->smem);
    }

    int phases() const { return 1; }
    const char* phase_name(int) const { return "fused"; }

    void launch(int, const w04::DeviceAttn& io) {
        if constexpr (std::is_same_v<typename P::elem, float>)
            launch_with(io, io.Q, io.K, io.V);
        else
            launch_with(io, q_, k_, v_);
    }

protected:
    void launch_with(const w04::DeviceAttn& io, const typename P::elem* q, const typename P::elem* k,
                     const typename P::elem* v) {
        const int N = static_cast<int>(io.s.n);
        const dim3 grid((N + inst_->br - 1) / inst_->br, io.s.bh);
        inst_->fn<<<grid, inst_->threads, inst_->smem>>>(q, k, v, io.O, N, io.s.d,
                                                         1.0f / std::sqrt(static_cast<float>(io.s.d)));
    }

    const Instance<P>* inst_;
    const typename P::elem* q_ = nullptr;  // FP16 only: converted copies
    const typename P::elem* k_ = nullptr;
    const typename P::elem* v_ = nullptr;
};

// A template (P = PrecFP16) so that the FP16 kernels are only instantiated in
// the FP16 binary.
template <class P = PrecFP16>
class StageS3Half : public StageS3<P> {
public:
    StageS3Half(const w04::AttnShape& s, const w04::BenchArgs& a) : StageS3<P>(s, a), n_(s.elems()) {
        for (__half** p : {&qh_, &kh_, &vh_}) W04_CUDA_CHECK(cudaMalloc(p, n_ * sizeof(__half)));
        this->q_ = qh_;
        this->k_ = kh_;
        this->v_ = vh_;
    }
    ~StageS3Half() {
        cudaFree(qh_);
        cudaFree(kh_);
        cudaFree(vh_);
    }
    StageS3Half(const StageS3Half&) = delete;
    StageS3Half& operator=(const StageS3Half&) = delete;

    const char* prepare_name() const { return "convert Q,K,V FP32->FP16"; }
    void prepare(const w04::DeviceAttn& io) {
        const std::size_t n4 = n_ / 4;
        const unsigned blocks = static_cast<unsigned>(std::min<std::size_t>((n4 + 255) / 256, 4096));
        convert_qkv_half<__half><<<blocks, 256>>>(io.Q, io.K, io.V, qh_, kh_, vh_, n4);
    }

private:
    std::size_t n_;
    __half *qh_ = nullptr, *kh_ = nullptr, *vh_ = nullptr;
};

using StageS3FP16 = StageS3Half<PrecFP16>;
using StageS3TF32 = StageS3<PrecTF32>;

template <class Stage, class P>
int run_s3(int argc, char** argv, const char* stage_id, const char* variant, w04::Precision prec) {
    w04::GpuStageSpec spec;
    spec.meta = {stage_id, "S3_HWACC", variant, "GPU", prec};
    spec.tunable = true;  // defaults: w04_tuned.hpp (decision C29)
    spec.tunable_warps = true;
    spec.native_warps = [](int br) { return br / 16; };  // the P4 mapping, one strip per warp
    spec.validate = [](const w04::BenchArgs& a) -> std::string {
        if (a.d > kDmax || a.d % 8 != 0)
            return "d=" + std::to_string(a.d) + " is not supported (d <= 64 and d % 8 == 0)";
        return unsupported_reason<P>(a.br, a.bc, a.warps);
    };
    return w04::run_gpu_stage<Stage>(argc, argv, spec);
}

}  // namespace w04s3
