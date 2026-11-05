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
// Warp layout (FlashAttention-2 split of Q): Br/16 warps per block; warp w
// owns query rows 16w … 16w+15 of the block for the whole kernel, so the
// softmax of a row never leaves its warp and the steps between Q·Kᵀ and P̃·V
// need only __syncwarp().
//
//   prologue : Q_i staged through shared memory into register fragments
//              (d/WK matrix_a fragments per warp), kept for the whole loop
//   per K/V tile j:
//     load K_j, V_j                                           __syncthreads (A)
//     S strip (16×Bc) = Q_w·K_jᵀ   Bc/16 accumulators, K as col_major matrix_b
//     store S strip -> Sw (FP32, shared)                       __syncwarp
//     online softmax in FP32: lane -> row lane%16, half lane/16 (Bc/2 columns,
//       float4 reads), max and sum combined over the lane pair by one shuffle;
//       P̃ -> Pw (half, FP16) or in place in Sw (TF32)            __syncwarp
//     PV (16×d) = P̃_w·V_j          4 accumulators, V as row_major matrix_b
//     store PV -> Sw (scratch)                                 __syncwarp
//     O = α·O + PV, per lane in registers (32 floats: row lane%16, columns
//       32·(lane/16) …), so O never depends on the WMMA accumulator layout,
//       which is not specified
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
//   * Br, Bc ∈ {16, 32, 64} (compile-time instantiations)
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
};

template <class P, int Br, int Bc>
struct Layout {
    static_assert(Br % 16 == 0 && Bc % 16 == 0, "Br and Bc must be multiples of 16");
    using elem = typename P::elem;
    static constexpr int kWarps = Br / 16;
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

template <class P, int Br, int Bc>
__global__ void __launch_bounds__(Layout<P, Br, Bc>::kThreads)
flash_fwd_tc(const typename P::elem* __restrict__ Q, const typename P::elem* __restrict__ K,
             const typename P::elem* __restrict__ V, float* __restrict__ O, int N, int d, float scale) {
    using L = Layout<P, Br, Bc>;
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

    // Q_i -> registers (one 16×64 strip per warp), staged through shared memory.
    load_rows<elem, Br, kKVld>(Qstage, Q + head, i0, N, d, tid, L::kThreads);
    __syncthreads();
    wmma::fragment<wmma::matrix_a, 16, 16, WK, frag_t, wmma::row_major> qf[kDmax / WK];
#pragma unroll
    for (int k = 0; k < kDmax / WK; ++k) {
        wmma::load_matrix_sync(qf[k], Qstage + warp * 16 * kKVld + k * WK, kKVld);
        P::round(qf[k]);
    }
    __syncthreads();  // staging area free for K/V

    float m = -FLT_MAX, l = 0.0f;
    float o[32];
#pragma unroll
    for (int q = 0; q < 32; ++q) o[q] = 0.0f;

    for (int j0 = 0; j0 < N; j0 += Bc) {
        load_rows<elem, Bc, kKVld>(Ks, K + head, j0, N, d, tid, L::kThreads);
        load_rows<elem, Bc, kKVld>(Vs, V + head, j0, N, d, tid, L::kThreads);
        __syncthreads();  // (A) K_j, V_j visible

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
                    wmma::load_matrix_sync(kf, Ks + n * 16 * kKVld + k * WK, kKVld);
                    P::round(kf);
                    wmma::mma_sync(acc[n], qf[k], kf, acc[n]);
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
        const float m_new = fmaxf(m, mt);
        float rs = 0.0f;
#pragma unroll
        for (int q = 0; q < kHalfCols; ++q) {
            const float p = (jbase + q < N) ? expf(s[q] - m_new) : 0.0f;
            s[q] = p;
            rs += p;
        }
        rs += __shfl_xor_sync(kFull, rs, 16);
        const float alpha = expf(m - m_new);  // 0 on the first tile (m = −FLT_MAX)
        l = alpha * l + rs;
        m = m_new;
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
                    wmma::load_matrix_sync(pf, Pw + k, kPld);
                else
                    wmma::load_matrix_sync(pf, Sw + k, kSld);
                P::round(pf);
#pragma unroll
                for (int t = 0; t < kDmax / 16; ++t) {
                    wmma::fragment<wmma::matrix_b, 16, 16, WK, frag_t, wmma::row_major> vf;
                    wmma::load_matrix_sync(vf, Vs + k * kKVld + t * 16, kKVld);
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
                o[q] = alpha * o[q] + v.x;
                o[q + 1] = alpha * o[q + 1] + v.y;
                o[q + 2] = alpha * o[q + 2] + v.z;
                o[q + 3] = alpha * o[q + 3] + v.w;
            }
        }
        __syncthreads();  // (B) K_j, V_j and Sw free for the next tile
    }

    // O_i = O_i / l_i (the only global write).
    const int i = i0 + warp * 16 + r;
    if (i < N) {
        float* Orow = O + head + static_cast<std::size_t>(i) * d;
#pragma unroll
        for (int q = 0; q < 32; ++q) {
            const int c = hsel * 32 + q;
            if (c < d) Orow[c] = o[q] / l;
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
    int br, bc;
    Fn fn;
    std::size_t smem;
    int threads;
};

template <class P, int Br, int Bc>
constexpr Instance<P> make_instance() {
    return Instance<P>{Br, Bc, &flash_fwd_tc<P, Br, Bc>, Layout<P, Br, Bc>::bytes, Layout<P, Br, Bc>::kThreads};
}

template <class P>
const Instance<P>* find_instance(int br, int bc) {
    static constexpr Instance<P> kInstances[] = {
        make_instance<P, 16, 16>(), make_instance<P, 16, 32>(), make_instance<P, 16, 64>(),
        make_instance<P, 32, 16>(), make_instance<P, 32, 32>(), make_instance<P, 32, 64>(),
        make_instance<P, 64, 16>(), make_instance<P, 64, 32>(), make_instance<P, 64, 64>(),
    };
    for (const Instance<P>& in : kInstances)
        if (in.br == br && in.bc == bc) return &in;
    return nullptr;
}

// TF32 stage; the FP16 stage below adds the half buffers and the prepare step.
template <class P>
class StageS3 {
public:
    StageS3(const w04::AttnShape&, const w04::BenchArgs& a) : inst_(find_instance<P>(a.br, a.bc)) {
        cudaDeviceProp prop{};
        W04_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        if (inst_->smem > prop.sharedMemPerBlockOptin) {
            std::fprintf(stderr, "S3: Br=%d Bc=%d needs %zu B shared memory, opt-in limit is %zu B\n", inst_->br,
                         inst_->bc, inst_->smem, static_cast<std::size_t>(prop.sharedMemPerBlockOptin));
            std::exit(2);
        }
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
    spec.tunable = true;
    spec.default_br = 64;  // decision C24 (same tile as the S2b 64×64 variant; E7 retunes)
    spec.default_bc = 64;
    spec.validate = [](const w04::BenchArgs& a) -> std::string {
        if (a.d > kDmax || a.d % 8 != 0)
            return "d=" + std::to_string(a.d) + " is not supported (d <= 64 and d % 8 == 0)";
        if (!find_instance<P>(a.br, a.bc))
            return "Br=" + std::to_string(a.br) + " Bc=" + std::to_string(a.bc) +
                   " is not instantiated (Br, Bc in {16, 32, 64})";
        return "";
    };
    return w04::run_gpu_stage<Stage>(argc, argv, spec);
}

}  // namespace w04s3
