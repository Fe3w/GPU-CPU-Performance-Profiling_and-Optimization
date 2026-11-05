// -----------------------------------------------------------------------------
// W04 ATTENTION — GPU S2b SWOPT FUSED: one FlashAttention-2-style kernel, FP32
//
// Change versus S2a: the three kernels (QKᵀ → softmax → PV) become one, and the
// N×N matrix S/P is never written to global memory. Each block owns Br query
// rows of one head:
//
//   load Q_i (Br×d) into shared memory once
//   m_i = −FLT_MAX, l_i = 0, O_i = 0                         (registers)
//   for each K/V tile j of Bc rows:
//       load K_j, V_j into shared memory
//       S_ij  = Q_i·K_jᵀ·(1/√d)                              (registers)
//       m_new = max(m_i, rowmax(S_ij))
//       P̃_ij  = exp(S_ij − m_new)
//       l_i   = exp(m_i − m_new)·l_i + rowsum(P̃_ij)
//       O_i   = exp(m_i − m_new)·O_i + P̃_ij·V_j              (P̃ staged in smem)
//       m_i   = m_new
//   O_i = O_i / l_i  → global memory (the only write)
//
// Thread mapping: 256 threads (16×16) for every tile size. Thread (ty, tx) owns
// the S elements at rows i = ty + 16a (a < Br/16) and columns j = tx + 16b
// (b < Bc/16), and the O elements at the same rows and columns c = tx + 16b
// (b < 4). Per step of the d loop a thread loads Br/16 + Bc/16 values from
// shared memory and does (Br/16)·(Bc/16) FMAs (2 FMA per load at 64×64, 1 at
// 32×32; S2a: 0.5). The 16 threads of one row form a half-warp, so row max and
// row sum are reduced with shuffles (no shared memory, no barrier).
//
// Shared memory (dynamic, per instantiation): Qs[Br][65], Ks[Bc][65],
// Vs[Bc][64], Ps[Br][Bc+1]. The +1 pads make the column-wise K reads and the
// two-row Q/P reads of a warp conflict-free. 64×64 needs 66,304 B and uses the
// opt-in limit; 32×32 needs 29,056 B.
//
// Known limitations:
//   * SIMT FP32 only (Tensor Cores are S3)
//   * three __syncthreads() per K/V tile, no double buffering or prefetch of
//     the next K/V tile
//   * P̃ goes through shared memory because a thread holds only Bc/16 columns
//     of its rows but needs all Bc for P̃·V
//   * d ≤ 64 (shorter heads are zero-padded to 64, so d < 64 wastes work)
//   * Br, Bc ∈ {16, 32, 64} (compile-time instantiations)
//   * K and V are re-read from global memory (mostly L2) once per query block
// -----------------------------------------------------------------------------
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "w04_gpu_stage.cuh"

namespace {

constexpr int kDmax = 64;              // head dimension the tiles are sized for
constexpr int kQKStride = kDmax + 1;   // padded row length of Qs and Ks
constexpr int kThreads = 256;          // 16 × 16
constexpr unsigned kFull = 0xffffffffu;

template <int Br, int Bc>
struct Tile {
    static_assert(Br % 16 == 0 && Bc % 16 == 0, "Br and Bc must be multiples of 16");
    static constexpr int TM = Br / 16;
    static constexpr int TN = Bc / 16;
    static constexpr int kPStride = Bc + 1;
    static constexpr std::size_t q_floats = static_cast<std::size_t>(Br) * kQKStride;
    static constexpr std::size_t k_floats = static_cast<std::size_t>(Bc) * kQKStride;
    static constexpr std::size_t v_floats = static_cast<std::size_t>(Bc) * kDmax;
    static constexpr std::size_t p_floats = static_cast<std::size_t>(Br) * kPStride;
    static constexpr std::size_t bytes = (q_floats + k_floats + v_floats + p_floats) * sizeof(float);
};

// Reduction over the 16 threads (one half-warp) that share a row.
__device__ __forceinline__ float row_max16(float v) {
#pragma unroll
    for (int off = 8; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(kFull, v, off, 16));
    return v;
}
__device__ __forceinline__ float row_sum16(float v) {
#pragma unroll
    for (int off = 8; off > 0; off >>= 1) v += __shfl_xor_sync(kFull, v, off, 16);
    return v;
}

template <int Br, int Bc>
__global__ void __launch_bounds__(kThreads)
flash_fwd_fp32(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
               float* __restrict__ O, int N, int d, float scale) {
    using T = Tile<Br, Bc>;
    constexpr int TM = T::TM, TN = T::TN, kPStride = T::kPStride;
    extern __shared__ float smem[];
    float* Qs = smem;
    float* Ks = Qs + T::q_floats;
    float* Vs = Ks + T::k_floats;
    float* Ps = Vs + T::v_floats;

    const int tx = threadIdx.x, ty = threadIdx.y, tid = ty * 16 + tx;
    const int h = blockIdx.y, i0 = blockIdx.x * Br;
    const std::size_t head = static_cast<std::size_t>(h) * N * d;
    const float* Qh = Q + head;
    const float* Kh = K + head;
    const float* Vh = V + head;

    // Q_i → shared memory once (rows ≥ N and columns ≥ d are zero).
    for (int e = tid; e < Br * kDmax; e += kThreads) {
        const int r = e / kDmax, c = e % kDmax, i = i0 + r;
        Qs[r * kQKStride + c] = (i < N && c < d) ? Qh[static_cast<std::size_t>(i) * d + c] : 0.0f;
    }

    float m[TM], l[TM], acc[TM][4];
#pragma unroll
    for (int a = 0; a < TM; ++a) {
        m[a] = -FLT_MAX;
        l[a] = 0.0f;
#pragma unroll
        for (int b = 0; b < 4; ++b) acc[a][b] = 0.0f;
    }

    for (int j0 = 0; j0 < N; j0 += Bc) {
        // K_j, V_j → shared memory (rows ≥ N and columns ≥ d are zero).
        for (int e = tid; e < Bc * kDmax; e += kThreads) {
            const int r = e / kDmax, c = e % kDmax, j = j0 + r;
            const bool ok = j < N && c < d;
            const std::size_t g = static_cast<std::size_t>(j) * d + c;
            Ks[r * kQKStride + c] = ok ? Kh[g] : 0.0f;
            Vs[r * kDmax + c] = ok ? Vh[g] : 0.0f;
        }
        __syncthreads();  // (1) K_j, V_j (and, on the first tile, Q_i) visible

        // S_ij = Q_i·K_jᵀ, accumulated in the same c order as S2a.
        float S[TM][TN];
#pragma unroll
        for (int a = 0; a < TM; ++a)
#pragma unroll
            for (int b = 0; b < TN; ++b) S[a][b] = 0.0f;
#pragma unroll 16
        for (int c = 0; c < kDmax; ++c) {
            float q[TM], k[TN];
#pragma unroll
            for (int a = 0; a < TM; ++a) q[a] = Qs[(ty + 16 * a) * kQKStride + c];
#pragma unroll
            for (int b = 0; b < TN; ++b) k[b] = Ks[(tx + 16 * b) * kQKStride + c];
#pragma unroll
            for (int a = 0; a < TM; ++a)
#pragma unroll
                for (int b = 0; b < TN; ++b) S[a][b] += q[a] * k[b];
        }

        // Online softmax on the tile; P̃ → shared memory.
#pragma unroll
        for (int a = 0; a < TM; ++a) {
            float mt = -FLT_MAX;
#pragma unroll
            for (int b = 0; b < TN; ++b) {
                S[a][b] *= scale;
                if (j0 + tx + 16 * b < N) mt = fmaxf(mt, S[a][b]);
            }
            const float m_new = fmaxf(m[a], row_max16(mt));
            float rs = 0.0f;
#pragma unroll
            for (int b = 0; b < TN; ++b) {
                const float p = (j0 + tx + 16 * b < N) ? expf(S[a][b] - m_new) : 0.0f;
                rs += p;
                Ps[(ty + 16 * a) * kPStride + tx + 16 * b] = p;
            }
            const float alpha = expf(m[a] - m_new);  // 0 on the first tile (m = −FLT_MAX)
            l[a] = alpha * l[a] + row_sum16(rs);
#pragma unroll
            for (int b = 0; b < 4; ++b) acc[a][b] *= alpha;
            m[a] = m_new;
        }
        __syncthreads();  // (2) P̃_ij visible

        // O_i += P̃_ij·V_j
#pragma unroll 8
        for (int jj = 0; jj < Bc; ++jj) {
            float p[TM], v[4];
#pragma unroll
            for (int a = 0; a < TM; ++a) p[a] = Ps[(ty + 16 * a) * kPStride + jj];
#pragma unroll
            for (int b = 0; b < 4; ++b) v[b] = Vs[jj * kDmax + tx + 16 * b];
#pragma unroll
            for (int a = 0; a < TM; ++a)
#pragma unroll
                for (int b = 0; b < 4; ++b) acc[a][b] += p[a] * v[b];
        }
        __syncthreads();  // (3) Ks, Vs, Ps free for the next tile
    }

    // O_i = O_i / l_i (the only global write).
    float* Oh = O + head;
#pragma unroll
    for (int a = 0; a < TM; ++a) {
        const int i = i0 + ty + 16 * a;
        if (i >= N) continue;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const int c = tx + 16 * b;
            if (c < d) Oh[static_cast<std::size_t>(i) * d + c] = acc[a][b] / l[a];
        }
    }
}

using KernelFn = void (*)(const float*, const float*, const float*, float*, int, int, float);

struct Instance {
    int br, bc;
    KernelFn fn;
    std::size_t smem;
};

template <int Br, int Bc>
constexpr Instance make_instance() {
    return Instance{Br, Bc, &flash_fwd_fp32<Br, Bc>, Tile<Br, Bc>::bytes};
}

constexpr Instance kInstances[] = {
    make_instance<16, 16>(), make_instance<16, 32>(), make_instance<16, 64>(),
    make_instance<32, 16>(), make_instance<32, 32>(), make_instance<32, 64>(),
    make_instance<64, 16>(), make_instance<64, 32>(), make_instance<64, 64>(),
};

const Instance* find_instance(int br, int bc) {
    for (const Instance& in : kInstances)
        if (in.br == br && in.bc == bc) return &in;
    return nullptr;
}

class StageS2b {
public:
    // No workspace: the only device buffers are Q, K, V and O (owned by the harness).
    StageS2b(const w04::AttnShape&, const w04::BenchArgs& a) : inst_(find_instance(a.br, a.bc)) {
        cudaDeviceProp prop{};
        W04_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        if (inst_->smem > prop.sharedMemPerBlockOptin) {
            std::fprintf(stderr, "S2b: Br=%d Bc=%d needs %zu B shared memory, opt-in limit is %zu B\n", inst_->br,
                         inst_->bc, inst_->smem, static_cast<std::size_t>(prop.sharedMemPerBlockOptin));
            std::exit(2);
        }
        W04_CUDA_CHECK(cudaFuncSetAttribute(inst_->fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(inst_->smem)));
        std::printf("tile    : Br=%d Bc=%d, %d threads, %zu B dynamic shared memory per block\n", inst_->br,
                    inst_->bc, kThreads, inst_->smem);
    }

    int phases() const { return 1; }
    const char* phase_name(int) const { return "fused"; }

    void launch(int, const w04::DeviceAttn& io) {
        const int N = static_cast<int>(io.s.n);
        const dim3 grid((N + inst_->br - 1) / inst_->br, io.s.bh);
        const dim3 block(16, 16);
        inst_->fn<<<grid, block, inst_->smem>>>(io.Q, io.K, io.V, io.O, N, io.s.d,
                                                1.0f / std::sqrt(static_cast<float>(io.s.d)));
    }

private:
    const Instance* inst_;
};

}  // namespace

int main(int argc, char** argv) {
    w04::GpuStageSpec spec;
    spec.meta = {"W04_ATTENTION_GPU_S2b_SWOPT_FUSED", "S2b_SWOPT", "FUSED", "GPU", w04::Precision::FP32};
    spec.tunable = true;
    spec.default_br = 32;  // decision C18 (provisional until the E7 autotune): fits the 48 KiB default
    spec.default_bc = 32;
    spec.validate = [](const w04::BenchArgs& a) -> std::string {
        if (a.d > kDmax) return "d=" + std::to_string(a.d) + " is not supported (d <= 64)";
        if (!find_instance(a.br, a.bc))
            return "Br=" + std::to_string(a.br) + " Bc=" + std::to_string(a.bc) +
                   " is not instantiated (Br, Bc in {16, 32, 64})";
        return "";
    };
    return w04::run_gpu_stage<StageS2b>(argc, argv, spec);
}
