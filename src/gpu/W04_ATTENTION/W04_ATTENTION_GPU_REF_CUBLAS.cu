// -----------------------------------------------------------------------------
// W04 ATTENTION — GPU REF_CUBLAS: vendor unfused ceiling (not a stage)
//
//   1. S = (1/√d)·Q·Kᵀ   cublasSgemmStridedBatched, one GEMM per head
//   2. P = softmax(S)    the S2a warp-per-row kernel (w04_softmax_warp.cuh)
//   3. O = P·V           cublasSgemmStridedBatched
//
// Same data flow as S2a (S/P materialized in DRAM), so S2a's time as a % of this
// one measures the quality of the hand-written GEMMs only.
//
// Row-major in a column-major library: a row-major r×c matrix X is the
// column-major c×r matrix Xᵀ. Hence
//   Sᵀ = K·Qᵀ  → gemm(T, N, m=N, n=N, k=d, A=K lda=d, B=Q ldb=d, C=S ldc=N)
//   Oᵀ = Vᵀ·Pᵀ → gemm(N, N, m=d, n=N, k=N, A=V lda=d, B=P ldb=N, C=O ldc=d)
//
// Math mode is CUBLAS_DEFAULT_MATH: FP32 GEMMs without TF32 Tensor Cores
// (TF32 needs CUBLAS_TF32_TENSOR_OP_MATH). A TF32 result would also fail the
// 1e-4 FP32 tolerance (TF32 has a 10-bit mantissa), so the correctness test
// guards this.
//
// Known limitations:
//   * the peak-memory figure includes cuBLAS's own workspace
//   * cuBLAS picks its kernels by heuristic; they may differ between N
// -----------------------------------------------------------------------------
#include <cublas_v2.h>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

#include "w04_gpu_stage.cuh"
#include "w04_softmax_warp.cuh"

#define W04_CUBLAS_CHECK(call)                                                             \
    do {                                                                                   \
        const cublasStatus_t st__ = (call);                                                \
        if (st__ != CUBLAS_STATUS_SUCCESS) {                                               \
            std::fprintf(stderr, "cuBLAS error %d (%s) at %s:%d\n", static_cast<int>(st__), \
                         #call, __FILE__, __LINE__);                                       \
            std::exit(EXIT_FAILURE);                                                       \
        }                                                                                  \
    } while (0)

namespace {

class StageCublas {
public:
    StageCublas(const w04::AttnShape& s, const w04::BenchArgs&) {
        W04_CUDA_CHECK(cudaMalloc(&S_, static_cast<std::size_t>(s.bh) * s.n * s.n * sizeof(float)));
        W04_CUBLAS_CHECK(cublasCreate(&handle_));
        W04_CUBLAS_CHECK(cublasSetMathMode(handle_, CUBLAS_DEFAULT_MATH));
        W04_CUBLAS_CHECK(cublasSetStream(handle_, 0));
        int ver = 0;
        W04_CUBLAS_CHECK(cublasGetVersion(handle_, &ver));
        std::printf("cublas  : version %d, CUBLAS_DEFAULT_MATH (FP32, no TF32)\n", ver);
    }
    ~StageCublas() {
        cublasDestroy(handle_);
        cudaFree(S_);
    }
    StageCublas(const StageCublas&) = delete;
    StageCublas& operator=(const StageCublas&) = delete;

    int phases() const { return 3; }
    const char* phase_name(int p) const { return p == 0 ? "qk" : p == 1 ? "softmax" : "pv"; }

    void launch(int p, const w04::DeviceAttn& io) {
        const int N = static_cast<int>(io.s.n), d = io.s.d, bh = io.s.bh;
        const long long strideNd = static_cast<long long>(N) * d;
        const long long strideNN = static_cast<long long>(N) * N;
        if (p == 0) {
            const float alpha = 1.0f / std::sqrt(static_cast<float>(d)), beta = 0.0f;
            W04_CUBLAS_CHECK(cublasSgemmStridedBatched(handle_, CUBLAS_OP_T, CUBLAS_OP_N, N, N, d, &alpha,
                                                       io.K, d, strideNd, io.Q, d, strideNd, &beta, S_, N,
                                                       strideNN, bh));
        } else if (p == 1) {
            w04::launch_softmax_rows_warp(S_, static_cast<std::size_t>(bh) * N, N);
        } else {
            const float alpha = 1.0f, beta = 0.0f;
            W04_CUBLAS_CHECK(cublasSgemmStridedBatched(handle_, CUBLAS_OP_N, CUBLAS_OP_N, d, N, N, &alpha, io.V,
                                                       d, strideNd, S_, N, strideNN, &beta, io.O, d, strideNd,
                                                       bh));
        }
    }

private:
    float* S_ = nullptr;
    cublasHandle_t handle_ = nullptr;
};

}  // namespace

int main(int argc, char** argv) {
    w04::GpuStageSpec spec;
    spec.meta = {"W04_ATTENTION_GPU_REF_CUBLAS", "REF", "CUBLAS", "GPU", w04::Precision::FP32};
    return w04::run_gpu_stage<StageCublas>(argc, argv, spec);
}
