// -----------------------------------------------------------------------------
// W04 ATTENTION — GPU S3 HWACC TENSOR CORE, TF32 variant
//
// Change versus S2b: Q·Kᵀ and P̃·V run as WMMA m16n16k8 with precision::tf32
// on the Tensor Cores (FP32 inputs, FP32 accumulators); softmax and O stay
// FP32. Every A/B fragment is rounded with __float_to_tf32 (round to nearest),
// which W03-S3 does not do. No prepare step.
// Default tile Br = Bc = 64 (4 warps, 52,224 B shared memory: opt-in limit,
// 1 block per SM).
//
// Kernel, warp layout and known limitations: w04_s3_tensor_core.cuh.
// -----------------------------------------------------------------------------
#include "w04_s3_tensor_core.cuh"

int main(int argc, char** argv) {
    return w04s3::run_s3<w04s3::StageS3TF32, w04s3::PrecTF32>(
        argc, argv, "W04_ATTENTION_GPU_S3_HWACC_TENSOR_CORE_TF32", "TENSOR_CORE_TF32", w04::Precision::TF32);
}
