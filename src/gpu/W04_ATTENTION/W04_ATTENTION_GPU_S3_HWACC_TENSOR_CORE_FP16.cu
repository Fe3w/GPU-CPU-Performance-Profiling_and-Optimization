// -----------------------------------------------------------------------------
// W04 ATTENTION — GPU S3 HWACC TENSOR CORE, FP16 variant
//
// Change versus S2b: Q·Kᵀ and P̃·V run as WMMA m16n16k16 on the Tensor Cores
// with FP16 inputs and FP32 accumulators; softmax and O stay FP32. Q, K and V
// are converted to half once per input upload by an untimed prepare step
// (timed separately, CSV prep_ms); P̃ is rounded to half before P̃·V.
// Default tile Br = Bc = 64 (4 warps, 45,056 B shared memory).
//
// Kernel, warp layout and known limitations: w04_s3_tensor_core.cuh.
// Extra limitation of this variant: 3 half copies of the inputs
// (+1.5 × |Q| device memory).
// -----------------------------------------------------------------------------
#include "w04_s3_tensor_core.cuh"

int main(int argc, char** argv) {
    return w04s3::run_s3<w04s3::StageS3FP16, w04s3::PrecFP16>(
        argc, argv, "W04_ATTENTION_GPU_S3_HWACC_TENSOR_CORE_FP16", "TENSOR_CORE_FP16", w04::Precision::FP16);
}
