// devinfo.cu — prints the GPU facts the talk uses, straight from the driver.
// Build: nvcc -O2 -o devinfo devinfo.cu      (demo.sh hw does this for you)
#include <cstdio>
#include <cuda_runtime.h>

static int attr(cudaDeviceAttr a, int dev) { int v = 0; cudaDeviceGetAttribute(&v, a, dev); return v; }

int main() {
    int dev = 0;
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) { std::printf("No CUDA device\n"); return 1; }
    std::printf("GPU                         : %s (CC %d.%d)\n", p.name, p.major, p.minor);
    std::printf("SMs                         : %d\n", attr(cudaDevAttrMultiProcessorCount, dev));
    std::printf("Max threads per SM          : %d  (= %d warps)\n",
                attr(cudaDevAttrMaxThreadsPerMultiProcessor, dev),
                attr(cudaDevAttrMaxThreadsPerMultiProcessor, dev) / 32);
    std::printf("Registers per SM (32-bit)   : %d\n", attr(cudaDevAttrMaxRegistersPerMultiprocessor, dev));
    std::printf("Shared memory per SM        : %d KB\n", attr(cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev) / 1024);
    std::printf("L2 cache                    : %d KB\n", attr(cudaDevAttrL2CacheSize, dev) / 1024);
    std::printf("Memory bus width            : %d-bit\n", attr(cudaDevAttrGlobalMemoryBusWidth, dev));
    std::printf("Global memory               : %.1f GB\n", p.totalGlobalMem / 1073741824.0);
    int regs = 56, warpRegs = regs * 32;
    int fit = attr(cudaDevAttrMaxRegistersPerMultiprocessor, dev) / warpRegs;
    int maxw = attr(cudaDevAttrMaxThreadsPerMultiProcessor, dev) / 32;
    std::printf("\nOccupancy example (W03 S3): %d regs/thread -> %d regs/warp -> %d warps fit of %d = %.0f%%\n",
                regs, warpRegs, fit, maxw, 100.0 * fit / maxw);
    return 0;
}
