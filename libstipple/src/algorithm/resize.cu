#include "algorithm/algorithm.cuh"
#include "algorithm/shapefunctions.cuh"
#include "utils/diagnostics.cuh"
#include "utils/math.cuh"
#include <cuda.h>

__device__ float cubicHermite(float A, float B, float C, float D, float t) {
    const float a = -A / 2.0f + (3.0f * B) / 2.0f - (3.0f * C) / 2.0f + D / 2.0f;
    const float b = A - (5.0f * B) / 2.0f + 2.0f * C - D / 2.0f;
    const float c = -A / 2.0f + C / 2.0f;
    const float d = B;
    return a * t * t * t + b * t * t + c * t + d;
}

__global__ void resizeDensityMapsKernel(
    KernelDensityMaps dstDensityMaps,
    const KernelDensityMaps srcDensityMaps,
    float factor,
    int borderWidth) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int layerIndex = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= dstDensityMaps.width() || y >= dstDensityMaps.height() || layerIndex >= dstDensityMaps.layers()) {
        return;
    }

    const float xSource = static_cast<float>(x) / factor - 0.5f - borderWidth;
    const float ySource = static_cast<float>(y) / factor - 0.5f - borderWidth;
    const int xFloor = __float2int_rd(xSource);
    const int yFloor = __float2int_rd(ySource);
    const float xFract = xSource - xFloor;
    const float yFract = ySource - yFloor;

    float samples[16];
    for (int xOffset = -1; xOffset <= 2; xOffset++) {
        for (int yOffset = -1; yOffset <= 2; yOffset++) {
            float sampleX = clampf(static_cast<float>(xFloor + xOffset), 0.0f, static_cast<float>(srcDensityMaps.width()) - 1.0f);
            float sampleY = clampf(static_cast<float>(yFloor + yOffset), 0.0f, static_cast<float>(srcDensityMaps.height()) - 1.0f);
            samples[yOffset * 4 + xOffset + 5] = srcDensityMaps(layerIndex, sampleX, sampleY);
        }
    }

    const float a = cubicHermite(samples[0], samples[1], samples[2], samples[3], xFract);
    const float b = cubicHermite(samples[4], samples[5], samples[6], samples[7], xFract);
    const float c = cubicHermite(samples[8], samples[9], samples[10], samples[11], xFract);
    const float d = cubicHermite(samples[12], samples[13], samples[14], samples[15], xFract);

    dstDensityMaps(layerIndex, x, y) = clampf(cubicHermite(a, b, c, d, yFract), 0.0f, 1.0f);
}

DensityMaps resizeDensityMaps(const DensityMaps& srcDensityMaps, float factor, int borderWidth) {
    DensityMaps dstDensityMaps(
        srcDensityMaps.layers(),
        (srcDensityMaps.width() + borderWidth * 2) * factor,
        (srcDensityMaps.height() + borderWidth * 2) * factor);

    dim3 blockSize(8, 8, 8);
    dim3 gridSize(
        (dstDensityMaps.width() + blockSize.x - 1) / blockSize.x,
        (dstDensityMaps.height() + blockSize.y - 1) / blockSize.y,
        (dstDensityMaps.layers() + blockSize.z - 1) / blockSize.z);
    resizeDensityMapsKernel<<<gridSize, blockSize>>>(
        dstDensityMaps,
        srcDensityMaps,
        factor,
        static_cast<int>(borderWidth));
    cuda_debug_synchronize();

    return std::move(dstDensityMaps);
}
