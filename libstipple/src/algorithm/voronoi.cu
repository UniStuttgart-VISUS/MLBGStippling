#include "algorithm/algorithm.cuh"
#include "algorithm/shapefunctions.cuh"
#include "utils/colormap.cuh"
#include "utils/diagnostics.cuh"
#include <cuda.h>
#include <thrust/unique.h>

__global__ void linearSearchKernel(
    KernelCellMaps cellMaps,
    const KernelStipples stipples) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int layerIndex = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= cellMaps.width() || y >= cellMaps.height() || layerIndex >= cellMaps.layers()) {
        return;
    }

    const float2 p = make_float2(x, y);
    float minDistance = FLT_MAX;
    int minIndex = 0;
    for (int i = 0; i < stipples.size(); i++) {
        if (stipples.layerIndex(i) == layerIndex) {
            const float distance = sdStipple(p, stipples[i]);
            if (distance < minDistance) {
                minDistance = distance;
                minIndex = i;
            }
        }
    }

    cellMaps(layerIndex, x, y) = minIndex;
}

__global__ void jumpFloodingSeedKernel(
    KernelCellMaps cellMaps,
    const KernelStipples stipples) {
    const int stippleIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (stippleIndex >= stipples.size()) {
        return;
    }

    int px = static_cast<int>(stipples[stippleIndex].center.x - 0.5f);
    int py = static_cast<int>(stipples[stippleIndex].center.y - 0.5f);
    if (px >= 0 && py >= 0 && px < cellMaps.width() && py < cellMaps.height()) {
        int layerIndex = stipples.layerIndex(stippleIndex);
        cellMaps(layerIndex, px, py) = stippleIndex;
        cellMaps(layerIndex + cellMaps.layers(), px, py) = stippleIndex;
    }
}

__launch_bounds__(1024, 2) __global__ void jumpFloodingStepKernel(
    KernelCellMaps cellMaps,
    const KernelStipples stipples,
    const int stepLength,
    const int layerOffsetPing,
    const int layerOffsetPong) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int layerIndex = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= cellMaps.width() || y >= cellMaps.height() || layerIndex >= cellMaps.layers()) {
        return;
    }

    float minDistance = FLT_MAX;
    int minCellIndex = -1;
    for (int oy = -1; oy <= 1; ++oy) {
        for (int ox = -1; ox <= 1; ++ox) {
            const int2 q = make_int2(x + ox * stepLength, y + oy * stepLength);
            if (q.x >= 0 && q.y >= 0 && q.x < cellMaps.width() && q.y < cellMaps.height()) {
                const int cellIndex = static_cast<const KernelCellMaps>(cellMaps)(layerIndex + layerOffsetPing, q.x, q.y);
                if (cellIndex != -1) {
                    const float distance = sdStipple(make_float2(x, y), stipples[cellIndex]);
                    if (distance < minDistance || (distance == minDistance && ox == 0 && oy == 0)) {
                        minDistance = distance;
                        minCellIndex = cellIndex;
                    }
                }
            }
        }
    }

    if (minCellIndex != -1) {
        cellMaps(layerIndex + layerOffsetPong, x, y) = minCellIndex;
    }
}

__global__ void voronoiMergeKernel(
    KernelCellMaps mergedCellMaps,
    const KernelCellMaps cellMaps,
    const KernelStipples stipples) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int layerIndex = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= cellMaps.width() || y >= cellMaps.height() || layerIndex >= cellMaps.layers()) {
        return;
    }

    const float2 p = make_float2(x, y);
    float minDistance = FLT_MAX;
    int minCellIndex = -1;
    int minLayerIndex = -1;
    for (int otherLayerIndex = 0; otherLayerIndex < cellMaps.layers(); ++otherLayerIndex) {
        const int cellIndex = cellMaps(otherLayerIndex, x, y);
        if (cellIndex != -1) {
            const float distance = sdStipple(p, stipples[cellIndex]);
            if (distance < minDistance) {
                minDistance = distance;
                minCellIndex = cellIndex;
                minLayerIndex = stipples.layerIndex(cellIndex);
            }
        }
    }

    mergedCellMaps(layerIndex, x, y) = (minLayerIndex == layerIndex) ? minCellIndex : -1;
}

__global__ void voronoiOuterInitKernel(
    KernelCellMaps outerMaps,
    const KernelCellMaps cellMaps,
    const KernelStipples stipples) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cellMaps.width() || y >= cellMaps.height()) {
        return;
    }

    const float2 p = make_float2(x, y);
    float minDistance = FLT_MAX;
    for (int layerIndex = 0; layerIndex < cellMaps.layers(); ++layerIndex) {
        const int cellIndex = cellMaps(layerIndex, x, y);
        if (cellIndex != -1) {
            const float distance = sdStipple(p, stipples[cellIndex]);
            if (distance < minDistance) {
                minDistance = distance;
            }
        }
    }

    const int outerIndex = y * cellMaps.width() + x;
    outerMaps(0, x, y) = (minDistance > 0.0f) ? outerIndex : -1;
}

__global__ void voronoiOuterFloodKernel(
    KernelCellMaps outerMaps,
    int* __restrict__ changes) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= outerMaps.width() || y >= outerMaps.height()) {
        return;
    }

    int minOuterIndex = outerMaps(0, x, y);
    if (minOuterIndex != -1) {
        for (int oy = -1; oy <= 1; ++oy) {
            for (int ox = -1; ox <= 1; ++ox) {
                const int xx = x + ox;
                const int yy = y + oy;
                if (xx >= 0 && yy >= 0 && xx < outerMaps.width() && yy < outerMaps.height()) {
                    int outerIndex = outerMaps(0, xx, yy);
                    if (outerIndex != -1 && outerIndex < minOuterIndex) {
                        minOuterIndex = outerIndex;
                        *changes = 1;
                    }
                }
            }
        }
        outerMaps(0, x, y) = minOuterIndex;
    }
}

__global__ void voronoiOuterReassignKernel(
    KernelCellMaps outerMaps,
    int* __restrict__ indices,
    int indicesSize) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= outerMaps.width() || y >= outerMaps.height()) {
        return;
    }

    int outerIndex = outerMaps(0, x, y);
    if (outerIndex != -1) {
        for (int newIndex = 0; newIndex < indicesSize; ++newIndex) {
            if (indices[newIndex] == outerIndex) {
                outerMaps(0, x, y) = newIndex;
                break;
            }
        }
    }
}

__device__ int uniqueBucket(int* bucket0, int bucketSize, int index) {
    for (int offset = 0; offset < bucketSize; ++offset) {
        int old = atomicCAS(bucket0 + offset, -1, index);
        if (old == -1 || old == index) {
            // Successfully written or already found.
            return offset;
        }
    }
    // Bucket is full.
    return -1;
}

__global__ void voronoiIntersectNaturalNeighborsKernel(
    int* error,
    int* nnIndexMap,
    float* nnWeightMap,
    const int bucketSize,
    const int kernelHeight,
    const int kernelWidth,
    const KernelCellMaps mergedCellMaps,
    const KernelStipples stipples) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= mergedCellMaps.width() || y >= mergedCellMaps.height()) {
        return;
    }

    const float SampleSize = 1.6f; // Size of stipple +60% to accomodate for unfavorably subpixel-placed stipples.
    const int bucketIndex0 = y * bucketSize * mergedCellMaps.width() + x * bucketSize;

    // Intersect a kernel-sized block of neighbor pixels.
    for (int yy = max(y - kernelHeight, 0); yy < min(y + kernelHeight, static_cast<int>(mergedCellMaps.height())); ++yy) {
        for (int xx = max(x - kernelWidth, 0); xx < min(x + kernelWidth, static_cast<int>(mergedCellMaps.width())); ++xx) {
            const float nnDistance = lengthf(make_float2(x - 0.5f - xx, y - 0.5f - yy)) - SampleSize;
            const int cellIndex = mergedCellMaps(0, xx, yy);
            if (cellIndex != -1) {
                const float distance = sdStipple(make_float2(xx, yy), stipples[cellIndex]);
                if (nnDistance <= distance) {
                    // Update or insert bucket and increment weight (non-normalized).
                    int bucketIndex = uniqueBucket(nnIndexMap + bucketIndex0, bucketSize, cellIndex);
                    if (bucketIndex != -1) {
                        atomicAdd(&nnWeightMap[bucketIndex0 + bucketIndex], 1.0f);
                    } else {
                        *error = 2;
                    }
                }
            } else {
                *error = 1;
            }
        }
    }
}

template <int Mode>
__global__ void voronoiDisplayKernel(
    KernelColorMaps colorMaps,
    const KernelCellMaps cellMaps,
    const KernelStipples stipples) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int layerIndex = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= cellMaps.width() || y >= cellMaps.height() || layerIndex >= cellMaps.layers()) {
        return;
    }

    const int cellIndex = cellMaps(layerIndex, x, y);
    if (cellIndex >= 0) {
        if (Mode == 0) {
            Color color;
            color.argb = static_cast<std::uint32_t>(cellIndex) | 0xFF000000;
            colorMaps(layerIndex, x, y) = color;
        } else if (Mode == 1) {
            colorMaps(layerIndex, x, y) = cellIndexToColor(cellIndex);
        } else if (Mode == 2) {
            const float distance = sdStipple(make_float2(x, y), stipples[cellIndex]);
            colorMaps(layerIndex, x, y) = sdToColor(distance);
        }
    } else {
        colorMaps(layerIndex, x, y) = Color(255, 0, 255);
    }
}

void writeCellMaps(const char* prefix, const CellMaps& cellMaps, const Stipples& stipples) {
    std::vector<Color> distanceMap(cellMaps.width() * cellMaps.height(), Color());
    ColorMaps distanceMaps(cellMaps.layers(), cellMaps.width(), cellMaps.height());
    voronoiDisplay(distanceMaps, cellMaps, stipples, 0);
    distanceMaps.copyToHost([&](auto layerIndex, auto* scan0) {
        if (!scan0) {
            return distanceMap.data();
        }
        char filename[256];
        sprintf(filename, "%s_%d.ppm", prefix, layerIndex);
        FILE* fp = fopen(filename, "wb");
        fprintf(fp, "P6\n%d %d\n255\n", distanceMaps.width(), distanceMaps.height());
        for (int y = 0; y < distanceMaps.height(); ++y) {
            for (int x = 0; x < distanceMaps.width(); ++x) {
                Color color = scan0[y * distanceMaps.width() + x];
                static unsigned char rgb[3];
                rgb[0] = color.r();
                rgb[1] = color.g();
                rgb[2] = color.b();
                fwrite(rgb, 1, 3, fp);
            }
        }
        fclose(fp);
        return distanceMap.data();
    });
}

void voronoi(CellMaps& cellMaps, const Stipples& stipples, VoronoiAlgorithm algorithm) {
    assert(cellMaps.layers() == stipples.layers() && "Number of layers must be equal");
    assert(!stipples.empty() && "Empty list of stipples");

    switch (algorithm) {
    case VoronoiAlgorithm::Search: {
        dim3 blockSize(16, 16, 1);
        dim3 gridSize((cellMaps.width() + blockSize.x - 1) / blockSize.x,
            (cellMaps.height() + blockSize.y - 1) / blockSize.y,
            (cellMaps.layers() + blockSize.z - 1) / blockSize.z);
        linearSearchKernel<<<gridSize, blockSize>>>(cellMaps, stipples);
        break;
    }
    case VoronoiAlgorithm::JumpFlooding: {
        // Ensure we have enough space for ping-ponging buffers.
        if (cellMaps.capacity() < cellMaps.size() * 2) {
            cellMaps.reserve(cellMaps.size() * 2);
        }
        // Place seeds.
        cellMaps.setOnDevice(-1);
        int seedBlockSize = 32;
        int seedGridSize = (stipples.size() + seedBlockSize - 1) / seedBlockSize;
        jumpFloodingSeedKernel<<<seedGridSize, seedBlockSize>>>(cellMaps, stipples);
        cuda_debug_synchronize();
        // Iterate steps.
        dim3 stepBlockSize(32, 32, 1);
        dim3 stepGridSize(
            (cellMaps.width() + stepBlockSize.x - 1) / stepBlockSize.x,
            (cellMaps.height() + stepBlockSize.y - 1) / stepBlockSize.y,
            (cellMaps.layers() + stepBlockSize.z - 1) / stepBlockSize.z);
        const int passes = static_cast<int>(ceil(log2(max(cellMaps.width(), cellMaps.height()))));
        int layerOffsetA = (passes % 2 == 0) ? 0 : cellMaps.layers();
        int layerOffsetB = (passes % 2 == 0) ? cellMaps.layers() : 0;
        for (int step = passes - 1; step >= 0; --step) {
            const int stepLength = 1 << step;
            jumpFloodingStepKernel<<<stepGridSize, stepBlockSize>>>(
                cellMaps, stipples, stepLength, layerOffsetA, layerOffsetB);
            std::swap(layerOffsetA, layerOffsetB);
        }
        break;
    }
    }
    cuda_debug_synchronize();
}

void voronoiMerge(CellMaps& mergedCellMaps, const CellMaps& cellMaps, const Stipples& stipples) {
    assert(mergedCellMaps.layers() == stipples.layers() && "Number of layers must be equal");
    assert(cellMaps.layers() == stipples.layers() && "Number of layers must be equal");
    assert(!stipples.empty() && "Empty list of stipples");

    dim3 blockSize(8, 8, 3);
    dim3 gridSize((cellMaps.width() + blockSize.x - 1) / blockSize.x,
        (cellMaps.height() + blockSize.y - 1) / blockSize.y,
        (cellMaps.layers() + blockSize.z - 1) / blockSize.z);
    voronoiMergeKernel<<<gridSize, blockSize>>>(
        mergedCellMaps,
        cellMaps,
        stipples);
    cuda_debug_synchronize();
}

void voronoiOuter(CellMaps& outerMaps, int& outerIndicesSize, const CellMaps& cellMaps, const Stipples& stipples) {
    assert(outerMaps.layers() > 0 && "Number of layers must be greater than zero");
    assert(cellMaps.layers() == stipples.layers() && "Number of layers must be equal");
    assert(!stipples.empty() && "Empty list of stipples");

    dim3 initBlockSize(8, 8);
    dim3 initGridSize((cellMaps.width() + initBlockSize.x - 1) / initBlockSize.x,
        (cellMaps.height() + initBlockSize.y - 1) / initBlockSize.y);
    voronoiOuterInitKernel<<<initGridSize, initBlockSize>>>(
        outerMaps,
        cellMaps,
        stipples);
    cuda_debug_synchronize();

    int* dChanges = nullptr;
    cuda_unwrap(cudaMalloc(&dChanges, sizeof(int)));

    dim3 floodBlockSize(8, 8);
    dim3 floodGridSize((outerMaps.width() + floodBlockSize.x - 1) / floodBlockSize.x,
        (outerMaps.height() + floodBlockSize.y - 1) / floodBlockSize.y);
    while (true) {
        cuda_unwrap(cudaMemset(dChanges, 0, sizeof(int)));

        voronoiOuterFloodKernel<<<floodGridSize, floodBlockSize>>>(
            outerMaps,
            dChanges);
        cuda_debug_synchronize();

        int changes;
        cuda_unwrap(cudaMemcpy(&changes, dChanges, sizeof(int), cudaMemcpyDeviceToHost));
        if (changes == 0) {
            break;
        }
    }

    cuda_unwrap(cudaFree(dChanges));

    CellMaps indices(1, outerMaps.width(), outerMaps.height());
    int mapSize = outerMaps.width() * outerMaps.height();
    thrust::device_ptr<int> outerMapBegin(outerMaps.data());
    thrust::device_ptr<int> outerMapEnd(outerMaps.data() + mapSize);
    thrust::device_ptr<int> indicesBegin(indices.data());
    thrust::device_ptr<int> indicesEnd(indices.data() + mapSize);

    thrust::copy(thrust::device, outerMapBegin, outerMapEnd, indicesBegin);
    thrust::sort(thrust::device, indicesBegin, indicesEnd);
    indicesEnd = thrust::unique(thrust::device, indicesBegin, indicesEnd);
    outerIndicesSize = indicesEnd - indicesBegin;

    dim3 reassignBlockSize(8, 8);
    dim3 reassignGridSize((outerMaps.width() + reassignBlockSize.x - 1) / reassignBlockSize.x,
        (outerMaps.height() + reassignBlockSize.y - 1) / reassignBlockSize.y);
    voronoiOuterReassignKernel<<<reassignGridSize, reassignBlockSize>>>(
        outerMaps,
        thrust::raw_pointer_cast(indicesBegin),
        outerIndicesSize);
    cuda_debug_synchronize();
}

bool voronoiIntersectNaturalNeighbors(
    Map<int>& nnIndexMap,
    Map<float>& nnWeightMap,
    const int bucketSize,
    const int kernelSize,
    const CellMaps& mergedCellMaps,
    const Stipples& stipples) {

    const int bucketMapSize = bucketSize * mergedCellMaps.width() * mergedCellMaps.height();
    thrust::device_vector<int> dNNIndexMap(bucketMapSize, -1);
    thrust::device_vector<float> dNNWeightMap(bucketMapSize, 0.0f);

    int* dError = nullptr;
    cuda_unwrap(cudaMalloc(&dError, sizeof(int)));
    cuda_unwrap(cudaMemset(dError, 0, sizeof(int)));

    dim3 blockSize(8, 8, 1);
    dim3 gridSize((mergedCellMaps.width() + blockSize.x - 1) / blockSize.x,
        (mergedCellMaps.height() + blockSize.y - 1) / blockSize.y, 1);
    voronoiIntersectNaturalNeighborsKernel<<<gridSize, blockSize>>>(
        dError,
        thrust::raw_pointer_cast(dNNIndexMap.data()),
        thrust::raw_pointer_cast(dNNWeightMap.data()),
        bucketSize,
        kernelSize,
        kernelSize,
        mergedCellMaps,
        stipples);
    cuda_debug_synchronize();

    int error;
    cuda_unwrap(cudaMemcpy(&error, dError, sizeof(int), cudaMemcpyDeviceToHost));
    cuda_unwrap(cudaFree(dError));

    if (error != 0) {
        assert(false && "Error");
        return false;
    }

    nnIndexMap.width = bucketSize * mergedCellMaps.width();
    nnIndexMap.height = mergedCellMaps.height();
    nnIndexMap.pixels.resize(nnIndexMap.width * nnIndexMap.height);
    thrust::copy(dNNIndexMap.begin(), dNNIndexMap.end(), nnIndexMap.pixels.begin());

    nnWeightMap.width = bucketSize * mergedCellMaps.width();
    nnWeightMap.height = mergedCellMaps.height();
    nnWeightMap.pixels.resize(nnWeightMap.width * nnWeightMap.height);
    thrust::copy(dNNWeightMap.begin(), dNNWeightMap.end(), nnWeightMap.pixels.begin());

    // Normalize weights.
    for (int y = 0; y < mergedCellMaps.height(); ++y) {
        for (int x = 0; x < mergedCellMaps.width(); ++x) {
            double sum = 0.0;
            for (int i = 0; i < bucketSize; ++i) {
                sum += nnWeightMap.pixels[y * bucketSize * mergedCellMaps.width() + x * bucketSize + i];
            }
            for (int i = 0; i < bucketSize; ++i) {
                nnWeightMap.pixels[y * bucketSize * mergedCellMaps.width() + x * bucketSize + i] /= sum;
            }
        }
    }

    return true;
}

void voronoiDisplay(ColorMaps& distanceMaps, const CellMaps& cellMaps, const Stipples& stipples, const int displayMode) {
    assert(distanceMaps.layers() == cellMaps.layers() && "Number of layers must be equal");

    dim3 blockSize(8, 8, 1);
    dim3 gridSize((cellMaps.width() + blockSize.x - 1) / blockSize.x,
        (cellMaps.height() + blockSize.y - 1) / blockSize.y,
        (cellMaps.layers() + blockSize.z - 1) / blockSize.z);
    if (displayMode == 0) {
        voronoiDisplayKernel<0><<<gridSize, blockSize>>>(distanceMaps, cellMaps, stipples);
    } else if (displayMode == 1) {
        voronoiDisplayKernel<1><<<gridSize, blockSize>>>(distanceMaps, cellMaps, stipples);
    } else {
        voronoiDisplayKernel<2><<<gridSize, blockSize>>>(distanceMaps, cellMaps, stipples);
    }
    cuda_debug_synchronize();
}
