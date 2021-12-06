#include "algorithm/algorithm.cuh"
#include "algorithm/shapefunctions.cuh"
#include "utils/diagnostics.cuh"
#include <builtin_types.h>
#include <cuda.h>
#include <thrust/device_vector.h>

struct CellAccumulator {
    float m00;
    float m10;
    float m01;
    float m11;
    float m20;
    float m02;

    float intersection;
    float stippleIntersection;

    float voronoiArea;
    float voronoiEmptyArea;

    __host__ __device__ inline CellAccumulator()
        : m00(0.0f)
        , m10(0.0f)
        , m01(0.0f)
        , m11(0.0f)
        , m20(0.0f)
        , m02(0.0f)
        , intersection(0.0f)
        , stippleIntersection(0.0f)
        , voronoiArea(0.0f)
        , voronoiEmptyArea(0.0f) { }
};

__global__ void reduceCellQuantitiesKernel(
    CellAccumulator* __restrict__ accumulators,
    const KernelStipples stipples,
    const KernelCellMaps cellMaps,
    const KernelDensityMaps densityMaps) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int layerIndex = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= cellMaps.width() || y >= cellMaps.height() || layerIndex >= cellMaps.layers()) {
        return;
    }

    const int cellIndex = cellMaps(layerIndex, x, y);
    if (cellIndex < 0) {
        // Occurs for coupled cells from foreign layers.
        return;
    }

    const float density = densityMaps(layerIndex, x, y);

    // Accumulate raw density moments.
    atomicAdd(&accumulators[cellIndex].m00, density);
    atomicAdd(&accumulators[cellIndex].m10, x * density);
    atomicAdd(&accumulators[cellIndex].m01, y * density);
    atomicAdd(&accumulators[cellIndex].m11, x * y * density);
    atomicAdd(&accumulators[cellIndex].m20, x * x * density);
    atomicAdd(&accumulators[cellIndex].m02, y * y * density);

    // Accumulate cell intersections (note that cellIndex == stippleIndex,
    // also note that we operate on a merged voronoi diagram in case of couplings).
    const float distance = sdStipple(make_float2(x, y), stipples[cellIndex]);
    const float inStipple = (distance <= 0.0f) ? 1.0f : 0.0f;
    atomicAdd(&accumulators[cellIndex].intersection, inStipple * density);
    atomicAdd(&accumulators[cellIndex].stippleIntersection, inStipple);

    // Accumulate voronoi area.
    const float DensityEpsilon = 1.0f / 255.0f;
    atomicAdd(&accumulators[cellIndex].voronoiArea, 1.0f);
    if (density < DensityEpsilon) {
        atomicAdd(&accumulators[cellIndex].voronoiEmptyArea, 1.0f);
    }
}

__global__ void mapCellQuantitiesKernel(
    KernelCells cells,
    const CellAccumulator* __restrict__ accumulators,
    const KernelStipples stipples) {
    const int cellIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (cellIndex >= cells.size()) {
        return;
    }

    const CellAccumulator accumulator = accumulators[cellIndex];

    if (accumulator.m00 <= 0.0f) {
        cells[cellIndex].densityArea = 0.0f;
        cells[cellIndex].densityCentroid.x = 0.0f;
        cells[cellIndex].densityCentroid.y = 0.0f;
        cells[cellIndex].densityOrientation = 0.0f;

        // Compute error ellipse and mean squared error from central moments.
        cells[cellIndex].densityMajorAxis = 0.0f;
        cells[cellIndex].densityMinorAxis = 0.0f;
        cells[cellIndex].densityMeanSquaredError = 0.0f;

        cells[cellIndex].densityIntersection = 0.0f;
    } else {
        // Copy density area from raw moment.
        cells[cellIndex].densityArea = accumulator.m00;

        // Compute centroid from raw moments.
        const float cx = accumulator.m10 / accumulator.m00;
        const float cy = accumulator.m01 / accumulator.m00;
        cells[cellIndex].densityCentroid.x = cx + 0.5f;
        cells[cellIndex].densityCentroid.y = cy + 0.5f;

        // Compute central moments from raw moments.
        const float u20 = accumulator.m20 / accumulator.m00 - cx * cx;
        const float u11 = 2.0f * (accumulator.m11 / accumulator.m00 - cx * cy);
        const float u02 = accumulator.m02 / accumulator.m00 - cy * cy;

        // Compute orientation from central moments.
        cells[cellIndex].densityOrientation = atan2(u11, u20 - u02) / 2.0f;

        // Compute error ellipse and mean squared error from central moments.
        cells[cellIndex].densityMajorAxis = sqrtf(8.0f * (u20 + u02 + sqrtf(u11 * u11 + (u20 - u02) * (u20 - u02)))) / 2.0f;
        cells[cellIndex].densityMinorAxis = sqrtf(8.0f * (u20 + u02 - sqrtf(u11 * u11 + (u20 - u02) * (u20 - u02)))) / 2.0f;
        cells[cellIndex].densityMeanSquaredError = u20 + u02;

        // Copy density intersection.
        cells[cellIndex].densityIntersection = accumulator.intersection;
    }

    // Copy voronoi area.
    cells[cellIndex].voronoiArea = accumulator.voronoiArea;
    cells[cellIndex].voronoiEmptyArea = accumulator.voronoiEmptyArea;

    // Copy stipple intersection.
    cells[cellIndex].stippleIntersection = accumulator.stippleIntersection;

    // Compute analytical stipple area and its half increment (estimating the discrediting error).
    const float stippleArea = aStipple(stipples[cellIndex].size, stipples[cellIndex].shape,
        stipples[cellIndex].shapeParameter, stipples[cellIndex].shapeRadius);
    cells[cellIndex].stippleArea = stippleArea;
    cells[cellIndex].stippleAreaSigma = aStipple(stipples[cellIndex].size + 0.5f, stipples[cellIndex].shape,
                                            stipples[cellIndex].shapeParameter, stipples[cellIndex].shapeRadius)
        - stippleArea;

    // Initialize merge cell search.
    AtomicPair pair;
    pair.floats[0] = FLT_MAX;
    pair.ints[1] = -1;
    cells[cellIndex].mergeCell = pair;

    // Copy layer index (note that cellIndex == stippleIndex).
    cells.layerIndex(cellIndex) = stipples.layerIndex(cellIndex);
}

__device__ unsigned long long int atomicMinPair(unsigned long long int* address, float value, int index) {
    AtomicPair pair;
    pair.floats[0] = value;
    pair.ints[1] = index;
    AtomicPair test;
    test.ulong = *address;
    while (test.floats[0] > value) {
        test.ulong = atomicCAS(address, test.ulong, pair.ulong);
    }
    return test.ulong;
}

__global__ void findMergeCellsKernel(
    KernelCells cells,
    const KernelCellMaps cellMaps) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int layerIndex = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= cellMaps.width() || y >= cellMaps.height() || layerIndex >= cellMaps.layers()) {
        return;
    }

    const int cellIndex = cellMaps(layerIndex, x, y);
    if (cellIndex < 0) {
        // Occurs for coupled cells from foreign layers.
        return;
    }

    const float2 centroid = make_float2(cells[cellIndex].densityCentroid.x, cells[cellIndex].densityCentroid.y);
    for (int oy = -1; oy <= 1; ++oy) {
        for (int ox = -1; ox <= 1; ++ox) {
            const int2 q = make_int2(x + ox, y + oy);
            if (q.x != x && q.y != y && q.x >= 0 && q.y >= 0 && q.x < cellMaps.width() && q.y < cellMaps.height()) {
                const int otherCellIndex = cellMaps(layerIndex, q.x, q.y);
                if (otherCellIndex >= 0 && otherCellIndex != cellIndex) {
                    const float2 otherCentroid = make_float2(cells[otherCellIndex].densityCentroid.x, cells[otherCellIndex].densityCentroid.y);
                    atomicMinPair(&(cells[cellIndex].mergeCell.ulong), lengthf(centroid - otherCentroid), otherCellIndex);
                }
            }
        }
    }
}

void collectCells(
    Cells& cells,
    const CellMaps& cellMaps,
    const DensityMaps& densityMaps,
    const Stipples& stipples) {
    assert(stipples.layers() == densityMaps.layers() && "Number of layers must be equal");
    assert(stipples.layers() == cellMaps.layers() && "Number of layers must be equal");
    assert(stipples.layers() == cells.layers() && "Number of layers must be equal");

    thrust::device_vector<CellAccumulator> dAccumulators(stipples.size(), CellAccumulator());

    // Run reduction kernel.
    dim3 blockSizeReduce(8, 8, 8);
    dim3 gridSizeReduce((cellMaps.width() + blockSizeReduce.x - 1) / blockSizeReduce.x,
        (cellMaps.height() + blockSizeReduce.y - 1) / blockSizeReduce.y,
        (cellMaps.layers() + blockSizeReduce.z - 1) / blockSizeReduce.z);
    reduceCellQuantitiesKernel<<<gridSizeReduce, blockSizeReduce>>>(
        thrust::raw_pointer_cast(dAccumulators.data()),
        stipples,
        cellMaps,
        densityMaps);
    cuda_debug_synchronize();

    // Make sure there is enough space for cells.
    if (cells.capacity() < stipples.size()) {
        cells.reserve(stipples.size() * Cells::GrowthFactor);
    }
    cells.setSize(stipples.size());

    // Run mapping kernel.
    int blockSizeMap = 8;
    int gridSizeMap = (cells.size() + blockSizeMap - 1) / blockSizeMap;
    mapCellQuantitiesKernel<<<gridSizeMap, blockSizeMap>>>(
        cells,
        thrust::raw_pointer_cast(dAccumulators.data()),
        stipples);
    cuda_debug_synchronize();

    // Find cells for merging (canidates).
    findMergeCellsKernel<<<gridSizeReduce, blockSizeReduce>>>(
        cells,
        cellMaps);
    cuda_debug_synchronize();
}
