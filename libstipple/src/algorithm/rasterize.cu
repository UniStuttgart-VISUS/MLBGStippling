#include "algorithm/algorithm.cuh"
#include "algorithm/shapefunctions.cuh"
#include "utils/colormap.cuh"
#include "utils/diagnostics.cuh"
#include "utils/hash.cuh"
#include <cuda.h>
#include <nvfunctional>

// Assume a minimum rounding error based on the size of a byte.
__device__ const float DensityEpsilon = 1.0f / 255.0f;

struct OuterAccumulator {
    float m00;
    float m01;
    float m10;

    float area;
    int transparency;

    __host__ __device__ inline OuterAccumulator()
        : m00(0.0f)
        , m01(0.0f)
        , m10(0.0f)
        , area(0.0f)
        , transparency(0) { }
};

__global__ void reduceOuterKernel(
    OuterAccumulator* __restrict__ accumulators,
    const KernelDensityMaps densityMaps,
    const KernelCellMaps outerMaps) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= outerMaps.width() || y >= outerMaps.height()) {
        return;
    }

    int outerIndex = outerMaps(0, x, y);
    if (outerIndex != -1) {
        // Accumulate density (weighted area).
        float densitySum = 0.0f;
        for (int layerIndex = 0; layerIndex < densityMaps.layers(); ++layerIndex) {
            const int index = outerIndex * outerMaps.layers() + layerIndex;
            const float density = densityMaps(layerIndex, x, y);
            atomicAdd(&accumulators[index].m00, density);
            atomicAdd(&accumulators[index].m10, x * density);
            atomicAdd(&accumulators[index].m01, y * density);
            atomicAdd(&accumulators[index].area, 1.0f);
            densitySum += density;
        }
        // Test if this pixel has partition of unity properties.
        if (densitySum < 1.0f - DensityEpsilon) {
            const int index = outerIndex * outerMaps.layers();
            atomicAdd(&accumulators[index].transparency, 1);
        }
    }
}

template <unsigned int TopK>
__device__ void findMaxK(float* maxValues, int* maxIndices, int size, nvstd::function<float(int)> access) {
    assert(maxValues[0] == -FLT_MAX && maxIndices[0] == -1 && maxValues[TopK - 1] == -FLT_MAX && maxIndices[TopK - 1] == -1 && "Initalize properly");
    for (int index = 0; index < size; ++index) {
        const float value = access(index);
        for (int k = 0; k < TopK; k++) {
            if (maxValues[k] < value) {
                for (int kk = TopK - 1; kk > k; kk--) {
                    maxValues[kk] = maxValues[kk - 1];
                    maxIndices[kk] = maxIndices[kk - 1];
                }
                maxValues[k] = value;
                maxIndices[k] = index;
                break;
            }
        }
    }
}

__device__ void findMaxFuzzy(int* maxIndex, int size, nvstd::function<float(int)> access, nvstd::function<Point(int, int)> centerOf, float minDelta) {
    assert(*maxIndex == -1 && "Initalize properly");
    const int TopK = 2;
    float maxValues[TopK] = { -FLT_MAX, -FLT_MAX };
    int maxIndices[TopK] = { -1, -1 };
    findMaxK<TopK>(maxValues, maxIndices, size, access);
    if (maxIndices[0] != -1 && maxIndices[1] != -1) {
        const Point center = centerOf(maxIndices[0], maxIndices[1]);
        const float random = hash_u2f(center.x, center.y);
        const float delta = fabsf(maxValues[0] - maxValues[1]);
        const float maxInfluence = 0.5f;
        if (random < smoothstepf(minDelta, 0.0f, delta) * maxInfluence) {
            *maxIndex = maxIndices[1];
        } else {
            *maxIndex = maxIndices[0];
        }
    } else {
        *maxIndex = maxIndices[0];
    }
}

// Rasterize stippling to reduce overdraw-induced errors so that:
//  - If density does not from a partition of unity: complement to one with transparency as highest density
//  - Identify highest-density layer as background.
//  - Render from back to front, i.e., highest to lowest density.
//  - Have the expected density take the entire area.
template <bool DrawStipples, bool DrawBackground, int MaxLayers>
__global__ void rasterizeKernel(
    Color* __restrict__ canvas,
    const OuterAccumulator* __restrict__ outerAccumulators,
    const KernelDensityMaps densityMaps,
    const KernelCellMaps cellMaps,
    const KernelCellMaps outerMaps,
    const KernelCells cells,
    const KernelStipples stipples,
    const float renderScale) {
    const int renderWidth = cellMaps.width() * renderScale;
    const int renderHeight = cellMaps.height() * renderScale;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= renderWidth || y >= renderHeight) {
        return;
    }

    const float2 pf = (make_float2(x, y) + 0.5f) / renderScale - 0.5f;
    const int2 p = make_int2(static_cast<int>(rintf(pf.x)), static_cast<int>(rintf(pf.y)));

    // Find closest stipple per layer.
    int minStippleIndices[MaxLayers] = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };
    float stippleDensities[MaxLayers] = { -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX };
    for (int layerIndex = 0; layerIndex < cellMaps.layers(); ++layerIndex) {
        // Fetch index.
        const int cellIndex = cellMaps(layerIndex, p.x, p.y);
        if (cellIndex != -1) {
            minStippleIndices[layerIndex] = cellIndex;
            // Fetch stipple density (non-combined Voronoi diagram).
            const Cell& cell = cells[cellIndex];
            stippleDensities[layerIndex] = cell.densityIntersection / cell.stippleIntersection;
        }
    }

    bool foundBackground = true;
    int outerIndex = outerMaps(0, p.x, p.y);
    if (DrawStipples) {
        // Super sample a bit to allow for anti-aliased stipples.
        for (int oy = -1; oy <= 1; ++oy) {
            for (int ox = -1; ox <= 1; ++ox) {
                const int2 q = make_int2(p.x + ox, p.y + oy);
                if (outerIndex == -1 && q.x >= 0 && q.y >= 0 && q.x < outerMaps.width() && q.y < outerMaps.height()) {
                    outerIndex = outerMaps(0, q.x, q.y);
                }
            }
        }
    }
    if (outerIndex != -1 && outerAccumulators[outerIndex * cellMaps.layers()].transparency == 0) {
        // This is a partition of unity. Thus, trigger background search.
        foundBackground = false;
    }

    Color color(0, 0, 0, 0);
    while (true) {
        // Take (soft delete) layer with the highest density.
        int maxDensityLayerIndex = -1;
        findMaxFuzzy(
            &maxDensityLayerIndex, cellMaps.layers(),
            [&](int layerIndex) { return stippleDensities[layerIndex]; },
            [&](int layerIndexA, int layerIndexB) {
                const auto& stippleA = stipples[minStippleIndices[layerIndexA]];
                const auto& stippleB = stipples[minStippleIndices[layerIndexB]];
                return (stippleA.center + stippleB.center) * (stippleA.size + stippleA.size);
            },
            DensityEpsilon * 0.5f);
        stippleDensities[maxDensityLayerIndex] = -FLT_MAX;

        if (maxDensityLayerIndex == -1) {
            // No more colors found.
            break;
        }
        if (DrawBackground && !foundBackground) {
            // Use (filled) space between stipples as background.
            Color outerColor(0, 0, 0, 0);
            if (outerIndex != -1) {
                int maxOuterDensityLayerIndex = -1;
                findMaxFuzzy(
                    &maxOuterDensityLayerIndex, cellMaps.layers(),
                    [&](int layerIndex) {
                        const auto& outerAccumulator = outerAccumulators[outerIndex * cellMaps.layers() + layerIndex];
                        return outerAccumulator.m00 / outerAccumulator.area;
                    },
                    [&](int layerIndexA, int layerIndexB) {
                        // Compute average centroid from raw moments.
                        const auto& outerAccumulatorA = outerAccumulators[outerIndex * cellMaps.layers() + layerIndexA];
                        const auto& outerAccumulatorB = outerAccumulators[outerIndex * cellMaps.layers() + layerIndexB];
                        const Point cA(outerAccumulatorA.m10 / outerAccumulatorA.m00, outerAccumulatorA.m01 / outerAccumulatorA.m00);
                        const Point cB(outerAccumulatorB.m10 / outerAccumulatorB.m00, outerAccumulatorB.m01 / outerAccumulatorB.m00);
                        return (cA + cB) * 2.0f;
                    },
                    DensityEpsilon * 0.5f);
                if (maxOuterDensityLayerIndex != -1) {
                    outerColor = stipples[minStippleIndices[maxOuterDensityLayerIndex]].color.rgb();
                }
            }
            color = outerColor;
        }
        if (DrawStipples) {
            // Alpha blend as foreground.
            const Stipple& stipple = stipples[minStippleIndices[maxDensityLayerIndex]];
            const float distance = sdStipple(pf, stipple);
            const float radius = sdStippleWidth(pf, 1.0f / renderScale, stipple) * 0.5f;
            const float innerRadius = (DrawBackground && color.a() == 0) ? 0.0f : radius;
            const float alpha = smoothstepf(-radius, innerRadius, -distance);
            if (color.a() == 0) {
                // Avoid blending against uninitialized.
                color = Color(stipple.color.r(), stipple.color.g(), stipple.color.b(), clampf(alpha * 255.0f, 0.0f, 255.0f));
            } else {
                // Alpha blend other stipples.
                color = color.mix(stipple.color.rgb(), clampf(alpha * 255.0f, 0.0f, 255.0f));
            }
        }
        foundBackground = true;
    }

    const int canvasIndex = y * renderWidth + x;
    canvas[canvasIndex] = color;
}

void rasterize(
    Map<Color>& canvas,
    const DensityMaps& densityMaps,
    const CellMaps& cellMaps,
    const CellMaps& outerMaps,
    const int outerIndicesSize,
    const Cells& cells, const Stipples& stipples,
    const float renderScale,
    const RasterMode mode) {
    thrust::device_vector<OuterAccumulator>
        dOuterAccumulators(outerIndicesSize * cellMaps.layers(), OuterAccumulator());
    const int renderWidth = cellMaps.width() * renderScale;
    const int renderHeight = cellMaps.height() * renderScale;
    thrust::device_vector<Color> dCanvas(renderWidth * renderHeight, Color());

    dim3 outerBlockSize(8, 8);
    dim3 outerGridSize((outerMaps.width() + outerBlockSize.x - 1) / outerBlockSize.x,
        (outerMaps.height() + outerBlockSize.y - 1) / outerBlockSize.y);
    reduceOuterKernel<<<outerGridSize, outerBlockSize>>>(
        thrust::raw_pointer_cast(dOuterAccumulators.data()),
        densityMaps,
        outerMaps);
    cuda_debug_synchronize();

    const int MaxLayers = 10;
    assert(cellMaps.layers() <= MaxLayers && "Too many layers");
    dim3 renderBlockSize(8, 8);
    dim3 renderGridSize((renderWidth + renderBlockSize.x - 1) / renderBlockSize.x,
        (renderHeight + renderBlockSize.y - 1) / renderBlockSize.y);
    switch (mode) {
    case RasterMode::StipplesWithBackground:
        rasterizeKernel<true, true, MaxLayers><<<renderGridSize, renderBlockSize>>>(
            thrust::raw_pointer_cast(dCanvas.data()),
            thrust::raw_pointer_cast(dOuterAccumulators.data()),
            densityMaps,
            cellMaps,
            outerMaps,
            cells,
            stipples,
            renderScale);
        break;
    case RasterMode::Stipples:
        rasterizeKernel<true, false, MaxLayers><<<renderGridSize, renderBlockSize>>>(
            thrust::raw_pointer_cast(dCanvas.data()),
            thrust::raw_pointer_cast(dOuterAccumulators.data()),
            densityMaps,
            cellMaps,
            outerMaps,
            cells,
            stipples,
            renderScale);
        break;
    case RasterMode::Background:
        rasterizeKernel<false, true, MaxLayers><<<renderGridSize, renderBlockSize>>>(
            thrust::raw_pointer_cast(dCanvas.data()),
            thrust::raw_pointer_cast(dOuterAccumulators.data()),
            densityMaps,
            cellMaps,
            outerMaps,
            cells,
            stipples,
            renderScale);
        break;
    }
    cuda_debug_synchronize();

    canvas.width = renderWidth;
    canvas.height = renderHeight;
    canvas.pixels.resize(dCanvas.size());
    thrust::copy(dCanvas.begin(), dCanvas.end(), canvas.pixels.begin());
}
