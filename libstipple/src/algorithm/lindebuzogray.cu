#include "algorithm/algorithm.cuh"
#include "algorithm/easingfunctions.cuh"
#include "algorithm/shapefunctions.cuh"
#include "utils/diagnostics.cuh"
#include "utils/hash.cuh"
#include "utils/math.cuh"
#include <cuda.h>

__device__ __forceinline__ float errorReduction(unsigned int hashIndex, float reductionChance) {
    return (hash_u2f(hashIndex, 0) - 1.0f) * reductionChance + 1.0;
}

__device__ __forceinline__ Point splitJitter(unsigned int hashIndex, float spread, Point splitVector) {
    float unitSpreadX = 0.5f + hash_u2f(hashIndex, 1);
    float unitSpreadY = 0.5f + hash_u2f(hashIndex, 2);
    return Point((0.5f * spread + 1.0f) * splitVector.x, (0.5f * spread + 1.0f) * splitVector.y);
}

__device__ __forceinline__ bool shouldCouple(unsigned int hashIndex, float avgStippleDensity) {
    //return true;
    //return hash_u2f(hashIndex, 3) < 0.5f;
    return hash_u2f(hashIndex, 3) < avgStippleDensity;
}

struct AverageModel {
    __device__ __forceinline__ static float estimateStippleSize(const Cell& cell, const LindeBuzoGrayLayer& layer) {
        if (cell.voronoiArea > 0.0f) {
            if (layer.sizeMin != layer.sizeMax) {
                // Estimate size based on infinitely liquid ink (density) in the Voronoi cell.
                const float densityVoronoiRatio = cell.densityArea / cell.voronoiArea;
                return eEaseSize(layer.sizeFunction, layer.sizeMin, layer.sizeMax, densityVoronoiRatio);
            } else {
                return layer.sizeMin;
            }
        } else {
            return 0.0f;
        }
    }
};

struct AdjustedAverageModel {
    __device__ __forceinline__ static float estimateStippleSize(const Cell& cell, const LindeBuzoGrayLayer& layer) {
        if (cell.voronoiArea > 0.0f) {
            if (layer.sizeMin != layer.sizeMax) {
                // Try to de-bias the painting area before computing the ratio (boundary cells often
                // have a strong bias towards emptiness).
                const float totalArea = cell.voronoiArea - smoothstepf(0.2f, 0.3f, cell.voronoiEmptyArea / cell.voronoiArea) * cell.voronoiEmptyArea;
                if (totalArea > 0.0f) {
                    const float areaRatio = cell.densityArea / totalArea;
                    return eEaseSize(layer.sizeFunction, layer.sizeMin, layer.sizeMax, areaRatio);
                } else {
                    return 0.0f;
                }
            } else {
                return layer.sizeMin;
            }
        } else {
            return 0.0f;
        }
    }
};

/// This model minimizes the difference between "density in a cell" and "ink of a stipple".
/// Note that this model is flawed: a stipple can exceed its cell and thus break cell statistics.
struct DifferenceModel {
    __device__ __forceinline__ static float epsilon(const Cell& cell, float hysteresis) {
        return hysteresis / 2.0f * cell.stippleArea;
    }

    __device__ __forceinline__ static bool shouldSplit(const Cell& cell, float hysteresis) {
        return cell.densityArea - cell.stippleArea > epsilon(cell, hysteresis);
    }

    __device__ __forceinline__ static bool shouldRemove(const Cell& cell, float hysteresis) {
        return cell.densityArea - cell.stippleArea < -epsilon(cell, hysteresis);
    }

    __device__ __forceinline__ static bool hasPriority(const int cellIndex, const int mergeCellIndex) {
        return false;
    }
};

/// This model is based on the idea of convolution (splitting density-stipple intersection and its complements).
///
/// Density inside the cell that "can be soaked as ink" by a stipple:
///   freeDensity = cell.densityArea - cell.densityIntersection;
/// Ink inside the cell (i.e. drawn), but not covered by density:
///   freeInk = cell.stippleIntersection - cell.densityIntersection;
/// Now we can define:
///   delta = freeDensity - freeInk;
///   - > 0: more density than ink (residual density).
///   - < 0: less density than ink (missing ink).
///   - == 0: balanced representation.
/// Ink that exceeds the cell boundaries.
///   violatingInk = cell.stippleArea - cell.stippleIntersection;
/// Then:
///  - inkError = [delta, violatingInk]
struct ConvolutionModel {
    using Base = DifferenceModel;

    __device__ __forceinline__ static float delta(const Cell& cell) {
        return cell.densityArea - cell.stippleIntersection;
    }

    __device__ __forceinline__ static float violatingInk(const Cell& cell) {
        return fmaxf(0.0f, cell.stippleArea - cell.stippleIntersection);
    }

    __device__ __forceinline__ static bool shouldSplit(const Cell& cell, float hysteresis, float alpha) {
        return delta(cell) > Base::epsilon(cell, hysteresis) + alpha * violatingInk(cell);
    }

    __device__ __forceinline__ static bool shouldRemove(const Cell& cell, float hysteresis, float alpha) {
        const float d = delta(cell);
        if (d >= 0.0f) {
            return false;
        } else {
            return d < -Base::epsilon(cell, hysteresis) + alpha * violatingInk(cell);
        }
    }

    __device__ __forceinline__ static bool shouldSplit(const Cell& cell, float hysteresis) {
        return shouldSplit(cell, hysteresis, 1.0f);
    }

    __device__ __forceinline__ static bool shouldRemove(const Cell& cell, float hysteresis) {
        return shouldRemove(cell, hysteresis, 1.0f);
    }

    __device__ __forceinline__ static bool hasPriority(const int cellIndex, const int mergeCellIndex) {
        return mergeCellIndex < cellIndex && mergeCellIndex != -1;
    }
};

// This model eleminates overlap at lower densities while ignoring it at
// higher densities, i.e.:
//   - average ink == 0.0: stipple shape matters more,
//   - average ink == 1.0: converting all density to ink matters more.
struct ConvolutionFillingModel {
    using Base = DifferenceModel;
    using Convolution = ConvolutionModel;

    __device__ __forceinline__ static float alpha(const Cell& cell) {
        const float averageDensity = cell.densityArea / cell.voronoiArea;
        return smoothstepf(1.0f, 29.0f / 32.0f, averageDensity);
    }

    __device__ __forceinline__ static bool shouldSplit(const Cell& cell, float hysteresis) {
        return ConvolutionModel::shouldSplit(cell, hysteresis, alpha(cell));
    }

    __device__ __forceinline__ static bool shouldRemove(const Cell& cell, float hysteresis) {
        return ConvolutionModel::shouldRemove(cell, hysteresis, alpha(cell));
    }

    __device__ __forceinline__ static bool hasPriority(const int cellIndex, const int mergeCellIndex) {
        return Convolution::hasPriority(cellIndex, mergeCellIndex);
    }
};

/// This model is asymmetric (conservative in splitting and rigorous in removing) to achieve packing.
__device__ const float PackingBalance = 4.0f;
__device__ const float PackingInf = 100.0f;
struct ConvolutionPackingModel {
    using Base = DifferenceModel;
    using Convolution = ConvolutionModel;

    __device__ __forceinline__ static bool shouldSplit(const Cell& cell, float hysteresis) {
        return ConvolutionModel::shouldSplit(cell, hysteresis * PackingBalance, PackingInf * PackingBalance);
    }

    __device__ __forceinline__ static bool shouldRemove(const Cell& cell, float hysteresis) {
        return ConvolutionModel::shouldRemove(cell, hysteresis / PackingBalance, PackingInf / PackingBalance);
    }

    __device__ __forceinline__ static bool hasPriority(const int cellIndex, const int mergeCellIndex) {
        return mergeCellIndex < cellIndex && mergeCellIndex != -1;
    }
};

struct StepResult {
    int splits = 0;
    int removals = 0;
    int merges = 0;
    int moves = 0;
    float totalDensityMeanSquaredError = 0.0f;
    float totalIntersectionError = 0.0f;
    float totalCellExcess = 0.0f;
    float totalStippleExcess = 0.0f;
};

template <typename StippleModel, typename SizeModel>
__global__ void lindeBuzoGrayStepKernel(
    StepResult* __restrict__ result,
    KernelStipples stipples,
    const KernelCells cells,
    const KernelCells coupledCells,
    const LindeBuzoGrayLayer* __restrict__ layers,
    float hysteresis,
    float errorReductionChance,
    float splitJitterSpread,
    float width, float height,
    int remainingIterations,
    unsigned int hashOffset) {
    const int cellIndex = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int hashIndex = cellIndex + hashOffset;
    if (cellIndex >= cells.size()) {
        return;
    }

    const Cell& cell = cells[cellIndex];
    const Cell& coupledCell = coupledCells.empty() ? cells[cellIndex] : coupledCells[cellIndex];
    const int layerIndex = cells.layerIndex(cellIndex);
    const LindeBuzoGrayLayer& layer = layers[layerIndex];
    const float avgStippleDensity = cell.densityIntersection / cell.stippleIntersection;

    // Egocentric decision.
    struct Ego {
        bool shouldSplit;
        bool shouldRemove;
        float stippleSize;
    };
    const Ego ego = [&] {
        // Jitter error tolerance to avoid local optima.
        const float cellHysteresis = hysteresis * errorReduction(hashIndex, errorReductionChance);
        // Compute what should happen to this cell.
        const float stippleSize = SizeModel::estimateStippleSize(cell, layer);
        return Ego {
            StippleModel::shouldSplit(cell, cellHysteresis) && stippleSize > 0.0f,
            StippleModel::shouldRemove(cell, cellHysteresis) || stippleSize <= 0.0f,
            stippleSize
        };
    }();
    //XXX: assert(!(ego.shouldSplit && ego.shouldRemove) && "Split and remove must not contradict");

    // Joint decision with merge neighbor.
    struct Joint {
        bool shouldMerge;
        float stippleSize;
        Point centroid;
    };
    const Joint joint = [&] {
        const int mergeCellIndex = cell.mergeCell.ints[1];
        if (mergeCellIndex != -1) {
            // Compute what should (approximately) happen to the merge neighbor cell.
            const Cell& mergeCell = cells[mergeCellIndex];
            const float mergeStippleSize = SizeModel::estimateStippleSize(mergeCell, layer);
            const bool mergeShouldSplit = StippleModel::shouldSplit(mergeCell, hysteresis) && mergeStippleSize > 0.0f;
            const bool mergeShouldRemove = StippleModel::shouldRemove(mergeCell, hysteresis) || mergeStippleSize <= 0.0f;

            // Do joint statistics.
            const Cell& coupledMergeCell = coupledCells.empty() ? cells[mergeCellIndex] : coupledCells[mergeCellIndex];
            Point centroid;
            if (coupledCell.voronoiArea > 0.0f && coupledMergeCell.voronoiArea > 0.0f && shouldCouple(hashIndex, avgStippleDensity)) {
                centroid = (coupledCell.densityCentroid + coupledMergeCell.densityCentroid) * 0.5f;
            } else {
                centroid = (cell.densityCentroid + mergeCell.densityCentroid) * 0.5f;
            }
            const Cell jointCell = Cell {
                cell.densityArea + mergeCell.densityArea,
                centroid,
                nanf(""), nanf(""), nanf(""), nanf(""), nanf(""),
                cell.voronoiArea + mergeCell.voronoiArea,
                cell.voronoiEmptyArea + mergeCell.voronoiEmptyArea,
                cell.stippleArea + mergeCell.stippleArea,
                cell.stippleAreaSigma + mergeCell.stippleAreaSigma,
                cell.stippleIntersection + mergeCell.stippleIntersection,
                { -1, -1 }
            };
            const float jointStippleSize = SizeModel::estimateStippleSize(jointCell, layer);
            const bool jointShouldRemove = StippleModel::shouldRemove(jointCell, hysteresis) || jointStippleSize <= 0.0f;

            return Joint {
                StippleModel::hasPriority(cellIndex, mergeCellIndex) && (mergeShouldRemove || ego.shouldRemove) && !jointShouldRemove,
                jointStippleSize,
                centroid
            };
        } else {
            return Joint {
                false, nanf(""), Point()
            };
        }
    }();

    if (ego.shouldSplit && remainingIterations > 2) {
        // Cell is too large, thus split along the half plane.
        Point splitVector = Point(0.5f * ego.stippleSize, 0.0f);
        splitVector = splitVector.rotated(cell.densityOrientation);

        // Repel centroids from half plane with some jitter to avoid local optima.
        Point centroid1 = cell.densityCentroid + splitJitter(hashIndex, splitJitterSpread, splitVector);
        Point centroid2 = cell.densityCentroid - splitJitter(hashIndex, splitJitterSpread, splitVector);

        // Clamp to bounds.
        centroid1 = centroid1.clamped(Point(0.0f, 0.0f), Point(width, height));
        centroid2 = centroid2.clamped(Point(0.0f, 0.0f), Point(width, height));

        stipples[cellIndex * 2] = Stipple { layer.color, StippleState::New, layer.shape, layer.shapeParameter, layer.shapeRadius, ego.stippleSize, centroid1, cell.densityOrientation };
        stipples[cellIndex * 2 + 1] = Stipple { layer.color, StippleState::New, layer.shape, layer.shapeParameter, layer.shapeRadius, ego.stippleSize, centroid2, cell.densityOrientation };
        stipples.layerIndex(cellIndex * 2) = layerIndex;
        stipples.layerIndex(cellIndex * 2 + 1) = layerIndex;
        atomicAdd(&result->splits, 1);
    } else if (ego.shouldRemove && !joint.shouldMerge && remainingIterations > 1) {
        // Cell is too small, thus remove.
        stipples.layerIndex(cellIndex * 2) = -1;
        stipples.layerIndex(cellIndex * 2 + 1) = -1;
        atomicAdd(&result->removals, 1);
    } else if (joint.shouldMerge && remainingIterations > 1) {
        // Cell should serve as a sink, thus merge and move.
        stipples[cellIndex * 2] = Stipple { layer.color, StippleState::Merged, layer.shape, layer.shapeParameter, layer.shapeRadius, joint.stippleSize, joint.centroid, cell.densityOrientation };
        stipples.layerIndex(cellIndex * 2) = layerIndex;
        stipples.layerIndex(cellIndex * 2 + 1) = -1;
        atomicAdd(&result->merges, 1);
    } else {
        // Cell is within acceptable range, thus keep and move.
        Point centroid;
        if (coupledCell.voronoiArea > 0.0f && shouldCouple(hashIndex, avgStippleDensity)) {
            centroid = coupledCell.densityCentroid;
        } else {
            centroid = cell.densityCentroid;
        }
        stipples[cellIndex * 2] = Stipple { layer.color, StippleState::Moved, layer.shape, layer.shapeParameter, layer.shapeRadius, ego.stippleSize, centroid, cell.densityOrientation };
        stipples.layerIndex(cellIndex * 2) = layerIndex;
        stipples.layerIndex(cellIndex * 2 + 1) = -1;
        atomicAdd(&result->moves, 1);
    }

    if (cell.voronoiArea > 0.0f) {
        atomicAdd(&result->totalDensityMeanSquaredError, cell.densityMeanSquaredError);
        atomicAdd(&result->totalIntersectionError, fabsf(cell.densityIntersection - cell.stippleIntersection));
        atomicAdd(&result->totalCellExcess, cell.densityArea - cell.densityIntersection);
        atomicAdd(&result->totalStippleExcess, fmaxf(0.0f, cell.stippleArea - cell.stippleIntersection));
    }
}

typedef thrust::tuple<thrust::device_ptr<int>, thrust::device_ptr<Stipple>> LayerIndexedStipple;
typedef thrust::tuple<int, Stipple> DereferencedLayerIndexedStipple;

struct IsNegativeLayer {
    __host__ __device__ bool operator()(const DereferencedLayerIndexedStipple& tuple) {
        return thrust::get<0>(tuple) == -1;
    }
};

class LindeBuzoGrayPrivate {
public:
    LindeBuzoGrayPrivate() = default;

    LindeBuzoGrayPrivate(LindeBuzoGrayPrivate const&) = delete;
    LindeBuzoGrayPrivate& operator=(LindeBuzoGrayPrivate const&) = delete;

    LindeBuzoGrayPrivate(LindeBuzoGrayPrivate&&) = delete;
    LindeBuzoGrayPrivate& operator=(LindeBuzoGrayPrivate&&) = delete;

    int hashOffset = 0;
    float hysteresis = 0.0f;
    int iteration = 0;
    int width;
    int height;
    LindeBuzoGrayOptions options;
    thrust::device_vector<LindeBuzoGrayLayer> dLayers;

    StepResult* dResult = nullptr;
    float lastTDMSE = 0.0f;
};

LindeBuzoGray::LindeBuzoGray(int width, int height,
    const LindeBuzoGrayOptions& options, thrust::device_vector<LindeBuzoGrayLayer>&& layers)
    : p(std::make_unique<LindeBuzoGrayPrivate>()) {
    p->width = width;
    p->height = height;
    p->options = options;
    p->dLayers = std::move(layers);
    // Allocate result.
    cuda_unwrap(cudaMalloc(&p->dResult, sizeof(StepResult)));
}

LindeBuzoGray::~LindeBuzoGray() {
    if (p) {
        cuda_unwrap(cudaFree(p->dResult));
    }
}

LindeBuzoGray::LindeBuzoGray(LindeBuzoGray&&) = default;

LindeBuzoGray& LindeBuzoGray::operator=(LindeBuzoGray&&) = default;

LindeBuzoGrayResult LindeBuzoGray::step(Stipples& stipples, const Cells& cells, const Cells& coupledCells) {
    // Make sure there is enough space for scattering.
    auto scatterSize = cells.size() * 2;
    if (stipples.capacity() < scatterSize) {
        stipples.reserve(cells.size() * Stipples::GrowthFactor);
    }

    // Reset result.
    StepResult defaultResult;
    cuda_unwrap(cudaMemcpy(p->dResult, &defaultResult, sizeof(StepResult), cudaMemcpyHostToDevice));

    // Compute hysteresis for this iteration.
    switch (p->options.hysteresisFunction) {
    case HysteresisFunction::Constant: {
        p->hysteresis = p->options.hysteresisStart;
        break;
    }
    case HysteresisFunction::Linear:
    case HysteresisFunction::LinearNoMSE: {
        float hysteresisSlope = (p->options.hysteresisMax - p->options.hysteresisStart) / p->options.maxIterations;
        p->hysteresis = p->options.hysteresisStart + hysteresisSlope * p->iteration;
        break;
    }
    case HysteresisFunction::ExponentialNoMSE: {
        float hysteresisExp = logf(p->options.hysteresisMax / p->options.hysteresisStart) / p->options.maxIterations;
        p->hysteresis = p->options.hysteresisStart * exp(hysteresisExp * p->iteration);
        break;
    }
    }

    // Offset hash.
    p->hashOffset += cells.size();

    // Scattering stipple analysis.
    int blockSize = 64;
    int gridSize = (cells.size() + blockSize - 1) / blockSize;
    const LindeBuzoGrayLayer* dLayers = thrust::raw_pointer_cast(p->dLayers.data());
    const int remainingIterations = p->options.forcedCooldown ? (p->options.maxIterations - p->iteration) : 3;
    // clang-format off
#define MODEL_CALL(stippleEnumerator, sizeEnumerator)                                                                   \
    (p->options.stippleModel == StippleModel::stippleEnumerator && p->options.sizeModel == SizeModel::sizeEnumerator) { \
        lindeBuzoGrayStepKernel<stippleEnumerator##Model, sizeEnumerator##Model><<<gridSize, blockSize>>>(              \
            p->dResult, stipples, cells, coupledCells, dLayers, p->hysteresis,                                          \
            p->options.errorReductionChance, p->options.splitJitterSpread,                                              \
            static_cast<float>(p->width), static_cast<float>(p->height), remainingIterations, p->hashOffset);       \
        cuda_debug_synchronize();                                                                                       \
    }
    if MODEL_CALL(Difference, Average)
    else if MODEL_CALL(Difference, AdjustedAverage) 
    else if MODEL_CALL(Convolution, Average)
    else if MODEL_CALL(Convolution, AdjustedAverage)
    else if MODEL_CALL(ConvolutionFilling, Average)
    else if MODEL_CALL(ConvolutionFilling, AdjustedAverage)
    else if MODEL_CALL(ConvolutionPacking, Average)
    else if MODEL_CALL(ConvolutionPacking, AdjustedAverage)
    else {
        assert(false && "Unexpected model");
    }
#undef IF_CALL
    // clang-format on

    StepResult result;
    cuda_unwrap(cudaMemcpy(&result, p->dResult, sizeof(StepResult), cudaMemcpyDeviceToHost));

    // Compact stipples (note that remove_if() retains a tail we do not care about).
    thrust::device_ptr<int> layerIndices0(stipples.layerIndices());
    thrust::device_ptr<Stipple> stipples0(stipples.data());
    auto zipBegin = thrust::make_zip_iterator(thrust::make_tuple(layerIndices0, stipples0));
    auto zipEnd = thrust::make_zip_iterator(thrust::make_tuple(layerIndices0 + scatterSize, stipples0 + scatterSize));
    auto newEnd = thrust::remove_if(thrust::device, zipBegin, zipEnd, IsNegativeLayer());
    stipples.setSize(result.splits * 2 + result.merges + result.moves);

    p->iteration++;

    bool scheduleStable;
    switch (p->options.hysteresisFunction) {
    case HysteresisFunction::Constant:
    case HysteresisFunction::Linear: {
        const float ratioEps = 0.001f;
        const float ratio = std::abs(result.totalDensityMeanSquaredError - p->lastTDMSE) / result.totalDensityMeanSquaredError;
        scheduleStable = (ratio < ratioEps);
        p->lastTDMSE = result.totalDensityMeanSquaredError;
        break;
    }
    case HysteresisFunction::LinearNoMSE:
    case HysteresisFunction::ExponentialNoMSE: {
        scheduleStable = true;
        break;
    }
    }
    const bool amountStable = (result.splits == 0 && result.removals == 0 && result.merges == 0);
    const bool done = (scheduleStable && amountStable) || (p->iteration >= p->options.maxIterations);

    return std::move(LindeBuzoGrayResult {
        done,
        p->iteration,
        stipples.size(),
        result.splits,
        result.removals,
        result.merges,
        result.moves,
        result.totalDensityMeanSquaredError,
        result.totalIntersectionError,
        result.totalCellExcess,
        result.totalStippleExcess });
}
