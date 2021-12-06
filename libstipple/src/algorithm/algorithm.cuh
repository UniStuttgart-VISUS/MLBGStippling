#ifndef ALGORITHM_ALGORITHM_CUH
#define ALGORITHM_ALGORITHM_CUH

#include "algorithm/algorithm.h"
#include "collections/layermaps.cuh"
#include "collections/layervector.cuh"
#include "map.h"
#include <memory>
#include <vector>

#include <thrust/device_vector.h>

typedef union {
    float floats[2];
    int ints[2];
    unsigned long long int ulong;
} AtomicPair;

struct Cell {
    float densityArea;
    Point densityCentroid;
    float densityOrientation;
    float densityMajorAxis;
    float densityMinorAxis;
    float densityMeanSquaredError;
    float densityIntersection;

    float voronoiArea;
    float voronoiEmptyArea;

    float stippleArea;
    float stippleAreaSigma;
    float stippleIntersection;

    AtomicPair mergeCell;
};

typedef LayerVector<Cell, int, true> Cells;
typedef LayerVector<Cell, int, false> KernelCells;
typedef LayerVector<Stipple, int, true> Stipples;
typedef LayerVector<Stipple, int, false> KernelStipples;
typedef LayerMaps<int, true> CellMaps;
typedef LayerMaps<int, false> KernelCellMaps;
typedef LayerMaps<float, true> DensityMaps;
typedef LayerMaps<float, false> KernelDensityMaps;
typedef LayerMaps<Color, true> ColorMaps;
typedef LayerMaps<Color, false> KernelColorMaps;

DensityMaps resizeDensityMaps(const DensityMaps& srcDensityMaps, float factor, int borderWidth);

void voronoi(CellMaps& cellMaps, const Stipples& stipples, VoronoiAlgorithm algorithm);

void voronoiMerge(CellMaps& mergedCellMaps, const CellMaps& cellMaps, const Stipples& stipples);

void voronoiOuter(CellMaps& outerMaps, int& outerIndicesSize, const CellMaps& cellMaps, const Stipples& stipples);

bool voronoiIntersectNaturalNeighbors(Map<int>& nnIndexMap, Map<float>& nnWeightMap, const int bucketSize, const int kernelSize, const CellMaps& mergedCellMaps, const Stipples& stipples);

void voronoiDisplay(ColorMaps& distanceMaps, const CellMaps& cellMaps, const Stipples& stipples, const int displayMode);

void collectCells(Cells& cells, const CellMaps& cellMaps, const DensityMaps& densityMaps, const Stipples& stipples);

class LindeBuzoGrayPrivate;

class LindeBuzoGray {
public:
    explicit LindeBuzoGray(int width, int height,
        const LindeBuzoGrayOptions& options, thrust::device_vector<LindeBuzoGrayLayer>&& layers);

    ~LindeBuzoGray();

    LindeBuzoGray(LindeBuzoGray const&) = delete;
    LindeBuzoGray& operator=(LindeBuzoGray const&) = delete;

    LindeBuzoGray(LindeBuzoGray&&);
    LindeBuzoGray& operator=(LindeBuzoGray&&);

    LindeBuzoGrayResult step(Stipples& stipples, const Cells& cells, const Cells& coupledCells);

private:
    std::unique_ptr<LindeBuzoGrayPrivate> p;
};

void rasterize(Map<Color>& canvas, const DensityMaps& densityMaps, const CellMaps& cellMaps, const CellMaps& outerMaps, const int outerIndicesSize, const Cells& cells, const Stipples& stipples, const float renderScale, const RasterMode mode);

#endif
