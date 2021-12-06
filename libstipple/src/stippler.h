#ifndef STIPPLER_H
#define STIPPLER_H

#include "algorithm/algorithm.h"
#include "map.h"
#include <cassert>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

enum class StippleAlgorithm {
    /// Linde-Buzo-Gray stippling for one layer.
    LBG = 0,
    /// Linde-Buzo-Gray stippling for multiple coupled layers.
    CoupledLBG = 1
};

struct StipplerLayer {
    int initialStipples = 100;

    Map<float> density;

    LindeBuzoGrayLayer lbg;
};

struct StipplerOptions {
    StippleAlgorithm stippleAlgorithm = StippleAlgorithm::CoupledLBG;
    VoronoiAlgorithm voronoiAlgorithm = VoronoiAlgorithm::JumpFlooding;

    float voronoiScale = 2.0f;
    int borderWidth = 0;

    LindeBuzoGrayOptions lbg;

    LindeBuzoGrayLayer adjustedLBGLayer(const LindeBuzoGrayLayer& lbgLayer) const {
        assert((lbgLayer.sizeMin <= lbgLayer.sizeMax) && "Minimum stipple size should be smaller or equal to maximum");
        LindeBuzoGrayLayer adjustedLayer(lbgLayer);
        if (adjustedLayer.shape == StippleShape::Line || adjustedLayer.shape == StippleShape::RoundedLine) {
            assert((lbgLayer.shapeParameter <= lbgLayer.sizeMin) && "Line width should be smaller or equal to minimum stipple size");
            adjustedLayer.shapeParameter *= voronoiScale;
        }
        adjustedLayer.shapeRadius *= voronoiScale;
        adjustedLayer.sizeMin *= voronoiScale;
        adjustedLayer.sizeMax *= voronoiScale;
        return adjustedLayer;
    }
};

struct NaturalNeighborData {
    Map<int> positions;

    Map<int> indexMap;
    Map<float> weightMap;
};

class StipplerPrivate;

class Stippler {
public:
    using IterationCallback = std::function<void(const std::vector<std::vector<Stipple>>&, const LindeBuzoGrayResult&)>;

    Stippler();
    ~Stippler();

    Stippler(const Stippler&) = delete;
    Stippler& operator=(const Stippler&) = delete;

    void resetLayers(const std::vector<StipplerLayer>& layers);

    const StipplerOptions& options() const;
    void setOptions(const StipplerOptions& options);

    IterationCallback iterationCallback() const;
    void setIterationCallback(IterationCallback cb);

    std::vector<std::vector<Stipple>> stipple();

    std::optional<NaturalNeighborData> computeNaturalNeighbors(const int bucketSize, const int kernelSize);

    Map<Color> rasterize(float renderScale, RasterMode mode);

private:
    std::unique_ptr<StipplerPrivate> p;

    StipplerOptions m_options;
    IterationCallback m_iterationCallback;
    bool m_stippledOnce;
};

#endif
