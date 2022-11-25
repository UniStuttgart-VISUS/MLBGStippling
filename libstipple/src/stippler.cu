#include "algorithm/algorithm.cuh"
#include "stippler.h"
#include <algorithm>
#include <cassert>
#include <random>

#include <thrust/host_vector.h>

class StipplerPrivate {
public:
    StipplerPrivate(
        const StipplerOptions& options,
        const std::vector<StipplerLayer>& layers)
        : m_densityMaps(makeDensityMaps(layers, options))
        , m_stipples(genStipples(layers, options, m_densityMaps.width(), m_densityMaps.height()))
        , m_cells(m_stipples.layers(), m_stipples.size())
        , m_coupledCells(m_stipples.layers(), m_stipples.size())
        , m_cellMaps(m_densityMaps.layers(), m_densityMaps.width(), m_densityMaps.height())
        , m_coupledCellMaps(m_densityMaps.layers(), m_densityMaps.width(), m_densityMaps.height())
        , m_lbg(makeLindeBuzoGray(layers, options)) { }

    static DensityMaps makeDensityMaps(const std::vector<StipplerLayer>& layers, const StipplerOptions& options) {
        assert((layers[0].density.width > 0 && layers[0].density.height > 0) && "Image size must be greater than zero");
        DensityMaps maps(layers.size(), layers[0].density.width, layers[0].density.height);
        maps.copyToDevice(layers.begin(), layers.end(), [](const auto& layer) { return layer.density.pixels.data(); });
        if (options.voronoiScale != 1.0f || options.borderWidth > 0) {
            return std::move(::resizeDensityMaps(maps, options.voronoiScale, options.borderWidth));
        }
        return std::move(maps);
    }

    static Stipples genStipples(const std::vector<StipplerLayer>& layers, const StipplerOptions& options, int width, int height) {
        const std::uint64_t seed = 1231;
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> centerXDis(0.01f * width, 0.99f * width);
        std::uniform_real_distribution<float> centerYDis(0.01f * height, 0.99f * height);
        std::vector<std::tuple<int, Stipple>> hostStipples;
        for (int layerIndex = 0; layerIndex < layers.size(); ++layerIndex) {
            const auto& layer = layers[layerIndex];
            const auto lbgLayer = options.adjustedLBGLayer(layer.lbg);
            std::uniform_real_distribution<float> sizeDis(lbgLayer.sizeMin, lbgLayer.sizeMax);
            std::generate_n(
                std::back_insert_iterator<std::vector<std::tuple<int, Stipple>>>(hostStipples),
                layer.initialStipples, [&]() {
                    return std::make_tuple(
                        layerIndex,
                        Stipple {
                            lbgLayer.color,
                            StippleState::New,
                            lbgLayer.shape,
                            lbgLayer.shapeParameter,
                            lbgLayer.shapeRadius,
                            sizeDis(gen),
                            Point(centerXDis(gen), centerYDis(gen)),
                            0.0f,
                        });
                });
        }

        Stipples stipples(layers.size(), hostStipples.size());
        stipples.copyToDevice(hostStipples.begin(), hostStipples.end());
        return std::move(stipples);
    }

    LindeBuzoGray makeLindeBuzoGray(
        const std::vector<StipplerLayer>& layers, const StipplerOptions& options) {
        thrust::host_vector<LindeBuzoGrayLayer> lbgLayers;
        std::transform(layers.begin(), layers.end(), std::back_inserter(lbgLayers),
            [options](const auto& layer) {
                return options.adjustedLBGLayer(layer.lbg);
            });
        thrust::device_vector<LindeBuzoGrayLayer> dLbgLayers = lbgLayers;
        LindeBuzoGray lbg(m_densityMaps.width(), m_densityMaps.height(), options.lbg, std::move(dLbgLayers));
        return std::move(lbg);
    }

    void updateDensityMaps(
        const StipplerOptions& options,
        const std::vector<StipplerLayer>& layers) {
        // TODO: assert if layer size or options do not match.
        m_densityMaps = makeDensityMaps(layers, options);
        m_lbg.softRewind();
    }

    std::vector<std::vector<Stipple>> stipple(StippleAlgorithm stippleAlgorithm, VoronoiAlgorithm voronoiAlgorithm, Stippler::IterationCallback cb) {
        LindeBuzoGrayResult result;
        do {
            if (m_stipples.empty()) {
                // printf("Recovering...\n");
                //  Zero stipples usually occur because of poor parameters or optimization - attempt a blunt recovery.
                std::mt19937 gen;
                std::uniform_real_distribution<float> centerXDis(m_densityMaps.width() / 4, (m_densityMaps.width() * 3) / 4);
                std::uniform_real_distribution<float> centerYDis(m_densityMaps.height() / 4, (m_densityMaps.height() * 3) / 4);
                std::vector<std::tuple<int, Stipple>> hostStipples;
                for (int layerIndex = 0; layerIndex < m_densityMaps.layers(); ++layerIndex) {
                    hostStipples.push_back(std::make_tuple(
                        layerIndex,
                        Stipple {
                            Color(255, 0, 0, 255),
                            StippleState::New,
                            StippleShape::Circle,
                            1.0f,
                            0.0f,
                            0.1f,
                            Point(centerXDis(gen), centerYDis(gen)),
                            0.0f,
                        }));
                }
                m_stipples.setSize(m_densityMaps.layers());
                m_stipples.copyToDevice(hostStipples.begin(), hostStipples.end());
            }
            switch (stippleAlgorithm) {
            case StippleAlgorithm::LBG:
                ::voronoi(m_cellMaps, m_stipples, voronoiAlgorithm);
                ::collectCells(m_cells, m_cellMaps, m_densityMaps, m_stipples);
                result = m_lbg.step(m_stipples, m_cells, Cells());
                break;
            case StippleAlgorithm::CoupledLBG:
                ::voronoi(m_cellMaps, m_stipples, voronoiAlgorithm);
                ::voronoiMerge(m_coupledCellMaps, m_cellMaps, m_stipples);
                ::collectCells(m_cells, m_cellMaps, m_densityMaps, m_stipples);
                ::collectCells(m_coupledCells, m_coupledCellMaps, m_densityMaps, m_stipples);
                result = m_lbg.step(m_stipples, m_cells, m_coupledCells);
                break;
            }
            if (cb) {
                std::vector<std::vector<Stipple>> layerStipples(m_stipples.layers(), std::vector<Stipple>());
                m_stipples.copyToHost([&](auto layerIndex, const auto& stipple) {
                    layerStipples[layerIndex].push_back(stipple);
                });
                cb(layerStipples, result);
            }
        } while (!result.done);

        std::vector<std::vector<Stipple>> layerStipples(m_stipples.layers(), std::vector<Stipple>());
        m_stipples.copyToHost([&](auto layerIndex, const auto& stipple) {
            layerStipples[layerIndex].push_back(stipple);
        });
        return std::move(layerStipples);
    }

    Map<Color> rasterize(float renderScale, RasterMode mode, VoronoiAlgorithm voronoiAlgorithm) {
        Map<Color> canvas;
        if (!m_stipples.empty()) {
            ::voronoi(m_cellMaps, m_stipples, voronoiAlgorithm);
            int outerIndicesSize = -1;
            ::voronoiOuter(m_coupledCellMaps, outerIndicesSize, m_cellMaps, m_stipples);
            ::collectCells(m_cells, m_cellMaps, m_densityMaps, m_stipples);
            ::rasterize(canvas, m_densityMaps, m_cellMaps, m_coupledCellMaps, outerIndicesSize, m_cells, m_stipples, renderScale, mode);
        }
        return std::move(canvas);
    }

    std::optional<NaturalNeighborData> computeNaturalNeighbors(VoronoiAlgorithm voronoiAlgorithm, const int bucketSize, const int kernelSize) {
        NaturalNeighborData naturalNeighborData;
        if (!m_stipples.empty()) {
            ::voronoi(m_cellMaps, m_stipples, voronoiAlgorithm);
            ::voronoiMerge(m_coupledCellMaps, m_cellMaps, m_stipples);

            // TODO: morton/z-curve-based reordering.

            // Compute intersection with the original cell map.
            bool ok = voronoiIntersectNaturalNeighbors(naturalNeighborData.indexMap, naturalNeighborData.weightMap,
                bucketSize, kernelSize, m_coupledCellMaps, m_stipples);
            if (!ok) {
                return std::nullopt;
            }

            // Copy positions.
            naturalNeighborData.positions.width = 2;
            naturalNeighborData.positions.height = m_stipples.size();
            naturalNeighborData.positions.pixels.assign(naturalNeighborData.positions.width * naturalNeighborData.positions.height, -1);
            int stippleIndex = 0;
            m_stipples.copyToHost([&](auto layerIndex, const auto& stipple) {
                if (layerIndex == 0) {
                    naturalNeighborData.positions.pixels[stippleIndex * 2] = std::round(stipple.center.x);
                    naturalNeighborData.positions.pixels[stippleIndex * 2 + 1] = std::round(stipple.center.y);
                    stippleIndex++;
                }
            });
        }
        return std::move(naturalNeighborData);
    }

private:
    DensityMaps m_densityMaps;
    Stipples m_stipples;
    Cells m_cells;
    Cells m_coupledCells;
    CellMaps m_cellMaps;
    CellMaps m_coupledCellMaps;
    LindeBuzoGray m_lbg;
};

Stippler::Stippler()
    : p(nullptr)
    , m_stippledOnce(false) {
}

Stippler::~Stippler() = default;

void Stippler::resetLayers(const std::vector<StipplerLayer>& layers, bool keepState) {
    if (!keepState || !m_stippledOnce) {
        p = std::make_unique<StipplerPrivate>(m_options, layers);
        m_stippledOnce = false;
    } else {
        p->updateDensityMaps(m_options, layers);
    }
}

const StipplerOptions& Stippler::options() const {
    return m_options;
}

void Stippler::setOptions(const StipplerOptions& options) {
    m_options = options;
    p.reset(nullptr);
}

Stippler::IterationCallback Stippler::iterationCallback() const {
    return m_iterationCallback;
}

void Stippler::setIterationCallback(Stippler::IterationCallback cb) {
    m_iterationCallback = cb;
}

std::vector<std::vector<Stipple>> Stippler::stipple() {
    assert(p && "Stippler not initialized (set layers)");
    m_stippledOnce = true;
    return std::move(p->stipple(m_options.stippleAlgorithm, m_options.voronoiAlgorithm, m_iterationCallback));
}

std::optional<NaturalNeighborData> Stippler::computeNaturalNeighbors(const int bucketSize, const int kernelSize) {
    if (p && m_stippledOnce) {
        return std::move(p->computeNaturalNeighbors(m_options.voronoiAlgorithm, bucketSize, kernelSize));
    } else {
        return NaturalNeighborData();
    }
}

Map<Color> Stippler::rasterize(float renderScale, RasterMode mode) {
    if (p && m_stippledOnce) {
        return std::move(p->rasterize(renderScale, mode, m_options.voronoiAlgorithm));
    } else {
        return Map<Color>();
    }
}
