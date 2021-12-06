#include "algorithm/algorithm.cuh"

#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include "ext/catch.h"
#include "ext/lodepng.h"
#include <fstream>
#include <vector>

Stipple ts(float ox, float oy, StippleShape shape, int width, int height) {
    return Stipple {
        Color(0, 0, 0),
        StippleState::New,
        shape,
        1.0f,
        1.0f,
        1.0f,
        Point(width / 2.0f + ox * width, height / 2.0f + oy * height),
        0.0f,
    };
}

std::vector<std::tuple<int, Stipple>> fixtureStipples(StippleShape shape, int width, int height) {
    return std::vector<std::tuple<int, Stipple>>({
        { 0, ts(0.125f, 0.0f, shape, width, height) },
        { 0, ts(0.0f, 0.125f, shape, width, height) },
        { 0, ts(-0.125f, 0, shape, width, height) },
        { 0, ts(0.0f, -0.125f, shape, width, height) },
        { 1, ts(0.125f, 0.125f, shape, width, height) },
        { 1, ts(-0.125f, -0.125f, shape, width, height) },
        { 1, ts(-0.125f, 0.125f, shape, width, height) },
        { 1, ts(0.125f, -0.125f, shape, width, height) },
    });
}

std::vector<std::tuple<int, Stipple>> randomStipples(StippleShape shape, int layers, int stipplesPerLayer, int width, int height) {
    std::seed_seq seq { 1, 2, 3, 1 };
    std::mt19937 gen(seq);
    std::uniform_real_distribution<float> disX(0.0f, width / 2.0f);
    std::uniform_real_distribution<float> disY(0.0f, height / 2.0f);
    std::vector<std::tuple<int, Stipple>> stipples(layers * stipplesPerLayer);
    for (int layer = 0; layer < layers; ++layer) {
        std::generate_n(stipples.begin(), stipplesPerLayer, [&]() {
            return std::make_tuple(layer, ts(disX(gen), disY(gen), shape, width, height));
        });
    }
    return std::move(stipples);
}

std::shared_ptr<Stipples> prepareStipples(const std::vector<std::tuple<int, Stipple>>& hostStipples) {
    int maxLayerIndex = 0;
    for (int i = 0; i < hostStipples.size(); ++i) {
        maxLayerIndex = std::max(maxLayerIndex, std::get<0>(hostStipples[i]));
    }
    auto stipples = std::make_shared<Stipples>(maxLayerIndex + 1, static_cast<int>(hostStipples.size()));
    stipples->copyToDevice(hostStipples.begin(), hostStipples.end());
    return std::move(stipples);
}

std::function<std::shared_ptr<CellMaps>()> prepareVoronoi(const std::shared_ptr<Stipples>& stipples, int w, int h, VoronoiAlgorithm algorithm) {
    auto cellMaps = std::make_shared<CellMaps>(stipples->layers(), w, h);
    return [stipples, cellMaps, algorithm]() -> std::shared_ptr<CellMaps> {
        voronoi(*cellMaps, *stipples, algorithm);
        cudaDeviceSynchronize();
        return cellMaps;
    };
}

void benchmarkVoronoi(VoronoiAlgorithm algorithm) {
    struct ResolutionConfig {
        int w;
        int h;
    };
    const ResolutionConfig resolutionConfigs[] = {
        { 256, 256 },
        { 1920, 1080 }
    };

    struct StippleConfig {
        int layers;
        int stipplesPerLayer;
    };
    const StippleConfig stippleConfigs[] = {
        { 1, 1000 },
        { 2, 1000 },
    };

    for (auto sc : stippleConfigs) {
        for (auto rc : resolutionConfigs) {
            auto stipples = prepareStipples(randomStipples(StippleShape::Circle, sc.layers, sc.stipplesPerLayer, rc.w, rc.h));
            auto run = prepareVoronoi(stipples, rc.w, rc.h, algorithm);
            BENCHMARK(std::to_string(sc.layers) + std::string(" # ") + std::to_string(sc.layers * sc.stipplesPerLayer) + std::string(" @ ")
                + std::to_string(rc.w) + std::string("x") + std::to_string(rc.h)) {
                return run();
            };
        }
    }
}

void writeCellMaps(std::shared_ptr<CellMaps> cellMaps, const std::string& name) {
    const std::uint32_t colorBrewer12Paired[] = {
        0xffe3cea6,
        0xffb4781f,
        0xff8adfb2,
        0xff2ca033,
        0xff999afb,
        0xff1c1ae3,
        0xff6fbffd,
        0xff007fff,
        0xffd6b2ca,
        0xff9a3d6a,
        0xff99ffff,
        0xff2859b1,
    };
    std::vector<int> hostLayer(cellMaps->width() * cellMaps->height(), 0);
    cellMaps->copyToHost([&](auto layerIndex, auto* scan0) {
        if (scan0) {
            for (int i = 0; i < cellMaps->width() * cellMaps->height(); ++i) {
                if (scan0[i] < 0) {
                    scan0[i] = 0x00000000;
                } else {
                    scan0[i] = (colorBrewer12Paired[scan0[i] % 12]);
                }
            }
            std::string filename = "voronoi-" + name + "-" + std::to_string(layerIndex) + ".png";
            lodepng::State state;
            state.info_png.color.colortype = LCT_RGB;
            state.info_png.color.bitdepth = 8;
            state.info_raw.colortype = LCT_RGBA;
            state.info_raw.bitdepth = 8;
            state.encoder.auto_convert = 0;
            std::vector<unsigned char> out;
            unsigned int error = lodepng::encode(out, reinterpret_cast<unsigned char*>(scan0),
                cellMaps->width(), cellMaps->height(), state);
            INFO(lodepng_error_text(error));
            REQUIRE(error == 0);
            std::ofstream file(filename, std::ios::trunc | std::ios::out | std::ofstream::binary);
            std::copy(out.begin(), out.end(), std::ostreambuf_iterator<char>(file));
        }
        return hostLayer.data();
    });
}

TEST_CASE("Search", "[voronoi]") {
    auto stipples = prepareStipples(fixtureStipples(StippleShape::Circle, 64, 64));
    auto run = prepareVoronoi(stipples, 64, 64, VoronoiAlgorithm::Search);
    writeCellMaps(run(), "search");

    benchmarkVoronoi(VoronoiAlgorithm::Search);
}

TEST_CASE("JumpFlooding", "[voronoi]") {
    auto stipples = prepareStipples(fixtureStipples(StippleShape::Circle, 64, 64));
    auto run = prepareVoronoi(stipples, 64, 64, VoronoiAlgorithm::JumpFlooding);
    writeCellMaps(run(), "jumpflooding");

    benchmarkVoronoi(VoronoiAlgorithm::JumpFlooding);
}

TEST_CASE("Merge", "[voronoi]") {
    auto stipples = prepareStipples(fixtureStipples(StippleShape::Circle, 64, 64));
    auto run = prepareVoronoi(stipples, 64, 64, VoronoiAlgorithm::Search);
    auto cellMaps = run();
    writeCellMaps(cellMaps, "merge-in");
    /*
    auto mergedCellMaps = std::make_shared<CellMaps>(cellMaps->layers(), cellMaps->width(), cellMaps->height());
    std::vector<Coupling> couplings = {
        Coupling { 0.0f, 1.0f },
        Coupling { 0.0f, 1.0f },
        Coupling { 0.0f, 1.0f },
        Coupling { 0.0f, 1.0f }
    };
    thrust::device_vector<Coupling> dCouplings = couplings;
    voronoiMerge(*mergedCellMaps, *cellMaps, *stipples, dCouplings, 0.0f);
    writeCellMaps(mergedCellMaps, "merge-out");
   */
}
