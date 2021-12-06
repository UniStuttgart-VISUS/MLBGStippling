#ifndef ALGORITHM_ALGORITHM_H
#define ALGORITHM_ALGORITHM_H

#include "stipple.h"

enum class VoronoiAlgorithm {
    Search = 0,
    JumpFlooding = 1
};

enum class SizeFunction {
    Linear = 0,
    QuadraticIn = 1,
    QuadraticOut = 2,
    QuadraticInOut = 3,
    ExponentialIn = 4,
    ExponentialOut = 5,
    ExponentialInOut = 6
};

struct LindeBuzoGrayLayer {
    Color color = Color(0, 0, 0);

    StippleShape shape = StippleShape::Circle;
    float shapeParameter = 1.5f;
    float shapeRadius = 0.25f;

    float sizeMin = 4.0f;
    float sizeMax = 4.0f;
    SizeFunction sizeFunction = SizeFunction::Linear;
};

struct Coupling {
    float weight = 0.0f;
    float chance = 0.0f;
};

enum class HysteresisFunction {
    Constant = 0,
    Linear = 1,
    LinearNoMSE = 2,
    ExponentialNoMSE = 3
};

enum class StippleModel {
    Difference = 0,
    Convolution = 1,
    ConvolutionFilling = 2,
    ConvolutionPacking = 3,
};

enum class SizeModel {
    Average = 0,
    AdjustedAverage = 1,
};

struct LindeBuzoGrayOptions {
    StippleModel stippleModel = StippleModel::Difference;
    SizeModel sizeModel = SizeModel::AdjustedAverage;

    int maxIterations = 50;
    bool forcedCooldown = false;

    float hysteresisStart = 0.01f;
    float hysteresisMax = 1.0f;
    HysteresisFunction hysteresisFunction = HysteresisFunction::Linear;

    float splitJitterSpread = 0.0f;
    float errorReductionChance = 0.0f;
};

struct LindeBuzoGrayResult {
    bool done = false;
    int iteration = 0;
    int total = 0;
    int splits = 0;
    int removals = 0;
    int merges = 0;
    int moves = 0;
    float totalDensityMeanSquaredError = 0.0f;
    float totalIntersectionError = 0.0f;
    float totalCellExcess = 0.0f;
    float totalStippleExcess = 0.0f;
};

enum class RasterMode {
    Stipples = 0,
    Background = 1,
    StipplesWithBackground = 2,
};

#endif
