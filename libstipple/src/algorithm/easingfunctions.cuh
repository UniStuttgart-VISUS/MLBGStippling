#ifndef ALGORITHM_EASINGFUNCTIONS_CUH
#define ALGORITHM_EASINGFUNCTIONS_CUH

#include "algorithm.h"

__device__ inline float eLinear(float t) {
    return t;
}

__device__ inline float eQuadraticIn(float t) {
    return t * t;
}

__device__ inline float eQuadraticOut(float t) {
    return -(t * (t - 2.0f));
}

__device__ inline float eQuadraticInOut(float t) {
    return (t < 0.5f) ? 2.0f * t * t : (-2.0f * t * t) + (4.0f * t) - 1.0f;
}

__device__ inline float eExponentialIn(float t) {
    return (t == 0.0f) ? 0.0f : powf(2.0f, 10.0f * (t - 1.0f));
}

__device__ inline float eExponentialOut(float t) {
    return (t == 1.0f) ? 1.0f : 1.0f - powf(2.0f, -10.0f * t);
}

__device__ inline float eExponentialInOut(float t) {
    if (t == 0.0f || t == 1.0f) {
        return t;
    } else if (t < 0.5f) {
        return 0.5f * powf(2.0f, (20.0f * t) - 10.0f);
    } else {
        return -0.5f * powf(2.0f, (-20.0f * t) + 10.0f) + 1.0f;
    }
}

__device__ inline float eEaseSize(SizeFunction sizeFunction, float sizeMin, float sizeMax, float t) {
    float alpha;
    switch (sizeFunction) {
    case SizeFunction::Linear:
        alpha = eLinear(t);
        break;
    case SizeFunction::QuadraticIn:
        alpha = eQuadraticIn(t);
        break;
    case SizeFunction::QuadraticOut:
        alpha = eQuadraticOut(t);
        break;
    case SizeFunction::QuadraticInOut:
        alpha = eQuadraticInOut(t);
        break;
    case SizeFunction::ExponentialIn:
        alpha = eExponentialIn(t);
        break;
    case SizeFunction::ExponentialOut:
        alpha = eExponentialOut(t);
        break;
    case SizeFunction::ExponentialInOut:
        alpha = eExponentialInOut(t);
        break;
    default:
        alpha = NAN;
        break;
    };
    return (1.0f - alpha) * sizeMin + alpha * sizeMax;
}

#endif
