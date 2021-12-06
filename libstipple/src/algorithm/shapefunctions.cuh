#ifndef ALGORITHM_SHAPEFUNCTIONS_CUH
#define ALGORITHM_SHAPEFUNCTIONS_CUH

#include "stipple.h"
#include "utils/math.cuh"
#include <cuda.h>

// https://www.iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm
__device__ inline float sdCircle(float2 p, float size) {
    return lengthf(p) - size / 2.0f;
}

// https://www.iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm
__device__ inline float sdRectangle(float2 p, float2 b) {
    const float2 d = fabsf(p) - b;
    return lengthf(fmaxf(d, make_float2(0.0f, 0.0f))) + fminf(fmaxf(d.x, d.y), 0.0f);
}

// https://www.iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm
__device__ inline float sdRhombus(float2 p, float2 b) {
    const float2 q = fabsf(p);
    const float h = fminf(fmaxf((-2.0f * ndotf(q, b) + ndotf(b, b)) / dotf(b, b), -1.0f), 1.0f);
    const float d = lengthf(q - 0.5f * b * make_float2(1.0f - h, 1.0f + h));
    return copysignf(d, q.x * b.y + q.y * b.x - b.x * b.y);
}

#if 0 // Analytic signed-distance ellipse.

// https://www.iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm
__device__ inline float sdEllipse(float2 p, float2 ab) {
    float l = ab.y * ab.y - ab.x * ab.x;
    float m = ab.x * p.x / l;
    float m2 = m * m;
    float n = ab.y * p.y / l;
    float n2 = n * n;
    float c = (m2 + n2 - 1.0f) / 3.0f;
    float c3 = c * c * c;
    float q = c3 + m2 * n2 * 2.0f;
    float d = c3 + m2 * n2;
    float g = m + m * n2;
    float co;
    if (d < 0.0f) {
        float h = acosf(q / c3) / 3.0f;
        float s = cosf(h);
        float t = sinf(h) * sqrtf(3.0f);
        float rx = sqrtf(-c * (s + t + 2.0f) + m2);
        float ry = sqrtf(-c * (s - t + 2.0f) + m2);
        co = (ry + copysignf(1.0f, l) * rx + fabsf(g) / (rx * ry) - m) / 2.0f;
    } else {
        float h = 2.0f * m * n * sqrtf(d);
        float s = powf(fabsf(q + h), 1.0f / 3.0f) * copysignf(1.0f, q + h);
        float u = powf(fabsf(q - h), 1.0f / 3.0f) * copysignf(1.0f, q - h);
        float rx = -s - u - c * 4.0f + 2.0f * m2;
        float ry = (s - u) * sqrtf(3.0f);
        float rm = sqrtf(rx * rx + ry * ry);
        co = (ry / sqrtf(rm - rx) + 2.0 * g / rm - m) / 2.0f;
    }
    float rx = ab.x * co;
    float ry = ab.y * sqrtf(1.0f - co * co);
    return hypotf(rx - p.x, ry - p.y) * copysignf(1.0f, p.y - ry);
}

#else

// [1] Maisonobe, L. "Quick computation of the distance between a point and an ellipse." (2006): 1-14.
__device__ inline float sdEllipse(float2 p, float2 ab) {
    float2 t = make_float2(0.707f, 0.707f);
    for (int _i = 0; _i < 2; ++_i) {
        float2 e = make_float2(
            (ab.x * ab.x - ab.y * ab.y) * powf(t.x, 3.0f) / ab.x,
            (ab.y * ab.y - ab.x * ab.x) * powf(t.y, 3.0f) / ab.y);
        float2 q = fabsf(p) - e;
        t = fminf(fmaxf((q * lengthf((ab * t) - e) / lengthf(q) + e) / ab, make_float2(0.0f, 0.0f)), make_float2(1.0f, 1.0f));
        t /= lengthf(t);
    }
    t = copysignf(ab * t, p);
    return copysignf(lengthf(p - t), lengthf(p) - lengthf(t));
}

#endif

// https://www.iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm
__device__ inline float sdTriangle(float2 p, float s) {
    const float k = sqrtf(3.0f);
    p.x = fabsf(p.x) - s;
    p.y = p.y + s / k;
    if (p.x + k * p.y > 0.0) {
        p = make_float2(p.x - k * p.y, -k * p.x - p.y) / 2.0f;
    }
    p.x -= fminf(fmaxf(p.x, -2.0f * s), 0.0f);
    return -hypotf(p.x, p.y) * copysignf(1.0f, p.y);
}

__device__ inline float sdStipple(float2 p, const Stipple& stipple) {
    // Translate.
    float2 pt = make_float2(
        stipple.center.x - 0.5f - p.x,
        stipple.center.y - 0.5f - p.y);
    if (stipple.shape != StippleShape::Circle) {
        // Rotate.
        pt = make_float2(
            pt.x * cosf(-stipple.rotation) - pt.y * sinf(-stipple.rotation),
            pt.x * sinf(-stipple.rotation) + pt.y * cosf(-stipple.rotation));
    }
    switch (stipple.shape) {
    case StippleShape::Circle: {
        return sdCircle(pt, stipple.size);
    }
    case StippleShape::Line: {
        const float2 b = make_float2(
            stipple.size / 2.0f,
            stipple.shapeParameter / 2.0f);
        return sdRectangle(pt, b);
    }
    case StippleShape::Rectangle: {
        const float2 b = make_float2(
            stipple.size * stipple.shapeParameter / 2.0f,
            stipple.size / 2.0f);
        return sdRectangle(pt, b);
    }
    case StippleShape::Rhombus: {
        const float2 b = make_float2(
            stipple.size / 2.0f * stipple.shapeParameter,
            stipple.size / 2.0f);
        return sdRhombus(pt, b);
    }
    case StippleShape::Ellipse: {
        float2 ab;
        if (p.x > p.y) {
            ab.x = stipple.size / 2.0f;
            ab.y = stipple.size * stipple.shapeParameter / 2.0f;
        } else {
            ab.x = stipple.size * stipple.shapeParameter / 2.0f;
            ab.y = stipple.size / 2.0f;
        }
        return sdEllipse(pt, ab);
    }
    case StippleShape::Triangle: {
        return sdTriangle(pt, stipple.size / 2.0f);
    }
    case StippleShape::RoundedLine: {
        const float2 b = make_float2(
            stipple.size / 2.0f - stipple.shapeRadius * (stipple.shapeParameter / 2.0f),
            stipple.shapeParameter / 2.0f - stipple.shapeRadius * (stipple.shapeParameter / 2.0f));
        return sdRectangle(pt, b) - stipple.shapeRadius * (stipple.shapeParameter / 2.0f);
    }
    case StippleShape::RoundedRectangle: {
        const float2 b = make_float2(
            (stipple.size / 2.0f) * (stipple.shapeParameter - stipple.shapeRadius),
            (stipple.size / 2.0f) * (1.0f - stipple.shapeRadius));
        return sdRectangle(pt, b) - stipple.shapeRadius * (stipple.size / 2.0f);
    }
    case StippleShape::RoundedRhombus: {
        const float2 b = make_float2(
            (1.0f - stipple.shapeRadius) * stipple.size / 2.0f * stipple.shapeParameter,
            (1.0f - stipple.shapeRadius) * stipple.size / 2.0f);
        return sdRhombus(pt, b) - stipple.shapeRadius * stipple.size * stipple.shapeParameter / (sqrtf(1.0f + stipple.shapeParameter * stipple.shapeParameter) * 2.0f);
    }
    case StippleShape::RoundedTriangle: {
        return sdTriangle(pt, (1.0f - stipple.shapeRadius) * stipple.size / 2.0f) - stipple.shapeRadius * stipple.size * sqrtf(3.0f) / 6.0f;
    }
    default:
        return NAN;
    }
}

__device__ inline float sdStippleWidth(float2 p, float eps, const Stipple& stipple) {
    const float dF = sdStipple(p, stipple);
    const float dx = sdStipple(p + make_float2(eps, 0.0f), stipple);
    const float dy = sdStipple(p + make_float2(0.0f, eps), stipple);
    return fabsf(dF - dx) + fabsf(dF - dy);
}

__device__ inline float aStipple(float size, StippleShape shape, float shapeParameter, float shapeRadius) {
    switch (shape) {
    case StippleShape::Circle:
        return static_cast<float>(M_PI) * (size / 2.0f) * (size / 2.0f);
    case StippleShape::Line:
        return size * shapeParameter;
    case StippleShape::Rectangle:
        return size * size * shapeParameter;
    case StippleShape::Rhombus:
        return size * size * shapeParameter / 2.0f;
    case StippleShape::Ellipse:
        return static_cast<float>(M_PI) * powf(size / 2.0f, 2.0f) * shapeParameter;
    case StippleShape::Triangle:
        return sqrtf(3.0f) * size * size / 4.0f;
    case StippleShape::RoundedLine:
        return (size * shapeParameter) - (4.0f - static_cast<float>(M_PI)) * (shapeRadius * shapeRadius * shapeParameter * shapeParameter / 4.0f);
    case StippleShape::RoundedRectangle:
        return (size * size * shapeParameter) - (4.0f - static_cast<float>(M_PI)) * (shapeRadius * shapeRadius * size * size / 4.0f);
    case StippleShape::RoundedRhombus: {
        //inscribed circle radius = a*b/(sqrt(a*a+b*b)*2), where a = stipple.size and b = stipple.size * stipple.shapeParameter
        const float r_inscribed = shapeRadius * size * shapeParameter / (sqrtf((1.0f + shapeParameter * shapeParameter)) * 2.0f);
        //base(size * size * shapeParameter / 2.0f) - cutoff((shapeRadius * size) * (shapeRadius * size) * shapeParameter / 2.0f) + circle(static_cast<float>(M_PI) * r_inscribed * r_inscribed)
        return (1.0f - shapeRadius * shapeRadius) * size * size * shapeParameter / 2.0f + (static_cast<float>(M_PI) * r_inscribed * r_inscribed);
    }
    case StippleShape::RoundedTriangle: {
        const float r_inscribed = shapeRadius * size * sqrtf(3.0f) / 6.0f;
        //base(sqrtf(3.0f) * size * size / 4.0f) - cutoff(sqrtf(3.0f) * (shapeRadius * size) * (shapeRadius * size) / 4.0f) + circle(static_cast<float>(M_PI) * r_inscribed * r_inscribed)
        return (1.0f - shapeRadius * shapeRadius) * sqrtf(3.0f) * size * size / 4.0f + (static_cast<float>(M_PI) * r_inscribed * r_inscribed);
    }
    default:
        return NAN;
    }
}

#endif
