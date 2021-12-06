#ifndef UTILS_MATH_CUH
#define UTILS_MATH_CUH

#include <builtin_types.h>

__device__ inline float2 operator-(float2& a) {
    return make_float2(-a.x, -a.y);
}

__device__ inline float2 operator+(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

__device__ inline void operator+=(float2& a, float2 b) {
    a.x += b.x;
    a.y += b.y;
}

__device__ inline float2 operator+(float a, float2 b) {
    return make_float2(a + b.x, a + b.y);
}

__device__ inline float2 operator+(float2 a, float b) {
    return make_float2(a.x + b, a.y + b);
}

__device__ inline void operator+=(float2& a, float b) {
    a.x += b;
    a.y += b;
}

__device__ inline float2 operator-(float2 a, float2 b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

__device__ inline void operator-=(float2& a, float2 b) {
    a.x -= b.x;
    a.y -= b.y;
}

__device__ inline float2 operator-(float2 a, float b) {
    return make_float2(a.x - b, a.y - b);
}

__device__ inline float2 operator-(float a, float2 b) {
    return make_float2(a - b.x, a - b.y);
}

__device__ inline void operator-=(float2& a, float b) {
    a.x -= b;
    a.y -= b;
}

__device__ inline float2 operator*(float2 a, float2 b) {
    return make_float2(a.x * b.x, a.y * b.y);
}

__device__ inline void operator*=(float2& a, float2 b) {
    a.x *= b.x;
    a.y *= b.y;
}

__device__ inline float2 operator*(float a, float2 b) {
    return make_float2(a * b.x, a * b.y);
}

__device__ inline float2 operator*(float2 a, float b) {
    return make_float2(a.x * b, a.y * b);
}

__device__ inline void operator*=(float2& a, float b) {
    a.x *= b;
    a.y *= b;
}

__device__ inline float2 operator/(float2 a, float2 b) {
    return make_float2(a.x / b.x, a.y / b.y);
}

__device__ inline void operator/=(float2& a, float2 b) {
    a.x /= b.x;
    a.y /= b.y;
}

__device__ inline float2 operator/(float a, float2 b) {
    return make_float2(a / b.x, a / b.y);
}

__device__ inline float2 operator/(float2 a, float b) {
    return make_float2(a.x / b, a.y / b);
}

__device__ inline void operator/=(float2& a, float b) {
    a.x /= b;
    a.y /= b;
}

__device__ inline float2 fabsf(float2 a) { return make_float2(fabsf(a.x), fabsf(a.y)); }

__device__ inline float2 copysignf(float2 a, float2 b) { return make_float2(copysignf(a.x, b.x), copysignf(a.y, b.y)); }

__device__ inline float2 fmaxf(float2 a, float2 b) { return make_float2(fmaxf(a.x, b.x), fmaxf(a.y, b.y)); }

__device__ inline float2 fminf(float2 a, float2 b) { return make_float2(fminf(a.x, b.x), fminf(a.y, b.y)); }

__device__ inline float dotf(float2 a, float2 b) { return a.x * b.x + a.y * b.y; }

__device__ inline float dotf(float4 a, float4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }

__device__ inline float ndotf(float2 a, float2 b) { return a.x * b.x - a.y * b.y; }

__device__ inline float lengthf(float2 a) { return hypotf(a.x, a.y); }

__device__ inline float clampf(float x, float a, float b) { return fmaxf(a, fminf(b, x)); }

__device__ inline float lerpf(float a, float b, float t) { return (1.0f - t) * a + t * b; }

__device__ inline float stepf(float edge, float x) {
    return x < edge ? 0.0f : 1.0f;
}

__device__ inline float smoothstepf(float edge0, float edge1, float x) {
    x = clampf((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return x * x * (3.0f - 2.0f * x);
}

class SMin {
public:
    __device__ SMin(const float smoothingFactor = 2.0f)
        : smoothingFactor(smoothingFactor)
        , minDistance(FLT_MAX)
        , count(0) { }

    __device__ inline void add(const float distance) {
        if (count == 0) {
            minDistance = distance;
        } else {
            const float greater = fmaxf(-smoothingFactor * minDistance, -smoothingFactor * distance);
            const float smaller = fminf(-smoothingFactor * minDistance, -smoothingFactor * distance);
            minDistance = -(greater + log2f(1.0f + exp2f(smaller - greater))) / smoothingFactor;
        }
        count++;
    }

    __device__ inline float get() const {
        return minDistance - (log2f(1.0f / count) / smoothingFactor);
    }

private:
    const float smoothingFactor;
    float minDistance;
    int count;
};

#endif
