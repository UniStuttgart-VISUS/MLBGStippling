#ifndef UTILS_COLORMAP_CUH
#define UTILS_COLORMAP_CUH

#include "../color.h"
#include "hash.cuh"
#include "math.cuh"

// Polynomial approximation for the Turbo colormap based on the one provided by Google.
__device__ inline float3 turboColormap(float x) {
    const float4 kRedVec4 = make_float4(0.13572138f, 4.61539260f, -42.66032258f, 132.13108234f);
    const float4 kGreenVec4 = make_float4(0.09140261f, 2.19418839f, 4.84296658f, -14.18503333f);
    const float4 kBlueVec4 = make_float4(0.10667330f, 12.64194608f, -60.58204836f, 110.36276771f);
    const float2 kRedVec2 = make_float2(-152.94239396f, 59.28637943f);
    const float2 kGreenVec2 = make_float2(4.27729857f, 2.82956604f);
    const float2 kBlueVec2 = make_float2(-89.90310912f, 27.34824973f);

    x = clampf(x, 0.0, 1.0);
    float4 v4 = make_float4(1.0f, x, x * x, x * x * x);
    float2 v2 = make_float2(v4.z, v4.w) * v4.z;
    return make_float3(
        dotf(v4, kRedVec4) + dotf(v2, kRedVec2),
        dotf(v4, kGreenVec4) + dotf(v2, kGreenVec2),
        dotf(v4, kBlueVec4) + dotf(v2, kBlueVec2));
}

__device__ inline Color sdToColor(float sd) {
    const float t = copysignf(fmodf(fabsf(sd) / 5.0f, 5.0f), sd);
    float3 color = turboColormap(0.5f + t * 0.5f);
    return Color(
        clampf(color.x * 255.0f, 0.0f, 255.0f),
        clampf(color.y * 255.0f, 0.0f, 255.0f),
        clampf(color.z * 255.0f, 0.0f, 255.0f));
}

__device__ inline Color cellIndexToColor(int cellIndex) {
    if (cellIndex >= 0) {
        float3 cellVector = hash_u2f3(cellIndex >> 16, cellIndex & 0xFFFF);
        return Color(
            fminf(255.0f, cellVector.x * 255.0f),
            fminf(255.0f, cellVector.y * 255.0f),
            fminf(255.0f, cellVector.z * 255.0f));
    } else {
        return Color(0, 0, 0);
    }
}

#endif
