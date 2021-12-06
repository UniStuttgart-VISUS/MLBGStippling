#ifndef UTILS_HASH_CUH
#define UTILS_HASH_CUH

#include <builtin_types.h>

// See also:
//   https://www.shadertoy.com/view/Xt3cDn
// and
//   https://www.shadertoy.com/view/XlGcRh

//#define hash_u2f iq3_u2f
#define hash_u2f pcg2d_u2f
#define hash_u2f3 iq3_u2f3

// Integer Hash - III
// - Inigo Quilez, Integer Hash - III, 2017
__device__ inline unsigned int iq3_u2(uint2 p) {
    p.x = 1103515245U * ((p.x >> 1U) ^ (p.y));
    p.y = 1103515245U * ((p.y >> 1U) ^ (p.x));
    unsigned int h32 = 1103515245U * ((p.x) ^ (p.y >> 3U));
    return h32 ^ (h32 >> 16);
}

__device__ inline float iq3_u2f(unsigned int x, unsigned int y) {
    return iq3_u2(make_uint2(x, y)) * (1.0f / float(0xffffffffU));
}

__device__ inline float3 iq3_u2f3(unsigned int x, unsigned int y) {
    const unsigned int h = iq3_u2(make_uint2(x, y));
    return make_float3(
        ((h >> 1) & 0x7fffffffU) / float(0x7fffffffU),
        (((h * 16807U) >> 1) & 0x7fffffffU) / float(0x7fffffffU),
        (((h * 48271U) >> 1) & 0x7fffffffU) / float(0x7fffffffU));
}

// https://www.pcg-random.org/
__device__ inline uint2 pcg2d_u2u2(uint2 v) {
    v.x = v.x * 1664525U + 1013904223U;
    v.y = v.y * 1664525U + 1013904223U;

    v.x += v.y * 1664525U;
    v.y += v.x * 1664525U;

    v.x = v.x ^ (v.x >> 16U);
    v.y = v.y ^ (v.y >> 16U);

    v.x += v.y * 1664525U;
    v.y += v.x * 1664525U;

    v.x = v.x ^ (v.x >> 16U);
    v.y = v.y ^ (v.y >> 16U);

    return v;
}

__device__ inline float pcg2d_u2f(unsigned int x, unsigned int y) {
    return pcg2d_u2u2(make_uint2(x, y)).x * (1.0f / float(0xffffffffU));
}

#endif
