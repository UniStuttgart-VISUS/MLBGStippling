#ifndef COLOR_H
#define COLOR_H

#include "defines.h"
#include <cstdint>

struct Color {
    Color() = default;

    CUDA_HOST_DEVICE inline Color(std::uint8_t r, std::uint8_t g, std::uint8_t b, std::uint8_t a = 255)
        : argb((a << 24) | (r << 16) | (g << 8) | b) { }

    CUDA_HOST_DEVICE inline bool operator==(const Color& other) const {
        return argb == other.argb;
    }

    CUDA_HOST_DEVICE inline bool operator!=(const Color& other) const {
        return argb != other.argb;
    }

    CUDA_HOST_DEVICE inline bool operator<(const Color& other) const {
        return argb < other.argb;
    }

    CUDA_HOST_DEVICE inline std::uint8_t a() const {
        return (argb >> 24) & 0xFF;
    }

    CUDA_HOST_DEVICE inline std::uint8_t r() const {
        return (argb >> 16) & 0xFF;
    }

    CUDA_HOST_DEVICE inline std::uint8_t g() const {
        return (argb >> 8) & 0xFF;
    }

    CUDA_HOST_DEVICE inline std::uint8_t b() const {
        return argb & 0xFF;
    }

    CUDA_HOST_DEVICE inline Color rgb() const {
        return Color(argb | 0xFF000000);
    }

    CUDA_HOST_DEVICE inline Color mix(Color that, std::uint32_t t) const {
        const std::uint32_t s = 255 - t;
        return Color(((b() * s + that.b() * t) >> 8)
            | ((g() * s + that.g() * t) & ~0xff)
            | (((r() * s + that.r() * t) << 8) & ~0xffff)
            | (((a() * s + that.a() * t) << 16) & ~0xffffff));
    }

    std::uint32_t argb;

private:
    CUDA_HOST_DEVICE inline Color(std::uint32_t argb)
        : argb(argb) { }
};

#endif
