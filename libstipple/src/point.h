#ifndef POINT_H
#define POINT_H

#include "defines.h"
#include <math.h>

struct Point {
    Point() = default;

    CUDA_HOST_DEVICE inline Point(float x, float y)
        : x(x)
        , y(y) {};

    CUDA_HOST_DEVICE inline Point operator+(const Point& other) const {
        return Point(x + other.x, y + other.y);
    }

    CUDA_HOST_DEVICE inline Point& operator+=(const Point& other) {
        x += other.x;
        y += other.y;
        return *this;
    }

    CUDA_HOST_DEVICE inline Point operator-(const Point& other) const {
        return Point(x - other.x, y - other.y);
    }

    CUDA_HOST_DEVICE inline Point& operator-=(const Point& other) {
        x -= other.x;
        y -= other.y;
        return *this;
    }

    CUDA_HOST_DEVICE inline Point& operator*(const float& other) {
        x *= other;
        y *= other;
        return *this;
    }

    CUDA_HOST_DEVICE inline Point& operator/(const float& other) {
        x /= other;
        y /= other;
        return *this;
    }

    CUDA_HOST_DEVICE inline Point clamped(Point minPoint, Point maxPoint) {
        auto clamp = [](float x, float minX, float maxX) {
            return x > maxX ? maxX : (x < minX ? minX : x);
        };
        return Point(
            clamp(this->x, minPoint.x, maxPoint.x),
            clamp(this->y, minPoint.y, maxPoint.y));
    }

    CUDA_HOST_DEVICE inline Point rotated(float radians) {
        return Point(this->x * cos(radians) - this->y * sin(radians),
            this->y * cos(radians) + this->x * sin(radians));
    }

    float x;
    float y;
};

#endif
