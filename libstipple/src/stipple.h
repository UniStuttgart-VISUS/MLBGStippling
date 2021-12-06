#ifndef STIPPLE_H
#define STIPPLE_H

#include "color.h"
#include "point.h"

enum class StippleState : unsigned char {
    New = 0,
    Merged = 1,
    Moved = 2,
};

enum class StippleShape : unsigned char {
    Circle = 0,
    Line = 1,
    Rectangle = 2,
    Rhombus = 3,
    Ellipse = 4,
    Triangle = 5,
    RoundedLine = 6,
    RoundedRectangle = 7,
    RoundedRhombus = 8,
    RoundedTriangle = 9
};

struct Stipple {
    Color color;
    StippleState state;

    StippleShape shape;
    float shapeParameter;
    float shapeRadius;

    float size;

    Point center;
    float rotation;
};

#endif
