#ifndef MAP_H
#define MAP_H

#include <vector>

template <typename T>
struct Map {
    int width = 0;
    int height = 0;
    std::vector<T> pixels;
};

#endif
