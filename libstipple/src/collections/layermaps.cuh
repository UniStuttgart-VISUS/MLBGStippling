#ifndef COLLECTIONS_LAYERMAPS_H
#define COLLECTIONS_LAYERMAPS_H

#include "utils/diagnostics.cuh"
#include <cstdint>
#include <cuda_runtime.h>
#include <memory>
#include <vector>

template <typename T>
class LayerMapsBase {
public:
    __host__ __device__ inline int layers() const { return m_layers; }
    __host__ __device__ inline int width() const { return m_width; }
    __host__ __device__ inline int height() const { return m_height; }

    __host__ __device__ inline int size() const {
        return m_width * m_height * m_layers;
    }

    __host__ __device__ inline int capacity() const {
        return m_capacity;
    }

    __device__ inline const T operator()(int layerIndex, int x, int y) const {
        //return m_data[layerIndex * m_width * m_height + y * m_width + x];
        return __ldg(m_data + (layerIndex * m_width * m_height + y * m_width + x));
    }

    __device__ inline T& operator()(int layerIndex, int x, int y) {
        return m_data[layerIndex * m_width * m_height + y * m_width + x];
    }

protected:
    LayerMapsBase(int layers, int width, int height, T* __restrict__ data)
        : m_layers(layers)
        , m_width(width)
        , m_height(height)
        , m_capacity(layers * width * height)
        , m_data(data) { }

    ~LayerMapsBase() = default;

    int m_layers;
    int m_width;
    int m_height;
    int m_capacity;
    T* __restrict__ m_data;
};

template <typename T, bool Owned>
class LayerMaps;

template <typename T>
class LayerMaps<T, false> : public LayerMapsBase<T> {
public:
    friend class LayerMaps<T, true>;
	
    inline LayerMaps() = delete;

    inline ~LayerMaps() = default;

    template <typename U, bool O>
    inline LayerMaps(LayerMaps<U, O> const& other)
        : LayerMapsBase<T>(other.m_layers, other.m_width, other.m_height, other.m_data) {
        this->m_capacity = other.m_capacity;
    }
    template <typename U, bool O>
    inline LayerMaps& operator=(LayerMaps<U, O> const& other) {
        this->m_layers = other.m_layers;
        this->m_width = other.m_width;
        this->m_height = other.m_height;
        this->m_capacity = other.m_capacity;
        this->m_data = other.m_data;
        return *this;
    }
};

template <typename T>
class LayerMaps<T, true> : public LayerMapsBase<T> {
public:
    friend class LayerMaps<T, false>;

    __host__ inline explicit LayerMaps(int layers, int width, int height)
        : LayerMapsBase<T>(layers, width, height, nullptr) {
#pragma warning(suppress : 4068)
#pragma nv_diag_suppress = restrict_qualifier_dropped
        cuda_unwrap(cudaMalloc((void**)&this->m_data, this->size() * sizeof(T)));
#pragma warning(suppress : 4068)
#pragma nv_diag_default = restrict_qualifier_dropped
    }

    __host__ inline ~LayerMaps() {
        cuda_unwrap(cudaFree(this->m_data));
    }

    inline LayerMaps(LayerMaps const&) = delete;
    inline LayerMaps& operator=(LayerMaps const&) = delete;

    inline LayerMaps(LayerMaps&& other)
        : LayerMapsBase<T>(other.m_layers, other.m_width, other.m_height, other.m_data) {
        this->m_capacity = other.m_capacity;
        other.m_data = nullptr;
    }
    inline LayerMaps& operator=(LayerMaps&& other) {
        this->m_layers = other.m_layers;
        this->m_width = other.m_width;
        this->m_height = other.m_height;
        this->m_capacity = other.m_capacity;
        this->m_data = other.m_data;
        other.m_data = nullptr;
        return *this;
    }

    __host__ inline T* data() {
        return this->m_data;
    }

    __host__ inline const T* data() const {
        return this->m_data;
    }

    __host__ inline void reserve(int capacity) {
        if (capacity < this->m_capacity) {
            return;
        }
        T* newData;
        cuda_unwrap(cudaMalloc(&newData, capacity * sizeof(T)));
        cuda_unwrap(cudaMemcpy(newData, this->m_data, this->size() * sizeof(T), cudaMemcpyDeviceToDevice));
        cuda_unwrap(cudaFree(this->m_data));
        this->m_capacity = capacity;
        this->m_data = newData;
    }

    __host__ inline void setOnDevice(T value) {
        cuda_unwrap(cudaMemset(this->m_data, value, this->m_capacity * sizeof(T)));
    }

    template <class InputIt, class UnaryOperation>
    __host__ inline void copyToDevice(InputIt begin, InputIt end, UnaryOperation unaryOp) {
        int layerIndex = 0;
        for (InputIt it = begin; it != end; ++it) {
            cuda_unwrap(cudaMemcpy(&this->m_data[this->m_width * this->m_height * layerIndex], unaryOp(*it),
                this->m_width * this->m_height * sizeof(T), cudaMemcpyHostToDevice));
            ++layerIndex;
        }
    }

    template <class BinaryOp>
    __host__ inline void copyToHost(BinaryOp binaryOp) const {
        T* scan0 = binaryOp(int(), static_cast<T*>(nullptr));
        for (int layerIndex = 0; layerIndex < this->m_layers; ++layerIndex) {
            cuda_unwrap(cudaMemcpy(scan0, &this->m_data[this->m_width * this->m_height * layerIndex],
                this->m_width * this->m_height * sizeof(T), cudaMemcpyDeviceToHost));
            scan0 = binaryOp(layerIndex, scan0);
            if (!scan0) {
                break;
            }
        }
    }
};

#endif
