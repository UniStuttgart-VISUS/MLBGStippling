#ifndef COLLECTIONS_LAYERVECTOR_CUH
#define COLLECTIONS_LAYERVECTOR_CUH

#include "utils/diagnostics.cuh"
#include <algorithm>
#include <cuda_runtime.h>
#include <tuple>
#include <vector>

template <typename T, typename I>
class LayerVectorBase {
public:
    static const int GrowthFactor = 4;

    __host__ __device__ inline int layers() const {
        return m_layers;
    }

    __host__ __device__ inline int size() const {
        return m_size;
    }

    __host__ __device__ inline int capacity() const {
        return m_capacity;
    }

    __host__ __device__ inline bool empty() const {
        return m_layers == 0 || m_size == 0;
    }

    __device__ inline const I layerIndex(int itemIndex) const {
        //return m_layerIndices[itemIndex];
        return __ldg(m_layerIndices + itemIndex);
    }

    __device__ inline I& layerIndex(int itemIndex) {
        return m_layerIndices[itemIndex];
    }

    __device__ inline const T& operator[](int itemIndex) const {
        return m_data[itemIndex];
    }

    __device__ inline T& operator[](int itemIndex) {
        return m_data[itemIndex];
    }

    __host__ inline void setSize(int size) {
        assert(size <= m_capacity && "Size must be below capacity");
        m_size = size;
    }

protected:
    LayerVectorBase(int layers, int size, I* __restrict__ layerIndices, T* __restrict__ data)
        : m_layers(layers)
        , m_size(size)
        , m_capacity(std::max(8192, m_size * GrowthFactor))
        , m_layerIndices(layerIndices)
        , m_data(data) {}

    ~LayerVectorBase() = default;

    int m_layers;
    int m_size;
    int m_capacity;
    I* __restrict__ m_layerIndices;
    T* __restrict__ m_data;
};

template <typename T, typename I, bool Owned>
class LayerVector;

template <typename T, typename I>
class LayerVector<T, I, false> : public LayerVectorBase<T, I> {
public:
    friend class LayerVector<T, I, true>;

    inline LayerVector() = delete;

    inline ~LayerVector() = default;

    template <typename U, typename J, bool O>
    inline LayerVector(LayerVector<U, J, O> const& other)
        : LayerVectorBase<T, I>(other.m_layers, other.m_size, other.m_layerIndices, other.m_data) {
        this->m_capacity = other.m_capacity;
    }
    template <typename U, typename J, bool O>
    inline LayerVector& operator=(LayerVector<U, J, O> const& other) {
        this->m_layers = other.m_layers;
        this->m_size = other.m_size;
        this->m_capacity = other.m_capacity;
        this->m_layerIndices = other.m_layerIndices;
        this->m_data = other.m_data;
        return *this;
    }
};

template <typename T, typename I>
class LayerVector<T, I, true> : public LayerVectorBase<T, I> {
public:
    friend class LayerVector<T, I, false>;

    __host__ inline explicit LayerVector()
        : LayerVectorBase<T, I>(0, 0, nullptr, nullptr) {
    }

    __host__ inline explicit LayerVector(int layers, int size)
        : LayerVectorBase<T, I>(layers, size, nullptr, nullptr) {
#pragma warning(suppress : 4068)
#pragma nv_diag_suppress = restrict_qualifier_dropped
        cuda_unwrap(cudaMalloc((void**)&this->m_layerIndices, this->m_capacity * sizeof(I)));
        cuda_unwrap(cudaMalloc((void**)&this->m_data, this->m_capacity * sizeof(T)));
#pragma warning(suppress : 4068)
#pragma nv_diag_default = restrict_qualifier_dropped
    }

    __host__ inline ~LayerVector() {
        cuda_unwrap(cudaFree(this->m_layerIndices));
        cuda_unwrap(cudaFree(this->m_data));
    }

    inline LayerVector(LayerVector const&) = delete;
    inline LayerVector& operator=(LayerVector const&) = delete;

    inline LayerVector(LayerVector&& other)
        : LayerVectorBase<T, I>(other.m_layers, other.m_size, other.m_layerIndices, other.m_data) {
        this->m_capacity = other.m_capacity;
        other.m_layerIndices = nullptr;
        other.m_data = nullptr;
    }
    inline LayerVector& operator=(LayerVector&& other) {
        this->m_layers = other.m_layers;
        this->m_size = other.m_size;
        this->m_capacity = other.m_capacity;
        this->m_layerIndices = other.m_layerIndices;
        this->m_data = other.m_data;
        other.m_layerIndices = nullptr;
        other.m_data = nullptr;
        return *this;
    }

    __host__ inline I* layerIndices() {
        return this->m_layerIndices;
    }

    __host__ inline const I* layerIndices() const {
        return this->m_layerIndices;
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
        I* newLayerIndices;
        T* newData;
        cuda_unwrap(cudaMalloc(&newLayerIndices, capacity * sizeof(I)));
        cuda_unwrap(cudaMalloc(&newData, capacity * sizeof(T)));
        cuda_unwrap(cudaMemcpy(newLayerIndices, this->m_layerIndices, this->m_size * sizeof(I), cudaMemcpyDeviceToDevice));
        cuda_unwrap(cudaMemcpy(newData, this->m_data, this->m_size * sizeof(T), cudaMemcpyDeviceToDevice));
        cuda_unwrap(cudaFree(this->m_layerIndices));
        cuda_unwrap(cudaFree(this->m_data));
        this->m_capacity = capacity;
        this->m_layerIndices = newLayerIndices;
        this->m_data = newData;
    }

    template <class InputIt, class UnaryOperation>
    __host__ inline void copyToDevice(InputIt begin, InputIt end, UnaryOperation unaryOp) {
        int index = 0;
        std::vector<I> hostLayerIndices(this->m_size);
        std::vector<T> hostData(this->m_size);
        for (InputIt it = begin; it != end; ++it) {
            auto tuple = unaryOp(*it);
            hostLayerIndices[index] = std::get<0>(tuple);
            hostData[index] = std::get<1>(tuple);
            ++index;
        }
        cuda_unwrap(cudaMemcpy(this->m_layerIndices, hostLayerIndices.data(),
            this->m_size * sizeof(I), cudaMemcpyHostToDevice));
        cuda_unwrap(cudaMemcpy(this->m_data, hostData.data(),
            this->m_size * sizeof(T), cudaMemcpyHostToDevice));
    }

    template <class InputIt>
    __host__ inline void copyToDevice(InputIt begin, InputIt end) {
        copyToDevice(begin, end, [](const auto& tuple) { return tuple; });
    }

    template <class BinaryOp>
    __host__ inline void copyToHost(BinaryOp binaryOp) const {
        std::vector<I> hostLayerIndices(this->m_size);
        std::vector<T> hostData(this->m_size);
        cuda_unwrap(cudaMemcpy(hostLayerIndices.data(), this->m_layerIndices, this->m_size * sizeof(I), cudaMemcpyDeviceToHost));
        cuda_unwrap(cudaMemcpy(hostData.data(), this->m_data, this->m_size * sizeof(T), cudaMemcpyDeviceToHost));
        for (int index = 0; index < this->m_size; ++index) {
            binaryOp(hostLayerIndices[index], hostData[index]);
        }
    }
};

#endif
