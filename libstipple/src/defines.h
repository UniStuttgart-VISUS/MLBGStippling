#ifndef DEFINES_H
#define DEFINES_H

#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif

#endif
