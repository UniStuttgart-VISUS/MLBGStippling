#ifndef UTILS_DIAGNOSTICS_CUH
#define UTILS_DIAGNOSTICS_CUH

#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(USE_NVTOOLSEXT)
#include <nvToolsExt.h>
#endif

#define cuda_unwrap(expr) \
    { cuda_unwraperr((expr), __FILE__, __LINE__); }

#if defined(USE_NVTOOLSEXT) || defined(_DEBUG)
// (1) Kernel launch argument error
// (2) Kernel execution error
#define cuda_debug_synchronize()                                     \
    {                                                                \
        cuda_unwraperr(cudaPeekAtLastError(), __FILE__, __LINE__);   \
        cuda_unwraperr(cudaDeviceSynchronize(), __FILE__, __LINE__); \
    }
#else
#define cuda_debug_synchronize()
#endif

inline void cuda_unwraperr(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "error: \"%s\" at\n  %s:%d\n", cudaGetErrorString(code), file, line);
        fflush(stderr);
        assert(false && "CUDA error");
        if (abort) {
            exit(code);
        }
    }
}

inline void cuda_debug_push(const char* name) {
#if defined(USE_NVTOOLSEXT)
    nvtxRangePush(name);
#endif
}

inline void cuda_debug_pop() {
#if defined(USE_NVTOOLSEXT)
    nvtxRangePop();
#endif
}

#endif
