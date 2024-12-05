
#ifndef ICLA_INTERNAL_H
#define ICLA_INTERNAL_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <ctype.h>

#if defined( _WIN32 ) || defined( _WIN64 )

    #include "icla_winthread.h"
    #include <windows.h>
    #include <limits.h>
    #include <io.h>

    #ifndef __NVCC__

        #include <float.h>
        #define copysign(x,y) _copysign(x,y)
        #define isnan(x)      _isnan(x)
        #define isinf(x)      ( ! _finite(x) && ! _isnan(x) )
        #define isfinite(x)   _finite(x)

        #define snprintf _snprintf

    #endif

#else

    #include <pthread.h>
    #include <unistd.h>
    #include <inttypes.h>

    #define HAVE_PTHREAD_KEY

#endif

#include "pthread_barrier.h"

#include "icla_v2.h"
#include "icla_lapack.h"
#include "icla_operators.h"
#include "icla_threadsetting.h"

struct icla_queue
{
#ifdef __cplusplus
public:

    icla_device_t   device()          { return device__;   }

    #ifdef ICLA_HAVE_CUDA

    cudaStream_t        cuda_stream()       { return stream__;   }
    cublasHandle_t      cublas_handle()     { return cublas__;   }
    cusparseHandle_t    cusparse_handle()   { return cusparse__; }
    cusolverDnHandle_t  cusolverdn_handle() { return cusolverdn__; }

    #endif

    void setup_ptrArray() {
        if(ptrArray__ == NULL) {
            icla_malloc((void**)&(ptrArray__), 3 * maxbatch__ * sizeof(void*));
            assert( ptrArray__ != NULL);
            dAarray__ = ptrArray__;
            dBarray__ = dAarray__ + maxbatch__;
            dCarray__ = dBarray__ + maxbatch__;
        }
    }

    #ifdef ICLA_HAVE_HIP

    hipStream_t      hip_stream()      { return stream__; };
    hipblasHandle_t  hipblas_handle()  { return hipblas__; };
    hipsparseHandle_t hipsparse_handle() { return hipsparse__; };
    hipsolverHandle_t hipsolver_handle() { return hipsolver__;}

    #endif

    void** get_dAarray() {
        if(ptrArray__ == NULL) setup_ptrArray();
        return dAarray__;
    }

    void** get_dBarray() {
        if(ptrArray__ == NULL) setup_ptrArray();
        return dBarray__;
    }

    void** get_dCarray() {
        if(ptrArray__ == NULL) setup_ptrArray();
        return dCarray__;
    }

    icla_int_t get_maxBatch() {return (icla_int_t)maxbatch__; }

protected:
    friend
    void icla_queue_create_internal(
        icla_device_t device, icla_queue_t* queuePtr,
        const char* func, const char* file, int line );

    #ifdef ICLA_HAVE_CUDA
    friend
    void icla_queue_create_from_cuda_internal(
        icla_device_t   device,
        cudaStream_t     stream,
        cublasHandle_t   cublas_handle,
        cusparseHandle_t cusparse_handle,
        icla_queue_t*   queuePtr,
        const char* func, const char* file, int line );
    #endif

    #ifdef ICLA_HAVE_HIP
    friend
    void icla_queue_create_from_hip_internal(
        icla_device_t    device,
        hipStream_t       stream,
        hipblasHandle_t   hipblas_handle,
        hipsparseHandle_t hipsparse_handle,
        icla_queue_t*    queuePtr,
        const char* func, const char* file, int line );
    #endif

    friend
    void icla_queue_destroy_internal(
        icla_queue_t queue,
        const char* func, const char* file, int line );
#endif

    int              own__;
    icla_device_t   device__;

    int              maxbatch__;

    void**           ptrArray__;

    void**           dAarray__;

    void**           dBarray__;

    void**           dCarray__;

    #ifdef ICLA_HAVE_CUDA
    cudaStream_t     stream__;
    cublasHandle_t   cublas__;
    cusparseHandle_t cusparse__;
    cusolverDnHandle_t cusolverdn__;

    #endif

    #ifdef ICLA_HAVE_HIP
    hipStream_t      stream__;

    hipblasHandle_t  hipblas__;
    hipsparseHandle_t hipsparse__;
    hipsolverHandle_t hipsolver__;

    #endif
};

#ifdef __cplusplus
extern "C" {
#endif

icla_queue_t iclablasGetQueue();

#ifdef __cplusplus
}
#endif

#if defined(linux) || defined(__linux) || defined(__linux__)
#if defined(__GNUC_EXCL__) || defined(__GNUC__)
#define ICLA_HAVE_WEAK    1
#endif
#endif

#ifndef max
#define max(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef min
#define min(a, b) ((a) < (b) ? (a) : (b))
#endif

#define ICLA_UNUSED(var)  ((void)var)

#endif

