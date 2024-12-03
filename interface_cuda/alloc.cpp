
#include <stdlib.h>
#include <stdio.h>

#ifdef DEBUG_MEMORY
#include <map>
#include <mutex>

#endif

#include <cuda_runtime.h>

#include "icla_v2.h"
#include "icla_internal.h"
#include "error.h"

#ifdef DEBUG_MEMORY
std::mutex                g_pointers_mutex;

std::map< void*, size_t > g_pointers_dev;
std::map< void*, size_t > g_pointers_cpu;
std::map< void*, size_t > g_pointers_pin;
#endif

extern "C" icla_int_t
icla_malloc( icla_ptr* ptrPtr, size_t size )
{

    if ( size == 0 )
        size = sizeof(iclaDoubleComplex);
    if ( cudaSuccess != cudaMalloc( ptrPtr, size )) {
        return ICLA_ERR_DEVICE_ALLOC;
    }

    #ifdef DEBUG_MEMORY
    g_pointers_mutex.lock();
    g_pointers_dev[ *ptrPtr ] = size;
    g_pointers_mutex.unlock();
    #endif

    return ICLA_SUCCESS;
}

extern "C" icla_int_t
icla_free_internal( icla_ptr ptr,
    const char* func, const char* file, int line )
{
    #ifdef DEBUG_MEMORY
    g_pointers_mutex.lock();
    if ( ptr != NULL && g_pointers_dev.count( ptr ) == 0 ) {
        fprintf( stderr, "icla_free( %p ) that wasn't allocated with icla_malloc.\n", ptr );
    }
    else {
        g_pointers_dev.erase( ptr );
    }
    g_pointers_mutex.unlock();
    #endif

    cudaError_t err = cudaFree( ptr );
    check_xerror( err, func, file, line );
    if ( err != cudaSuccess ) {
        return ICLA_ERR_INVALID_PTR;
    }
    return ICLA_SUCCESS;
}

extern "C" icla_int_t
icla_malloc_cpu( void** ptrPtr, size_t size )
{

    if ( size == 0 )
        size = sizeof(iclaDoubleComplex);
#if 1
#if defined( _WIN32 ) || defined( _WIN64 )
    *ptrPtr = _aligned_malloc( size, 64 );
    if ( *ptrPtr == NULL ) {
        return ICLA_ERR_HOST_ALLOC;
    }
#else
    int err = posix_memalign( ptrPtr, 64, size );
    if ( err != 0 ) {
        *ptrPtr = NULL;
        return ICLA_ERR_HOST_ALLOC;
    }
#endif
#else
    *ptrPtr = malloc( size );
    if ( *ptrPtr == NULL ) {
        return ICLA_ERR_HOST_ALLOC;
    }
#endif

    #ifdef DEBUG_MEMORY
    g_pointers_mutex.lock();
    g_pointers_cpu[ *ptrPtr ] = size;
    g_pointers_mutex.unlock();
    #endif

    return ICLA_SUCCESS;
}

extern "C" icla_int_t
icla_free_cpu( void* ptr )
{
    #ifdef DEBUG_MEMORY
    g_pointers_mutex.lock();
    if ( ptr != NULL && g_pointers_cpu.count( ptr ) == 0 ) {
        fprintf( stderr, "icla_free_cpu( %p ) that wasn't allocated with icla_malloc_cpu.\n", ptr );
    }
    else {
        g_pointers_cpu.erase( ptr );
    }
    g_pointers_mutex.unlock();
    #endif

#if defined( _WIN32 ) || defined( _WIN64 )
    _aligned_free( ptr );
#else
    free( ptr );
#endif
    return ICLA_SUCCESS;
}

extern "C" icla_int_t
icla_malloc_pinned( void** ptrPtr, size_t size )
{

    if ( size == 0 )
        size = sizeof(iclaDoubleComplex);
    if ( cudaSuccess != cudaHostAlloc( ptrPtr, size, cudaHostAllocPortable )) {
        return ICLA_ERR_HOST_ALLOC;
    }

    #ifdef DEBUG_MEMORY
    g_pointers_mutex.lock();
    g_pointers_pin[ *ptrPtr ] = size;
    g_pointers_mutex.unlock();
    #endif

    return ICLA_SUCCESS;
}

extern "C" icla_int_t
icla_free_pinned_internal( void* ptr,
    const char* func, const char* file, int line )
{
    #ifdef DEBUG_MEMORY
    g_pointers_mutex.lock();
    if ( ptr != NULL && g_pointers_pin.count( ptr ) == 0 ) {
        fprintf( stderr, "icla_free_pinned( %p ) that wasn't allocated with icla_malloc_pinned.\n", ptr );
    }
    else {
        g_pointers_pin.erase( ptr );
    }
    g_pointers_mutex.unlock();
    #endif

    cudaError_t err = cudaFreeHost( ptr );
    check_xerror( err, func, file, line );
    if ( cudaSuccess != err ) {
        return ICLA_ERR_INVALID_PTR;
    }
    return ICLA_SUCCESS;
}

extern "C" icla_int_t
icla_mem_info(size_t * freeMem, size_t * totalMem) {
    cudaMemGetInfo(freeMem, totalMem);
    return ICLA_SUCCESS;
}

extern "C" icla_int_t
icla_memset(void * ptr, int value, size_t count) {
    return cudaMemset(ptr, value, count);
}

extern "C" icla_int_t
icla_memset_async(void * ptr, int value, size_t count, icla_queue_t queue) {
#ifdef ICLA_HAVE_CUDA

    return cudaMemsetAsync(ptr, value, count, queue->cuda_stream());
#elif defined(ICLA_HAVE_HIP)
    return hipMemsetAsync(ptr, value, count, queue->hip_stream());
#endif
}

