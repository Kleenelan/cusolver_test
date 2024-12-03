
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <cuda_runtime.h>

#include "icla_internal.h"
#include "error.h"

#if defined(ICLA_HAVE_CUDA) || defined(ICLA_HAVE_HIP)
#ifndef ICLA_NO_V1

#ifndef ICLA_NO_V1
    extern icla_queue_t* g_null_queues;

    #ifdef HAVE_PTHREAD_KEY
    extern pthread_key_t g_icla_queue_key;
    #else
    extern icla_queue_t g_icla_queue;
    #endif
#endif

extern int g_icla_devices_cnt;

extern "C" void
icla_device_sync()
{
    cudaError_t err;
    err = cudaDeviceSynchronize();
    check_error( err );
    ICLA_UNUSED( err );
}

extern "C" icla_int_t
iclablasSetKernelStream( icla_queue_t queue )
{
    icla_int_t info = 0;
    #ifdef HAVE_PTHREAD_KEY
    info = pthread_setspecific( g_icla_queue_key, queue );
    #else
    g_icla_queue = queue;
    #endif
    return info;
}

extern "C" icla_int_t
iclablasGetKernelStream( icla_queue_t *queue_ptr )
{
    #ifdef HAVE_PTHREAD_KEY
    *queue_ptr = (icla_queue_t) pthread_getspecific( g_icla_queue_key );
    #else
    *queue_ptr = g_icla_queue;
    #endif
    return 0;
}

extern "C"
icla_queue_t iclablasGetQueue()
{
    icla_queue_t queue;
    #ifdef HAVE_PTHREAD_KEY
    queue = (icla_queue_t) pthread_getspecific( g_icla_queue_key );
    #else
    queue = g_icla_queue;
    #endif
    if ( queue == NULL ) {
        icla_device_t dev;
        icla_getdevice( &dev );
        if ( dev >= g_icla_devices_cnt || g_null_queues == NULL ) {
            fprintf( stderr, "Error: %s requires icla_init() to be called first for ICLA v1 compatability.\n",
                     __func__ );
            return NULL;
        }

        if ( g_null_queues[dev] == NULL ) {
            #ifdef ICLA_HAVE_CUDA
            icla_queue_create_from_cuda( dev, NULL, NULL, NULL, &g_null_queues[dev] );
            #elif defined(ICLA_HAVE_HIP)
            icla_queue_create_from_hip( dev, NULL, NULL, NULL, &g_null_queues[dev] );
            #endif

            assert( g_null_queues[dev] != NULL );
        }
        queue = g_null_queues[dev];
    }
    assert( queue != NULL );
    return queue;
}

extern "C" void
icla_queue_create_v1_internal(
    icla_queue_t* queue_ptr,
    const char* func, const char* file, int line )
{
    int device;
    cudaError_t err;
    err = cudaGetDevice( &device );
    check_xerror( err, func, file, line );
    ICLA_UNUSED( err );

    icla_queue_create_internal( device, queue_ptr, func, file, line );
}

#endif

#endif

