/*
    -- ICLA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates
*/

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

// these are included already in icla_internal.h & other headers
#include <cuda_runtime.h>
//#include <cublas_v2.h>

#include "icla_internal.h"
#include "error.h"


#if defined(ICLA_HAVE_CUDA) || defined(ICLA_HAVE_HIP)
#ifndef ICLA_NO_V1

// -----------------------------------------------------------------------------
// globals
// see interface.cpp for definitions

#ifndef ICLA_NO_V1
    extern icla_queue_t* g_null_queues;

    #ifdef HAVE_PTHREAD_KEY
    extern pthread_key_t g_icla_queue_key;
    #else
    extern icla_queue_t g_icla_queue;
    #endif
#endif // ICLA_NO_V1


// -----------------------------------------------------------------------------
extern int g_icla_devices_cnt;


// =============================================================================
// device support

/***************************************************************************//**
    @deprecated
    Synchronize the current device.
    This functionality does not exist in OpenCL, so it is deprecated for CUDA, too.

    @ingroup icla_device
*******************************************************************************/
extern "C" void
icla_device_sync()
{
    cudaError_t err;
    err = cudaDeviceSynchronize();
    check_error( err );
    ICLA_UNUSED( err );
}


// =============================================================================
// queue support

/***************************************************************************//**
    @deprecated

    Sets the current global ICLA v1 queue for kernels to execute in.
    In ICLA v2, all kernels take queue as an argument, so this is deprecated.
    If compiled with ICLA_NO_V1, this is not defined.

    @param[in]
    queue       Queue to set as current global ICLA v1 queue.

    @return ICLA_SUCCESS if successful

    @ingroup icla_queue
*******************************************************************************/
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


/***************************************************************************//**
    @deprecated

    Gets the current global ICLA v1 queue for kernels to execute in.
    In ICLA v2, all kernels take queue as an argument, so this is deprecated.
    If compiled with ICLA_NO_V1, this is not defined.

    @param[out]
    queue_ptr    On output, set to the current global ICLA v1 queue.

    @return ICLA_SUCCESS if successful

    @ingroup icla_queue
*******************************************************************************/
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


/***************************************************************************//**
    @deprecated

    Gets the current global ICLA v1 queue for kernels to execute in.
    Unlike iclablasGetKernelStream(), if the current queue is NULL,
    this will return a special ICLA queue that has a NULL CUDA stream.
    This allows ICLA v1 wrappers to call v2 kernels with a non-NULL queue.

    In ICLA v2, all kernels take queue as an argument, so this is deprecated.
    If compiled with ICLA_NO_V1, this is not defined.

    @return Current global ICLA v1 queue.

    @ingroup icla_queue
*******************************************************************************/
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
        // create queue w/ NULL stream first time that NULL queue is used
        if ( g_null_queues[dev] == NULL ) {
            #ifdef ICLA_HAVE_CUDA
            icla_queue_create_from_cuda( dev, NULL, NULL, NULL, &g_null_queues[dev] );
            #elif defined(ICLA_HAVE_HIP)
            icla_queue_create_from_hip( dev, NULL, NULL, NULL, &g_null_queues[dev] );
            #endif
            //printf( "dev %lld create queue %p\n", (long long) dev, (void*) g_null_queues[dev] );
            assert( g_null_queues[dev] != NULL );
        }
        queue = g_null_queues[dev];
    }
    assert( queue != NULL );
    return queue;
}


/******************************************************************************/
// @deprecated
// ICLA v1 version that doesn't take device ID.
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

#endif // not ICLA_NO_V1
#endif // ICLA_HAVE_CUDA
