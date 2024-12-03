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

#include <map>

#if __cplusplus >= 201103  // C++11 standard
#include <mutex>
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(ICLA_WITH_MKL)
#include <mkl_service.h>
#endif

#if defined(ICLA_WITH_ACML)
#include <acml.h>
#endif

#include <cuda_runtime.h>

// defining ICLA_LAPACK_H is a hack to NOT include icla_lapack.h
// via icla_internal.h here, since it conflicts with acml.h and we don't
// need lapack here, but we want acml.h for the acmlversion() function.
#define ICLA_LAPACK_H

#include "icla_internal.h"
#include "error.h"

#define MAX_BATCHCOUNT    (65534)

#if defined(ICLA_HAVE_CUDA) || defined(ICLA_HAVE_HIP)

#ifdef DEBUG_MEMORY
// defined in alloc.cpp
extern std::map< void*, size_t > g_pointers_dev;
extern std::map< void*, size_t > g_pointers_cpu;
extern std::map< void*, size_t > g_pointers_pin;
#endif

// -----------------------------------------------------------------------------
// prototypes
extern "C" void
icla_warn_leaks( const std::map< void*, size_t >& pointers, const char* type );


// -----------------------------------------------------------------------------
// constants

// bit flags
enum {
    own_none     = 0x0000,
    own_stream   = 0x0001,
    own_cublas   = 0x0002,
    own_cusparse = 0x0004,
    own_opencl   = 0x0008,
    own_hip      = 0x0010,
    own_hipblas  = 0x0020,
    own_hipsparse= 0x0040
};


// -----------------------------------------------------------------------------
// globals
#if __cplusplus >= 201103  // C++11 standard
    static std::mutex g_mutex;
#else
    // without C++11, wrap pthread mutex
    class PthreadMutex {
    public:
        PthreadMutex()
        {
            int err = pthread_mutex_init( &mutex, NULL );
            if ( err ) {
                fprintf( stderr, "pthread_mutex_init failed: %d\n", err );
            }
        }

        ~PthreadMutex()
        {
            int err = pthread_mutex_destroy( &mutex );
            if ( err ) {
                fprintf( stderr, "pthread_mutex_destroy failed: %d\n", err );
            }
        }

        void lock()
        {
            int err = pthread_mutex_lock( &mutex );
            if ( err ) {
                fprintf( stderr, "pthread_mutex_lock failed: %d\n", err );
            }
        }

        void unlock()
        {
            int err = pthread_mutex_unlock( &mutex );
            if ( err ) {
                fprintf( stderr, "pthread_mutex_unlock failed: %d\n", err );
            }
        }

    private:
        pthread_mutex_t mutex;
    };

    static PthreadMutex g_mutex;
#endif

// count of (init - finalize) calls
static int g_init = 0;

#ifndef ICLA_NO_V1
    icla_queue_t* g_null_queues = NULL;

    #ifdef HAVE_PTHREAD_KEY
    pthread_key_t g_icla_queue_key;
    #else
    icla_queue_t g_icla_queue = NULL;
    #endif
#endif // ICLA_NO_V1


// -----------------------------------------------------------------------------
// subset of the CUDA device properties, set by icla_init()
struct icla_device_info
{
    size_t memory;
    size_t shmem_block;       // maximum shared memory per thread block in bytes
    size_t shmem_block_optin; // maximum shared memory per thread block in bytes with opt-in
    size_t shmem_multiproc;   // maximum shared memory per multiprocessor in bytes
    icla_int_t gpu_arch;
    icla_int_t multiproc_count;        // number of multiprocessors
    icla_int_t num_threads_block;      // max. #threads per block
    icla_int_t num_threads_multiproc;  // max. #threads per sm
};

int g_icla_devices_cnt = 0;
struct icla_device_info* g_icla_devices = NULL;


// =============================================================================
// initialization

/***************************************************************************//**
    Initializes the ICLA library.
    Caches information about available CUDA devices.

    Every icla_init call must be paired with a icla_finalize call.
    Only one thread needs to call icla_init and icla_finalize,
    but every thread may call it. If n threads call icla_init,
    the n-th call to icla_finalize will release resources.

    When renumbering CUDA devices, call cudaSetValidDevices before calling icla_init.
    When setting CUDA device flags, call cudaSetDeviceFlags before calling icla_init.

    @retval ICLA_SUCCESS
    @retval ICLA_ERR_UNKNOWN
    @retval ICLA_ERR_HOST_ALLOC

    @see icla_finalize

    @ingroup icla_init
*******************************************************************************/
extern "C" icla_int_t
icla_init()
{
    icla_int_t info = 0;

    g_mutex.lock();
    {
        if ( g_init == 0 ) {
            // query number of devices
            cudaError_t err;
            g_icla_devices_cnt = 0;
            err = cudaGetDeviceCount( &g_icla_devices_cnt );
            if ( err != 0 && err != cudaErrorNoDevice ) {
                info = ICLA_ERR_UNKNOWN;
                goto cleanup;
            }

            // allocate list of devices
            size_t size;
            size = max( 1, g_icla_devices_cnt ) * sizeof(struct icla_device_info);
            icla_malloc_cpu( (void**) &g_icla_devices, size );
            if ( g_icla_devices == NULL ) {
                info = ICLA_ERR_HOST_ALLOC;
                goto cleanup;
            }
            memset( g_icla_devices, 0, size );

            // query each device
            for( int dev=0; dev < g_icla_devices_cnt; ++dev ) {
                cudaDeviceProp prop;
                err = cudaGetDeviceProperties( &prop, dev );
                if ( err != 0 ) {
                    info = ICLA_ERR_UNKNOWN;
                }
                else {
                    g_icla_devices[dev].memory                = prop.totalGlobalMem;
                    g_icla_devices[dev].num_threads_block     = prop.maxThreadsPerBlock;
                    g_icla_devices[dev].num_threads_multiproc = prop.maxThreadsPerMultiProcessor;
                    g_icla_devices[dev].multiproc_count       = prop.multiProcessorCount;
                    g_icla_devices[dev].shmem_block           = prop.sharedMemPerBlock;

                    g_icla_devices[dev].gpu_arch          = prop.major*100 + prop.minor*10;
                    #ifdef ICLA_HAVE_CUDA
                    g_icla_devices[dev].shmem_multiproc   = prop.sharedMemPerMultiprocessor;
                    // dynamic shared memory in CUDA has a special opt-in since CUDA 9
                    #if CUDA_VERSION >= 9000
                    g_icla_devices[dev].shmem_block_optin = prop.sharedMemPerBlockOptin;
                    #else
                    g_icla_devices[dev].shmem_block_optin = prop.sharedMemPerBlock;
                    #endif

                    #elif defined(ICLA_HAVE_HIP)
                    g_icla_devices[dev].shmem_multiproc   = prop.maxSharedMemoryPerMultiProcessor;
                    g_icla_devices[dev].shmem_block_optin = prop.sharedMemPerBlock;
                    #endif
                }
            }

            #ifndef ICLA_NO_V1
                #ifdef HAVE_PTHREAD_KEY
                    // create thread-specific key
                    // currently, this is needed only for ICLA v1 compatability
                    // see icla_init, iclablas(Set|Get)KernelStream, iclaGetQueue
                    info = pthread_key_create( &g_icla_queue_key, NULL );
                    if ( info != 0 ) {
                        info = ICLA_ERR_UNKNOWN;
                        goto cleanup;
                    }
                #endif

                // ----- queues with NULL streams (for backwards compatability with ICLA 1.x)
                // allocate array of queues with NULL stream
                size = max( 1, g_icla_devices_cnt ) * sizeof(icla_queue_t);
                icla_malloc_cpu( (void**) &g_null_queues, size );
                if ( g_null_queues == NULL ) {
                    info = ICLA_ERR_HOST_ALLOC;
                    goto cleanup;
                }
                memset( g_null_queues, 0, size );
            #endif // ICLA_NO_V1
        }
cleanup:
        g_init += 1;  // increment (init - finalize) count
    }
    g_mutex.unlock();

    return info;
}


/***************************************************************************//**
    Frees information used by the ICLA library.
    @see icla_init

    @ingroup icla_init
*******************************************************************************/
extern "C" icla_int_t
icla_finalize()
{
    icla_int_t info = 0;

    g_mutex.lock();
    {
        if ( g_init <= 0 ) {
            info = ICLA_ERR_NOT_INITIALIZED;
        }
        else {
            g_init -= 1;  // decrement (init - finalize) count
            if ( g_init == 0 ) {
                info = 0;

                if ( g_icla_devices != NULL ) {
                    icla_free_cpu( g_icla_devices );
                    g_icla_devices = NULL;
                }

                #ifndef ICLA_NO_V1
                if ( g_null_queues != NULL ) {
                    for( int dev=0; dev < g_icla_devices_cnt; ++dev ) {
                        icla_queue_destroy( g_null_queues[dev] );
                        g_null_queues[dev] = NULL;
                    }
                    icla_free_cpu( g_null_queues );
                    g_null_queues = NULL;
                }

                #ifdef HAVE_PTHREAD_KEY
                    pthread_key_delete( g_icla_queue_key );
                #endif
                #endif // ICLA_NO_V1

                #ifdef DEBUG_MEMORY
                icla_warn_leaks( g_pointers_dev, "device" );
                icla_warn_leaks( g_pointers_cpu, "CPU" );
                icla_warn_leaks( g_pointers_pin, "CPU pinned" );
                #endif
            }
        }
    }
    g_mutex.unlock();

    return info;
}


// =============================================================================
// testing and debugging support

#ifdef DEBUG_MEMORY
/***************************************************************************//**
    If DEBUG_MEMORY is defined at compile time, prints warnings when
    icla_finalize() is called for any GPU device, CPU, or CPU pinned
    allocations that were not freed.

    @param[in]
    pointers    Hash table mapping allocated pointers to size.

    @param[in]
    type        String describing type of pointers (GPU, CPU, etc.)

    @ingroup icla_testing
*******************************************************************************/
extern "C" void
icla_warn_leaks( const std::map< void*, size_t >& pointers, const char* type )
{
    if ( pointers.size() > 0 ) {
        fprintf( stderr, "Warning: ICLA detected memory leak of %llu %s pointers:\n",
                 (long long unsigned) pointers.size(), type );
        std::map< void*, size_t >::const_iterator iter;
        for( iter = pointers.begin(); iter != pointers.end(); ++iter ) {
            fprintf( stderr, "    pointer %p, size %lu\n", iter->first, iter->second );
        }
    }
}
#endif


/***************************************************************************//**
    Print ICLA version, CUDA version, LAPACK/BLAS library version,
    available GPU devices, number of threads, date, etc.
    Used in testing.
    @ingroup icla_testing
*******************************************************************************/
extern "C" void
icla_print_environment()
{
    icla_int_t major, minor, micro;
    icla_version( &major, &minor, &micro );

    printf( "%% ICLA %lld.%lld.%lld %s %lld-bit icla_int_t, %lld-bit pointer.\n",
            (long long) major, (long long) minor, (long long) micro,
            ICLA_VERSION_STAGE,
            (long long) (8*sizeof(icla_int_t)),
            (long long) (8*sizeof(void*)) );

/* CUDA */

#if defined(ICLA_HAVE_CUDA)

    printf( "%% Compiled for CUDA architectures %s\n", ICLA_CUDA_ARCH );

    // CUDA, OpenCL, OpenMP, MKL, ACML versions all printed on same line
    int cuda_runtime=0, cuda_driver=0;
    cudaError_t err;
    err = cudaDriverGetVersion( &cuda_driver );
    check_error( err );
    err = cudaRuntimeGetVersion( &cuda_runtime );
    if ( err != cudaErrorNoDevice ) {
        check_error( err );
    }
    printf( "%% CUDA runtime %d, driver %d. ", cuda_runtime, cuda_driver );

#endif

/* HIP */

#if defined(ICLA_HAVE_HIP)
    // TODO: add more specifics here

    int hip_runtime=0, hip_driver=0;
    hipError_t err;
    err = hipDriverGetVersion( &hip_driver );
    check_error( err );
    err = hipRuntimeGetVersion( &hip_runtime );
    if ( err != hipErrorNoDevice ) {
        check_error( err );
    }

    printf("%% HIP runtime %d, driver %d. ", hip_runtime, hip_driver );
#endif


/* OpenMP */

#if defined(_OPENMP)
    int omp_threads = 0;
    #pragma omp parallel
    {
        omp_threads = omp_get_num_threads();
    }
    printf( "OpenMP threads %d. ", omp_threads );
#else
    printf( "ICLA not compiled with OpenMP. " );
#endif

#if defined(ICLA_WITH_MKL)
    MKLVersion mkl_version;
    mkl_get_version( &mkl_version );
    printf( "MKL %d.%d.%d, MKL threads %d. ",
            mkl_version.MajorVersion,
            mkl_version.MinorVersion,
            mkl_version.UpdateVersion,
            mkl_get_max_threads() );
#endif

#if defined(ICLA_WITH_ACML)
    // ACML 4 doesn't have acml_build parameter
    int acml_major, acml_minor, acml_patch, acml_build;
    acmlversion( &acml_major, &acml_minor, &acml_patch, &acml_build );
    printf( "ACML %d.%d.%d.%d ", acml_major, acml_minor, acml_patch, acml_build );
#endif

    printf( "\n" );

    // print devices
    int ndevices = 0;
    err = cudaGetDeviceCount( &ndevices );
    if ( err != cudaErrorNoDevice ) {
        check_error( err );
    }
    for( int dev = 0; dev < ndevices; ++dev ) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties( &prop, dev );
        check_error( err );

        #ifdef ICLA_HAVE_CUDA
        printf( "%% device %d: %s, %.1f MHz clock, %.1f MiB memory, capability %d.%d\n",
                dev,
                prop.name,
                prop.clockRate / 1000.,
                prop.totalGlobalMem / (1024.*1024.),
                prop.major,
                prop.minor );

        int arch = prop.major*100 + prop.minor*10;
        if ( arch < ICLA_CUDA_ARCH_MIN ) {
            printf("\n"
                   "==============================================================================\n"
                   "WARNING: ICLA was compiled only for CUDA capability %.1f and higher;\n"
                   "device %d has only capability %.1f; some routines will not run correctly!\n"
                   "==============================================================================\n\n",
                   ICLA_CUDA_ARCH_MIN/100., dev, arch/100. );
        }
        #endif

        #ifdef ICLA_HAVE_HIP
        printf( "%% device %d: %s, %.1f MHz clock, %.1f MiB memory, gcn arch %s\n",
                dev,
                prop.name,
                prop.clockRate / 1000.,
                prop.totalGlobalMem / (1024.*1024.),
                prop.gcnArchName );
        #endif
    }

    ICLA_UNUSED( err );
    time_t t = time( NULL );
    printf( "%% %s", ctime( &t ));
}

#if CUDA_VERSION >= 11000
#define icla_memoryType() type
#else
#define icla_memoryType() memoryType
#endif

/***************************************************************************//**
    For debugging purposes, determines whether a pointer points to CPU or GPU memory.

    On CUDA architecture 2.0 cards with unified addressing, CUDA can tell if
    it is a device pointer or pinned host pointer.
    For malloc'd host pointers, cudaPointerGetAttributes returns error,
    implying it is a (non-pinned) host pointer.

    On older cards, this cannot determine if it is CPU or GPU memory.

    @param[in] A    pointer to test

    @return  1:  if A is a device pointer (definitely),
    @return  0:  if A is a host   pointer (definitely or inferred from error),
    @return -1:  if unknown.

    @ingroup icla_util
*******************************************************************************/
extern "C" icla_int_t
icla_is_devptr( const void* A )
{
    cudaError_t err;
    cudaDeviceProp prop;
    cudaPointerAttributes attr;
    int dev;  // must be int
    err = cudaGetDevice( &dev );
    if ( ! err ) {
        err = cudaGetDeviceProperties( &prop, dev );

        #ifdef ICLA_HAVE_CUDA
        if ( ! err && prop.unifiedAddressing ) {
        #elif defined(ICLA_HAVE_HIP)
        // in HIP, assume all can.
        // There's no corresponding property, and examples show no need to check any properties
        if ( ! err ) {
        #endif

            // I think the cudaPointerGetAttributes prototype is wrong, missing const (mgates)
            err = cudaPointerGetAttributes( &attr, const_cast<void*>( A ));
            if ( ! err ) {
                // definitely know type
                #ifdef ICLA_HAVE_CUDA
                  #if CUDA_VERSION >= 11000
                    return (attr.type == cudaMemoryTypeDevice);
                  #else
                    return (attr.memoryType == cudaMemoryTypeDevice);
                  #endif

                #elif defined(ICLA_HAVE_HIP)
		  #if ROCM_VERSION >= 60000
		    return (attr.type == hipMemoryTypeDevice);
		  #else
                    return (attr.memoryType == hipMemoryTypeDevice);
		  #endif
                #endif
            }
            else if ( err == cudaErrorInvalidValue ) {
                // clear error; see http://icl.cs.utk.edu/icla/forum/viewtopic.php?f=2&t=529
                cudaGetLastError();
                // infer as host pointer
                return 0;
            }
        }
    }
    // clear error
    cudaGetLastError();
    // unknown, e.g., device doesn't support unified addressing
    return -1;
}


// =============================================================================
// device support

/***************************************************************************//**
    Returns CUDA architecture capability for the current device.
    This requires icla_init() to be called first to cache the information.
    Version is an integer xyz, where x is major, y is minor, and z is micro,
    the same as __CUDA_ARCH__. Thus for architecture 1.3.0 it returns 130.

    @return CUDA_ARCH for the current device.

    @ingroup icla_device
*******************************************************************************/
extern "C" icla_int_t
icla_getdevice_arch()
{
    int dev;
    cudaError_t err;
    err = cudaGetDevice( &dev );
    check_error( err );
    ICLA_UNUSED( err );
    if ( g_icla_devices == NULL || dev < 0 || dev >= g_icla_devices_cnt ) {
        fprintf( stderr, "Error in %s: ICLA not initialized (call icla_init() first) or bad device\n", __func__ );
        return 0;
    }
    return g_icla_devices[dev].gpu_arch;
}


/***************************************************************************//**
    Fills in devices array with the available devices.
    (This makes much more sense in OpenCL than in CUDA.)

    @param[out]
    devices     Array of dimension (size).
                On output, devices[0, ..., num_dev-1] contain device IDs.
                Entries >= num_dev are not touched.

    @param[in]
    size        Dimension of the array devices.

    @param[out]
    num_dev     Number of devices, limited to size.

    @ingroup icla_device
*******************************************************************************/
extern "C" void
icla_getdevices(
    icla_device_t* devices,
    icla_int_t  size,
    icla_int_t* num_dev )
{
    cudaError_t err;
    int cnt;
    err = cudaGetDeviceCount( &cnt );
    check_error( err );
    ICLA_UNUSED( err );

    cnt = min( cnt, int(size) );
    for( int i = 0; i < cnt; ++i ) {
        devices[i] = i;
    }
    *num_dev = cnt;
}


/***************************************************************************//**
    Get the current device.

    @param[out]
    device      On output, device ID of the current device.
                Each thread has its own current device.

    @ingroup icla_device
*******************************************************************************/
extern "C" void
icla_getdevice( icla_device_t* device )
{
    int dev;
    cudaError_t err;
    err = cudaGetDevice( &dev );
    *device = dev;
    check_error( err );
    ICLA_UNUSED( err );
}


/***************************************************************************//**
    Set the current device.

    @param[in]
    device      Device ID to set as the current device.
                Each thread has its own current device.

    @ingroup icla_device
*******************************************************************************/
extern "C" void
icla_setdevice( icla_device_t device )
{
    cudaError_t err;
    err = cudaSetDevice( int(device) );
    check_error( err );
    ICLA_UNUSED( err );
}

/***************************************************************************//**
    Returns the multiprocessor count for the current device.
    This requires icla_init() to be called first to cache the information.

    @return the multiprocessor count for the current device.

    @ingroup icla_device
*******************************************************************************/
extern "C" icla_int_t
icla_getdevice_multiprocessor_count()
{
    int dev;
    cudaError_t err;
    err = cudaGetDevice( &dev );
    check_error( err );
    ICLA_UNUSED( err );
    if ( g_icla_devices == NULL || dev < 0 || dev >= g_icla_devices_cnt ) {
        fprintf( stderr, "Error in %s: ICLA not initialized (call icla_init() first) or bad device\n", __func__ );
        return 0;
    }
    return g_icla_devices[dev].multiproc_count;
}

/***************************************************************************//**
    Returns the maximum number of threads per block for the current device.
    This requires icla_init() to be called first to cache the information.

    @return the maximum number of threads per block for the current device.

    @ingroup icla_device
*******************************************************************************/
extern "C" icla_int_t
icla_getdevice_num_threads_block()
{
    int dev;
    cudaError_t err;
    err = cudaGetDevice( &dev );
    check_error( err );
    ICLA_UNUSED( err );
    if ( g_icla_devices == NULL || dev < 0 || dev >= g_icla_devices_cnt ) {
        fprintf( stderr, "Error in %s: ICLA not initialized (call icla_init() first) or bad device\n", __func__ );
        return 0;
    }
    return g_icla_devices[dev].num_threads_block;
}

/***************************************************************************//**
    Returns the maximum number of threads per multiprocessor for the current device.
    This requires icla_init() to be called first to cache the information.

    @return the maximum number of threads per multiprocessor for the current device.

    @ingroup icla_device
*******************************************************************************/
extern "C" icla_int_t
icla_getdevice_num_threads_multiprocessor()
{
    int dev;
    cudaError_t err;
    err = cudaGetDevice( &dev );
    check_error( err );
    ICLA_UNUSED( err );
    if ( g_icla_devices == NULL || dev < 0 || dev >= g_icla_devices_cnt ) {
        fprintf( stderr, "Error in %s: ICLA not initialized (call icla_init() first) or bad device\n", __func__ );
        return 0;
    }
    return g_icla_devices[dev].num_threads_multiproc;
}

/***************************************************************************//**
    Returns the maximum shared memory per block (in bytes) for the current device.
    This requires icla_init() to be called first to cache the information.

    @return the maximum shared memory per block (in bytes) for the current device.

    @ingroup icla_device
*******************************************************************************/
extern "C" size_t
icla_getdevice_shmem_block()
{
    int dev;
    cudaError_t err;
    err = cudaGetDevice( &dev );
    check_error( err );
    ICLA_UNUSED( err );
    if ( g_icla_devices == NULL || dev < 0 || dev >= g_icla_devices_cnt ) {
        fprintf( stderr, "Error in %s: ICLA not initialized (call icla_init() first) or bad device\n", __func__ );
        return 0;
    }
    return g_icla_devices[dev].shmem_block;
}

/***************************************************************************//**
    Returns the maximum shared memory per block (in bytes) with a special opt-in
    for the current device.
    This requires icla_init() to be called first to cache the information.

    @return the maximum shared memory per block (in bytes) with a special opt-in
    for the current device.

    @ingroup icla_device
*******************************************************************************/
extern "C" size_t
icla_getdevice_shmem_block_optin()
{
    int dev;
    cudaError_t err;
    err = cudaGetDevice( &dev );
    check_error( err );
    ICLA_UNUSED( err );
    if ( g_icla_devices == NULL || dev < 0 || dev >= g_icla_devices_cnt ) {
        fprintf( stderr, "Error in %s: ICLA not initialized (call icla_init() first) or bad device\n", __func__ );
        return 0;
    }
    return g_icla_devices[dev].shmem_block_optin;
}

/***************************************************************************//**
    Returns the maximum shared memory multiprocessor (in bytes) for the current device.
    This requires icla_init() to be called first to cache the information.

    @return the maximum shared memory per multiprocessor (in bytes) for the current device.

    @ingroup icla_device
*******************************************************************************/
extern "C" size_t
icla_getdevice_shmem_multiprocessor()
{
    int dev;
    cudaError_t err;
    err = cudaGetDevice( &dev );
    check_error( err );
    ICLA_UNUSED( err );
    if ( g_icla_devices == NULL || dev < 0 || dev >= g_icla_devices_cnt ) {
        fprintf( stderr, "Error in %s: ICLA not initialized (call icla_init() first) or bad device\n", __func__ );
        return 0;
    }
    return g_icla_devices[dev].shmem_multiproc;
}


/***************************************************************************//**
    @param[in]
    queue           Queue to query.

    @return         Amount of free memory in bytes available on the device
                    associated with the queue.

    @ingroup icla_queue
*******************************************************************************/
extern "C" size_t
icla_mem_size( icla_queue_t queue )
{
    // CUDA would only need a device ID, but OpenCL requires a queue.
    size_t freeMem, totalMem;
    icla_device_t orig_dev;
    icla_getdevice( &orig_dev );
    icla_setdevice( icla_queue_get_device( queue ));
    cudaError_t err = cudaMemGetInfo( &freeMem, &totalMem );
    check_error( err );
    ICLA_UNUSED( err );
    icla_setdevice( orig_dev );
    return freeMem;
}


// =============================================================================
// queue support

/***************************************************************************//**
    @param[in]
    queue       Queue to query.

    @return Device ID associated with the ICLA queue.

    @ingroup icla_queue
*******************************************************************************/
extern "C"
icla_int_t
icla_queue_get_device( icla_queue_t queue )
{
    return queue->device();
}


#ifdef ICLA_HAVE_CUDA
/***************************************************************************//**
    @param[in]
    queue       Queue to query.

    @return CUDA stream associated with the ICLA queue.

    @ingroup icla_queue
*******************************************************************************/
extern "C"
cudaStream_t
icla_queue_get_cuda_stream( icla_queue_t queue )
{
    return queue->cuda_stream();
}


/***************************************************************************//**
    @param[in]
    queue       Queue to query.

    @return cuBLAS handle associated with the ICLA queue.
            ICLA assumes the handle's stream will not be modified.

    @ingroup icla_queue
*******************************************************************************/

extern "C"
cublasHandle_t
icla_queue_get_cublas_handle( icla_queue_t queue )
{
    return queue->cublas_handle();
}

/***************************************************************************//**
    @param[in]
    queue       Queue to query.

    @return cuSparse handle associated with the ICLA queue.
            ICLA assumes the handle's stream will not be modified.

    @ingroup icla_queue
*******************************************************************************/
extern "C"
cusparseHandle_t
icla_queue_get_cusparse_handle( icla_queue_t queue )
{
    return queue->cusparse_handle();
}

#elif defined(ICLA_HAVE_HIP)

/***************************************************************************//**
    @param[in]
    queue       Queue to query.

    @return HIP stream associated with the ICLA queue.

    @ingroup icla_queue
*******************************************************************************/
extern "C"
hipStream_t
icla_queue_get_hip_stream( icla_queue_t queue )
{
    return queue->hip_stream();
}


/***************************************************************************//**
    @param[in]
    queue       Queue to query.

    @return hipBLAS handle associated with the ICLA queue.
            ICLA assumes the handle's stream will not be modified.

    @ingroup icla_queue
*******************************************************************************/

extern "C"
hipblasHandle_t
icla_queue_get_hipblas_handle( icla_queue_t queue )
{
    return queue->hipblas_handle();
}

/***************************************************************************//**
    @param[in]
    queue       Queue to query.

    @return hipSparse handle associated with the ICLA queue.
            ICLA assumes the handle's stream will not be modified.

    @ingroup icla_queue
*******************************************************************************/
extern "C"
cusparseHandle_t
icla_queue_get_hipsparse_handle( icla_queue_t queue )
{
    return queue->hipsparse_handle();
}



#endif



/***************************************************************************//**
    @fn icla_queue_create( device, queue_ptr )

    icla_queue_create( device, queue_ptr ) is the preferred alias to this
    function.

    Creates a new ICLA queue, with associated CUDA stream, cuBLAS handle,
    and cuSparse handle.

    This is the ICLA v2 version which takes a device ID.

    @param[in]
    device          Device to create queue on.

    @param[out]
    queue_ptr       On output, the newly created queue.

    @ingroup icla_queue
*******************************************************************************/
extern "C" void
icla_queue_create_internal(
    icla_device_t device, icla_queue_t* queue_ptr,
    const char* func, const char* file, int line )
{
    icla_queue_t queue;
    icla_malloc_cpu( (void**)&queue, sizeof(*queue) );
    assert( queue != NULL );
    *queue_ptr = queue;

    queue->own__      = own_none;
    queue->device__   = device;
    queue->stream__   = NULL;
    queue->ptrArray__ = NULL;
    queue->dAarray__  = NULL;
    queue->dBarray__  = NULL;
    queue->dCarray__  = NULL;

#if defined(ICLA_HAVE_CUDA)
    queue->cublas__   = NULL;
    queue->cusparse__ = NULL;
#elif defined(ICLA_HAVE_HIP)
    queue->hipblas__  = NULL;
    queue->hipsparse__ = NULL;
#endif
    queue->maxbatch__ = MAX_BATCHCOUNT;

    icla_setdevice( device );

    cudaError_t err;
    err = cudaStreamCreate( &queue->stream__ );
    check_xerror( err, func, file, line );
    queue->own__ |= own_stream;

#if defined(ICLA_HAVE_CUDA)
    cublasStatus_t stat;
    stat = cublasCreate( &queue->cublas__ );
    check_xerror( stat, func, file, line );
    queue->own__ |= own_cublas;
    stat = cublasSetStream( queue->cublas__, queue->stream__ );
    check_xerror( stat, func, file, line );

    cusparseStatus_t stat2;
    stat2 = cusparseCreate( &queue->cusparse__ );
    check_xerror( stat2, func, file, line );
    queue->own__ |= own_cusparse;
    stat2 = cusparseSetStream( queue->cusparse__, queue->stream__ );
    check_xerror( stat2, func, file, line );
#elif defined(ICLA_HAVE_HIP)

    hipblasStatus_t stat;
    stat = hipblasCreate( &queue->hipblas__ );
    check_xerror( stat, func, file, line );
    queue->own__ |= own_hipblas;
    stat = hipblasSetStream( queue->hipblas__, queue->stream__ );
    check_xerror( stat, func, file, line );

    hipsparseStatus_t stat2;
    stat2 = hipsparseCreate( &queue->hipsparse__ );
    check_xerror( stat2, func, file, line );
    queue->own__ |= own_hipsparse;
    stat2 = hipsparseSetStream( queue->hipsparse__, queue->stream__ );
    check_xerror( stat2, func, file, line );

#endif

    ICLA_UNUSED( err );
    ICLA_UNUSED( stat );
    ICLA_UNUSED( stat2 );
}


/***************************************************************************//**
    @fn icla_queue_create_from_cuda( device, cuda_stream, cublas_handle, cusparse_handle, queue_ptr )

    Warning: non-portable outside of CUDA. Use with discretion.

    Creates a new ICLA queue, using the given CUDA stream, cuBLAS handle, and
    cuSparse handle. The caller retains ownership of the given stream and
    handles, so must free them after destroying the queue;
    see icla_queue_destroy().

    ICLA sets the stream on the cuBLAS and cuSparse handles, and assumes
    it will not be changed while ICLA is running.

    @param[in]
    device          Device to create queue on.

    @param[in]
    cuda_stream     CUDA stream to use, even if NULL (the so-called default stream).

    @param[in]
    cublas_handle   cuBLAS handle to use. If NULL, a new handle is created.

    @param[in]
    cusparse_handle cuSparse handle to use. If NULL, a new handle is created.

    @param[out]
    queue_ptr       On output, the newly created queue.

    @ingroup icla_queue
*******************************************************************************/
#ifdef ICLA_HAVE_CUDA
extern "C" void
icla_queue_create_from_cuda_internal(
    icla_device_t   device,
    cudaStream_t     cuda_stream,
    cublasHandle_t   cublas_handle,
    cusparseHandle_t cusparse_handle,
    icla_queue_t*   queue_ptr,
    const char* func, const char* file, int line )
{
    icla_queue_t queue;
    icla_malloc_cpu( (void**)&queue, sizeof(*queue) );
    assert( queue != NULL );
    *queue_ptr = queue;

    queue->own__      = own_none;
    queue->device__   = device;
    queue->stream__   = NULL;
    queue->cublas__   = NULL;
    queue->cusparse__ = NULL;
    queue->ptrArray__ = NULL;
    queue->dAarray__  = NULL;
    queue->dBarray__  = NULL;
    queue->dCarray__  = NULL;
    queue->maxbatch__ = MAX_BATCHCOUNT;

    icla_setdevice( device );

    // stream can be NULL
    queue->stream__ = cuda_stream;

    // allocate cublas handle if given as NULL
    cublasStatus_t stat;
    if ( cublas_handle == NULL ) {
        stat  = cublasCreate( &cublas_handle );
        check_xerror( stat, func, file, line );
        queue->own__ |= own_cublas;
    }
    queue->cublas__ = cublas_handle;
    stat  = cublasSetStream( queue->cublas__, queue->stream__ );
    check_xerror( stat, func, file, line );

    // allocate cusparse handle if given as NULL
    cusparseStatus_t stat2;
    if ( cusparse_handle == NULL ) {
        stat2 = cusparseCreate( &cusparse_handle );
        check_xerror( stat, func, file, line );
        queue->own__ |= own_cusparse;
    }
    queue->cusparse__ = cusparse_handle;
    stat2 = cusparseSetStream( queue->cusparse__, queue->stream__ );
    check_xerror( stat2, func, file, line );

    ICLA_UNUSED( stat );
    ICLA_UNUSED( stat2 );

}
#endif


/***************************************************************************//**
    @fn icla_queue_create_from_hip( device, hip_stream, hipblas_handle, hipsparse_handle, queue_ptr )

    Warning: non-portable outside of CUDA. Use with discretion.

    Creates a new ICLA queue, using the given CUDA stream, cuBLAS handle, and
    cuSparse handle. The caller retains ownership of the given stream and
    handles, so must free them after destroying the queue;
    see icla_queue_destroy().

    ICLA sets the stream on the cuBLAS and cuSparse handles, and assumes
    it will not be changed while ICLA is running.

    @param[in]
    device          Device to create queue on.

    @param[in]
    cuda_stream     CUDA stream to use, even if NULL (the so-called default stream).

    @param[in]
    cublas_handle   cuBLAS handle to use. If NULL, a new handle is created.

    @param[in]
    cusparse_handle cuSparse handle to use. If NULL, a new handle is created.

    @param[out]
    queue_ptr       On output, the newly created queue.

    @ingroup icla_queue
*******************************************************************************/
#ifdef ICLA_HAVE_HIP
extern "C" void
icla_queue_create_from_hip_internal(
    icla_device_t    device,
    hipStream_t       hip_stream,
    hipblasHandle_t   hipblas_handle,
    hipsparseHandle_t hipsparse_handle,
    icla_queue_t*    queue_ptr,
    const char* func, const char* file, int line )
{
    icla_queue_t queue;
    icla_malloc_cpu( (void**)&queue, sizeof(*queue) );
    assert( queue != NULL );
    *queue_ptr = queue;

    queue->own__      = own_none;
    queue->device__   = device;
    queue->stream__   = NULL;

    queue->ptrArray__ = NULL;
    queue->dAarray__  = NULL;
    queue->dBarray__  = NULL;
    queue->dCarray__  = NULL;

    queue->hipblas__  = NULL;
    queue->hipsparse__= NULL;
    queue->maxbatch__ = MAX_BATCHCOUNT;

    icla_setdevice( device );

    // stream can be NULL
    queue->stream__ = hip_stream;

    // allocate cublas handle if given as NULL
    hipblasStatus_t stat;
    if ( hipblas_handle == NULL ) {
        stat  = hipblasCreate( &hipblas_handle );
        check_xerror( stat, func, file, line );
        queue->own__ |= own_hipblas;
    }
    queue->hipblas__ = hipblas_handle;
    stat  = hipblasSetStream( queue->hipblas__, queue->stream__ );
    check_xerror( stat, func, file, line );

    // allocate cusparse handle if given as NULL
    hipsparseStatus_t stat2;
    if ( hipsparse_handle == NULL ) {
        stat2 = hipsparseCreate( &hipsparse_handle );
        check_xerror( stat, func, file, line );
        queue->own__ |= own_hipsparse;
    }
    queue->hipsparse__ = hipsparse_handle;
    stat2 = hipsparseSetStream( queue->hipsparse__, queue->stream__ );
    check_xerror( stat2, func, file, line );

    ICLA_UNUSED( stat );
    ICLA_UNUSED( stat2 );
}
#endif



/***************************************************************************//**
    @fn icla_queue_destroy( queue )

    Destroys a queue, freeing its resources.

    If the queue was created with icla_queue_create_from_cuda(), the CUDA
    stream, cuBLAS handle, and cuSparse handle given there are NOT freed -- the
    caller retains ownership. However, if ICLA allocated the handles, ICLA
    will free them here.

    @param[in]
    queue           Queue to destroy.

    @ingroup icla_queue
*******************************************************************************/
extern "C" void
icla_queue_destroy_internal(
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    if ( queue != NULL ) {
    #if defined(ICLA_HAVE_CUDA)
        if ( queue->cublas__ != NULL && (queue->own__ & own_cublas)) {
            cublasStatus_t stat = cublasDestroy( queue->cublas__ );
            check_xerror( stat, func, file, line );
            ICLA_UNUSED( stat );
        }
        if ( queue->cusparse__ != NULL && (queue->own__ & own_cusparse)) {
            cusparseStatus_t stat = cusparseDestroy( queue->cusparse__ );
            check_xerror( stat, func, file, line );
            ICLA_UNUSED( stat );
        }
    #elif defined(ICLA_HAVE_HIP)

        if ( queue->hipblas__ != NULL && (queue->own__ & own_hipblas)) {
            hipblasStatus_t stat = hipblasDestroy( queue->hipblas__ );
            check_xerror( stat, func, file, line );
            ICLA_UNUSED( stat );
        }
        if ( queue->hipsparse__ != NULL && (queue->own__ & own_hipsparse)) {
            hipsparseStatus_t stat = hipsparseDestroy( queue->hipsparse__ );
            check_xerror( stat, func, file, line );
            ICLA_UNUSED( stat );
        }
    #endif
        if ( queue->stream__ != NULL && (queue->own__ & own_stream)) {
            cudaError_t err = cudaStreamDestroy( queue->stream__ );
            check_xerror( err, func, file, line );
            ICLA_UNUSED( err );
        }

        if( queue->ptrArray__ != NULL ) icla_free( queue->ptrArray__ );

        queue->own__      = own_none;
        queue->device__   = -1;
        queue->stream__   = NULL;
        queue->ptrArray__ = NULL;
        queue->dAarray__  = NULL;
        queue->dBarray__  = NULL;
        queue->dCarray__  = NULL;

    #if defined(ICLA_HAVE_CUDA)
        queue->cublas__   = NULL;
        queue->cusparse__ = NULL;
    #elif defined(ICLA_HAVE_HIP)
        queue->hipblas__  = NULL;
        queue->hipsparse__= NULL;
    #endif

        icla_free_cpu( queue );
    }
}


/***************************************************************************//**
    @fn icla_queue_sync( queue )

    Synchronizes with a queue. The CPU blocks until all operations on the queue
    are finished.

    @param[in]
    queue           Queue to synchronize.

    @ingroup icla_queue
*******************************************************************************/
extern "C" void
icla_queue_sync_internal(
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    cudaError_t err;
    if ( queue != NULL ) {
        err = cudaStreamSynchronize( queue->cuda_stream() );
    }
    else {
        err = cudaStreamSynchronize( NULL );
    }
    check_xerror( err, func, file, line );
    ICLA_UNUSED( err );
}


// =============================================================================
// event support

/***************************************************************************//**
    Creates a GPU event.

    @param[in]
    event           On output, the newly created event.

    @ingroup icla_event
*******************************************************************************/
extern "C" void
icla_event_create( icla_event_t* event )
{
    cudaError_t err;
    err = cudaEventCreate( event );
    check_error( err );
    ICLA_UNUSED( err );
}


/***************************************************************************//**
    Creates a GPU event, without timing support. May improve performance

    @param[in]
    event           On output, the newly created event.

    @ingroup icla_event
*******************************************************************************/
extern "C" void
icla_event_create_untimed( icla_event_t* event )
{
    cudaError_t err;
    err = cudaEventCreateWithFlags( event, cudaEventDisableTiming );
    check_error( err );
    ICLA_UNUSED( err );
}



/***************************************************************************//*
    Destroys a GPU event, freeing its resources.

    @param[in]
    event           Event to destroy.

    @ingroup icla_event
*******************************************************************************/
extern "C" void
icla_event_destroy( icla_event_t event )
{
    if ( event != NULL ) {
        cudaError_t err;
        err = cudaEventDestroy( event );
        check_error( err );
        ICLA_UNUSED( err );
    }
}


/***************************************************************************//**
    Records an event into the queue's execution stream.
    The event will trigger when all previous operations on this queue finish.

    @param[in]
    event           Event to record.

    @param[in]
    queue           Queue to execute in.

    @ingroup icla_event
*******************************************************************************/
extern "C" void
icla_event_record( icla_event_t event, icla_queue_t queue )
{
    cudaError_t err;
    err = cudaEventRecord( event, queue->cuda_stream() );
    check_error( err );
    ICLA_UNUSED( err );
}


/***************************************************************************//**
    Synchronizes with an event. The CPU blocks until the event triggers.

    @param[in]
    event           Event to synchronize with.

    @ingroup icla_event
*******************************************************************************/
extern "C" void
icla_event_sync( icla_event_t event )
{
    cudaError_t err;
    err = cudaEventSynchronize( event );
    check_error( err );
    ICLA_UNUSED( err );
}


/***************************************************************************//**
    Synchronizes a queue with an event. The queue blocks until the event
    triggers. The CPU does not block.

    @param[in]
    event           Event to synchronize with.

    @param[in]
    queue           Queue to synchronize.

    @ingroup icla_event
*******************************************************************************/
extern "C" void
icla_queue_wait_event( icla_queue_t queue, icla_event_t event )
{
    cudaError_t err;
    err = cudaStreamWaitEvent( queue->cuda_stream(), event, 0 );
    check_error( err );
    ICLA_UNUSED( err );
}

#endif // ICLA_HAVE_CUDA or ICLA_HAVE_HIP
