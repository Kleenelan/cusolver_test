
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <map>

#if __cplusplus >= 201103

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

#define ICLA_LAPACK_H

#include "icla_internal.h"
#include "error.h"

#define MAX_BATCHCOUNT    (65534)

#if defined(ICLA_HAVE_CUDA) || defined(ICLA_HAVE_HIP)

#ifdef DEBUG_MEMORY

extern std::map< void*, size_t > g_pointers_dev;
extern std::map< void*, size_t > g_pointers_cpu;
extern std::map< void*, size_t > g_pointers_pin;
#endif

extern "C" void
icla_warn_leaks( const std::map< void*, size_t >& pointers, const char* type );

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

#if __cplusplus >= 201103

    static std::mutex g_mutex;
#else

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

static int g_init = 0;

#ifndef ICLA_NO_V1
    icla_queue_t* g_null_queues = NULL;

    #ifdef HAVE_PTHREAD_KEY
    pthread_key_t g_icla_queue_key;
    #else
    icla_queue_t g_icla_queue = NULL;
    #endif
#endif

struct icla_device_info
{
    size_t memory;
    size_t shmem_block;

    size_t shmem_block_optin;

    size_t shmem_multiproc;

    icla_int_t gpu_arch;
    icla_int_t multiproc_count;

    icla_int_t num_threads_block;

    icla_int_t num_threads_multiproc;

};

int g_icla_devices_cnt = 0;
struct icla_device_info* g_icla_devices = NULL;

extern "C" icla_int_t
icla_init()
{
    icla_int_t info = 0;

    g_mutex.lock();
    {
        if ( g_init == 0 ) {

            cudaError_t err;
            g_icla_devices_cnt = 0;
            err = cudaGetDeviceCount( &g_icla_devices_cnt );
            if ( err != 0 && err != cudaErrorNoDevice ) {
                info = ICLA_ERR_UNKNOWN;
                goto cleanup;
            }

            size_t size;
            size = max( 1, g_icla_devices_cnt ) * sizeof(struct icla_device_info);
            icla_malloc_cpu( (void**) &g_icla_devices, size );
            if ( g_icla_devices == NULL ) {
                info = ICLA_ERR_HOST_ALLOC;
                goto cleanup;
            }
            memset( g_icla_devices, 0, size );

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

                    info = pthread_key_create( &g_icla_queue_key, NULL );
                    if ( info != 0 ) {
                        info = ICLA_ERR_UNKNOWN;
                        goto cleanup;
                    }
                #endif

                size = max( 1, g_icla_devices_cnt ) * sizeof(icla_queue_t);
                icla_malloc_cpu( (void**) &g_null_queues, size );
                if ( g_null_queues == NULL ) {
                    info = ICLA_ERR_HOST_ALLOC;
                    goto cleanup;
                }
                memset( g_null_queues, 0, size );
            #endif

        }
cleanup:
        g_init += 1;

    }
    g_mutex.unlock();

    return info;
}

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
            g_init -= 1;

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
                #endif

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

#ifdef DEBUG_MEMORY

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

#if defined(ICLA_HAVE_CUDA)

    printf( "%% Compiled for CUDA architectures %s\n", ICLA_CUDA_ARCH );

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

#if defined(ICLA_HAVE_HIP)

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

    int acml_major, acml_minor, acml_patch, acml_build;
    acmlversion( &acml_major, &acml_minor, &acml_patch, &acml_build );
    printf( "ACML %d.%d.%d.%d ", acml_major, acml_minor, acml_patch, acml_build );
#endif

    printf( "\n" );

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

extern "C" icla_int_t
icla_is_devptr( const void* A )
{
    cudaError_t err;
    cudaDeviceProp prop;
    cudaPointerAttributes attr;
    int dev;

    err = cudaGetDevice( &dev );
    if ( ! err ) {
        err = cudaGetDeviceProperties( &prop, dev );

        #ifdef ICLA_HAVE_CUDA
        if ( ! err && prop.unifiedAddressing ) {
        #elif defined(ICLA_HAVE_HIP)

        if ( ! err ) {
        #endif

            err = cudaPointerGetAttributes( &attr, const_cast<void*>( A ));
            if ( ! err ) {

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

                cudaGetLastError();

                return 0;
            }
        }
    }

    cudaGetLastError();

    return -1;
}

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

extern "C" void
icla_setdevice( icla_device_t device )
{
    cudaError_t err;
    err = cudaSetDevice( int(device) );
    check_error( err );
    ICLA_UNUSED( err );
}

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

extern "C" size_t
icla_mem_size( icla_queue_t queue )
{

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

extern "C"
icla_int_t
icla_queue_get_device( icla_queue_t queue )
{
    return queue->device();
}

#ifdef ICLA_HAVE_CUDA

extern "C"
cudaStream_t
icla_queue_get_cuda_stream( icla_queue_t queue )
{
    return queue->cuda_stream();
}

extern "C"
cublasHandle_t
icla_queue_get_cublas_handle( icla_queue_t queue )
{
    return queue->cublas_handle();
}

extern "C"
cusparseHandle_t
icla_queue_get_cusparse_handle( icla_queue_t queue )
{
    return queue->cusparse_handle();
}

#elif defined(ICLA_HAVE_HIP)

extern "C"
hipStream_t
icla_queue_get_hip_stream( icla_queue_t queue )
{
    return queue->hip_stream();
}

extern "C"
hipblasHandle_t
icla_queue_get_hipblas_handle( icla_queue_t queue )
{
    return queue->hipblas_handle();
}

extern "C"
cusparseHandle_t
icla_queue_get_hipsparse_handle( icla_queue_t queue )
{
    return queue->hipsparse_handle();
}

#endif

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

    queue->stream__ = cuda_stream;

    cublasStatus_t stat;
    if ( cublas_handle == NULL ) {
        stat  = cublasCreate( &cublas_handle );
        check_xerror( stat, func, file, line );
        queue->own__ |= own_cublas;
    }
    queue->cublas__ = cublas_handle;
    stat  = cublasSetStream( queue->cublas__, queue->stream__ );
    check_xerror( stat, func, file, line );

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

    queue->stream__ = hip_stream;

    hipblasStatus_t stat;
    if ( hipblas_handle == NULL ) {
        stat  = hipblasCreate( &hipblas_handle );
        check_xerror( stat, func, file, line );
        queue->own__ |= own_hipblas;
    }
    queue->hipblas__ = hipblas_handle;
    stat  = hipblasSetStream( queue->hipblas__, queue->stream__ );
    check_xerror( stat, func, file, line );

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

extern "C" void
icla_event_create( icla_event_t* event )
{
    cudaError_t err;
    err = cudaEventCreate( event );
    check_error( err );
    ICLA_UNUSED( err );
}

extern "C" void
icla_event_create_untimed( icla_event_t* event )
{
    cudaError_t err;
    err = cudaEventCreateWithFlags( event, cudaEventDisableTiming );
    check_error( err );
    ICLA_UNUSED( err );
}

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

extern "C" void
icla_event_record( icla_event_t event, icla_queue_t queue )
{
    cudaError_t err;
    err = cudaEventRecord( event, queue->cuda_stream() );
    check_error( err );
    ICLA_UNUSED( err );
}

extern "C" void
icla_event_sync( icla_event_t event )
{
    cudaError_t err;
    err = cudaEventSynchronize( event );
    check_error( err );
    ICLA_UNUSED( err );
}

extern "C" void
icla_queue_wait_event( icla_queue_t queue, icla_event_t event )
{
    cudaError_t err;
    err = cudaStreamWaitEvent( queue->cuda_stream(), event, 0 );
    check_error( err );
    ICLA_UNUSED( err );
}

#endif

