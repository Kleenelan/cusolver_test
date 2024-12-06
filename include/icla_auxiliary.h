
#ifndef ICLA_AUXILIARY_H
#define ICLA_AUXILIARY_H

#include "icla_types.h"

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif





icla_int_t icla_init( void );
icla_int_t icla_finalize( void );

#ifdef ICLA_HAVE_OPENCL
icla_int_t icla_init_opencl(
    cl_platform_id platform,
    cl_context context,
    icla_int_t setup_clBlas );

icla_int_t icla_finalize_opencl(
    icla_int_t finalize_clBlas );
#endif





void icla_version( icla_int_t* major, icla_int_t* minor, icla_int_t* micro );
void icla_print_environment();





real_Double_t icla_wtime( void );
real_Double_t icla_sync_wtime( icla_queue_t queue );






icla_int_t icla_buildconnection_mgpu(
    icla_int_t gnode[IclaMaxGPUs+2][IclaMaxGPUs+2],
    icla_int_t *ncmplx,
    icla_int_t ngpu );

void icla_indices_1D_bcyclic(
    icla_int_t nb, icla_int_t ngpu, icla_int_t dev,
    icla_int_t j0, icla_int_t j1,
    icla_int_t* dj0, icla_int_t* dj1 );

void icla_swp2pswp(
    icla_trans_t trans, icla_int_t n,
    icla_int_t *ipiv,
    icla_int_t *newipiv );





icla_int_t icla_get_smlsize_divideconquer();





icla_int_t
icla_malloc( icla_ptr *ptr_ptr, size_t bytes );

icla_int_t
icla_malloc_cpu( void **ptr_ptr, size_t bytes );

icla_int_t
icla_malloc_pinned( void **ptr_ptr, size_t bytes );

icla_int_t
icla_free_cpu( void *ptr );

#define icla_free( ptr ) \
        icla_free_internal( ptr, __func__, __FILE__, __LINE__ )

#define icla_free_pinned( ptr ) \
        icla_free_pinned_internal( ptr, __func__, __FILE__, __LINE__ )

icla_int_t
icla_free_internal(
    icla_ptr ptr,
    const char* func, const char* file, int line );

icla_int_t
icla_free_pinned_internal(
    void *ptr,
    const char* func, const char* file, int line );


icla_int_t
icla_mem_info(size_t* freeMem, size_t* totalMem);


icla_int_t
icla_memset(void * ptr, int value, size_t count);


icla_int_t
icla_memset_async(void * ptr, int value, size_t count, icla_queue_t queue);










static inline icla_int_t icla_imalloc( iclaInt_ptr           *ptr_ptr, size_t n ) { return icla_malloc( (icla_ptr*) ptr_ptr, n*sizeof(icla_int_t)        ); }


static inline icla_int_t icla_index_malloc( iclaIndex_ptr    *ptr_ptr, size_t n ) { return icla_malloc( (icla_ptr*) ptr_ptr, n*sizeof(icla_index_t)      ); }


static inline icla_int_t icla_uindex_malloc( iclaUIndex_ptr    *ptr_ptr, size_t n ) { return icla_malloc( (icla_ptr*) ptr_ptr, n*sizeof(icla_uindex_t)      ); }


static inline icla_int_t icla_smalloc( iclaFloat_ptr         *ptr_ptr, size_t n ) { return icla_malloc( (icla_ptr*) ptr_ptr, n*sizeof(float)              ); }


static inline icla_int_t icla_dmalloc( iclaDouble_ptr        *ptr_ptr, size_t n ) { return icla_malloc( (icla_ptr*) ptr_ptr, n*sizeof(double)             ); }


static inline icla_int_t icla_cmalloc( iclaFloatComplex_ptr  *ptr_ptr, size_t n ) { return icla_malloc( (icla_ptr*) ptr_ptr, n*sizeof(iclaFloatComplex)  ); }


static inline icla_int_t icla_zmalloc( iclaDoubleComplex_ptr *ptr_ptr, size_t n ) { return icla_malloc( (icla_ptr*) ptr_ptr, n*sizeof(iclaDoubleComplex) ); }











static inline icla_int_t icla_imalloc_cpu( icla_int_t        **ptr_ptr, size_t n ) { return icla_malloc_cpu( (void**) ptr_ptr, n*sizeof(icla_int_t)        ); }


static inline icla_int_t icla_index_malloc_cpu( icla_index_t **ptr_ptr, size_t n ) { return icla_malloc_cpu( (void**) ptr_ptr, n*sizeof(icla_index_t)      ); }


static inline icla_int_t icla_uindex_malloc_cpu( icla_uindex_t **ptr_ptr, size_t n ) { return icla_malloc_cpu( (void**) ptr_ptr, n*sizeof(icla_uindex_t)      ); }


static inline icla_int_t icla_smalloc_cpu( float              **ptr_ptr, size_t n ) { return icla_malloc_cpu( (void**) ptr_ptr, n*sizeof(float)              ); }


static inline icla_int_t icla_dmalloc_cpu( double             **ptr_ptr, size_t n ) { return icla_malloc_cpu( (void**) ptr_ptr, n*sizeof(double)             ); }


static inline icla_int_t icla_cmalloc_cpu( iclaFloatComplex  **ptr_ptr, size_t n ) { return icla_malloc_cpu( (void**) ptr_ptr, n*sizeof(iclaFloatComplex)  ); }


static inline icla_int_t icla_zmalloc_cpu( iclaDoubleComplex **ptr_ptr, size_t n ) { return icla_malloc_cpu( (void**) ptr_ptr, n*sizeof(iclaDoubleComplex) ); }











static inline icla_int_t icla_imalloc_pinned( icla_int_t        **ptr_ptr, size_t n ) { return icla_malloc_pinned( (void**) ptr_ptr, n*sizeof(icla_int_t)        ); }


static inline icla_int_t icla_index_malloc_pinned( icla_index_t **ptr_ptr, size_t n ) { return icla_malloc_pinned( (void**) ptr_ptr, n*sizeof(icla_index_t)      ); }


static inline icla_int_t icla_smalloc_pinned( float              **ptr_ptr, size_t n ) { return icla_malloc_pinned( (void**) ptr_ptr, n*sizeof(float)              ); }


static inline icla_int_t icla_dmalloc_pinned( double             **ptr_ptr, size_t n ) { return icla_malloc_pinned( (void**) ptr_ptr, n*sizeof(double)             ); }


static inline icla_int_t icla_cmalloc_pinned( iclaFloatComplex  **ptr_ptr, size_t n ) { return icla_malloc_pinned( (void**) ptr_ptr, n*sizeof(iclaFloatComplex)  ); }


static inline icla_int_t icla_zmalloc_pinned( iclaDoubleComplex **ptr_ptr, size_t n ) { return icla_malloc_pinned( (void**) ptr_ptr, n*sizeof(iclaDoubleComplex) ); }




icla_int_t icla_is_devptr( const void* ptr );





icla_int_t
icla_num_gpus( void );






icla_int_t
icla_getdevice_arch();



void
icla_getdevices(
    icla_device_t* devices,
    icla_int_t     size,
    icla_int_t*    num_dev );

void
icla_getdevice( icla_device_t* dev );

void
icla_setdevice( icla_device_t dev );

size_t
icla_mem_size( icla_queue_t queue );

icla_int_t
icla_getdevice_multiprocessor_count();

icla_int_t
icla_getdevice_num_threads_block();

icla_int_t
icla_getdevice_num_threads_multiprocessor();

size_t
icla_getdevice_shmem_block();

size_t
icla_getdevice_shmem_block_optin();

size_t
icla_getdevice_shmem_multiprocessor();




#define icla_queue_create(          device, queue_ptr ) \
        icla_queue_create_internal( device, queue_ptr, __func__, __FILE__, __LINE__ )

#define icla_queue_create_from_cuda(          device, cuda_stream, cublas_handle, cusparse_handle, queue_ptr ) \
        icla_queue_create_from_cuda_internal( device, cuda_stream, cublas_handle, cusparse_handle, queue_ptr, __func__, __FILE__, __LINE__ )

#define icla_queue_create_from_hip(           device, hip_stream, hipblas_handle, hipsparse_handle, queue_ptr ) \
        icla_queue_create_from_hip_internal( device, hip_stream, hipblas_handle, hipsparse_handle, queue_ptr, __func__, __FILE__, __LINE__ )

#define icla_queue_create_from_opencl(          device, cl_queue, queue_ptr ) \
        icla_queue_create_from_opencl_internal( device, cl_queue, queue_ptr, __func__, __FILE__, __LINE__ )

#define icla_queue_destroy( queue ) \
        icla_queue_destroy_internal( queue, __func__, __FILE__, __LINE__ )

#define icla_queue_sync( queue ) \
        icla_queue_sync_internal( queue, __func__, __FILE__, __LINE__ )

void
icla_queue_create_internal(
    icla_device_t device,
    icla_queue_t* queue_ptr,
    const char* func, const char* file, int line );

#ifdef ICLA_HAVE_CUDA
void
icla_queue_create_from_cuda_internal(
    icla_device_t   device,
    cudaStream_t     stream,
    cublasHandle_t   cublas,
    cusparseHandle_t cusparse,
    icla_queue_t*   queue_ptr,
    const char* func, const char* file, int line );
#endif


#ifdef ICLA_HAVE_HIP
void
icla_queue_create_from_hip_internal(
    icla_device_t    device,
    hipStream_t       stream,
    hipblasHandle_t   hipblas,
    hipsparseHandle_t hipsparse,
    icla_queue_t*    queue_ptr,
    const char* func, const char* file, int line );
#endif


#ifdef ICLA_HAVE_OPENCL
icla_int_t
icla_queue_create_from_opencl_internal(
    icla_device_t   device,
    cl_command_queue cl_queue,
    const char* func, const char* file, int line );
#endif

void
icla_queue_destroy_internal(
    icla_queue_t queue,
    const char* func, const char* file, int line );

void
icla_queue_sync_internal(
    icla_queue_t queue,
    const char* func, const char* file, int line );

icla_int_t
icla_queue_get_device( icla_queue_t queue );





void
icla_event_create( icla_event_t* event_ptr );

void
icla_event_create_untimed( icla_event_t* event_ptr );

void
icla_event_destroy( icla_event_t event );

void
icla_event_record( icla_event_t event, icla_queue_t queue );

void
icla_event_query( icla_event_t event );

void
icla_event_sync( icla_event_t event );

void
icla_queue_wait_event( icla_queue_t queue, icla_event_t event );





void icla_xerbla( const char *name, icla_int_t info );

const char* icla_strerror( icla_int_t error );





size_t icla_strlcpy( char *dst, const char *src, size_t size );








__host__ __device__
static inline icla_int_t icla_ceildiv( icla_int_t x, icla_int_t y )
{
    return (x + y - 1)/y;
}






__host__ __device__
static inline icla_int_t icla_roundup( icla_int_t x, icla_int_t y )
{
    return icla_ceildiv( x, y ) * y;
}









static inline float  icla_ssqrt( float  x ) { return sqrtf( x ); }


static inline double icla_dsqrt( double x ) { return sqrt( x ); }


iclaFloatComplex    icla_csqrt( iclaFloatComplex  x );


iclaDoubleComplex   icla_zsqrt( iclaDoubleComplex x );





void icla_iprint( icla_int_t m, icla_int_t n, const icla_int_t *A, icla_int_t lda );

void icla_iprint_gpu( icla_int_t m, icla_int_t n, icla_int_t* dA, icla_int_t ldda, icla_queue_t queue );

#ifdef __cplusplus
}
#endif


#endif
