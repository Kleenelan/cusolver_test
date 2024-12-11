
#include "icla_internal.h"
#include "error.h"

#include <cuda_runtime.h>

#if defined(ICLA_HAVE_CUDA) || defined(ICLA_HAVE_HIP)

extern "C" void
icla_setvector_internal(
    icla_int_t n, icla_int_t elemSize,
    void const* hx_src, icla_int_t incx,
    icla_ptr   dy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    cudaStream_t stream = NULL;
    if ( queue != NULL ) {
        stream = queue->cuda_stream();
    }
    cublasStatus_t status;
    status = cublasSetVectorAsync(
        int(n), int(elemSize),
        hx_src, int(incx),
        dy_dst, int(incy), stream );
    if ( queue != NULL )
        cudaStreamSynchronize( stream );
    check_xerror( status, func, file, line );
    ICLA_UNUSED( status );
}

extern "C" void
icla_setvector_async_internal(
    icla_int_t n, icla_int_t elemSize,
    void const* hx_src, icla_int_t incx,
    icla_ptr   dy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{

    cudaStream_t stream = NULL;
    if ( queue != NULL ) {
        stream = queue->cuda_stream();
    }
    else {
        fprintf( stderr, "Warning: %s got NULL queue\n", __func__ );
    }
    cublasStatus_t status;
    status = cublasSetVectorAsync(
        int(n), int(elemSize),
        hx_src, int(incx),
        dy_dst, int(incy), stream );
    check_xerror( status, func, file, line );
    ICLA_UNUSED( status );
}

extern "C" void
icla_getvector_internal(
    icla_int_t n, icla_int_t elemSize,
    icla_const_ptr dx_src, icla_int_t incx,
    void*           hy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    cudaStream_t stream = NULL;
    if ( queue != NULL ) {
        stream = queue->cuda_stream();
    }
    cublasStatus_t status;
    status = cublasGetVectorAsync(
        int(n), int(elemSize),
        dx_src, int(incx),
        hy_dst, int(incy), stream );
    if ( queue != NULL )
        cudaStreamSynchronize( stream );
    check_xerror( status, func, file, line );
    ICLA_UNUSED( status );
}

extern "C" void
icla_getvector_async_internal(
    icla_int_t n, icla_int_t elemSize,
    icla_const_ptr dx_src, icla_int_t incx,
    void*           hy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{

    cudaStream_t stream = NULL;
    if ( queue != NULL ) {
        stream = queue->cuda_stream();
    }
    else {
        fprintf( stderr, "Warning: %s got NULL queue\n", __func__ );
    }
    cublasStatus_t status;
    status = cublasGetVectorAsync(
        int(n), int(elemSize),
        dx_src, int(incx),
        hy_dst, int(incy), stream );
    check_xerror( status, func, file, line );
    ICLA_UNUSED( status );
}

extern "C" void
icla_copyvector_internal(
    icla_int_t n, icla_int_t elemSize,
    icla_const_ptr dx_src, icla_int_t incx,
    icla_ptr       dy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    cudaStream_t stream = NULL;
    if ( queue != NULL ) {
        stream = queue->cuda_stream();
    }
    if ( incx == 1 && incy == 1 ) {
        cudaError_t status;
        status = cudaMemcpyAsync(
            dy_dst,
            dx_src,
            int(n*elemSize), cudaMemcpyDeviceToDevice, stream );
        if ( queue != NULL )
            cudaStreamSynchronize( stream );
        check_xerror( status, func, file, line );
        ICLA_UNUSED( status );
    }
    else {
        icla_copymatrix_internal(
            1, n, elemSize, dx_src, incx, dy_dst, incy, queue, func, file, line );
    }
}

extern "C" void
icla_copyvector_async_internal(
    icla_int_t n, icla_int_t elemSize,
    icla_const_ptr dx_src, icla_int_t incx,
    icla_ptr       dy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{

    cudaStream_t stream = NULL;
    if ( queue != NULL ) {
        stream = queue->cuda_stream();
    }
    else {
        fprintf( stderr, "Warning: %s got NULL queue\n", __func__ );
    }
    if ( incx == 1 && incy == 1 ) {
        cudaError_t status;
        status = cudaMemcpyAsync(
            dy_dst,
            dx_src,
            int(n*elemSize), cudaMemcpyDeviceToDevice, stream );
        check_xerror( status, func, file, line );
        ICLA_UNUSED( status );
    }
    else {
        icla_copymatrix_async_internal(
            1, n, elemSize, dx_src, incx, dy_dst, incy, queue, func, file, line );
    }
}

extern "C" void
icla_setmatrix_internal(
    icla_int_t m, icla_int_t n, icla_int_t elemSize,
    void const* hA_src, icla_int_t lda,
    icla_ptr   dB_dst, icla_int_t lddb,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    cudaStream_t stream = NULL;
    if ( queue != NULL ) {
        stream = queue->cuda_stream();
    }
    cublasStatus_t status;
    status = cublasSetMatrixAsync(
        int(m), int(n), int(elemSize),
        hA_src, int(lda),
        dB_dst, int(lddb), stream );
    if ( queue != NULL )
        cudaStreamSynchronize( stream );
    check_xerror( status, func, file, line );
    ICLA_UNUSED( status );
}

extern "C" void
icla_setmatrix_async_internal(
    icla_int_t m, icla_int_t n, icla_int_t elemSize,
    void const* hA_src, icla_int_t lda,
    icla_ptr   dB_dst, icla_int_t lddb,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{

    cudaStream_t stream = NULL;
    if ( queue != NULL ) {
        stream = queue->cuda_stream();
    }
    else {
        fprintf( stderr, "Warning: %s got NULL queue\n", __func__ );
    }
    cublasStatus_t status;
    status = cublasSetMatrixAsync(
        int(m), int(n), int(elemSize),
        hA_src, int(lda),
        dB_dst, int(lddb), stream );
    check_xerror( status, func, file, line );
    ICLA_UNUSED( status );
}

extern "C" void
icla_getmatrix_internal(
    icla_int_t m, icla_int_t n, icla_int_t elemSize,
    icla_const_ptr dA_src, icla_int_t ldda,
    void*           hB_dst, icla_int_t ldb,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    cudaStream_t stream = NULL;
    if ( queue != NULL ) {
        stream = queue->cuda_stream();
    }
    cublasStatus_t status;
    status = cublasGetMatrixAsync(
        int(m), int(n), int(elemSize),
        dA_src, int(ldda),
        hB_dst, int(ldb), stream );
    if ( queue != NULL )
        cudaStreamSynchronize( stream );
    check_xerror( status, func, file, line );
    ICLA_UNUSED( status );
}

extern "C" void
icla_getmatrix_async_internal(
    icla_int_t m, icla_int_t n, icla_int_t elemSize,
    icla_const_ptr dA_src, icla_int_t ldda,
    void*           hB_dst, icla_int_t ldb,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    cudaStream_t stream = NULL;
    if ( queue != NULL ) {
        stream = queue->cuda_stream();
    }
    else {
        fprintf( stderr, "Warning: %s got NULL queue\n", __func__ );
    }
    cublasStatus_t status;
    status = cublasGetMatrixAsync(
        int(m), int(n), int(elemSize),
        dA_src, int(ldda),
        hB_dst, int(ldb), stream );
    check_xerror( status, func, file, line );
    ICLA_UNUSED( status );
}

extern "C" void
icla_copymatrix_internal(
    icla_int_t m, icla_int_t n, icla_int_t elemSize,
    icla_const_ptr dA_src, icla_int_t ldda,
    icla_ptr       dB_dst, icla_int_t lddb,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    cudaStream_t stream = NULL;
    if ( queue != NULL ) {
        stream = queue->cuda_stream();
    }
    cudaError_t status;
    status = cudaMemcpy2DAsync(
        dB_dst, int(lddb*elemSize),
        dA_src, int(ldda*elemSize),
        int(m*elemSize), int(n), cudaMemcpyDeviceToDevice, stream );
    if ( queue != NULL )
        cudaStreamSynchronize( stream );
    check_xerror( status, func, file, line );
    ICLA_UNUSED( status );
}

extern "C" void
icla_copymatrix_async_internal(
    icla_int_t m, icla_int_t n, icla_int_t elemSize,
    icla_const_ptr dA_src, icla_int_t ldda,
    icla_ptr       dB_dst, icla_int_t lddb,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{

    cudaStream_t stream = NULL;
    if ( queue != NULL ) {
        stream = queue->cuda_stream();
    }
    else {
        fprintf( stderr, "Warning: %s got NULL queue\n", __func__ );
    }
    cudaError_t status;
    status = cudaMemcpy2DAsync(
        dB_dst, int(lddb*elemSize),
        dA_src, int(ldda*elemSize),
        int(m*elemSize), int(n), cudaMemcpyDeviceToDevice, stream );
    check_xerror( status, func, file, line );
    ICLA_UNUSED( status );
}

#endif

