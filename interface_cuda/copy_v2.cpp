/*
    -- ICLA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates
*/
#include "icla_internal.h"
#include "error.h"

#include <cuda_runtime.h>

#if defined(ICLA_HAVE_CUDA) || defined(ICLA_HAVE_HIP)

// Generic, type-independent routines to copy data.
// Type-safe versions which avoid the user needing sizeof(...) are in headers;
// see icla_{s,d,c,z,i,index_}{set,get,copy}{matrix,vector}

/***************************************************************************//**
    @fn icla_setvector( n, elemSize, hx_src, incx, dy_dst, incy, queue )

    Copy vector hx_src on CPU host to dy_dst on GPU device.
    Elements may be arbitrary size.
    Type-safe versions set elemSize appropriately.

    This version synchronizes the queue after the transfer.
    See icla_setvector_async() for an asynchronous version.

    @param[in]
    n           Number of elements in vector.

    @param[in]
    elemSize    Size of each element, e.g., sizeof(double).

    @param[in]
    hx_src      Source array of dimension (1 + (n-1))*incx, on CPU host.

    @param[in]
    incx        Increment between elements of hx_src. incx > 0.

    @param[out]
    dy_dst      Destination array of dimension (1 + (n-1))*incy, on GPU device.

    @param[in]
    incy        Increment between elements of dy_dst. incy > 0.

    @param[in]
    queue       Queue to execute in.

    @ingroup icla_setvector
*******************************************************************************/
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


/***************************************************************************//**
    @fn icla_setvector_async( n, elemSize, hx_src, incx, dy_dst, incy, queue )

    Copy vector hx_src on CPU host to dy_dst on GPU device.
    Elements may be arbitrary size.
    Type-safe versions set elemSize appropriately.

    This version is asynchronous: it may return before the transfer finishes,
    if hx_src is pinned CPU memory.
    See icla_setvector() for a synchronous version.

    @param[in]
    n           Number of elements in vector.

    @param[in]
    elemSize    Size of each element, e.g., sizeof(double).

    @param[in]
    hx_src      Source array of dimension (1 + (n-1))*incx, on CPU host.

    @param[in]
    incx        Increment between elements of hx_src. incx > 0.

    @param[out]
    dy_dst      Destination array of dimension (1 + (n-1))*incy, on GPU device.

    @param[in]
    incy        Increment between elements of dy_dst. incy > 0.

    @param[in]
    queue       Queue to execute in.

    @ingroup icla_setvector
*******************************************************************************/
extern "C" void
icla_setvector_async_internal(
    icla_int_t n, icla_int_t elemSize,
    void const* hx_src, icla_int_t incx,
    icla_ptr   dy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    // for backwards compatability, accepts NULL queue to mean NULL stream.
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


/***************************************************************************//**
    @fn icla_getvector( n, elemSize, dx_src, incx, hy_dst, incy, queue )

    Copy vector dx_src on GPU device to hy_dst on CPU host.
    Elements may be arbitrary size.
    Type-safe versions set elemSize appropriately.

    This version synchronizes the queue after the transfer.
    See icla_getvector_async() for an asynchronous version.

    @param[in]
    n           Number of elements in vector.

    @param[in]
    elemSize    Size of each element, e.g., sizeof(double).

    @param[in]
    dx_src      Source array of dimension (1 + (n-1))*incx, on GPU device.

    @param[in]
    incx        Increment between elements of hx_src. incx > 0.

    @param[out]
    hy_dst      Destination array of dimension (1 + (n-1))*incy, on CPU host.

    @param[in]
    incy        Increment between elements of dy_dst. incy > 0.

    @param[in]
    queue       Queue to execute in.

    @ingroup icla_getvector
*******************************************************************************/
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


/***************************************************************************//**
    @fn icla_getvector_async( n, elemSize, dx_src, incx, hy_dst, incy, queue )

    Copy vector dx_src on GPU device to hy_dst on CPU host.
    Elements may be arbitrary size.
    Type-safe versions set elemSize appropriately.

    This version is asynchronous: it may return before the transfer finishes,
    if hy_dst is pinned CPU memory.
    See icla_getvector() for a synchronous version.

    @param[in]
    n           Number of elements in vector.

    @param[in]
    elemSize    Size of each element, e.g., sizeof(double).

    @param[in]
    dx_src      Source array of dimension (1 + (n-1))*incx, on GPU device.

    @param[in]
    incx        Increment between elements of hx_src. incx > 0.

    @param[out]
    hy_dst      Destination array of dimension (1 + (n-1))*incy, on CPU host.

    @param[in]
    incy        Increment between elements of dy_dst. incy > 0.

    @param[in]
    queue       Queue to execute in.

    @ingroup icla_getvector
*******************************************************************************/
extern "C" void
icla_getvector_async_internal(
    icla_int_t n, icla_int_t elemSize,
    icla_const_ptr dx_src, icla_int_t incx,
    void*           hy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    // for backwards compatability, accepts NULL queue to mean NULL stream.
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


/***************************************************************************//**
    @fn icla_copyvector( n, elemSize, dx_src, incx, dy_dst, incy, queue )

    Copy vector dx_src on GPU device to dy_dst on GPU device.
    Elements may be arbitrary size.
    Type-safe versions set elemSize appropriately.
    With CUDA unified addressing, dx and dy can be on different GPUs.

    This version synchronizes the queue after the transfer.
    See icla_copyvector_async() for an asynchronous version.

    @param[in]
    n           Number of elements in vector.

    @param[in]
    elemSize    Size of each element, e.g., sizeof(double).

    @param[in]
    dx_src      Source array of dimension (1 + (n-1))*incx, on GPU device.

    @param[in]
    incx        Increment between elements of hx_src. incx > 0.

    @param[out]
    dy_dst      Destination array of dimension (1 + (n-1))*incy, on GPU device.

    @param[in]
    incy        Increment between elements of dy_dst. incy > 0.

    @param[in]
    queue       Queue to execute in.

    @ingroup icla_copyvector
*******************************************************************************/
// TODO compare performance with cublasZcopy BLAS function.
// But this implementation can handle any element size, not just [sdcz] precisions.
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


/***************************************************************************//**
    @fn icla_copyvector_async( n, elemSize, dx_src, incx, dy_dst, incy, queue )

    Copy vector dx_src on GPU device to dy_dst on GPU device.
    Elements may be arbitrary size.
    Type-safe versions set elemSize appropriately.
    With CUDA unified addressing, dx and dy can be on different GPUs.

    This version is asynchronous: it may return before the transfer finishes.
    See icla_copyvector() for a synchronous version.

    @param[in]
    n           Number of elements in vector.

    @param[in]
    elemSize    Size of each element, e.g., sizeof(double).

    @param[in]
    dx_src      Source array of dimension (1 + (n-1))*incx, on GPU device.

    @param[in]
    incx        Increment between elements of hx_src. incx > 0.

    @param[out]
    dy_dst      Destination array of dimension (1 + (n-1))*incy, on GPU device.

    @param[in]
    incy        Increment between elements of dy_dst. incy > 0.

    @param[in]
    queue       Queue to execute in.

    @ingroup icla_copyvector
*******************************************************************************/
extern "C" void
icla_copyvector_async_internal(
    icla_int_t n, icla_int_t elemSize,
    icla_const_ptr dx_src, icla_int_t incx,
    icla_ptr       dy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    // for backwards compatability, accepts NULL queue to mean NULL stream.
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


/***************************************************************************//**
    @fn icla_setmatrix( m, n, elemSize, hA_src, lda, dB_dst, lddb, queue )

    Copy all or part of matrix hA_src on CPU host to dB_dst on GPU device.
    Elements may be arbitrary size.
    Type-safe versions set elemSize appropriately.

    This version synchronizes the queue after the transfer.
    See icla_setmatrix_async() for an asynchronous version.

    @param[in]
    m           Number of rows of matrix A. m >= 0.

    @param[in]
    n           Number of columns of matrix A. n >= 0.

    @param[in]
    elemSize    Size of each element, e.g., sizeof(double).

    @param[in]
    hA_src      Source array of dimension (lda,n), on CPU host.

    @param[in]
    lda         Leading dimension of matrix A. lda >= m.

    @param[out]
    dB_dst      Destination array of dimension (lddb,n), on GPU device.

    @param[in]
    lddb        Leading dimension of matrix B. lddb >= m.

    @param[in]
    queue       Queue to execute in.

    @ingroup icla_setmatrix
*******************************************************************************/
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


/***************************************************************************//**
    @fn icla_setmatrix_async( m, n, elemSize, hA_src, lda, dB_dst, lddb, queue )

    Copy all or part of matrix hA_src on CPU host to dB_dst on GPU device.
    Elements may be arbitrary size.
    Type-safe versions set elemSize appropriately.

    This version is asynchronous: it may return before the transfer finishes,
    if hA_src is pinned CPU memory.
    See icla_setmatrix() for a synchronous version.

    @param[in]
    m           Number of rows of matrix A. m >= 0.

    @param[in]
    n           Number of columns of matrix A. n >= 0.

    @param[in]
    elemSize    Size of each element, e.g., sizeof(double).

    @param[in]
    hA_src      Source array of dimension (lda,n), on CPU host.

    @param[in]
    lda         Leading dimension of matrix A. lda >= m.

    @param[out]
    dB_dst      Destination array of dimension (lddb,n), on GPU device.

    @param[in]
    lddb        Leading dimension of matrix B. lddb >= m.

    @param[in]
    queue       Queue to execute in.

    @ingroup icla_setmatrix
*******************************************************************************/
extern "C" void
icla_setmatrix_async_internal(
    icla_int_t m, icla_int_t n, icla_int_t elemSize,
    void const* hA_src, icla_int_t lda,
    icla_ptr   dB_dst, icla_int_t lddb,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    // for backwards compatability, accepts NULL queue to mean NULL stream.
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


/***************************************************************************//**
    @fn icla_getmatrix( m, n, elemSize, dA_src, ldda, hB_dst, ldb, queue )

    Copy all or part of matrix dA_src on GPU device to hB_dst on CPU host.
    Elements may be arbitrary size.
    Type-safe versions set elemSize appropriately.

    This version synchronizes the queue after the transfer.
    See icla_getmatrix_async() for an asynchronous version.

    @param[in]
    m           Number of rows of matrix A. m >= 0.

    @param[in]
    n           Number of columns of matrix A. n >= 0.

    @param[in]
    elemSize    Size of each element, e.g., sizeof(double).

    @param[in]
    dA_src      Source array of dimension (ldda,n), on GPU device.

    @param[in]
    ldda        Leading dimension of matrix A. ldda >= m.

    @param[out]
    hB_dst      Destination array of dimension (ldb,n), on CPU host.

    @param[in]
    ldb         Leading dimension of matrix B. ldb >= m.

    @param[in]
    queue       Queue to execute in.

    @ingroup icla_getmatrix
*******************************************************************************/
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


/***************************************************************************//**
    @fn icla_getmatrix_async( m, n, elemSize, dA_src, ldda, hB_dst, ldb, queue )

    Copy all or part of matrix dA_src on GPU device to hB_dst on CPU host.
    Elements may be arbitrary size.
    Type-safe versions set elemSize appropriately.

    This version is asynchronous: it may return before the transfer finishes,
    if hB_dst is pinned CPU memory.
    See icla_getmatrix() for a synchronous version.

    @param[in]
    m           Number of rows of matrix A. m >= 0.

    @param[in]
    n           Number of columns of matrix A. n >= 0.

    @param[in]
    elemSize    Size of each element, e.g., sizeof(double).

    @param[in]
    dA_src      Source array of dimension (ldda,n), on GPU device.

    @param[in]
    ldda        Leading dimension of matrix A. ldda >= m.

    @param[out]
    hB_dst      Destination array of dimension (ldb,n), on CPU host.

    @param[in]
    ldb         Leading dimension of matrix B. ldb >= m.

    @param[in]
    queue       Queue to execute in.

    @ingroup icla_getmatrix
*******************************************************************************/
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


/***************************************************************************//**
    @fn icla_copymatrix( m, n, elemSize, dA_src, ldda, dB_dst, lddb, queue )

    Copy all or part of matrix dA_src on GPU device to dB_dst on GPU device.
    Elements may be arbitrary size.
    Type-safe versions set elemSize appropriately.
    With CUDA unified addressing, dA and dB can be on different GPUs.

    This version synchronizes the queue after the transfer.
    See icla_copymatrix_async() for an asynchronous version.

    @param[in]
    m           Number of rows of matrix A. m >= 0.

    @param[in]
    n           Number of columns of matrix A. n >= 0.

    @param[in]
    elemSize    Size of each element, e.g., sizeof(double).

    @param[in]
    dA_src      Source array of dimension (ldda,n).

    @param[in]
    ldda        Leading dimension of matrix A. ldda >= m.

    @param[out]
    dB_dst      Destination array of dimension (lddb,n), on GPU device.

    @param[in]
    lddb        Leading dimension of matrix B. lddb >= m.

    @param[in]
    queue       Queue to execute in.

    @ingroup icla_copymatrix
*******************************************************************************/
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


/***************************************************************************//**
    @fn icla_copymatrix_async( m, n, elemSize, dA_src, ldda, dB_dst, lddb, queue )

    Copy all or part of matrix dA_src on GPU device to dB_dst on GPU device.
    Elements may be arbitrary size.
    Type-safe versions set elemSize appropriately.
    With CUDA unified addressing, dA and dB can be on different GPUs.

    This version is asynchronous: it may return before the transfer finishes.
    See icla_copyvector() for a synchronous version.

    @param[in]
    m           Number of rows of matrix A. m >= 0.

    @param[in]
    n           Number of columns of matrix A. n >= 0.

    @param[in]
    elemSize    Size of each element, e.g., sizeof(double).

    @param[in]
    dA_src      Source array of dimension (ldda,n), on GPU device.

    @param[in]
    ldda        Leading dimension of matrix A. ldda >= m.

    @param[out]
    dB_dst      Destination array of dimension (lddb,n), on GPU device.

    @param[in]
    lddb        Leading dimension of matrix B. lddb >= m.

    @param[in]
    queue       Queue to execute in.

    @ingroup icla_copymatrix
*******************************************************************************/
extern "C" void
icla_copymatrix_async_internal(
    icla_int_t m, icla_int_t n, icla_int_t elemSize,
    icla_const_ptr dA_src, icla_int_t ldda,
    icla_ptr       dB_dst, icla_int_t lddb,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    // for backwards compatability, accepts NULL queue to mean NULL stream.
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

#endif // ICLA_HAVE_CUDA
