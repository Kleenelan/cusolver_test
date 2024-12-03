/*
    -- ICLA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates
*/
#ifndef ICLA_NO_V1

#include "icla_internal.h"
#include "iclablas_v1.h"  // includes v1 prototypes; does NOT map routine names
#include "error.h"

#include <cuda_runtime.h>

#if defined(ICLA_HAVE_CUDA) || defined(ICLA_HAVE_HIP)

// These ICLA v1 routines are all deprecated.
// See copy_v2.cpp for documentation.

// Generic, type-independent routines to copy data.
// Type-safe versions which avoid the user needing sizeof(...) are in headers;
// see icla_{s,d,c,z,i,index_}{set,get,copy}{matrix,vector}

/******************************************************************************/
extern "C" void
icla_setvector_v1_internal(
    icla_int_t n, icla_int_t elemSize,
    void const* hx_src, icla_int_t incx,
    icla_ptr   dy_dst, icla_int_t incy,
    const char* func, const char* file, int line )
{
    icla_setvector_internal(
        n, elemSize,
        hx_src, incx,
        dy_dst, incy,
        iclablasGetQueue(),
        func, file, line );
}


/******************************************************************************/
extern "C" void
icla_getvector_v1_internal(
    icla_int_t n, icla_int_t elemSize,
    icla_const_ptr dx_src, icla_int_t incx,
    void*           hy_dst, icla_int_t incy,
    const char* func, const char* file, int line )
{
    icla_getvector_internal(
        n, elemSize,
        dx_src, incx,
        hy_dst, incy,
        iclablasGetQueue(),
        func, file, line );
}


/******************************************************************************/
extern "C" void
icla_copyvector_v1_internal(
    icla_int_t n, icla_int_t elemSize,
    icla_const_ptr dx_src, icla_int_t incx,
    icla_ptr       dy_dst, icla_int_t incy,
    const char* func, const char* file, int line )
{
    icla_copyvector_internal(
        n, elemSize,
        dx_src, incx,
        dy_dst, incy,
        iclablasGetQueue(),
        func, file, line );
}


/******************************************************************************/
extern "C" void
icla_setmatrix_v1_internal(
    icla_int_t m, icla_int_t n, icla_int_t elemSize,
    void const* hA_src, icla_int_t lda,
    icla_ptr   dB_dst, icla_int_t lddb,
    const char* func, const char* file, int line )
{
    cublasStatus_t status;
    status = cublasSetMatrix(
        int(m), int(n), int(elemSize),
        hA_src, int(lda),
        dB_dst, int(lddb) );
    check_xerror( status, func, file, line );
    ICLA_UNUSED( status );
}


/******************************************************************************/
extern "C" void
icla_getmatrix_v1_internal(
    icla_int_t m, icla_int_t n, icla_int_t elemSize,
    icla_const_ptr dA_src, icla_int_t ldda,
    void*           hB_dst, icla_int_t ldb,
    const char* func, const char* file, int line )
{
    cublasStatus_t status;
    status = cublasGetMatrix(
        int(m), int(n), int(elemSize),
        dA_src, int(ldda),
        hB_dst, int(ldb) );
    check_xerror( status, func, file, line );
    ICLA_UNUSED( status );
}


/******************************************************************************/
extern "C" void
icla_copymatrix_v1_internal(
    icla_int_t m, icla_int_t n, icla_int_t elemSize,
    icla_const_ptr dA_src, icla_int_t ldda,
    icla_ptr       dB_dst, icla_int_t lddb,
    const char* func, const char* file, int line )
{
    cudaError_t status;
    status = cudaMemcpy2D(
        dB_dst, int(lddb*elemSize),
        dA_src, int(ldda*elemSize),
        int(m*elemSize), int(n), cudaMemcpyDeviceToDevice );
    check_xerror( status, func, file, line );
    ICLA_UNUSED( status );
}

#endif // ICLA_HAVE_CUDA

#endif // ICLA_NO_V1
