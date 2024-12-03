/*
    -- ICLA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates
       @generated from interface_cuda/blas_z_v1.cpp, normal z -> s, Fri Nov 29 12:16:16 2024
*/
#ifndef ICLA_NO_V1

#include "icla_internal.h"
#include "iclablas_v1.h"  // includes v1 prototypes; does NOT map routine names
#include "error.h"

#define REAL

#ifdef ICLA_HAVE_CUDA

// These ICLA v1 routines are all deprecated.
// See blas_s_v2.cpp for documentation.


// =============================================================================
// Level 1 BLAS

/******************************************************************************/
extern "C" icla_int_t
icla_isamax_v1(
    icla_int_t n,
    iclaFloat_const_ptr dx, icla_int_t incx )
{
    return icla_isamax( n, dx, incx, iclablasGetQueue() );
}


/******************************************************************************/
extern "C" icla_int_t
icla_isamin_v1(
    icla_int_t n,
    iclaFloat_const_ptr dx, icla_int_t incx )
{
    return icla_isamin( n, dx, incx, iclablasGetQueue() );
}


/******************************************************************************/
extern "C" float
icla_sasum_v1(
    icla_int_t n,
    iclaFloat_const_ptr dx, icla_int_t incx )
{
    return icla_sasum( n, dx, incx, iclablasGetQueue() );
}


/******************************************************************************/
extern "C" void
icla_saxpy_v1(
    icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dx, icla_int_t incx,
    iclaFloat_ptr       dy, icla_int_t incy )
{
    icla_saxpy( n, alpha, dx, incx, dy, incy, iclablasGetQueue() );
}


/******************************************************************************/
extern "C" void
icla_scopy_v1(
    icla_int_t n,
    iclaFloat_const_ptr dx, icla_int_t incx,
    iclaFloat_ptr       dy, icla_int_t incy )
{
    icla_scopy( n, dx, incx, dy, incy, iclablasGetQueue() );
}


/******************************************************************************/
extern "C"
float icla_sdot_v1(
    icla_int_t n,
    iclaFloat_const_ptr dx, icla_int_t incx,
    iclaFloat_const_ptr dy, icla_int_t incy )
{
    return icla_sdot( n, dx, incx, dy, incy, iclablasGetQueue() );
}


/******************************************************************************/
#ifdef COMPLEX
extern "C"
float icla_sdot_v1(
    icla_int_t n,
    iclaFloat_const_ptr dx, icla_int_t incx,
    iclaFloat_const_ptr dy, icla_int_t incy )
{
    return icla_sdot( n, dx, incx, dy, incy, iclablasGetQueue() );
}
#endif // COMPLEX


/******************************************************************************/
extern "C" float
icla_snrm2_v1(
    icla_int_t n,
    iclaFloat_const_ptr dx, icla_int_t incx )
{
    return icla_snrm2( n, dx, incx, iclablasGetQueue() );
}


/******************************************************************************/
extern "C" void
icla_srot_v1(
    icla_int_t n,
    iclaFloat_ptr dx, icla_int_t incx,
    iclaFloat_ptr dy, icla_int_t incy,
    float c, float s )
{
    icla_srot( n, dx, incx, dy, incy, c, s, iclablasGetQueue() );
}


/******************************************************************************/
#ifdef COMPLEX
extern "C" void
icla_srot_v1(
    icla_int_t n,
    iclaFloat_ptr dx, icla_int_t incx,
    iclaFloat_ptr dy, icla_int_t incy,
    float c, float s )
{
    icla_srot( n, dx, incx, dy, incy, c, s, iclablasGetQueue() );
}
#endif // COMPLEX


/******************************************************************************/
#ifdef REAL
extern "C" void
icla_srotm_v1(
    icla_int_t n,
    float *dx, icla_int_t incx,
    float *dy, icla_int_t incy,
    const float *param )
{
    icla_srotm( n, dx, incx, dy, incy, param, iclablasGetQueue() );
}
#endif // REAL


/******************************************************************************/
#ifdef REAL
extern "C" void
icla_srotmg_v1(
    float *d1, float       *d2,
    float *x1, const float *y1,
    float *param )
{
    icla_srotmg( d1, d2, x1, y1, param, iclablasGetQueue() );
}
#endif // REAL


/******************************************************************************/
extern "C" void
icla_sscal_v1(
    icla_int_t n,
    float alpha,
    iclaFloat_ptr dx, icla_int_t incx )
{
    icla_sscal( n, alpha, dx, incx, iclablasGetQueue() );
}


/******************************************************************************/
#ifdef COMPLEX
extern "C" void
icla_sscal_v1(
    icla_int_t n,
    float alpha,
    iclaFloat_ptr dx, icla_int_t incx )
{
    icla_sscal( n, alpha, dx, incx, iclablasGetQueue() );
}
#endif // COMPLEX


/******************************************************************************/
extern "C" void
icla_sswap_v1(
    icla_int_t n,
    iclaFloat_ptr dx, icla_int_t incx,
    iclaFloat_ptr dy, icla_int_t incy )
{
    icla_sswap( n, dx, incx, dy, incy, iclablasGetQueue() );
}


// =============================================================================
// Level 2 BLAS

/******************************************************************************/
extern "C" void
icla_sgemv_v1(
    icla_trans_t transA,
    icla_int_t m, icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_const_ptr dx, icla_int_t incx,
    float beta,
    iclaFloat_ptr       dy, icla_int_t incy )
{
    icla_sgemv(
        transA,
        m, n,
        alpha, dA, ldda,
               dx, incx,
        beta,  dy, incy,
        iclablasGetQueue() );
}


/******************************************************************************/
extern "C" void
icla_sger_v1(
    icla_int_t m, icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dx, icla_int_t incx,
    iclaFloat_const_ptr dy, icla_int_t incy,
    iclaFloat_ptr       dA, icla_int_t ldda )
{
    icla_sger(
        m, n,
        alpha, dx, incx,
               dy, incy,
               dA, ldda,
        iclablasGetQueue() );
}


/******************************************************************************/
#ifdef COMPLEX
extern "C" void
icla_sger_v1(
    icla_int_t m, icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dx, icla_int_t incx,
    iclaFloat_const_ptr dy, icla_int_t incy,
    iclaFloat_ptr       dA, icla_int_t ldda )
{
    icla_sger(
        m, n,
        alpha, dx, incx,
               dy, incy,
               dA, ldda,
        iclablasGetQueue() );
}
#endif // COMPLEX


/******************************************************************************/
extern "C" void
icla_ssymv_v1(
    icla_uplo_t uplo,
    icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_const_ptr dx, icla_int_t incx,
    float beta,
    iclaFloat_ptr       dy, icla_int_t incy )
{
    icla_ssymv(
        uplo,
        n,
        alpha, dA, ldda,
               dx, incx,
        beta,  dy, incy,
        iclablasGetQueue() );
}


/******************************************************************************/
extern "C" void
icla_ssyr_v1(
    icla_uplo_t uplo,
    icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dx, icla_int_t incx,
    iclaFloat_ptr       dA, icla_int_t ldda )
{
    icla_ssyr(
        uplo,
        n,
        alpha, dx, incx,
               dA, ldda,
        iclablasGetQueue() );
}


/******************************************************************************/
extern "C" void
icla_ssyr2_v1(
    icla_uplo_t uplo,
    icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dx, icla_int_t incx,
    iclaFloat_const_ptr dy, icla_int_t incy,
    iclaFloat_ptr       dA, icla_int_t ldda )
{
    icla_ssyr2(
        uplo,
        n,
        alpha, dx, incx,
               dy, incy,
               dA, ldda,
        iclablasGetQueue() );
}


/******************************************************************************/
extern "C" void
icla_strmv_v1(
    icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t n,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_ptr       dx, icla_int_t incx )
{
    icla_strmv(
        uplo, trans, diag,
        n,
        dA, ldda,
        dx, incx,
        iclablasGetQueue() );
}


/******************************************************************************/
extern "C" void
icla_strsv_v1(
    icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t n,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_ptr       dx, icla_int_t incx )
{
    icla_strsv(
        uplo, trans, diag,
        n,
        dA, ldda,
        dx, incx,
        iclablasGetQueue() );
}


// =============================================================================
// Level 3 BLAS

/******************************************************************************/
extern "C" void
icla_sgemm_v1(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_const_ptr dB, icla_int_t lddb,
    float beta,
    iclaFloat_ptr       dC, icla_int_t lddc )
{
    icla_sgemm(
        transA, transB,
        m, n, k,
        alpha, dA, ldda,
               dB, lddb,
        beta,  dC, lddc,
        iclablasGetQueue() );
}


/******************************************************************************/
extern "C" void
icla_ssymm_v1(
    icla_side_t side, icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_const_ptr dB, icla_int_t lddb,
    float beta,
    iclaFloat_ptr       dC, icla_int_t lddc )
{
    icla_ssymm(
        side, uplo,
        m, n,
        alpha, dA, ldda,
               dB, lddb,
        beta,  dC, lddc,
        iclablasGetQueue() );
}


/******************************************************************************/
extern "C" void
icla_ssyrk_v1(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    float beta,
    iclaFloat_ptr       dC, icla_int_t lddc )
{
    icla_ssyrk(
        uplo, trans,
        n, k,
        alpha, dA, ldda,
        beta,  dC, lddc,
        iclablasGetQueue() );
}


/******************************************************************************/
extern "C" void
icla_ssyr2k_v1(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_const_ptr dB, icla_int_t lddb,
    float beta,
    iclaFloat_ptr       dC, icla_int_t lddc )
{
    icla_ssyr2k(
        uplo, trans,
        n, k,
        alpha, dA, ldda,
               dB, lddb,
        beta,  dC, lddc,
        iclablasGetQueue() );
}


/******************************************************************************/
#ifdef COMPLEX
extern "C" void
icla_ssymm_v1(
    icla_side_t side, icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_const_ptr dB, icla_int_t lddb,
    float beta,
    iclaFloat_ptr       dC, icla_int_t lddc )
{
    icla_ssymm(
        side, uplo,
        m, n,
        alpha, dA, ldda,
               dB, lddb,
        beta,  dC, lddc,
        iclablasGetQueue() );
}
#endif // COMPLEX


/******************************************************************************/
#ifdef COMPLEX
extern "C" void
icla_ssyrk_v1(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    float beta,
    iclaFloat_ptr       dC, icla_int_t lddc )
{
    icla_ssyrk(
        uplo, trans,
        n, k,
        alpha, dA, ldda,
        beta,  dC, lddc,
        iclablasGetQueue() );
}
#endif // COMPLEX


/******************************************************************************/
#ifdef COMPLEX
extern "C" void
icla_ssyr2k_v1(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_const_ptr dB, icla_int_t lddb,
    float beta,
    iclaFloat_ptr       dC, icla_int_t lddc )
{
    icla_ssyr2k(
        uplo, trans,
        n, k,
        alpha, dA, ldda,
               dB, lddb,
        beta,  dC, lddc,
        iclablasGetQueue() );
}
#endif // COMPLEX


/******************************************************************************/
extern "C" void
icla_strmm_v1(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_ptr       dB, icla_int_t lddb )
{
    icla_strmm(
        side, uplo, trans, diag,
        m, n,
        alpha, dA, ldda,
               dB, lddb,
        iclablasGetQueue() );
}


/******************************************************************************/
extern "C" void
icla_strsm_v1(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_ptr       dB, icla_int_t lddb )
{
    icla_strsm(
        side, uplo, trans, diag,
        m, n,
        alpha, dA, ldda,
               dB, lddb,
        iclablasGetQueue() );
}

#endif // ICLA_HAVE_CUDA

#undef REAL

#endif // ICLA_NO_V1
