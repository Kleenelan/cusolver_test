/*
    -- ICLA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates
       @generated from interface_cuda/blas_z_v1.cpp, normal z -> c, Fri Nov 29 12:16:16 2024
*/
#ifndef ICLA_NO_V1

#include "icla_internal.h"
#include "iclablas_v1.h"  // includes v1 prototypes; does NOT map routine names
#include "error.h"

#define COMPLEX

#ifdef ICLA_HAVE_CUDA

// These ICLA v1 routines are all deprecated.
// See blas_c_v2.cpp for documentation.


// =============================================================================
// Level 1 BLAS

/******************************************************************************/
extern "C" icla_int_t
icla_icamax_v1(
    icla_int_t n,
    iclaFloatComplex_const_ptr dx, icla_int_t incx )
{
    return icla_icamax( n, dx, incx, iclablasGetQueue() );
}


/******************************************************************************/
extern "C" icla_int_t
icla_icamin_v1(
    icla_int_t n,
    iclaFloatComplex_const_ptr dx, icla_int_t incx )
{
    return icla_icamin( n, dx, incx, iclablasGetQueue() );
}


/******************************************************************************/
extern "C" float
icla_scasum_v1(
    icla_int_t n,
    iclaFloatComplex_const_ptr dx, icla_int_t incx )
{
    return icla_scasum( n, dx, incx, iclablasGetQueue() );
}


/******************************************************************************/
extern "C" void
icla_caxpy_v1(
    icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex_ptr       dy, icla_int_t incy )
{
    icla_caxpy( n, alpha, dx, incx, dy, incy, iclablasGetQueue() );
}


/******************************************************************************/
extern "C" void
icla_ccopy_v1(
    icla_int_t n,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex_ptr       dy, icla_int_t incy )
{
    icla_ccopy( n, dx, incx, dy, incy, iclablasGetQueue() );
}


/******************************************************************************/
extern "C"
iclaFloatComplex icla_cdotc_v1(
    icla_int_t n,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex_const_ptr dy, icla_int_t incy )
{
    return icla_cdotc( n, dx, incx, dy, incy, iclablasGetQueue() );
}


/******************************************************************************/
#ifdef COMPLEX
extern "C"
iclaFloatComplex icla_cdotu_v1(
    icla_int_t n,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex_const_ptr dy, icla_int_t incy )
{
    return icla_cdotu( n, dx, incx, dy, incy, iclablasGetQueue() );
}
#endif // COMPLEX


/******************************************************************************/
extern "C" float
icla_scnrm2_v1(
    icla_int_t n,
    iclaFloatComplex_const_ptr dx, icla_int_t incx )
{
    return icla_scnrm2( n, dx, incx, iclablasGetQueue() );
}


/******************************************************************************/
extern "C" void
icla_crot_v1(
    icla_int_t n,
    iclaFloatComplex_ptr dx, icla_int_t incx,
    iclaFloatComplex_ptr dy, icla_int_t incy,
    float c, iclaFloatComplex s )
{
    icla_crot( n, dx, incx, dy, incy, c, s, iclablasGetQueue() );
}


/******************************************************************************/
#ifdef COMPLEX
extern "C" void
icla_csrot_v1(
    icla_int_t n,
    iclaFloatComplex_ptr dx, icla_int_t incx,
    iclaFloatComplex_ptr dy, icla_int_t incy,
    float c, float s )
{
    icla_csrot( n, dx, incx, dy, incy, c, s, iclablasGetQueue() );
}
#endif // COMPLEX


/******************************************************************************/
#ifdef REAL
extern "C" void
icla_crotm_v1(
    icla_int_t n,
    float *dx, icla_int_t incx,
    float *dy, icla_int_t incy,
    const float *param )
{
    icla_crotm( n, dx, incx, dy, incy, param, iclablasGetQueue() );
}
#endif // REAL


/******************************************************************************/
#ifdef REAL
extern "C" void
icla_crotmg_v1(
    float *d1, float       *d2,
    float *x1, const float *y1,
    float *param )
{
    icla_crotmg( d1, d2, x1, y1, param, iclablasGetQueue() );
}
#endif // REAL


/******************************************************************************/
extern "C" void
icla_cscal_v1(
    icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_ptr dx, icla_int_t incx )
{
    icla_cscal( n, alpha, dx, incx, iclablasGetQueue() );
}


/******************************************************************************/
#ifdef COMPLEX
extern "C" void
icla_csscal_v1(
    icla_int_t n,
    float alpha,
    iclaFloatComplex_ptr dx, icla_int_t incx )
{
    icla_csscal( n, alpha, dx, incx, iclablasGetQueue() );
}
#endif // COMPLEX


/******************************************************************************/
extern "C" void
icla_cswap_v1(
    icla_int_t n,
    iclaFloatComplex_ptr dx, icla_int_t incx,
    iclaFloatComplex_ptr dy, icla_int_t incy )
{
    icla_cswap( n, dx, incx, dy, incy, iclablasGetQueue() );
}


// =============================================================================
// Level 2 BLAS

/******************************************************************************/
extern "C" void
icla_cgemv_v1(
    icla_trans_t transA,
    icla_int_t m, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr       dy, icla_int_t incy )
{
    icla_cgemv(
        transA,
        m, n,
        alpha, dA, ldda,
               dx, incx,
        beta,  dy, incy,
        iclablasGetQueue() );
}


/******************************************************************************/
extern "C" void
icla_cgerc_v1(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex_const_ptr dy, icla_int_t incy,
    iclaFloatComplex_ptr       dA, icla_int_t ldda )
{
    icla_cgerc(
        m, n,
        alpha, dx, incx,
               dy, incy,
               dA, ldda,
        iclablasGetQueue() );
}


/******************************************************************************/
#ifdef COMPLEX
extern "C" void
icla_cgeru_v1(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex_const_ptr dy, icla_int_t incy,
    iclaFloatComplex_ptr       dA, icla_int_t ldda )
{
    icla_cgeru(
        m, n,
        alpha, dx, incx,
               dy, incy,
               dA, ldda,
        iclablasGetQueue() );
}
#endif // COMPLEX


/******************************************************************************/
extern "C" void
icla_chemv_v1(
    icla_uplo_t uplo,
    icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr       dy, icla_int_t incy )
{
    icla_chemv(
        uplo,
        n,
        alpha, dA, ldda,
               dx, incx,
        beta,  dy, incy,
        iclablasGetQueue() );
}


/******************************************************************************/
extern "C" void
icla_cher_v1(
    icla_uplo_t uplo,
    icla_int_t n,
    float alpha,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex_ptr       dA, icla_int_t ldda )
{
    icla_cher(
        uplo,
        n,
        alpha, dx, incx,
               dA, ldda,
        iclablasGetQueue() );
}


/******************************************************************************/
extern "C" void
icla_cher2_v1(
    icla_uplo_t uplo,
    icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex_const_ptr dy, icla_int_t incy,
    iclaFloatComplex_ptr       dA, icla_int_t ldda )
{
    icla_cher2(
        uplo,
        n,
        alpha, dx, incx,
               dy, incy,
               dA, ldda,
        iclablasGetQueue() );
}


/******************************************************************************/
extern "C" void
icla_ctrmv_v1(
    icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t n,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr       dx, icla_int_t incx )
{
    icla_ctrmv(
        uplo, trans, diag,
        n,
        dA, ldda,
        dx, incx,
        iclablasGetQueue() );
}


/******************************************************************************/
extern "C" void
icla_ctrsv_v1(
    icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t n,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr       dx, icla_int_t incx )
{
    icla_ctrsv(
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
icla_cgemm_v1(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_const_ptr dB, icla_int_t lddb,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr       dC, icla_int_t lddc )
{
    icla_cgemm(
        transA, transB,
        m, n, k,
        alpha, dA, ldda,
               dB, lddb,
        beta,  dC, lddc,
        iclablasGetQueue() );
}


/******************************************************************************/
extern "C" void
icla_csymm_v1(
    icla_side_t side, icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_const_ptr dB, icla_int_t lddb,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr       dC, icla_int_t lddc )
{
    icla_csymm(
        side, uplo,
        m, n,
        alpha, dA, ldda,
               dB, lddb,
        beta,  dC, lddc,
        iclablasGetQueue() );
}


/******************************************************************************/
extern "C" void
icla_csyrk_v1(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr       dC, icla_int_t lddc )
{
    icla_csyrk(
        uplo, trans,
        n, k,
        alpha, dA, ldda,
        beta,  dC, lddc,
        iclablasGetQueue() );
}


/******************************************************************************/
extern "C" void
icla_csyr2k_v1(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_const_ptr dB, icla_int_t lddb,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr       dC, icla_int_t lddc )
{
    icla_csyr2k(
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
icla_chemm_v1(
    icla_side_t side, icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_const_ptr dB, icla_int_t lddb,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr       dC, icla_int_t lddc )
{
    icla_chemm(
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
icla_cherk_v1(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    float alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    float beta,
    iclaFloatComplex_ptr       dC, icla_int_t lddc )
{
    icla_cherk(
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
icla_cher2k_v1(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_const_ptr dB, icla_int_t lddb,
    float beta,
    iclaFloatComplex_ptr       dC, icla_int_t lddc )
{
    icla_cher2k(
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
icla_ctrmm_v1(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr       dB, icla_int_t lddb )
{
    icla_ctrmm(
        side, uplo, trans, diag,
        m, n,
        alpha, dA, ldda,
               dB, lddb,
        iclablasGetQueue() );
}


/******************************************************************************/
extern "C" void
icla_ctrsm_v1(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr       dB, icla_int_t lddb )
{
    icla_ctrsm(
        side, uplo, trans, diag,
        m, n,
        alpha, dA, ldda,
               dB, lddb,
        iclablasGetQueue() );
}

#endif // ICLA_HAVE_CUDA

#undef COMPLEX

#endif // ICLA_NO_V1
