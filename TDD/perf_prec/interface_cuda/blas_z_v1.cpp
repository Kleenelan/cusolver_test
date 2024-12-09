
#ifndef ICLA_NO_V1

#include "icla_internal.h"
#include "iclablas_v1.h"

#include "error.h"

#define COMPLEX

#ifdef ICLA_HAVE_CUDA

extern "C" icla_int_t
icla_izamax_v1(
    icla_int_t n,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx )
{
    return icla_izamax( n, dx, incx, iclablasGetQueue() );
}

extern "C" icla_int_t
icla_izamin_v1(
    icla_int_t n,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx )
{
    return icla_izamin( n, dx, incx, iclablasGetQueue() );
}

extern "C" double
icla_dzasum_v1(
    icla_int_t n,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx )
{
    return icla_dzasum( n, dx, incx, iclablasGetQueue() );
}

extern "C" void
icla_zaxpy_v1(
    icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex_ptr       dy, icla_int_t incy )
{
    icla_zaxpy( n, alpha, dx, incx, dy, incy, iclablasGetQueue() );
}

extern "C" void
icla_zcopy_v1(
    icla_int_t n,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex_ptr       dy, icla_int_t incy )
{
    icla_zcopy( n, dx, incx, dy, incy, iclablasGetQueue() );
}

extern "C"
iclaDoubleComplex icla_zdotc_v1(
    icla_int_t n,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex_const_ptr dy, icla_int_t incy )
{
    return icla_zdotc( n, dx, incx, dy, incy, iclablasGetQueue() );
}

#ifdef COMPLEX
extern "C"
iclaDoubleComplex icla_zdotu_v1(
    icla_int_t n,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex_const_ptr dy, icla_int_t incy )
{
    return icla_zdotu( n, dx, incx, dy, incy, iclablasGetQueue() );
}
#endif

extern "C" double
icla_dznrm2_v1(
    icla_int_t n,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx )
{
    return icla_dznrm2( n, dx, incx, iclablasGetQueue() );
}

extern "C" void
icla_zrot_v1(
    icla_int_t n,
    iclaDoubleComplex_ptr dx, icla_int_t incx,
    iclaDoubleComplex_ptr dy, icla_int_t incy,
    double c, iclaDoubleComplex s )
{
    icla_zrot( n, dx, incx, dy, incy, c, s, iclablasGetQueue() );
}

#ifdef COMPLEX
extern "C" void
icla_zdrot_v1(
    icla_int_t n,
    iclaDoubleComplex_ptr dx, icla_int_t incx,
    iclaDoubleComplex_ptr dy, icla_int_t incy,
    double c, double s )
{
    icla_zdrot( n, dx, incx, dy, incy, c, s, iclablasGetQueue() );
}
#endif

#ifdef REAL
extern "C" void
icla_zrotm_v1(
    icla_int_t n,
    double *dx, icla_int_t incx,
    double *dy, icla_int_t incy,
    const double *param )
{
    icla_zrotm( n, dx, incx, dy, incy, param, iclablasGetQueue() );
}
#endif

#ifdef REAL
extern "C" void
icla_zrotmg_v1(
    double *d1, double       *d2,
    double *x1, const double *y1,
    double *param )
{
    icla_zrotmg( d1, d2, x1, y1, param, iclablasGetQueue() );
}
#endif

extern "C" void
icla_zscal_v1(
    icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_ptr dx, icla_int_t incx )
{
    icla_zscal( n, alpha, dx, incx, iclablasGetQueue() );
}

#ifdef COMPLEX
extern "C" void
icla_zdscal_v1(
    icla_int_t n,
    double alpha,
    iclaDoubleComplex_ptr dx, icla_int_t incx )
{
    icla_zdscal( n, alpha, dx, incx, iclablasGetQueue() );
}
#endif

extern "C" void
icla_zswap_v1(
    icla_int_t n,
    iclaDoubleComplex_ptr dx, icla_int_t incx,
    iclaDoubleComplex_ptr dy, icla_int_t incy )
{
    icla_zswap( n, dx, incx, dy, incy, iclablasGetQueue() );
}

extern "C" void
icla_zgemv_v1(
    icla_trans_t transA,
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dy, icla_int_t incy )
{
    icla_zgemv(
        transA,
        m, n,
        alpha, dA, ldda,
               dx, incx,
        beta,  dy, incy,
        iclablasGetQueue() );
}

extern "C" void
icla_zgerc_v1(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex_const_ptr dy, icla_int_t incy,
    iclaDoubleComplex_ptr       dA, icla_int_t ldda )
{
    icla_zgerc(
        m, n,
        alpha, dx, incx,
               dy, incy,
               dA, ldda,
        iclablasGetQueue() );
}

#ifdef COMPLEX
extern "C" void
icla_zgeru_v1(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex_const_ptr dy, icla_int_t incy,
    iclaDoubleComplex_ptr       dA, icla_int_t ldda )
{
    icla_zgeru(
        m, n,
        alpha, dx, incx,
               dy, incy,
               dA, ldda,
        iclablasGetQueue() );
}
#endif

extern "C" void
icla_zhemv_v1(
    icla_uplo_t uplo,
    icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dy, icla_int_t incy )
{
    icla_zhemv(
        uplo,
        n,
        alpha, dA, ldda,
               dx, incx,
        beta,  dy, incy,
        iclablasGetQueue() );
}

extern "C" void
icla_zher_v1(
    icla_uplo_t uplo,
    icla_int_t n,
    double alpha,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex_ptr       dA, icla_int_t ldda )
{
    icla_zher(
        uplo,
        n,
        alpha, dx, incx,
               dA, ldda,
        iclablasGetQueue() );
}

extern "C" void
icla_zher2_v1(
    icla_uplo_t uplo,
    icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex_const_ptr dy, icla_int_t incy,
    iclaDoubleComplex_ptr       dA, icla_int_t ldda )
{
    icla_zher2(
        uplo,
        n,
        alpha, dx, incx,
               dy, incy,
               dA, ldda,
        iclablasGetQueue() );
}

extern "C" void
icla_ztrmv_v1(
    icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t n,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr       dx, icla_int_t incx )
{
    icla_ztrmv(
        uplo, trans, diag,
        n,
        dA, ldda,
        dx, incx,
        iclablasGetQueue() );
}

extern "C" void
icla_ztrsv_v1(
    icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t n,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr       dx, icla_int_t incx )
{
    icla_ztrsv(
        uplo, trans, diag,
        n,
        dA, ldda,
        dx, incx,
        iclablasGetQueue() );
}

extern "C" void
icla_zgemm_v1(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dB, icla_int_t lddb,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dC, icla_int_t lddc )
{
    icla_zgemm(
        transA, transB,
        m, n, k,
        alpha, dA, ldda,
               dB, lddb,
        beta,  dC, lddc,
        iclablasGetQueue() );
}

extern "C" void
icla_zsymm_v1(
    icla_side_t side, icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dB, icla_int_t lddb,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dC, icla_int_t lddc )
{
    icla_zsymm(
        side, uplo,
        m, n,
        alpha, dA, ldda,
               dB, lddb,
        beta,  dC, lddc,
        iclablasGetQueue() );
}

extern "C" void
icla_zsyrk_v1(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dC, icla_int_t lddc )
{
    icla_zsyrk(
        uplo, trans,
        n, k,
        alpha, dA, ldda,
        beta,  dC, lddc,
        iclablasGetQueue() );
}

extern "C" void
icla_zsyr2k_v1(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dB, icla_int_t lddb,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dC, icla_int_t lddc )
{
    icla_zsyr2k(
        uplo, trans,
        n, k,
        alpha, dA, ldda,
               dB, lddb,
        beta,  dC, lddc,
        iclablasGetQueue() );
}

#ifdef COMPLEX
extern "C" void
icla_zhemm_v1(
    icla_side_t side, icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dB, icla_int_t lddb,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dC, icla_int_t lddc )
{
    icla_zhemm(
        side, uplo,
        m, n,
        alpha, dA, ldda,
               dB, lddb,
        beta,  dC, lddc,
        iclablasGetQueue() );
}
#endif

#ifdef COMPLEX
extern "C" void
icla_zherk_v1(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    double alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    double beta,
    iclaDoubleComplex_ptr       dC, icla_int_t lddc )
{
    icla_zherk(
        uplo, trans,
        n, k,
        alpha, dA, ldda,
        beta,  dC, lddc,
        iclablasGetQueue() );
}
#endif

#ifdef COMPLEX
extern "C" void
icla_zher2k_v1(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dB, icla_int_t lddb,
    double beta,
    iclaDoubleComplex_ptr       dC, icla_int_t lddc )
{
    icla_zher2k(
        uplo, trans,
        n, k,
        alpha, dA, ldda,
               dB, lddb,
        beta,  dC, lddc,
        iclablasGetQueue() );
}
#endif

extern "C" void
icla_ztrmm_v1(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr       dB, icla_int_t lddb )
{
    icla_ztrmm(
        side, uplo, trans, diag,
        m, n,
        alpha, dA, ldda,
               dB, lddb,
        iclablasGetQueue() );
}

extern "C" void
icla_ztrsm_v1(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr       dB, icla_int_t lddb )
{
    icla_ztrsm(
        side, uplo, trans, diag,
        m, n,
        alpha, dA, ldda,
               dB, lddb,
        iclablasGetQueue() );
}

#endif

#undef COMPLEX

#endif

