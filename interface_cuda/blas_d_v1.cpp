
#ifndef ICLA_NO_V1

#include "icla_internal.h"
#include "iclablas_v1.h"

#include "error.h"

#define REAL

#ifdef ICLA_HAVE_CUDA

extern "C" icla_int_t
icla_idamax_v1(
    icla_int_t n,
    iclaDouble_const_ptr dx, icla_int_t incx )
{
    return icla_idamax( n, dx, incx, iclablasGetQueue() );
}

extern "C" icla_int_t
icla_idamin_v1(
    icla_int_t n,
    iclaDouble_const_ptr dx, icla_int_t incx )
{
    return icla_idamin( n, dx, incx, iclablasGetQueue() );
}

extern "C" double
icla_dasum_v1(
    icla_int_t n,
    iclaDouble_const_ptr dx, icla_int_t incx )
{
    return icla_dasum( n, dx, incx, iclablasGetQueue() );
}

extern "C" void
icla_daxpy_v1(
    icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dx, icla_int_t incx,
    iclaDouble_ptr       dy, icla_int_t incy )
{
    icla_daxpy( n, alpha, dx, incx, dy, incy, iclablasGetQueue() );
}

extern "C" void
icla_dcopy_v1(
    icla_int_t n,
    iclaDouble_const_ptr dx, icla_int_t incx,
    iclaDouble_ptr       dy, icla_int_t incy )
{
    icla_dcopy( n, dx, incx, dy, incy, iclablasGetQueue() );
}

extern "C"
double icla_ddot_v1(
    icla_int_t n,
    iclaDouble_const_ptr dx, icla_int_t incx,
    iclaDouble_const_ptr dy, icla_int_t incy )
{
    return icla_ddot( n, dx, incx, dy, incy, iclablasGetQueue() );
}

#ifdef COMPLEX
extern "C"
double icla_ddot_v1(
    icla_int_t n,
    iclaDouble_const_ptr dx, icla_int_t incx,
    iclaDouble_const_ptr dy, icla_int_t incy )
{
    return icla_ddot( n, dx, incx, dy, incy, iclablasGetQueue() );
}
#endif

extern "C" double
icla_dnrm2_v1(
    icla_int_t n,
    iclaDouble_const_ptr dx, icla_int_t incx )
{
    return icla_dnrm2( n, dx, incx, iclablasGetQueue() );
}

extern "C" void
icla_drot_v1(
    icla_int_t n,
    iclaDouble_ptr dx, icla_int_t incx,
    iclaDouble_ptr dy, icla_int_t incy,
    double c, double s )
{
    icla_drot( n, dx, incx, dy, incy, c, s, iclablasGetQueue() );
}

#ifdef COMPLEX
extern "C" void
icla_drot_v1(
    icla_int_t n,
    iclaDouble_ptr dx, icla_int_t incx,
    iclaDouble_ptr dy, icla_int_t incy,
    double c, double s )
{
    icla_drot( n, dx, incx, dy, incy, c, s, iclablasGetQueue() );
}
#endif

#ifdef REAL
extern "C" void
icla_drotm_v1(
    icla_int_t n,
    double *dx, icla_int_t incx,
    double *dy, icla_int_t incy,
    const double *param )
{
    icla_drotm( n, dx, incx, dy, incy, param, iclablasGetQueue() );
}
#endif

#ifdef REAL
extern "C" void
icla_drotmg_v1(
    double *d1, double       *d2,
    double *x1, const double *y1,
    double *param )
{
    icla_drotmg( d1, d2, x1, y1, param, iclablasGetQueue() );
}
#endif

extern "C" void
icla_dscal_v1(
    icla_int_t n,
    double alpha,
    iclaDouble_ptr dx, icla_int_t incx )
{
    icla_dscal( n, alpha, dx, incx, iclablasGetQueue() );
}

#ifdef COMPLEX
extern "C" void
icla_dscal_v1(
    icla_int_t n,
    double alpha,
    iclaDouble_ptr dx, icla_int_t incx )
{
    icla_dscal( n, alpha, dx, incx, iclablasGetQueue() );
}
#endif

extern "C" void
icla_dswap_v1(
    icla_int_t n,
    iclaDouble_ptr dx, icla_int_t incx,
    iclaDouble_ptr dy, icla_int_t incy )
{
    icla_dswap( n, dx, incx, dy, incy, iclablasGetQueue() );
}

extern "C" void
icla_dgemv_v1(
    icla_trans_t transA,
    icla_int_t m, icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_const_ptr dx, icla_int_t incx,
    double beta,
    iclaDouble_ptr       dy, icla_int_t incy )
{
    icla_dgemv(
        transA,
        m, n,
        alpha, dA, ldda,
               dx, incx,
        beta,  dy, incy,
        iclablasGetQueue() );
}

extern "C" void
icla_dger_v1(
    icla_int_t m, icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dx, icla_int_t incx,
    iclaDouble_const_ptr dy, icla_int_t incy,
    iclaDouble_ptr       dA, icla_int_t ldda )
{
    icla_dger(
        m, n,
        alpha, dx, incx,
               dy, incy,
               dA, ldda,
        iclablasGetQueue() );
}

#ifdef COMPLEX
extern "C" void
icla_dger_v1(
    icla_int_t m, icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dx, icla_int_t incx,
    iclaDouble_const_ptr dy, icla_int_t incy,
    iclaDouble_ptr       dA, icla_int_t ldda )
{
    icla_dger(
        m, n,
        alpha, dx, incx,
               dy, incy,
               dA, ldda,
        iclablasGetQueue() );
}
#endif

extern "C" void
icla_dsymv_v1(
    icla_uplo_t uplo,
    icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_const_ptr dx, icla_int_t incx,
    double beta,
    iclaDouble_ptr       dy, icla_int_t incy )
{
    icla_dsymv(
        uplo,
        n,
        alpha, dA, ldda,
               dx, incx,
        beta,  dy, incy,
        iclablasGetQueue() );
}

extern "C" void
icla_dsyr_v1(
    icla_uplo_t uplo,
    icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dx, icla_int_t incx,
    iclaDouble_ptr       dA, icla_int_t ldda )
{
    icla_dsyr(
        uplo,
        n,
        alpha, dx, incx,
               dA, ldda,
        iclablasGetQueue() );
}

extern "C" void
icla_dsyr2_v1(
    icla_uplo_t uplo,
    icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dx, icla_int_t incx,
    iclaDouble_const_ptr dy, icla_int_t incy,
    iclaDouble_ptr       dA, icla_int_t ldda )
{
    icla_dsyr2(
        uplo,
        n,
        alpha, dx, incx,
               dy, incy,
               dA, ldda,
        iclablasGetQueue() );
}

extern "C" void
icla_dtrmv_v1(
    icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t n,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_ptr       dx, icla_int_t incx )
{
    icla_dtrmv(
        uplo, trans, diag,
        n,
        dA, ldda,
        dx, incx,
        iclablasGetQueue() );
}

extern "C" void
icla_dtrsv_v1(
    icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t n,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_ptr       dx, icla_int_t incx )
{
    icla_dtrsv(
        uplo, trans, diag,
        n,
        dA, ldda,
        dx, incx,
        iclablasGetQueue() );
}

extern "C" void
icla_dgemm_v1(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_const_ptr dB, icla_int_t lddb,
    double beta,
    iclaDouble_ptr       dC, icla_int_t lddc )
{
    icla_dgemm(
        transA, transB,
        m, n, k,
        alpha, dA, ldda,
               dB, lddb,
        beta,  dC, lddc,
        iclablasGetQueue() );
}

extern "C" void
icla_dsymm_v1(
    icla_side_t side, icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_const_ptr dB, icla_int_t lddb,
    double beta,
    iclaDouble_ptr       dC, icla_int_t lddc )
{
    icla_dsymm(
        side, uplo,
        m, n,
        alpha, dA, ldda,
               dB, lddb,
        beta,  dC, lddc,
        iclablasGetQueue() );
}

extern "C" void
icla_dsyrk_v1(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    double beta,
    iclaDouble_ptr       dC, icla_int_t lddc )
{
    icla_dsyrk(
        uplo, trans,
        n, k,
        alpha, dA, ldda,
        beta,  dC, lddc,
        iclablasGetQueue() );
}

extern "C" void
icla_dsyr2k_v1(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_const_ptr dB, icla_int_t lddb,
    double beta,
    iclaDouble_ptr       dC, icla_int_t lddc )
{
    icla_dsyr2k(
        uplo, trans,
        n, k,
        alpha, dA, ldda,
               dB, lddb,
        beta,  dC, lddc,
        iclablasGetQueue() );
}

#ifdef COMPLEX
extern "C" void
icla_dsymm_v1(
    icla_side_t side, icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_const_ptr dB, icla_int_t lddb,
    double beta,
    iclaDouble_ptr       dC, icla_int_t lddc )
{
    icla_dsymm(
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
icla_dsyrk_v1(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    double beta,
    iclaDouble_ptr       dC, icla_int_t lddc )
{
    icla_dsyrk(
        uplo, trans,
        n, k,
        alpha, dA, ldda,
        beta,  dC, lddc,
        iclablasGetQueue() );
}
#endif

#ifdef COMPLEX
extern "C" void
icla_dsyr2k_v1(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_const_ptr dB, icla_int_t lddb,
    double beta,
    iclaDouble_ptr       dC, icla_int_t lddc )
{
    icla_dsyr2k(
        uplo, trans,
        n, k,
        alpha, dA, ldda,
               dB, lddb,
        beta,  dC, lddc,
        iclablasGetQueue() );
}
#endif

extern "C" void
icla_dtrmm_v1(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_ptr       dB, icla_int_t lddb )
{
    icla_dtrmm(
        side, uplo, trans, diag,
        m, n,
        alpha, dA, ldda,
               dB, lddb,
        iclablasGetQueue() );
}

extern "C" void
icla_dtrsm_v1(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_ptr       dB, icla_int_t lddb )
{
    icla_dtrsm(
        side, uplo, trans, diag,
        m, n,
        alpha, dA, ldda,
               dB, lddb,
        iclablasGetQueue() );
}

#endif

#undef REAL

#endif

