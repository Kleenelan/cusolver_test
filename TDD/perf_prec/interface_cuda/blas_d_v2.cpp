
#include "icla_internal.h"
#include "error.h"

#define REAL

#define PRECISION_d

extern "C" icla_int_t
icla_idamax(
    icla_int_t n,
    iclaDouble_const_ptr dx, icla_int_t incx,
    icla_queue_t queue )
{
    int result;

    cublasIdamax( queue->cublas_handle(), int(n), (double*)dx, int(incx), &result );
    return result;
}

extern "C" icla_int_t
icla_idamin(
    icla_int_t n,
    iclaDouble_const_ptr dx, icla_int_t incx,
    icla_queue_t queue )
{
    int result;

    cublasIdamin( queue->cublas_handle(), int(n), (double*)dx, int(incx), &result );
    return result;
}

extern "C" double
icla_dasum(
    icla_int_t n,
    iclaDouble_const_ptr dx, icla_int_t incx,
    icla_queue_t queue )
{
    double result;
    cublasDasum( queue->cublas_handle(), int(n), (double*)dx, int(incx), &result );
    return result;
}

extern "C" void
icla_daxpy(
    icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dx, icla_int_t incx,
    iclaDouble_ptr       dy, icla_int_t incy,
    icla_queue_t queue )
{
    cublasDaxpy( queue->cublas_handle(), int(n), (double*)&alpha, (double*)dx, int(incx), (double*)dy, int(incy) );
}

extern "C" void
icla_dcopy(
    icla_int_t n,
    iclaDouble_const_ptr dx, icla_int_t incx,
    iclaDouble_ptr       dy, icla_int_t incy,
    icla_queue_t queue )
{
    cublasDcopy( queue->cublas_handle(), int(n), (double*)dx, int(incx), (double*)dy, int(incy) );
}

#ifdef COMPLEX

extern "C"
double icla_ddot(
    icla_int_t n,
    iclaDouble_const_ptr dx, icla_int_t incx,
    iclaDouble_const_ptr dy, icla_int_t incy,
    icla_queue_t queue )
{
    double result;
    cublasDdot( queue->cublas_handle(), int(n), (double*)dx, int(incx), (double*)dy, int(incy), (double*)&result );
    return result;
}
#endif

extern "C"
double icla_ddot(
    icla_int_t n,
    iclaDouble_const_ptr dx, icla_int_t incx,
    iclaDouble_const_ptr dy, icla_int_t incy,
    icla_queue_t queue )
{
    double result;
    cublasDdot( queue->cublas_handle(), int(n), (double*)dx, int(incx), (double*)dy, int(incy), (double*)&result );
    return result;
}

extern "C" double
icla_dnrm2(
    icla_int_t n,
    iclaDouble_const_ptr dx, icla_int_t incx,
    icla_queue_t queue )
{
    double result;
    cublasDnrm2( queue->cublas_handle(), int(n), (double*)dx, int(incx), &result );
    return result;
}

extern "C" void
icla_drot(
    icla_int_t n,
    iclaDouble_ptr dx, icla_int_t incx,
    iclaDouble_ptr dy, icla_int_t incy,
    double c, double s,
    icla_queue_t queue )
{
    cublasDrot( queue->cublas_handle(), int(n), (double*)dx, int(incx), (double*)dy, int(incy), &c, (double*)&s );
}

#ifdef COMPLEX

extern "C" void
icla_drot(
    icla_int_t n,
    iclaDouble_ptr dx, icla_int_t incx,
    iclaDouble_ptr dy, icla_int_t incy,
    double c, double s,
    icla_queue_t queue )
{
    cublasDrot( queue->cublas_handle(), int(n), (double*)dx, int(incx), (double*)dy, int(incy), &c, &s );
}
#endif

extern "C" void
icla_drotg(
    double *a, double *b,
    double             *c, double *s,
    icla_queue_t queue )
{
    cublasDrotg( queue->cublas_handle(), (double*)a, (double*)b, c, (double*)s );
}

#ifdef REAL

extern "C" void
icla_drotm(
    icla_int_t n,
    double *dx, icla_int_t incx,
    double *dy, icla_int_t incy,
    const double *param,
    icla_queue_t queue )
{
    cublasDrotm( queue->cublas_handle(), int(n), dx, int(incx), dy, int(incy), param );
}
#endif

#ifdef REAL

extern "C" void
icla_drotmg(
    double *d1, double       *d2,
    double *x1, const double *y1,
    double *param,
    icla_queue_t queue )
{
    cublasDrotmg( queue->cublas_handle(), d1, d2, x1, y1, param );
}
#endif

extern "C" void
icla_dscal(
    icla_int_t n,
    double alpha,
    iclaDouble_ptr dx, icla_int_t incx,
    icla_queue_t queue )
{
    cublasDscal( queue->cublas_handle(), int(n), (double*)&alpha, (double*)dx, int(incx) );
}

#ifdef COMPLEX

extern "C" void
icla_dscal(
    icla_int_t n,
    double alpha,
    iclaDouble_ptr dx, icla_int_t incx,
    icla_queue_t queue )
{
    cublasDscal( queue->cublas_handle(), int(n), &alpha, (double*)dx, int(incx) );
}
#endif

extern "C" void
icla_dswap(
    icla_int_t n,
    iclaDouble_ptr dx, icla_int_t incx,
    iclaDouble_ptr dy, icla_int_t incy,
    icla_queue_t queue )
{
    cublasDswap( queue->cublas_handle(), int(n), (double*)dx, int(incx), (double*)dy, int(incy) );
}

extern "C" void
icla_dgemv(
    icla_trans_t transA,
    icla_int_t m, icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_const_ptr dx, icla_int_t incx,
    double beta,
    iclaDouble_ptr       dy, icla_int_t incy,
    icla_queue_t queue )
{
    cublasDgemv(
        queue->cublas_handle(),
        cublas_trans_const( transA ),
        int(m), int(n),
        (double*)&alpha, (double*)dA, int(ldda),
                (double*)dx, int(incx),
        (double*)&beta,  (double*)dy, int(incy) );
}

#ifdef COMPLEX

extern "C" void
icla_dger(
    icla_int_t m, icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dx, icla_int_t incx,
    iclaDouble_const_ptr dy, icla_int_t incy,
    iclaDouble_ptr       dA, icla_int_t ldda,
    icla_queue_t queue )
{
    cublasDger(
        queue->cublas_handle(),
        int(m), int(n),
        (double*)&alpha, (double*)dx, int(incx),
                (double*)dy, int(incy),
                (double*)dA, int(ldda) );
}
#endif

extern "C" void
icla_dger(
    icla_int_t m, icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dx, icla_int_t incx,
    iclaDouble_const_ptr dy, icla_int_t incy,
    iclaDouble_ptr       dA, icla_int_t ldda,
    icla_queue_t queue )
{
    cublasDger(
        queue->cublas_handle(),
        int(m), int(n),
        (double*)&alpha, (double*)dx, int(incx),
                (double*)dy, int(incy),
                (double*)dA, int(ldda) );
}

#ifdef COMPLEX

extern "C" void
icla_dsymv(
    icla_uplo_t uplo,
    icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_const_ptr dx, icla_int_t incx,
    double beta,
    iclaDouble_ptr       dy, icla_int_t incy,
    icla_queue_t queue )
{
    cublasDsymv(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        int(n),
        (double*)&alpha, (double*)dA, int(ldda),
                (double*)dx, int(incx),
        (double*)&beta,  (double*)dy, int(incy) );
}
#endif

#ifdef COMPLEX

extern "C" void
icla_dsyr(
    icla_uplo_t uplo,
    icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dx, icla_int_t incx,
    iclaDouble_ptr       dA, icla_int_t ldda,
    icla_queue_t queue )
{
    cublasDsyr(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        int(n),
        (const double*)&alpha, (double*)dx, int(incx),
                (double*)dA, int(ldda) );
}
#endif

#ifdef COMPLEX

extern "C" void
icla_dsyr2(
    icla_uplo_t uplo,
    icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dx, icla_int_t incx,
    iclaDouble_const_ptr dy, icla_int_t incy,
    iclaDouble_ptr       dA, icla_int_t ldda,
    icla_queue_t queue )
{
    cublasDsyr2(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        int(n),
        (double*)&alpha, (double*)dx, int(incx),
                (double*)dy, int(incy),
                (double*)dA, int(ldda) );
}
#endif

extern "C" void
icla_dsymv(
    icla_uplo_t uplo,
    icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_const_ptr dx, icla_int_t incx,
    double beta,
    iclaDouble_ptr       dy, icla_int_t incy,
    icla_queue_t queue )
{
    cublasDsymv(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        int(n),
        (double*)&alpha, (double*)dA, int(ldda),
                (double*)dx, int(incx),
        (double*)&beta,  (double*)dy, int(incy) );
}

extern "C" void
icla_dsyr(
    icla_uplo_t uplo,
    icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dx, icla_int_t incx,
    iclaDouble_ptr       dA, icla_int_t ldda,
    icla_queue_t queue )
{
    cublasDsyr(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        int(n),
        (double*)&alpha, (double*)dx, int(incx),
                (double*)dA, int(ldda) );
}

extern "C" void
icla_dsyr2(
    icla_uplo_t uplo,
    icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dx, icla_int_t incx,
    iclaDouble_const_ptr dy, icla_int_t incy,
    iclaDouble_ptr       dA, icla_int_t ldda,
    icla_queue_t queue )
{
    cublasDsyr2(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        int(n),
        (double*)&alpha, (double*)dx, int(incx),
                (double*)dy, int(incy),
                (double*)dA, int(ldda) );
}

extern "C" void
icla_dtrmv(
    icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t n,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_ptr       dx, icla_int_t incx,
    icla_queue_t queue )
{
    cublasDtrmv(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        cublas_diag_const( diag ),
        int(n),
        (double*)dA, int(ldda),
        (double*)dx, int(incx) );
}

extern "C" void
icla_dtrsv(
    icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t n,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_ptr       dx, icla_int_t incx,
    icla_queue_t queue )
{
    cublasDtrsv(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        cublas_diag_const( diag ),
        int(n),
        (double*)dA, int(ldda),
        (double*)dx, int(incx) );
}

extern "C" void
icla_dgemm(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_const_ptr dB, icla_int_t lddb,
    double beta,
    iclaDouble_ptr       dC, icla_int_t lddc,
    icla_queue_t queue )
{
    cublasDgemm(
        queue->cublas_handle(),
        cublas_trans_const( transA ),
        cublas_trans_const( transB ),
        int(m), int(n), int(k),
        (double*)&alpha, (double*)dA, int(ldda),
                (double*)dB, int(lddb),
        (double*)&beta,  (double*)dC, int(lddc) );
}

#ifdef COMPLEX

extern "C" void
icla_dsymm(
    icla_side_t side, icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_const_ptr dB, icla_int_t lddb,
    double beta,
    iclaDouble_ptr       dC, icla_int_t lddc,
    icla_queue_t queue )
{
    cublasDsymm(
        queue->cublas_handle(),
        cublas_side_const( side ),
        cublas_uplo_const( uplo ),
        int(m), int(n),
        (double*)&alpha, (double*)dA, int(ldda),
                (double*)dB, int(lddb),
        (double*)&beta,  (double*)dC, int(lddc) );
}
#endif

#ifdef COMPLEX

extern "C" void
icla_dsyrk(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    double beta,
    iclaDouble_ptr       dC, icla_int_t lddc,
    icla_queue_t queue )
{
    cublasDsyrk(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        int(n), int(k),
        &alpha, (double*)dA, int(ldda),
        &beta,  (double*)dC, int(lddc) );
}
#endif

#ifdef COMPLEX

extern "C" void
icla_dsyr2k(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_const_ptr dB, icla_int_t lddb,
    double beta,
    iclaDouble_ptr       dC, icla_int_t lddc,
    icla_queue_t queue )
{
    cublasDsyr2k(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        int(n), int(k),
        (double*)&alpha, (double*)dA, int(ldda),
                (double*)dB, int(lddb),
        &beta,  (double*)dC, int(lddc) );
}
#endif

extern "C" void
icla_dsymm(
    icla_side_t side, icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_const_ptr dB, icla_int_t lddb,
    double beta,
    iclaDouble_ptr       dC, icla_int_t lddc,
    icla_queue_t queue )
{
    cublasDsymm(
        queue->cublas_handle(),
        cublas_side_const( side ),
        cublas_uplo_const( uplo ),
        int(m), int(n),
        (double*)&alpha, (double*)dA, int(ldda),
                (double*)dB, int(lddb),
        (double*)&beta,  (double*)dC, int(lddc) );
}

extern "C" void
icla_dsyrk(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    double beta,
    iclaDouble_ptr       dC, icla_int_t lddc,
    icla_queue_t queue )
{
    cublasDsyrk(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        int(n), int(k),
        (double*)&alpha, (double*)dA, int(ldda),
        (double*)&beta,  (double*)dC, int(lddc) );
}

extern "C" void
icla_dsyr2k(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_const_ptr dB, icla_int_t lddb,
    double beta,
    iclaDouble_ptr       dC, icla_int_t lddc,
    icla_queue_t queue )
{
    cublasDsyr2k(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        int(n), int(k),
        (double*)&alpha, (double*)dA, int(ldda),
                (double*)dB, int(lddb),
        (double*)&beta,  (double*)dC, int(lddc) );
}

extern "C" void
icla_dtrmm(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_ptr       dB, icla_int_t lddb,
    icla_queue_t queue )
{
    #ifdef ICLA_HAVE_HIP

        cublasDtrmm(
		    queue->cublas_handle(),
		    cublas_side_const( side ),
                    cublas_uplo_const( uplo ),
                    cublas_trans_const( trans ),
                    cublas_diag_const( diag ),
		    int(m), int(n),
		    (double*)&alpha, (const double*)dA, int(ldda),
		    (double*)dB, int(lddb)
    #if (ROCM_VERSION >= 60000)
		    , (double*)dB, int(lddb)
    #endif
		    );
    #else
        cublasDtrmm(
                    queue->cublas_handle(),
                    cublas_side_const( side ),
                    cublas_uplo_const( uplo ),
                    cublas_trans_const( trans ),
                    cublas_diag_const( diag ),
                    int(m), int(n),
                    &alpha, dA, int(ldda),
                    dB, int(lddb),
                    dB, int(lddb) );

    #endif
}

extern "C" void
icla_dtrsm(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_ptr       dB, icla_int_t lddb,
    icla_queue_t queue )
{
    cublasDtrsm(
        queue->cublas_handle(),
        cublas_side_const( side ),
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        cublas_diag_const( diag ),
        int(m), int(n),
        (double*)&alpha, (double*)dA, int(ldda),
                (double*)dB, int(lddb) );
}

#undef REAL
