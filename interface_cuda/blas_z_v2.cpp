
#include "icla_internal.h"
#include "error.h"

#define COMPLEX

#define PRECISION_z

extern "C" icla_int_t
icla_izamax(
    icla_int_t n,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    icla_queue_t queue )
{
    int result;

    cublasIzamax( queue->cublas_handle(), int(n), (cuDoubleComplex*)dx, int(incx), &result );
    return result;
}

extern "C" icla_int_t
icla_izamin(
    icla_int_t n,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    icla_queue_t queue )
{
    int result;

    cublasIzamin( queue->cublas_handle(), int(n), (cuDoubleComplex*)dx, int(incx), &result );
    return result;
}

extern "C" double
icla_dzasum(
    icla_int_t n,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    icla_queue_t queue )
{
    double result;
    cublasDzasum( queue->cublas_handle(), int(n), (cuDoubleComplex*)dx, int(incx), &result );
    return result;
}

extern "C" void
icla_zaxpy(
    icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex_ptr       dy, icla_int_t incy,
    icla_queue_t queue )
{
    cublasZaxpy( queue->cublas_handle(), int(n), (cuDoubleComplex*)&alpha, (cuDoubleComplex*)dx, int(incx), (cuDoubleComplex*)dy, int(incy) );
}

extern "C" void
icla_zcopy(
    icla_int_t n,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex_ptr       dy, icla_int_t incy,
    icla_queue_t queue )
{
    cublasZcopy( queue->cublas_handle(), int(n), (cuDoubleComplex*)dx, int(incx), (cuDoubleComplex*)dy, int(incy) );
}

#ifdef COMPLEX

extern "C"
iclaDoubleComplex icla_zdotc(
    icla_int_t n,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex_const_ptr dy, icla_int_t incy,
    icla_queue_t queue )
{
    iclaDoubleComplex result;
    cublasZdotc( queue->cublas_handle(), int(n), (cuDoubleComplex*)dx, int(incx), (cuDoubleComplex*)dy, int(incy), (cuDoubleComplex*)&result );
    return result;
}
#endif

extern "C"
iclaDoubleComplex icla_zdotu(
    icla_int_t n,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex_const_ptr dy, icla_int_t incy,
    icla_queue_t queue )
{
    iclaDoubleComplex result;
    cublasZdotu( queue->cublas_handle(), int(n), (cuDoubleComplex*)dx, int(incx), (cuDoubleComplex*)dy, int(incy), (cuDoubleComplex*)&result );
    return result;
}

extern "C" double
icla_dznrm2(
    icla_int_t n,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    icla_queue_t queue )
{
    double result;
    cublasDznrm2( queue->cublas_handle(), int(n), (cuDoubleComplex*)dx, int(incx), &result );
    return result;
}

extern "C" void
icla_zrot(
    icla_int_t n,
    iclaDoubleComplex_ptr dx, icla_int_t incx,
    iclaDoubleComplex_ptr dy, icla_int_t incy,
    double c, iclaDoubleComplex s,
    icla_queue_t queue )
{
    cublasZrot( queue->cublas_handle(), int(n), (cuDoubleComplex*)dx, int(incx), (cuDoubleComplex*)dy, int(incy), &c, (cuDoubleComplex*)&s );
}

#ifdef COMPLEX

extern "C" void
icla_zdrot(
    icla_int_t n,
    iclaDoubleComplex_ptr dx, icla_int_t incx,
    iclaDoubleComplex_ptr dy, icla_int_t incy,
    double c, double s,
    icla_queue_t queue )
{
    cublasZdrot( queue->cublas_handle(), int(n), (cuDoubleComplex*)dx, int(incx), (cuDoubleComplex*)dy, int(incy), &c, &s );
}
#endif

extern "C" void
icla_zrotg(
    iclaDoubleComplex *a, iclaDoubleComplex *b,
    double             *c, iclaDoubleComplex *s,
    icla_queue_t queue )
{
    cublasZrotg( queue->cublas_handle(), (cuDoubleComplex*)a, (cuDoubleComplex*)b, c, (cuDoubleComplex*)s );
}

#ifdef REAL

extern "C" void
icla_zrotm(
    icla_int_t n,
    double *dx, icla_int_t incx,
    double *dy, icla_int_t incy,
    const double *param,
    icla_queue_t queue )
{
    cublasZrotm( queue->cublas_handle(), int(n), dx, int(incx), dy, int(incy), param );
}
#endif

#ifdef REAL

extern "C" void
icla_zrotmg(
    double *d1, double       *d2,
    double *x1, const double *y1,
    double *param,
    icla_queue_t queue )
{
    cublasZrotmg( queue->cublas_handle(), d1, d2, x1, y1, param );
}
#endif

extern "C" void
icla_zscal(
    icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_ptr dx, icla_int_t incx,
    icla_queue_t queue )
{
    cublasZscal( queue->cublas_handle(), int(n), (cuDoubleComplex*)&alpha, (cuDoubleComplex*)dx, int(incx) );
}

#ifdef COMPLEX

extern "C" void
icla_zdscal(
    icla_int_t n,
    double alpha,
    iclaDoubleComplex_ptr dx, icla_int_t incx,
    icla_queue_t queue )
{
    cublasZdscal( queue->cublas_handle(), int(n), &alpha, (cuDoubleComplex*)dx, int(incx) );
}
#endif

extern "C" void
icla_zswap(
    icla_int_t n,
    iclaDoubleComplex_ptr dx, icla_int_t incx,
    iclaDoubleComplex_ptr dy, icla_int_t incy,
    icla_queue_t queue )
{
    cublasZswap( queue->cublas_handle(), int(n), (cuDoubleComplex*)dx, int(incx), (cuDoubleComplex*)dy, int(incy) );
}

extern "C" void
icla_zgemv(
    icla_trans_t transA,
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dy, icla_int_t incy,
    icla_queue_t queue )
{
    cublasZgemv(
        queue->cublas_handle(),
        cublas_trans_const( transA ),
        int(m), int(n),
        (cuDoubleComplex*)&alpha, (cuDoubleComplex*)dA, int(ldda),
                (cuDoubleComplex*)dx, int(incx),
        (cuDoubleComplex*)&beta,  (cuDoubleComplex*)dy, int(incy) );
}

#ifdef COMPLEX

extern "C" void
icla_zgerc(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex_const_ptr dy, icla_int_t incy,
    iclaDoubleComplex_ptr       dA, icla_int_t ldda,
    icla_queue_t queue )
{
    cublasZgerc(
        queue->cublas_handle(),
        int(m), int(n),
        (cuDoubleComplex*)&alpha, (cuDoubleComplex*)dx, int(incx),
                (cuDoubleComplex*)dy, int(incy),
                (cuDoubleComplex*)dA, int(ldda) );
}
#endif

extern "C" void
icla_zgeru(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex_const_ptr dy, icla_int_t incy,
    iclaDoubleComplex_ptr       dA, icla_int_t ldda,
    icla_queue_t queue )
{
    cublasZgeru(
        queue->cublas_handle(),
        int(m), int(n),
        (cuDoubleComplex*)&alpha, (cuDoubleComplex*)dx, int(incx),
                (cuDoubleComplex*)dy, int(incy),
                (cuDoubleComplex*)dA, int(ldda) );
}

#ifdef COMPLEX

extern "C" void
icla_zhemv(
    icla_uplo_t uplo,
    icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dy, icla_int_t incy,
    icla_queue_t queue )
{
    cublasZhemv(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        int(n),
        (cuDoubleComplex*)&alpha, (cuDoubleComplex*)dA, int(ldda),
                (cuDoubleComplex*)dx, int(incx),
        (cuDoubleComplex*)&beta,  (cuDoubleComplex*)dy, int(incy) );
}
#endif

#ifdef COMPLEX

extern "C" void
icla_zher(
    icla_uplo_t uplo,
    icla_int_t n,
    double alpha,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex_ptr       dA, icla_int_t ldda,
    icla_queue_t queue )
{
    cublasZher(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        int(n),
        (const double*)&alpha, (cuDoubleComplex*)dx, int(incx),
                (cuDoubleComplex*)dA, int(ldda) );
}
#endif

#ifdef COMPLEX

extern "C" void
icla_zher2(
    icla_uplo_t uplo,
    icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex_const_ptr dy, icla_int_t incy,
    iclaDoubleComplex_ptr       dA, icla_int_t ldda,
    icla_queue_t queue )
{
    cublasZher2(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        int(n),
        (cuDoubleComplex*)&alpha, (cuDoubleComplex*)dx, int(incx),
                (cuDoubleComplex*)dy, int(incy),
                (cuDoubleComplex*)dA, int(ldda) );
}
#endif

extern "C" void
icla_zsymv(
    icla_uplo_t uplo,
    icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dy, icla_int_t incy,
    icla_queue_t queue )
{
    cublasZsymv(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        int(n),
        (cuDoubleComplex*)&alpha, (cuDoubleComplex*)dA, int(ldda),
                (cuDoubleComplex*)dx, int(incx),
        (cuDoubleComplex*)&beta,  (cuDoubleComplex*)dy, int(incy) );
}

extern "C" void
icla_zsyr(
    icla_uplo_t uplo,
    icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex_ptr       dA, icla_int_t ldda,
    icla_queue_t queue )
{
    cublasZsyr(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        int(n),
        (cuDoubleComplex*)&alpha, (cuDoubleComplex*)dx, int(incx),
                (cuDoubleComplex*)dA, int(ldda) );
}

extern "C" void
icla_zsyr2(
    icla_uplo_t uplo,
    icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex_const_ptr dy, icla_int_t incy,
    iclaDoubleComplex_ptr       dA, icla_int_t ldda,
    icla_queue_t queue )
{
    cublasZsyr2(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        int(n),
        (cuDoubleComplex*)&alpha, (cuDoubleComplex*)dx, int(incx),
                (cuDoubleComplex*)dy, int(incy),
                (cuDoubleComplex*)dA, int(ldda) );
}

extern "C" void
icla_ztrmv(
    icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t n,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr       dx, icla_int_t incx,
    icla_queue_t queue )
{
    cublasZtrmv(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        cublas_diag_const( diag ),
        int(n),
        (cuDoubleComplex*)dA, int(ldda),
        (cuDoubleComplex*)dx, int(incx) );
}

extern "C" void
icla_ztrsv(
    icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t n,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr       dx, icla_int_t incx,
    icla_queue_t queue )
{
    cublasZtrsv(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        cublas_diag_const( diag ),
        int(n),
        (cuDoubleComplex*)dA, int(ldda),
        (cuDoubleComplex*)dx, int(incx) );
}

extern "C" void
icla_zgemm(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dB, icla_int_t lddb,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dC, icla_int_t lddc,
    icla_queue_t queue )
{
    cublasZgemm(
        queue->cublas_handle(),
        cublas_trans_const( transA ),
        cublas_trans_const( transB ),
        int(m), int(n), int(k),
        (cuDoubleComplex*)&alpha, (cuDoubleComplex*)dA, int(ldda),
                (cuDoubleComplex*)dB, int(lddb),
        (cuDoubleComplex*)&beta,  (cuDoubleComplex*)dC, int(lddc) );
}

#ifdef COMPLEX

extern "C" void
icla_zhemm(
    icla_side_t side, icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dB, icla_int_t lddb,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dC, icla_int_t lddc,
    icla_queue_t queue )
{
    cublasZhemm(
        queue->cublas_handle(),
        cublas_side_const( side ),
        cublas_uplo_const( uplo ),
        int(m), int(n),
        (cuDoubleComplex*)&alpha, (cuDoubleComplex*)dA, int(ldda),
                (cuDoubleComplex*)dB, int(lddb),
        (cuDoubleComplex*)&beta,  (cuDoubleComplex*)dC, int(lddc) );
}
#endif

#ifdef COMPLEX

extern "C" void
icla_zherk(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    double alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    double beta,
    iclaDoubleComplex_ptr       dC, icla_int_t lddc,
    icla_queue_t queue )
{
    cublasZherk(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        int(n), int(k),
        &alpha, (cuDoubleComplex*)dA, int(ldda),
        &beta,  (cuDoubleComplex*)dC, int(lddc) );
}
#endif

#ifdef COMPLEX

extern "C" void
icla_zher2k(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dB, icla_int_t lddb,
    double beta,
    iclaDoubleComplex_ptr       dC, icla_int_t lddc,
    icla_queue_t queue )
{
    cublasZher2k(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        int(n), int(k),
        (cuDoubleComplex*)&alpha, (cuDoubleComplex*)dA, int(ldda),
                (cuDoubleComplex*)dB, int(lddb),
        &beta,  (cuDoubleComplex*)dC, int(lddc) );
}
#endif

extern "C" void
icla_zsymm(
    icla_side_t side, icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dB, icla_int_t lddb,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dC, icla_int_t lddc,
    icla_queue_t queue )
{
    cublasZsymm(
        queue->cublas_handle(),
        cublas_side_const( side ),
        cublas_uplo_const( uplo ),
        int(m), int(n),
        (cuDoubleComplex*)&alpha, (cuDoubleComplex*)dA, int(ldda),
                (cuDoubleComplex*)dB, int(lddb),
        (cuDoubleComplex*)&beta,  (cuDoubleComplex*)dC, int(lddc) );
}

extern "C" void
icla_zsyrk(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dC, icla_int_t lddc,
    icla_queue_t queue )
{
    cublasZsyrk(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        int(n), int(k),
        (cuDoubleComplex*)&alpha, (cuDoubleComplex*)dA, int(ldda),
        (cuDoubleComplex*)&beta,  (cuDoubleComplex*)dC, int(lddc) );
}

extern "C" void
icla_zsyr2k(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dB, icla_int_t lddb,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dC, icla_int_t lddc,
    icla_queue_t queue )
{
    cublasZsyr2k(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        int(n), int(k),
        (cuDoubleComplex*)&alpha, (cuDoubleComplex*)dA, int(ldda),
                (cuDoubleComplex*)dB, int(lddb),
        (cuDoubleComplex*)&beta,  (cuDoubleComplex*)dC, int(lddc) );
}

extern "C" void
icla_ztrmm(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr       dB, icla_int_t lddb,
    icla_queue_t queue )
{
    #ifdef ICLA_HAVE_HIP

        cublasZtrmm(
		    queue->cublas_handle(),
		    cublas_side_const( side ),
                    cublas_uplo_const( uplo ),
                    cublas_trans_const( trans ),
                    cublas_diag_const( diag ),
		    int(m), int(n),
		    (cuDoubleComplex*)&alpha, (const cuDoubleComplex*)dA, int(ldda),
		    (cuDoubleComplex*)dB, int(lddb)
    #if (ROCM_VERSION >= 60000)
		    , (cuDoubleComplex*)dB, int(lddb)
    #endif
		    );
    #else
        cublasZtrmm(
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
icla_ztrsm(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr       dB, icla_int_t lddb,
    icla_queue_t queue )
{
    cublasZtrsm(
        queue->cublas_handle(),
        cublas_side_const( side ),
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        cublas_diag_const( diag ),
        int(m), int(n),
        (cuDoubleComplex*)&alpha, (cuDoubleComplex*)dA, int(ldda),
                (cuDoubleComplex*)dB, int(lddb) );
}

#undef COMPLEX
