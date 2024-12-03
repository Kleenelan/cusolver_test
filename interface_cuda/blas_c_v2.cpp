
#include "icla_internal.h"
#include "error.h"

#define COMPLEX

#define PRECISION_c

extern "C" icla_int_t
icla_icamax(
    icla_int_t n,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    icla_queue_t queue )
{
    int result;

    cublasIcamax( queue->cublas_handle(), int(n), (cuFloatComplex*)dx, int(incx), &result );
    return result;
}

extern "C" icla_int_t
icla_icamin(
    icla_int_t n,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    icla_queue_t queue )
{
    int result;

    cublasIcamin( queue->cublas_handle(), int(n), (cuFloatComplex*)dx, int(incx), &result );
    return result;
}

extern "C" float
icla_scasum(
    icla_int_t n,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    icla_queue_t queue )
{
    float result;
    cublasScasum( queue->cublas_handle(), int(n), (cuFloatComplex*)dx, int(incx), &result );
    return result;
}

extern "C" void
icla_caxpy(
    icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex_ptr       dy, icla_int_t incy,
    icla_queue_t queue )
{
    cublasCaxpy( queue->cublas_handle(), int(n), (cuFloatComplex*)&alpha, (cuFloatComplex*)dx, int(incx), (cuFloatComplex*)dy, int(incy) );
}

extern "C" void
icla_ccopy(
    icla_int_t n,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex_ptr       dy, icla_int_t incy,
    icla_queue_t queue )
{
    cublasCcopy( queue->cublas_handle(), int(n), (cuFloatComplex*)dx, int(incx), (cuFloatComplex*)dy, int(incy) );
}

#ifdef COMPLEX

extern "C"
iclaFloatComplex icla_cdotc(
    icla_int_t n,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex_const_ptr dy, icla_int_t incy,
    icla_queue_t queue )
{
    iclaFloatComplex result;
    cublasCdotc( queue->cublas_handle(), int(n), (cuFloatComplex*)dx, int(incx), (cuFloatComplex*)dy, int(incy), (cuFloatComplex*)&result );
    return result;
}
#endif

extern "C"
iclaFloatComplex icla_cdotu(
    icla_int_t n,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex_const_ptr dy, icla_int_t incy,
    icla_queue_t queue )
{
    iclaFloatComplex result;
    cublasCdotu( queue->cublas_handle(), int(n), (cuFloatComplex*)dx, int(incx), (cuFloatComplex*)dy, int(incy), (cuFloatComplex*)&result );
    return result;
}

extern "C" float
icla_scnrm2(
    icla_int_t n,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    icla_queue_t queue )
{
    float result;
    cublasScnrm2( queue->cublas_handle(), int(n), (cuFloatComplex*)dx, int(incx), &result );
    return result;
}

extern "C" void
icla_crot(
    icla_int_t n,
    iclaFloatComplex_ptr dx, icla_int_t incx,
    iclaFloatComplex_ptr dy, icla_int_t incy,
    float c, iclaFloatComplex s,
    icla_queue_t queue )
{
    cublasCrot( queue->cublas_handle(), int(n), (cuFloatComplex*)dx, int(incx), (cuFloatComplex*)dy, int(incy), &c, (cuFloatComplex*)&s );
}

#ifdef COMPLEX

extern "C" void
icla_csrot(
    icla_int_t n,
    iclaFloatComplex_ptr dx, icla_int_t incx,
    iclaFloatComplex_ptr dy, icla_int_t incy,
    float c, float s,
    icla_queue_t queue )
{
    cublasCsrot( queue->cublas_handle(), int(n), (cuFloatComplex*)dx, int(incx), (cuFloatComplex*)dy, int(incy), &c, &s );
}
#endif

extern "C" void
icla_crotg(
    iclaFloatComplex *a, iclaFloatComplex *b,
    float             *c, iclaFloatComplex *s,
    icla_queue_t queue )
{
    cublasCrotg( queue->cublas_handle(), (cuFloatComplex*)a, (cuFloatComplex*)b, c, (cuFloatComplex*)s );
}

#ifdef REAL

extern "C" void
icla_crotm(
    icla_int_t n,
    float *dx, icla_int_t incx,
    float *dy, icla_int_t incy,
    const float *param,
    icla_queue_t queue )
{
    cublasCrotm( queue->cublas_handle(), int(n), dx, int(incx), dy, int(incy), param );
}
#endif

#ifdef REAL

extern "C" void
icla_crotmg(
    float *d1, float       *d2,
    float *x1, const float *y1,
    float *param,
    icla_queue_t queue )
{
    cublasCrotmg( queue->cublas_handle(), d1, d2, x1, y1, param );
}
#endif

extern "C" void
icla_cscal(
    icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_ptr dx, icla_int_t incx,
    icla_queue_t queue )
{
    cublasCscal( queue->cublas_handle(), int(n), (cuFloatComplex*)&alpha, (cuFloatComplex*)dx, int(incx) );
}

#ifdef COMPLEX

extern "C" void
icla_csscal(
    icla_int_t n,
    float alpha,
    iclaFloatComplex_ptr dx, icla_int_t incx,
    icla_queue_t queue )
{
    cublasCsscal( queue->cublas_handle(), int(n), &alpha, (cuFloatComplex*)dx, int(incx) );
}
#endif

extern "C" void
icla_cswap(
    icla_int_t n,
    iclaFloatComplex_ptr dx, icla_int_t incx,
    iclaFloatComplex_ptr dy, icla_int_t incy,
    icla_queue_t queue )
{
    cublasCswap( queue->cublas_handle(), int(n), (cuFloatComplex*)dx, int(incx), (cuFloatComplex*)dy, int(incy) );
}

extern "C" void
icla_cgemv(
    icla_trans_t transA,
    icla_int_t m, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr       dy, icla_int_t incy,
    icla_queue_t queue )
{
    cublasCgemv(
        queue->cublas_handle(),
        cublas_trans_const( transA ),
        int(m), int(n),
        (cuFloatComplex*)&alpha, (cuFloatComplex*)dA, int(ldda),
                (cuFloatComplex*)dx, int(incx),
        (cuFloatComplex*)&beta,  (cuFloatComplex*)dy, int(incy) );
}

#ifdef COMPLEX

extern "C" void
icla_cgerc(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex_const_ptr dy, icla_int_t incy,
    iclaFloatComplex_ptr       dA, icla_int_t ldda,
    icla_queue_t queue )
{
    cublasCgerc(
        queue->cublas_handle(),
        int(m), int(n),
        (cuFloatComplex*)&alpha, (cuFloatComplex*)dx, int(incx),
                (cuFloatComplex*)dy, int(incy),
                (cuFloatComplex*)dA, int(ldda) );
}
#endif

extern "C" void
icla_cgeru(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex_const_ptr dy, icla_int_t incy,
    iclaFloatComplex_ptr       dA, icla_int_t ldda,
    icla_queue_t queue )
{
    cublasCgeru(
        queue->cublas_handle(),
        int(m), int(n),
        (cuFloatComplex*)&alpha, (cuFloatComplex*)dx, int(incx),
                (cuFloatComplex*)dy, int(incy),
                (cuFloatComplex*)dA, int(ldda) );
}

#ifdef COMPLEX

extern "C" void
icla_chemv(
    icla_uplo_t uplo,
    icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr       dy, icla_int_t incy,
    icla_queue_t queue )
{
    cublasChemv(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        int(n),
        (cuFloatComplex*)&alpha, (cuFloatComplex*)dA, int(ldda),
                (cuFloatComplex*)dx, int(incx),
        (cuFloatComplex*)&beta,  (cuFloatComplex*)dy, int(incy) );
}
#endif

#ifdef COMPLEX

extern "C" void
icla_cher(
    icla_uplo_t uplo,
    icla_int_t n,
    float alpha,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex_ptr       dA, icla_int_t ldda,
    icla_queue_t queue )
{
    cublasCher(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        int(n),
        (const float*)&alpha, (cuFloatComplex*)dx, int(incx),
                (cuFloatComplex*)dA, int(ldda) );
}
#endif

#ifdef COMPLEX

extern "C" void
icla_cher2(
    icla_uplo_t uplo,
    icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex_const_ptr dy, icla_int_t incy,
    iclaFloatComplex_ptr       dA, icla_int_t ldda,
    icla_queue_t queue )
{
    cublasCher2(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        int(n),
        (cuFloatComplex*)&alpha, (cuFloatComplex*)dx, int(incx),
                (cuFloatComplex*)dy, int(incy),
                (cuFloatComplex*)dA, int(ldda) );
}
#endif

extern "C" void
icla_csymv(
    icla_uplo_t uplo,
    icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr       dy, icla_int_t incy,
    icla_queue_t queue )
{
    cublasCsymv(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        int(n),
        (cuFloatComplex*)&alpha, (cuFloatComplex*)dA, int(ldda),
                (cuFloatComplex*)dx, int(incx),
        (cuFloatComplex*)&beta,  (cuFloatComplex*)dy, int(incy) );
}

extern "C" void
icla_csyr(
    icla_uplo_t uplo,
    icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex_ptr       dA, icla_int_t ldda,
    icla_queue_t queue )
{
    cublasCsyr(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        int(n),
        (cuFloatComplex*)&alpha, (cuFloatComplex*)dx, int(incx),
                (cuFloatComplex*)dA, int(ldda) );
}

extern "C" void
icla_csyr2(
    icla_uplo_t uplo,
    icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex_const_ptr dy, icla_int_t incy,
    iclaFloatComplex_ptr       dA, icla_int_t ldda,
    icla_queue_t queue )
{
    cublasCsyr2(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        int(n),
        (cuFloatComplex*)&alpha, (cuFloatComplex*)dx, int(incx),
                (cuFloatComplex*)dy, int(incy),
                (cuFloatComplex*)dA, int(ldda) );
}

extern "C" void
icla_ctrmv(
    icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t n,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr       dx, icla_int_t incx,
    icla_queue_t queue )
{
    cublasCtrmv(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        cublas_diag_const( diag ),
        int(n),
        (cuFloatComplex*)dA, int(ldda),
        (cuFloatComplex*)dx, int(incx) );
}

extern "C" void
icla_ctrsv(
    icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t n,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr       dx, icla_int_t incx,
    icla_queue_t queue )
{
    cublasCtrsv(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        cublas_diag_const( diag ),
        int(n),
        (cuFloatComplex*)dA, int(ldda),
        (cuFloatComplex*)dx, int(incx) );
}

extern "C" void
icla_cgemm(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_const_ptr dB, icla_int_t lddb,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr       dC, icla_int_t lddc,
    icla_queue_t queue )
{
    cublasCgemm(
        queue->cublas_handle(),
        cublas_trans_const( transA ),
        cublas_trans_const( transB ),
        int(m), int(n), int(k),
        (cuFloatComplex*)&alpha, (cuFloatComplex*)dA, int(ldda),
                (cuFloatComplex*)dB, int(lddb),
        (cuFloatComplex*)&beta,  (cuFloatComplex*)dC, int(lddc) );
}

#ifdef COMPLEX

extern "C" void
icla_chemm(
    icla_side_t side, icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_const_ptr dB, icla_int_t lddb,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr       dC, icla_int_t lddc,
    icla_queue_t queue )
{
    cublasChemm(
        queue->cublas_handle(),
        cublas_side_const( side ),
        cublas_uplo_const( uplo ),
        int(m), int(n),
        (cuFloatComplex*)&alpha, (cuFloatComplex*)dA, int(ldda),
                (cuFloatComplex*)dB, int(lddb),
        (cuFloatComplex*)&beta,  (cuFloatComplex*)dC, int(lddc) );
}
#endif

#ifdef COMPLEX

extern "C" void
icla_cherk(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    float alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    float beta,
    iclaFloatComplex_ptr       dC, icla_int_t lddc,
    icla_queue_t queue )
{
    cublasCherk(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        int(n), int(k),
        &alpha, (cuFloatComplex*)dA, int(ldda),
        &beta,  (cuFloatComplex*)dC, int(lddc) );
}
#endif

#ifdef COMPLEX

extern "C" void
icla_cher2k(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_const_ptr dB, icla_int_t lddb,
    float beta,
    iclaFloatComplex_ptr       dC, icla_int_t lddc,
    icla_queue_t queue )
{
    cublasCher2k(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        int(n), int(k),
        (cuFloatComplex*)&alpha, (cuFloatComplex*)dA, int(ldda),
                (cuFloatComplex*)dB, int(lddb),
        &beta,  (cuFloatComplex*)dC, int(lddc) );
}
#endif

extern "C" void
icla_csymm(
    icla_side_t side, icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_const_ptr dB, icla_int_t lddb,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr       dC, icla_int_t lddc,
    icla_queue_t queue )
{
    cublasCsymm(
        queue->cublas_handle(),
        cublas_side_const( side ),
        cublas_uplo_const( uplo ),
        int(m), int(n),
        (cuFloatComplex*)&alpha, (cuFloatComplex*)dA, int(ldda),
                (cuFloatComplex*)dB, int(lddb),
        (cuFloatComplex*)&beta,  (cuFloatComplex*)dC, int(lddc) );
}

extern "C" void
icla_csyrk(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr       dC, icla_int_t lddc,
    icla_queue_t queue )
{
    cublasCsyrk(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        int(n), int(k),
        (cuFloatComplex*)&alpha, (cuFloatComplex*)dA, int(ldda),
        (cuFloatComplex*)&beta,  (cuFloatComplex*)dC, int(lddc) );
}

extern "C" void
icla_csyr2k(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_const_ptr dB, icla_int_t lddb,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr       dC, icla_int_t lddc,
    icla_queue_t queue )
{
    cublasCsyr2k(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        int(n), int(k),
        (cuFloatComplex*)&alpha, (cuFloatComplex*)dA, int(ldda),
                (cuFloatComplex*)dB, int(lddb),
        (cuFloatComplex*)&beta,  (cuFloatComplex*)dC, int(lddc) );
}

extern "C" void
icla_ctrmm(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr       dB, icla_int_t lddb,
    icla_queue_t queue )
{
    #ifdef ICLA_HAVE_HIP

        cublasCtrmm(
		    queue->cublas_handle(),
		    cublas_side_const( side ),
                    cublas_uplo_const( uplo ),
                    cublas_trans_const( trans ),
                    cublas_diag_const( diag ),
		    int(m), int(n),
		    (cuFloatComplex*)&alpha, (const cuFloatComplex*)dA, int(ldda),
		    (cuFloatComplex*)dB, int(lddb)
    #if (ROCM_VERSION >= 60000)
		    , (cuFloatComplex*)dB, int(lddb)
    #endif
		    );
    #else
        cublasCtrmm(
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
icla_ctrsm(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr       dB, icla_int_t lddb,
    icla_queue_t queue )
{
    cublasCtrsm(
        queue->cublas_handle(),
        cublas_side_const( side ),
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        cublas_diag_const( diag ),
        int(m), int(n),
        (cuFloatComplex*)&alpha, (cuFloatComplex*)dA, int(ldda),
                (cuFloatComplex*)dB, int(lddb) );
}

#undef COMPLEX
