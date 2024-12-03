
#include "icla_internal.h"
#include "error.h"

#define REAL

#define PRECISION_s

extern "C" icla_int_t
icla_isamax(
    icla_int_t n,
    iclaFloat_const_ptr dx, icla_int_t incx,
    icla_queue_t queue )
{
    int result;

    cublasIsamax( queue->cublas_handle(), int(n), (float*)dx, int(incx), &result );
    return result;
}

extern "C" icla_int_t
icla_isamin(
    icla_int_t n,
    iclaFloat_const_ptr dx, icla_int_t incx,
    icla_queue_t queue )
{
    int result;

    cublasIsamin( queue->cublas_handle(), int(n), (float*)dx, int(incx), &result );
    return result;
}

extern "C" float
icla_sasum(
    icla_int_t n,
    iclaFloat_const_ptr dx, icla_int_t incx,
    icla_queue_t queue )
{
    float result;
    cublasSasum( queue->cublas_handle(), int(n), (float*)dx, int(incx), &result );
    return result;
}

extern "C" void
icla_saxpy(
    icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dx, icla_int_t incx,
    iclaFloat_ptr       dy, icla_int_t incy,
    icla_queue_t queue )
{
    cublasSaxpy( queue->cublas_handle(), int(n), (float*)&alpha, (float*)dx, int(incx), (float*)dy, int(incy) );
}

extern "C" void
icla_scopy(
    icla_int_t n,
    iclaFloat_const_ptr dx, icla_int_t incx,
    iclaFloat_ptr       dy, icla_int_t incy,
    icla_queue_t queue )
{
    cublasScopy( queue->cublas_handle(), int(n), (float*)dx, int(incx), (float*)dy, int(incy) );
}

#ifdef COMPLEX

extern "C"
float icla_sdot(
    icla_int_t n,
    iclaFloat_const_ptr dx, icla_int_t incx,
    iclaFloat_const_ptr dy, icla_int_t incy,
    icla_queue_t queue )
{
    float result;
    cublasSdot( queue->cublas_handle(), int(n), (float*)dx, int(incx), (float*)dy, int(incy), (float*)&result );
    return result;
}
#endif

extern "C"
float icla_sdot(
    icla_int_t n,
    iclaFloat_const_ptr dx, icla_int_t incx,
    iclaFloat_const_ptr dy, icla_int_t incy,
    icla_queue_t queue )
{
    float result;
    cublasSdot( queue->cublas_handle(), int(n), (float*)dx, int(incx), (float*)dy, int(incy), (float*)&result );
    return result;
}

extern "C" float
icla_snrm2(
    icla_int_t n,
    iclaFloat_const_ptr dx, icla_int_t incx,
    icla_queue_t queue )
{
    float result;
    cublasSnrm2( queue->cublas_handle(), int(n), (float*)dx, int(incx), &result );
    return result;
}

extern "C" void
icla_srot(
    icla_int_t n,
    iclaFloat_ptr dx, icla_int_t incx,
    iclaFloat_ptr dy, icla_int_t incy,
    float c, float s,
    icla_queue_t queue )
{
    cublasSrot( queue->cublas_handle(), int(n), (float*)dx, int(incx), (float*)dy, int(incy), &c, (float*)&s );
}

#ifdef COMPLEX

extern "C" void
icla_srot(
    icla_int_t n,
    iclaFloat_ptr dx, icla_int_t incx,
    iclaFloat_ptr dy, icla_int_t incy,
    float c, float s,
    icla_queue_t queue )
{
    cublasSrot( queue->cublas_handle(), int(n), (float*)dx, int(incx), (float*)dy, int(incy), &c, &s );
}
#endif

extern "C" void
icla_srotg(
    float *a, float *b,
    float             *c, float *s,
    icla_queue_t queue )
{
    cublasSrotg( queue->cublas_handle(), (float*)a, (float*)b, c, (float*)s );
}

#ifdef REAL

extern "C" void
icla_srotm(
    icla_int_t n,
    float *dx, icla_int_t incx,
    float *dy, icla_int_t incy,
    const float *param,
    icla_queue_t queue )
{
    cublasSrotm( queue->cublas_handle(), int(n), dx, int(incx), dy, int(incy), param );
}
#endif

#ifdef REAL

extern "C" void
icla_srotmg(
    float *d1, float       *d2,
    float *x1, const float *y1,
    float *param,
    icla_queue_t queue )
{
    cublasSrotmg( queue->cublas_handle(), d1, d2, x1, y1, param );
}
#endif

extern "C" void
icla_sscal(
    icla_int_t n,
    float alpha,
    iclaFloat_ptr dx, icla_int_t incx,
    icla_queue_t queue )
{
    cublasSscal( queue->cublas_handle(), int(n), (float*)&alpha, (float*)dx, int(incx) );
}

#ifdef COMPLEX

extern "C" void
icla_sscal(
    icla_int_t n,
    float alpha,
    iclaFloat_ptr dx, icla_int_t incx,
    icla_queue_t queue )
{
    cublasSscal( queue->cublas_handle(), int(n), &alpha, (float*)dx, int(incx) );
}
#endif

extern "C" void
icla_sswap(
    icla_int_t n,
    iclaFloat_ptr dx, icla_int_t incx,
    iclaFloat_ptr dy, icla_int_t incy,
    icla_queue_t queue )
{
    cublasSswap( queue->cublas_handle(), int(n), (float*)dx, int(incx), (float*)dy, int(incy) );
}

extern "C" void
icla_sgemv(
    icla_trans_t transA,
    icla_int_t m, icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_const_ptr dx, icla_int_t incx,
    float beta,
    iclaFloat_ptr       dy, icla_int_t incy,
    icla_queue_t queue )
{
    cublasSgemv(
        queue->cublas_handle(),
        cublas_trans_const( transA ),
        int(m), int(n),
        (float*)&alpha, (float*)dA, int(ldda),
                (float*)dx, int(incx),
        (float*)&beta,  (float*)dy, int(incy) );
}

#ifdef COMPLEX

extern "C" void
icla_sger(
    icla_int_t m, icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dx, icla_int_t incx,
    iclaFloat_const_ptr dy, icla_int_t incy,
    iclaFloat_ptr       dA, icla_int_t ldda,
    icla_queue_t queue )
{
    cublasSger(
        queue->cublas_handle(),
        int(m), int(n),
        (float*)&alpha, (float*)dx, int(incx),
                (float*)dy, int(incy),
                (float*)dA, int(ldda) );
}
#endif

extern "C" void
icla_sger(
    icla_int_t m, icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dx, icla_int_t incx,
    iclaFloat_const_ptr dy, icla_int_t incy,
    iclaFloat_ptr       dA, icla_int_t ldda,
    icla_queue_t queue )
{
    cublasSger(
        queue->cublas_handle(),
        int(m), int(n),
        (float*)&alpha, (float*)dx, int(incx),
                (float*)dy, int(incy),
                (float*)dA, int(ldda) );
}

#ifdef COMPLEX

extern "C" void
icla_ssymv(
    icla_uplo_t uplo,
    icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_const_ptr dx, icla_int_t incx,
    float beta,
    iclaFloat_ptr       dy, icla_int_t incy,
    icla_queue_t queue )
{
    cublasSsymv(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        int(n),
        (float*)&alpha, (float*)dA, int(ldda),
                (float*)dx, int(incx),
        (float*)&beta,  (float*)dy, int(incy) );
}
#endif

#ifdef COMPLEX

extern "C" void
icla_ssyr(
    icla_uplo_t uplo,
    icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dx, icla_int_t incx,
    iclaFloat_ptr       dA, icla_int_t ldda,
    icla_queue_t queue )
{
    cublasSsyr(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        int(n),
        (const float*)&alpha, (float*)dx, int(incx),
                (float*)dA, int(ldda) );
}
#endif

#ifdef COMPLEX

extern "C" void
icla_ssyr2(
    icla_uplo_t uplo,
    icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dx, icla_int_t incx,
    iclaFloat_const_ptr dy, icla_int_t incy,
    iclaFloat_ptr       dA, icla_int_t ldda,
    icla_queue_t queue )
{
    cublasSsyr2(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        int(n),
        (float*)&alpha, (float*)dx, int(incx),
                (float*)dy, int(incy),
                (float*)dA, int(ldda) );
}
#endif

extern "C" void
icla_ssymv(
    icla_uplo_t uplo,
    icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_const_ptr dx, icla_int_t incx,
    float beta,
    iclaFloat_ptr       dy, icla_int_t incy,
    icla_queue_t queue )
{
    cublasSsymv(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        int(n),
        (float*)&alpha, (float*)dA, int(ldda),
                (float*)dx, int(incx),
        (float*)&beta,  (float*)dy, int(incy) );
}

extern "C" void
icla_ssyr(
    icla_uplo_t uplo,
    icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dx, icla_int_t incx,
    iclaFloat_ptr       dA, icla_int_t ldda,
    icla_queue_t queue )
{
    cublasSsyr(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        int(n),
        (float*)&alpha, (float*)dx, int(incx),
                (float*)dA, int(ldda) );
}

extern "C" void
icla_ssyr2(
    icla_uplo_t uplo,
    icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dx, icla_int_t incx,
    iclaFloat_const_ptr dy, icla_int_t incy,
    iclaFloat_ptr       dA, icla_int_t ldda,
    icla_queue_t queue )
{
    cublasSsyr2(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        int(n),
        (float*)&alpha, (float*)dx, int(incx),
                (float*)dy, int(incy),
                (float*)dA, int(ldda) );
}

extern "C" void
icla_strmv(
    icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t n,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_ptr       dx, icla_int_t incx,
    icla_queue_t queue )
{
    cublasStrmv(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        cublas_diag_const( diag ),
        int(n),
        (float*)dA, int(ldda),
        (float*)dx, int(incx) );
}

extern "C" void
icla_strsv(
    icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t n,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_ptr       dx, icla_int_t incx,
    icla_queue_t queue )
{
    cublasStrsv(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        cublas_diag_const( diag ),
        int(n),
        (float*)dA, int(ldda),
        (float*)dx, int(incx) );
}

extern "C" void
icla_sgemm(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_const_ptr dB, icla_int_t lddb,
    float beta,
    iclaFloat_ptr       dC, icla_int_t lddc,
    icla_queue_t queue )
{
    cublasSgemm(
        queue->cublas_handle(),
        cublas_trans_const( transA ),
        cublas_trans_const( transB ),
        int(m), int(n), int(k),
        (float*)&alpha, (float*)dA, int(ldda),
                (float*)dB, int(lddb),
        (float*)&beta,  (float*)dC, int(lddc) );
}

#ifdef COMPLEX

extern "C" void
icla_ssymm(
    icla_side_t side, icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_const_ptr dB, icla_int_t lddb,
    float beta,
    iclaFloat_ptr       dC, icla_int_t lddc,
    icla_queue_t queue )
{
    cublasSsymm(
        queue->cublas_handle(),
        cublas_side_const( side ),
        cublas_uplo_const( uplo ),
        int(m), int(n),
        (float*)&alpha, (float*)dA, int(ldda),
                (float*)dB, int(lddb),
        (float*)&beta,  (float*)dC, int(lddc) );
}
#endif

#ifdef COMPLEX

extern "C" void
icla_ssyrk(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    float beta,
    iclaFloat_ptr       dC, icla_int_t lddc,
    icla_queue_t queue )
{
    cublasSsyrk(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        int(n), int(k),
        &alpha, (float*)dA, int(ldda),
        &beta,  (float*)dC, int(lddc) );
}
#endif

#ifdef COMPLEX

extern "C" void
icla_ssyr2k(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_const_ptr dB, icla_int_t lddb,
    float beta,
    iclaFloat_ptr       dC, icla_int_t lddc,
    icla_queue_t queue )
{
    cublasSsyr2k(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        int(n), int(k),
        (float*)&alpha, (float*)dA, int(ldda),
                (float*)dB, int(lddb),
        &beta,  (float*)dC, int(lddc) );
}
#endif

extern "C" void
icla_ssymm(
    icla_side_t side, icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_const_ptr dB, icla_int_t lddb,
    float beta,
    iclaFloat_ptr       dC, icla_int_t lddc,
    icla_queue_t queue )
{
    cublasSsymm(
        queue->cublas_handle(),
        cublas_side_const( side ),
        cublas_uplo_const( uplo ),
        int(m), int(n),
        (float*)&alpha, (float*)dA, int(ldda),
                (float*)dB, int(lddb),
        (float*)&beta,  (float*)dC, int(lddc) );
}

extern "C" void
icla_ssyrk(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    float beta,
    iclaFloat_ptr       dC, icla_int_t lddc,
    icla_queue_t queue )
{
    cublasSsyrk(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        int(n), int(k),
        (float*)&alpha, (float*)dA, int(ldda),
        (float*)&beta,  (float*)dC, int(lddc) );
}

extern "C" void
icla_ssyr2k(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_const_ptr dB, icla_int_t lddb,
    float beta,
    iclaFloat_ptr       dC, icla_int_t lddc,
    icla_queue_t queue )
{
    cublasSsyr2k(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        int(n), int(k),
        (float*)&alpha, (float*)dA, int(ldda),
                (float*)dB, int(lddb),
        (float*)&beta,  (float*)dC, int(lddc) );
}

extern "C" void
icla_strmm(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_ptr       dB, icla_int_t lddb,
    icla_queue_t queue )
{
    #ifdef ICLA_HAVE_HIP

        cublasStrmm(
		    queue->cublas_handle(),
		    cublas_side_const( side ),
                    cublas_uplo_const( uplo ),
                    cublas_trans_const( trans ),
                    cublas_diag_const( diag ),
		    int(m), int(n),
		    (float*)&alpha, (const float*)dA, int(ldda),
		    (float*)dB, int(lddb)
    #if (ROCM_VERSION >= 60000)
		    , (float*)dB, int(lddb)
    #endif
		    );
    #else
        cublasStrmm(
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
icla_strsm(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_ptr       dB, icla_int_t lddb,
    icla_queue_t queue )
{
    cublasStrsm(
        queue->cublas_handle(),
        cublas_side_const( side ),
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        cublas_diag_const( diag ),
        int(m), int(n),
        (float*)&alpha, (float*)dA, int(ldda),
                (float*)dB, int(lddb) );
}

#undef REAL
