
#include <limits>

#include "icla_internal.h"

#define REAL

const double ICLA_D_NAN
    = ICLA_D_MAKE( std::numeric_limits<double>::quiet_NaN(),
                    std::numeric_limits<double>::quiet_NaN() );

const double ICLA_D_INF
    = ICLA_D_MAKE( std::numeric_limits<double>::infinity(),
                    std::numeric_limits<double>::infinity() );

int icla_d_isnan( double x )
{
#ifdef COMPLEX
    return isnan( real( x )) ||
           isnan( imag( x ));
#else
    return isnan( x );
#endif
}

int icla_d_isinf( double x )
{
#ifdef COMPLEX
    return isinf( real( x )) ||
           isinf( imag( x ));
#else
    return isinf( x );
#endif
}

int icla_d_isnan_inf( double x )
{
#ifdef COMPLEX
    return isnan( real( x )) ||
           isnan( imag( x )) ||
           isinf( real( x )) ||
           isinf( imag( x ));
#else
    return isnan( x ) || isinf( x );
#endif
}

extern "C"
icla_int_t icla_dnan_inf(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    const double *A, icla_int_t lda,
    icla_int_t *cnt_nan,
    icla_int_t *cnt_inf )
{
    #define A(i_, j_) (A + (i_) + (j_)*lda)

    icla_int_t info = 0;
    if (uplo != iclaLower && uplo != iclaUpper && uplo != iclaFull)
        info = -1;
    else if (m < 0)
        info = -2;
    else if (n < 0)
        info = -3;
    else if (lda < m)
        info = -5;

    if (info != 0) {
        icla_xerbla( __func__, -(info) );
        return info;
    }

    int c_nan = 0;
    int c_inf = 0;

    if (uplo == iclaLower) {
        for (int j = 0; j < n; ++j) {
            for (int i = j; i < m; ++i) {

                if      (icla_d_isnan( *A(i,j) )) { c_nan++; }
                else if (icla_d_isinf( *A(i,j) )) { c_inf++; }
            }
        }
    }
    else if (uplo == iclaUpper) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m && i <= j; ++i) {

                if      (icla_d_isnan( *A(i,j) )) { c_nan++; }
                else if (icla_d_isinf( *A(i,j) )) { c_inf++; }
            }
        }
    }
    else if (uplo == iclaFull) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                if      (icla_d_isnan( *A(i,j) )) { c_nan++; }
                else if (icla_d_isinf( *A(i,j) )) { c_inf++; }
            }
        }
    }

    if (cnt_nan != NULL) { *cnt_nan = c_nan; }
    if (cnt_inf != NULL) { *cnt_inf = c_inf; }

    return (c_nan + c_inf);
}

extern "C"
icla_int_t icla_dnan_inf_gpu(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    icla_int_t *cnt_nan,
    icla_int_t *cnt_inf,
    icla_queue_t queue )
{
    icla_int_t info = 0;
    if (uplo != iclaLower && uplo != iclaUpper && uplo != iclaFull)
        info = -1;
    else if (m < 0)
        info = -2;
    else if (n < 0)
        info = -3;
    else if (ldda < m)
        info = -5;

    if (info != 0) {
        icla_xerbla( __func__, -(info) );
        return info;
    }

    icla_int_t lda = m;
    double* A;
    icla_dmalloc_cpu( &A, lda*n );

    icla_dgetmatrix( m, n, dA, ldda, A, lda, queue );

    icla_int_t cnt = icla_dnan_inf( uplo, m, n, A, lda, cnt_nan, cnt_inf );

    icla_free_cpu( A );
    return cnt;
}
