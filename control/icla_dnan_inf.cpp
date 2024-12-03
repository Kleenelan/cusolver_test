/*
    -- ICLA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates
       @generated from control/icla_znan_inf.cpp, normal z -> d, Fri Nov 29 12:16:14 2024

*/
#include <limits>

#include "icla_internal.h"

#define REAL


const double ICLA_D_NAN
    = ICLA_D_MAKE( std::numeric_limits<double>::quiet_NaN(),
                    std::numeric_limits<double>::quiet_NaN() );

const double ICLA_D_INF
    = ICLA_D_MAKE( std::numeric_limits<double>::infinity(),
                    std::numeric_limits<double>::infinity() );


/***************************************************************************//**
    @param[in] x    Scalar to test.
    @return true if either real(x) or imag(x) is NAN.
    @ingroup icla_nan_inf
*******************************************************************************/
int icla_d_isnan( double x )
{
#ifdef COMPLEX
    return isnan( real( x )) ||
           isnan( imag( x ));
#else
    return isnan( x );
#endif
}


/***************************************************************************//**
    @param[in] x    Scalar to test.
    @return true if either real(x) or imag(x) is INF.
    @ingroup icla_nan_inf
*******************************************************************************/
int icla_d_isinf( double x )
{
#ifdef COMPLEX
    return isinf( real( x )) ||
           isinf( imag( x ));
#else
    return isinf( x );
#endif
}


/***************************************************************************//**
    @param[in] x    Scalar to test.
    @return true if either real(x) or imag(x) is NAN or INF.
    @ingroup icla_nan_inf
*******************************************************************************/
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


/***************************************************************************//**
    Purpose
    -------
    icla_dnan_inf checks a matrix that is located on the CPU host
    for NAN (not-a-number) and INF (infinity) values.

    NAN is created by 0/0 and similar.
    INF is created by x/0 and similar, where x != 0.

    Arguments
    ---------
    @param[in]
    uplo    icla_uplo_t
            Specifies what part of the matrix A to check.
      -     = iclaUpper:  Upper triangular part of A
      -     = iclaLower:  Lower triangular part of A
      -     = iclaFull:   All of A

    @param[in]
    m       INTEGER
            The number of rows of the matrix A. m >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A. n >= 0.

    @param[in]
    A       DOUBLE PRECISION array, dimension (lda,n), on the CPU host.
            The m-by-n matrix to be printed.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A. lda >= m.

    @param[out]
    cnt_nan INTEGER*
            If non-NULL, on exit contains the number of NAN values in A.

    @param[out]
    cnt_inf INTEGER*
            If non-NULL, on exit contains the number of INF values in A.

    @return
      -     >= 0:  Returns number of NAN + number of INF values.
      -     <  0:  If it returns -i, the i-th argument had an illegal value,
                   or another error occured, such as memory allocation failed.

    @ingroup icla_nan_inf
*******************************************************************************/
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
            for (int i = j; i < m; ++i) {  // i >= j
                if      (icla_d_isnan( *A(i,j) )) { c_nan++; }
                else if (icla_d_isinf( *A(i,j) )) { c_inf++; }
            }
        }
    }
    else if (uplo == iclaUpper) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m && i <= j; ++i) {  // i <= j
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


/***************************************************************************//**
    Purpose
    -------
    icla_dnan_inf checks a matrix that is located on the CPU host
    for NAN (not-a-number) and INF (infinity) values.

    NAN is created by 0/0 and similar.
    INF is created by x/0 and similar, where x != 0.

    Arguments
    ---------
    @param[in]
    uplo    icla_uplo_t
            Specifies what part of the matrix A to check.
      -     = iclaUpper:  Upper triangular part of A
      -     = iclaLower:  Lower triangular part of A
      -     = iclaFull:   All of A

    @param[in]
    m       INTEGER
            The number of rows of the matrix A. m >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A. n >= 0.

    @param[in]
    dA      DOUBLE PRECISION array, dimension (ldda,n), on the GPU device.
            The m-by-n matrix to be printed.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array A. ldda >= m.

    @param[out]
    cnt_nan INTEGER*
            If non-NULL, on exit contains the number of NAN values in A.

    @param[out]
    cnt_inf INTEGER*
            If non-NULL, on exit contains the number of INF values in A.

    @param[in]
    queue   icla_queue_t
            Queue to execute in.

    @return
      -     >= 0:  Returns number of NAN + number of INF values.
      -     <  0:  If it returns -i, the i-th argument had an illegal value,
                   or another error occured, such as memory allocation failed.

    @ingroup icla_nan_inf
*******************************************************************************/
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
