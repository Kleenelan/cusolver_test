
#include <algorithm>

#include "icla_v2.h"
#include "icla_lapack.h"
#include "../control/icla_threadsetting.h"

#include "testings.h"

#define COMPLEX

#define A(i,j)  A[i + j*lda]

extern "C"
void icla_zmake_hermitian( icla_int_t N, iclaDoubleComplex* A, icla_int_t lda )
{
    icla_int_t i, j;
    for( i=0; i < N; ++i ) {
        A(i,i) = ICLA_Z_MAKE( ICLA_Z_REAL( A(i,i) ), 0. );
        for( j=0; j < i; ++j ) {
            A(j,i) = ICLA_Z_CONJ( A(i,j) );
        }
    }
}

extern "C"
void icla_zmake_hpd( icla_int_t N, iclaDoubleComplex* A, icla_int_t lda )
{
    icla_int_t i, j;
    for( i=0; i < N; ++i ) {
        A(i,i) = ICLA_Z_MAKE( ICLA_Z_REAL( A(i,i) ) + N, 0. );
        for( j=0; j < i; ++j ) {
            A(j,i) = ICLA_Z_CONJ( A(i,j) );
        }
    }
}

#ifdef COMPLEX

extern "C"
void icla_zmake_symmetric( icla_int_t N, iclaDoubleComplex* A, icla_int_t lda )
{
    icla_int_t i, j;
    for( i=0; i < N; ++i ) {
        for( j=0; j < i; ++j ) {
            A(j,i) =  A(i,j);
        }
    }
}

extern "C"
void icla_zmake_spd( icla_int_t N, iclaDoubleComplex* A, icla_int_t lda )
{
    icla_int_t i, j;
    for( i=0; i < N; ++i ) {
        A(i,i) = ICLA_Z_MAKE( ICLA_Z_REAL( A(i,i) ) + N, ICLA_Z_IMAG( A(i,i) ) );
        for( j=0; j < i; ++j ) {
            A(j,i) = A(i,j);
        }
    }
}
#endif

extern "C"
double safe_lapackf77_zlanhe(
    const char *norm, const char *uplo,
    const icla_int_t *n,
    const iclaDoubleComplex *A, const icla_int_t *lda,
    double *work )
{
    #ifdef ICLA_WITH_MKL

    icla_int_t la_threads = icla_get_lapack_numthreads();
    icla_set_lapack_numthreads( 1 );
    #endif

    double result = lapackf77_zlanhe( norm, uplo, n, A, lda, work );

    #ifdef ICLA_WITH_MKL

    icla_set_lapack_numthreads( la_threads );
    #endif

    return result;
}
