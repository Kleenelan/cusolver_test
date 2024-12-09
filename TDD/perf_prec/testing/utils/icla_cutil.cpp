
#include <algorithm>

#include "icla_v2.h"
#include "icla_lapack.h"
#include "../control/icla_threadsetting.h"

#include "testings.h"

#define COMPLEX

#define A(i,j)  A[i + j*lda]

extern "C"
void icla_cmake_hermitian( icla_int_t N, iclaFloatComplex* A, icla_int_t lda )
{
    icla_int_t i, j;
    for( i=0; i < N; ++i ) {
        A(i,i) = ICLA_C_MAKE( ICLA_C_REAL( A(i,i) ), 0. );
        for( j=0; j < i; ++j ) {
            A(j,i) = ICLA_C_CONJ( A(i,j) );
        }
    }
}

extern "C"
void icla_cmake_hpd( icla_int_t N, iclaFloatComplex* A, icla_int_t lda )
{
    icla_int_t i, j;
    for( i=0; i < N; ++i ) {
        A(i,i) = ICLA_C_MAKE( ICLA_C_REAL( A(i,i) ) + N, 0. );
        for( j=0; j < i; ++j ) {
            A(j,i) = ICLA_C_CONJ( A(i,j) );
        }
    }
}

#ifdef COMPLEX

extern "C"
void icla_cmake_symmetric( icla_int_t N, iclaFloatComplex* A, icla_int_t lda )
{
    icla_int_t i, j;
    for( i=0; i < N; ++i ) {
        for( j=0; j < i; ++j ) {
            A(j,i) =  A(i,j);
        }
    }
}

extern "C"
void icla_cmake_spd( icla_int_t N, iclaFloatComplex* A, icla_int_t lda )
{
    icla_int_t i, j;
    for( i=0; i < N; ++i ) {
        A(i,i) = ICLA_C_MAKE( ICLA_C_REAL( A(i,i) ) + N, ICLA_C_IMAG( A(i,i) ) );
        for( j=0; j < i; ++j ) {
            A(j,i) = A(i,j);
        }
    }
}
#endif

extern "C"
float safe_lapackf77_clanhe(
    const char *norm, const char *uplo,
    const icla_int_t *n,
    const iclaFloatComplex *A, const icla_int_t *lda,
    float *work )
{
    #ifdef ICLA_WITH_MKL

    icla_int_t la_threads = icla_get_lapack_numthreads();
    icla_set_lapack_numthreads( 1 );
    #endif

    float result = lapackf77_clanhe( norm, uplo, n, A, lda, work );

    #ifdef ICLA_WITH_MKL

    icla_set_lapack_numthreads( la_threads );
    #endif

    return result;
}
