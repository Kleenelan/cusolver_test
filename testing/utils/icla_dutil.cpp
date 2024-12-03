
#include <algorithm>

#include "icla_v2.h"
#include "icla_lapack.h"
#include "../control/icla_threadsetting.h"

#include "testings.h"

#define REAL

#define A(i,j)  A[i + j*lda]

extern "C"
void icla_dmake_symmetric( icla_int_t N, double* A, icla_int_t lda )
{
    icla_int_t i, j;
    for( i=0; i < N; ++i ) {
        A(i,i) = ICLA_D_MAKE( ICLA_D_REAL( A(i,i) ), 0. );
        for( j=0; j < i; ++j ) {
            A(j,i) = ICLA_D_CONJ( A(i,j) );
        }
    }
}

extern "C"
void icla_dmake_hpd( icla_int_t N, double* A, icla_int_t lda )
{
    icla_int_t i, j;
    for( i=0; i < N; ++i ) {
        A(i,i) = ICLA_D_MAKE( ICLA_D_REAL( A(i,i) ) + N, 0. );
        for( j=0; j < i; ++j ) {
            A(j,i) = ICLA_D_CONJ( A(i,j) );
        }
    }
}

#ifdef COMPLEX

extern "C"
void icla_dmake_symmetric( icla_int_t N, double* A, icla_int_t lda )
{
    icla_int_t i, j;
    for( i=0; i < N; ++i ) {
        for( j=0; j < i; ++j ) {
            A(j,i) =  A(i,j);
        }
    }
}

extern "C"
void icla_dmake_spd( icla_int_t N, double* A, icla_int_t lda )
{
    icla_int_t i, j;
    for( i=0; i < N; ++i ) {
        A(i,i) = ICLA_D_MAKE( ICLA_D_REAL( A(i,i) ) + N, ICLA_D_IMAG( A(i,i) ) );
        for( j=0; j < i; ++j ) {
            A(j,i) = A(i,j);
        }
    }
}
#endif

extern "C"
double safe_lapackf77_dlansy(
    const char *norm, const char *uplo,
    const icla_int_t *n,
    const double *A, const icla_int_t *lda,
    double *work )
{
    #ifdef ICLA_WITH_MKL

    icla_int_t la_threads = icla_get_lapack_numthreads();
    icla_set_lapack_numthreads( 1 );
    #endif

    double result = lapackf77_dlansy( norm, uplo, n, A, lda, work );

    #ifdef ICLA_WITH_MKL

    icla_set_lapack_numthreads( la_threads );
    #endif

    return result;
}
