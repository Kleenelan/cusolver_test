/*
    -- ICLA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates
       @generated from testing/icla_zgesvd_check.cpp, normal z -> c, Fri Nov 29 12:16:17 2024
*/

#include "icla_v2.h"
#include "icla_lapack.h"
#include "testings.h"

#define COMPLEX

/**
    Check the results following the LAPACK's [zcds]drvbd routine.
    A is factored as A = U diag(S) VT and the following 4 tests computed:
    (1)    | A - U diag(S) VT | / ( |A| max(m,n) )
    (2)    | I - U^H U   | / ( m )
    (3)    | I - VT VT^H | / ( n )
    (4)    S contains min(m,n) nonnegative values in decreasing order.
           (Return 0 if true, 1 if false.)

    If check is false, skips (1) - (3), but always does (4).
    ********************************************************************/
extern "C"
void check_cgesvd(
    icla_int_t check,
    icla_vec_t jobu,
    icla_vec_t jobv,
    icla_int_t m, icla_int_t n,
    iclaFloatComplex *A,  icla_int_t lda,
    float *S,
    iclaFloatComplex *U,  icla_int_t ldu,
    iclaFloatComplex *VT, icla_int_t ldv,
    float result[4] )
{
    float unused[1];
    const icla_int_t izero = 0;
    float eps = lapackf77_slamch( "E" );

    if ( jobu == iclaNoVec ) {
        U = NULL;
    }
    if ( jobv == iclaNoVec ) {
        VT = NULL;
    }

    // -1 indicates check not done
    result[0] = -1;
    result[1] = -1;
    result[2] = -1;
    result[3] = -1;

    icla_int_t min_mn = min(m, n);
    icla_int_t n_u  = (jobu == iclaAllVec ? m : min_mn);
    icla_int_t m_vt = (jobv == iclaAllVec ? n : min_mn);

    assert( lda >= m );
    assert( ldu >= m );
    assert( ldv >= m_vt );

    if ( check ) {
        // cbdt01 needs m+n
        // cunt01 prefers n*(n+1) to check U; m*(m+1) to check V
        icla_int_t lwork_err = m+n;
        if ( U != NULL ) {
            lwork_err = max( lwork_err, n_u*(n_u+1) );
        }
        if ( VT != NULL ) {
            lwork_err = max( lwork_err, m_vt*(m_vt+1) );
        }
        iclaFloatComplex *work_err;
        TESTING_CHECK( icla_cmalloc_cpu( &work_err, lwork_err ));

        // cbdt01 and cunt01 need max(m,n), depending
        float *rwork_err;
        TESTING_CHECK( icla_smalloc_cpu( &rwork_err, max(m,n) ));

        if ( U != NULL && VT != NULL ) {
            // since KD=0 (3rd arg), E is not referenced so pass unused (9th arg)
            lapackf77_cbdt01( &m, &n, &izero, A, &lda,
                              U, &ldu, S, unused, VT, &ldv,
                              work_err,
                              #ifdef COMPLEX
                              rwork_err,
                              #endif
                              &result[0] );
        }
        if ( U != NULL ) {
            lapackf77_cunt01( "Columns", &m,  &n_u, U,  &ldu, work_err, &lwork_err,
                              #ifdef COMPLEX
                              rwork_err,
                              #endif
                              &result[1] );
        }
        if ( VT != NULL ) {
            lapackf77_cunt01( "Rows",    &m_vt, &n, VT, &ldv, work_err, &lwork_err,
                              #ifdef COMPLEX
                              rwork_err,
                              #endif
                              &result[2] );
        }

        result[0] *= eps;
        result[1] *= eps;
        result[2] *= eps;

        icla_free_cpu( work_err );
        icla_free_cpu( rwork_err );
    }

    // check S is sorted
    result[3] = 0.;
    for (int j=0; j < min_mn-1; j++) {
        if ( S[j] < S[j+1] )
            result[3] = 1.;
        if ( S[j] < 0. )
            result[3] = 1.;
    }
    if (min_mn > 1 && S[min_mn-1] < 0.) {
        result[3] = 1.;
    }
}
