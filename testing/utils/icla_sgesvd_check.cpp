
#include "icla_v2.h"
#include "icla_lapack.h"
#include "testings.h"

#define REAL

extern "C"
void check_sgesvd(
    icla_int_t check,
    icla_vec_t jobu,
    icla_vec_t jobv,
    icla_int_t m, icla_int_t n,
    float *A,  icla_int_t lda,
    float *S,
    float *U,  icla_int_t ldu,
    float *VT, icla_int_t ldv,
    float result[4] )
{
    float unused[1];
    const icla_int_t izero = 0;
    float eps = lapackf77_slamch( "E" );

    if ( jobu == IclaNoVec ) {
        U = NULL;
    }
    if ( jobv == IclaNoVec ) {
        VT = NULL;
    }

    result[0] = -1;
    result[1] = -1;
    result[2] = -1;
    result[3] = -1;

    icla_int_t min_mn = min(m, n);
    icla_int_t n_u  = (jobu == IclaAllVec ? m : min_mn);
    icla_int_t m_vt = (jobv == IclaAllVec ? n : min_mn);

    assert( lda >= m );
    assert( ldu >= m );
    assert( ldv >= m_vt );

    if ( check ) {

        icla_int_t lwork_err = m+n;
        if ( U != NULL ) {
            lwork_err = max( lwork_err, n_u*(n_u+1) );
        }
        if ( VT != NULL ) {
            lwork_err = max( lwork_err, m_vt*(m_vt+1) );
        }
        float *work_err;
        TESTING_CHECK( icla_smalloc_cpu( &work_err, lwork_err ));

        float *rwork_err;
        TESTING_CHECK( icla_smalloc_cpu( &rwork_err, max(m,n) ));

        if ( U != NULL && VT != NULL ) {

            lapackf77_sbdt01( &m, &n, &izero, A, &lda,
                              U, &ldu, S, unused, VT, &ldv,
                              work_err,
                              #ifdef COMPLEX
                              rwork_err,
                              #endif
                              &result[0] );
        }
        if ( U != NULL ) {
            lapackf77_sort01( "Columns", &m,  &n_u, U,  &ldu, work_err, &lwork_err,
                              #ifdef COMPLEX
                              rwork_err,
                              #endif
                              &result[1] );
        }
        if ( VT != NULL ) {
            lapackf77_sort01( "Rows",    &m_vt, &n, VT, &ldv, work_err, &lwork_err,
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
