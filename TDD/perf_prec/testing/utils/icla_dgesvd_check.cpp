
#include "icla_v2.h"
#include "icla_lapack.h"
#include "testings.h"

#define REAL

extern "C"
void check_dgesvd(
    icla_int_t check,
    icla_vec_t jobu,
    icla_vec_t jobv,
    icla_int_t m, icla_int_t n,
    double *A,  icla_int_t lda,
    double *S,
    double *U,  icla_int_t ldu,
    double *VT, icla_int_t ldv,
    double result[4] )
{
    double unused[1];
    const icla_int_t izero = 0;
    double eps = lapackf77_dlamch( "E" );

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

    icla_int_t min_mn = min__(m, n);
    icla_int_t n_u  = (jobu == IclaAllVec ? m : min_mn);
    icla_int_t m_vt = (jobv == IclaAllVec ? n : min_mn);

    assert( lda >= m );
    assert( ldu >= m );
    assert( ldv >= m_vt );

    if ( check ) {

        icla_int_t lwork_err = m+n;
        if ( U != NULL ) {
            lwork_err = max__( lwork_err, n_u*(n_u+1) );
        }
        if ( VT != NULL ) {
            lwork_err = max__( lwork_err, m_vt*(m_vt+1) );
        }
        double *work_err;
        TESTING_CHECK( icla_dmalloc_cpu( &work_err, lwork_err ));

        double *rwork_err;
        TESTING_CHECK( icla_dmalloc_cpu( &rwork_err, max__(m,n) ));

        if ( U != NULL && VT != NULL ) {

            lapackf77_dbdt01( &m, &n, &izero, A, &lda,
                              U, &ldu, S, unused, VT, &ldv,
                              work_err,
                              #ifdef COMPLEX
                              rwork_err,
                              #endif
                              &result[0] );
        }
        if ( U != NULL ) {
            lapackf77_dort01( "Columns", &m,  &n_u, U,  &ldu, work_err, &lwork_err,
                              #ifdef COMPLEX
                              rwork_err,
                              #endif
                              &result[1] );
        }
        if ( VT != NULL ) {
            lapackf77_dort01( "Rows",    &m_vt, &n, VT, &ldv, work_err, &lwork_err,
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
