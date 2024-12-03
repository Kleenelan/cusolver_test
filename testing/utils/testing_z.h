/*
    -- ICLA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Mark Gates

       Utilities for testing.
*/
#ifndef TESTING_ICLA_Z_H
#define TESTING_ICLA_Z_H

#ifdef __cplusplus
extern "C" {
#endif

#define COMPLEX

void icla_zmake_symmetric( icla_int_t N, iclaDoubleComplex* A, icla_int_t lda );
void icla_zmake_hermitian( icla_int_t N, iclaDoubleComplex* A, icla_int_t lda );

void icla_zmake_spd( icla_int_t N, iclaDoubleComplex* A, icla_int_t lda );
void icla_zmake_hpd( icla_int_t N, iclaDoubleComplex* A, icla_int_t lda );

// work around MKL bug in multi-threaded lanhe/lansy
double safe_lapackf77_zlanhe(
    const char *norm, const char *uplo,
    const icla_int_t *n,
    const iclaDoubleComplex *A, const icla_int_t *lda,
    double *work );

#ifdef COMPLEX
static inline double icla_zlapy2( iclaDoubleComplex x )
{
    double xr = ICLA_Z_REAL( x );
    double xi = ICLA_Z_IMAG( x );
    return lapackf77_dlapy2( &xr, &xi );
}
#endif

void check_zgesvd(
    icla_int_t check,
    icla_vec_t jobu,
    icla_vec_t jobvt,
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex *A,  icla_int_t lda,
    double *S,
    iclaDoubleComplex *U,  icla_int_t ldu,
    iclaDoubleComplex *VT, icla_int_t ldv,
    double result[4] );

void check_zgeev(
    icla_vec_t jobvl,
    icla_vec_t jobvr,
    icla_int_t n,
    iclaDoubleComplex *A,  icla_int_t lda,
    #ifdef COMPLEX
    iclaDoubleComplex *w,
    #else
    double *wr, double *wi,
    #endif
    iclaDoubleComplex *VL, icla_int_t ldvl,
    iclaDoubleComplex *VR, icla_int_t ldvr,
    iclaDoubleComplex *work, icla_int_t lwork,
    #ifdef COMPLEX
    double *rwork, icla_int_t lrwork,
    #endif
    double result[4] );

//void icla_zgenerate_matrix(
//    icla_int_t matrix,
//    icla_int_t m, icla_int_t n,
//    icla_int_t iseed[4],
//    double* sigma,
//    iclaDoubleComplex* A, icla_int_t lda );

#undef COMPLEX

#ifdef __cplusplus
}
#endif

/******************************************************************************/
// C++ utility functions

//class icla_opts;

//void icla_generate_matrix(
//    icla_opts& opts,
//    icla_int_t iseed[4],
//    icla_int_t m, icla_int_t n,
//    double* sigma_ptr,
//    iclaDoubleComplex* A_ptr, icla_int_t lda );

#endif        //  #ifndef TESTING_ICLA_Z_H
