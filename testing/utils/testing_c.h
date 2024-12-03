/*
    -- ICLA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from testing/testing_z.h, normal z -> c, Fri Nov 29 12:16:14 2024
       @author Mark Gates

       Utilities for testing.
*/
#ifndef TESTING_ICLA_C_H
#define TESTING_ICLA_C_H

#ifdef __cplusplus
extern "C" {
#endif

#define COMPLEX

void icla_cmake_symmetric( icla_int_t N, iclaFloatComplex* A, icla_int_t lda );
void icla_cmake_hermitian( icla_int_t N, iclaFloatComplex* A, icla_int_t lda );

void icla_cmake_spd( icla_int_t N, iclaFloatComplex* A, icla_int_t lda );
void icla_cmake_hpd( icla_int_t N, iclaFloatComplex* A, icla_int_t lda );

// work around MKL bug in multi-threaded lanhe/lansy
float safe_lapackf77_clanhe(
    const char *norm, const char *uplo,
    const icla_int_t *n,
    const iclaFloatComplex *A, const icla_int_t *lda,
    float *work );

#ifdef COMPLEX
static inline float icla_clapy2( iclaFloatComplex x )
{
    float xr = ICLA_C_REAL( x );
    float xi = ICLA_C_IMAG( x );
    return lapackf77_slapy2( &xr, &xi );
}
#endif

void check_cgesvd(
    icla_int_t check,
    icla_vec_t jobu,
    icla_vec_t jobvt,
    icla_int_t m, icla_int_t n,
    iclaFloatComplex *A,  icla_int_t lda,
    float *S,
    iclaFloatComplex *U,  icla_int_t ldu,
    iclaFloatComplex *VT, icla_int_t ldv,
    float result[4] );

void check_cgeev(
    icla_vec_t jobvl,
    icla_vec_t jobvr,
    icla_int_t n,
    iclaFloatComplex *A,  icla_int_t lda,
    #ifdef COMPLEX
    iclaFloatComplex *w,
    #else
    float *wr, float *wi,
    #endif
    iclaFloatComplex *VL, icla_int_t ldvl,
    iclaFloatComplex *VR, icla_int_t ldvr,
    iclaFloatComplex *work, icla_int_t lwork,
    #ifdef COMPLEX
    float *rwork, icla_int_t lrwork,
    #endif
    float result[4] );

//void icla_cgenerate_matrix(
//    icla_int_t matrix,
//    icla_int_t m, icla_int_t n,
//    icla_int_t iseed[4],
//    float* sigma,
//    iclaFloatComplex* A, icla_int_t lda );

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
//    float* sigma_ptr,
//    iclaFloatComplex* A_ptr, icla_int_t lda );

#endif        //  #ifndef TESTING_ICLA_C_H
