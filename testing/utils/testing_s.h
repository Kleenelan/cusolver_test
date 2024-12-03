/*
    -- ICLA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from testing/testing_z.h, normal z -> s, Fri Nov 29 12:16:14 2024
       @author Mark Gates

       Utilities for testing.
*/
#ifndef TESTING_ICLA_S_H
#define TESTING_ICLA_S_H

#ifdef __cplusplus
extern "C" {
#endif

#define REAL

void icla_smake_symmetric( icla_int_t N, float* A, icla_int_t lda );
void icla_smake_symmetric( icla_int_t N, float* A, icla_int_t lda );

void icla_smake_spd( icla_int_t N, float* A, icla_int_t lda );
void icla_smake_hpd( icla_int_t N, float* A, icla_int_t lda );

// work around MKL bug in multi-threaded lanhe/lansy
float safe_lapackf77_slansy(
    const char *norm, const char *uplo,
    const icla_int_t *n,
    const float *A, const icla_int_t *lda,
    float *work );

#ifdef COMPLEX
static inline float icla_slapy2( float x )
{
    float xr = ICLA_S_REAL( x );
    float xi = ICLA_S_IMAG( x );
    return lapackf77_slapy2( &xr, &xi );
}
#endif

void check_sgesvd(
    icla_int_t check,
    icla_vec_t jobu,
    icla_vec_t jobvt,
    icla_int_t m, icla_int_t n,
    float *A,  icla_int_t lda,
    float *S,
    float *U,  icla_int_t ldu,
    float *VT, icla_int_t ldv,
    float result[4] );

void check_sgeev(
    icla_vec_t jobvl,
    icla_vec_t jobvr,
    icla_int_t n,
    float *A,  icla_int_t lda,
    #ifdef COMPLEX
    float *w,
    #else
    float *wr, float *wi,
    #endif
    float *VL, icla_int_t ldvl,
    float *VR, icla_int_t ldvr,
    float *work, icla_int_t lwork,
    #ifdef COMPLEX
    float *rwork, icla_int_t lrwork,
    #endif
    float result[4] );

//void icla_sgenerate_matrix(
//    icla_int_t matrix,
//    icla_int_t m, icla_int_t n,
//    icla_int_t iseed[4],
//    float* sigma,
//    float* A, icla_int_t lda );

#undef REAL

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
//    float* A_ptr, icla_int_t lda );

#endif        //  #ifndef TESTING_ICLA_S_H
