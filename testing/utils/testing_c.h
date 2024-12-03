
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

#undef COMPLEX

#ifdef __cplusplus
}
#endif

#endif

