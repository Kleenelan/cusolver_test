
#ifndef TESTING_ICLA_D_H
#define TESTING_ICLA_D_H

#ifdef __cplusplus
extern "C" {
#endif

#define REAL

void icla_dmake_symmetric( icla_int_t N, double* A, icla_int_t lda );
void icla_dmake_symmetric( icla_int_t N, double* A, icla_int_t lda );

void icla_dmake_spd( icla_int_t N, double* A, icla_int_t lda );
void icla_dmake_hpd( icla_int_t N, double* A, icla_int_t lda );

double safe_lapackf77_dlansy(
    const char *norm, const char *uplo,
    const icla_int_t *n,
    const double *A, const icla_int_t *lda,
    double *work );

#ifdef COMPLEX
static inline double icla_dlapy2( double x )
{
    double xr = ICLA_D_REAL( x );
    double xi = ICLA_D_IMAG( x );
    return lapackf77_dlapy2( &xr, &xi );
}
#endif

void check_dgesvd(
    icla_int_t check,
    icla_vec_t jobu,
    icla_vec_t jobvt,
    icla_int_t m, icla_int_t n,
    double *A,  icla_int_t lda,
    double *S,
    double *U,  icla_int_t ldu,
    double *VT, icla_int_t ldv,
    double result[4] );

void check_dgeev(
    icla_vec_t jobvl,
    icla_vec_t jobvr,
    icla_int_t n,
    double *A,  icla_int_t lda,
    #ifdef COMPLEX
    double *w,
    #else
    double *wr, double *wi,
    #endif
    double *VL, icla_int_t ldvl,
    double *VR, icla_int_t ldvr,
    double *work, icla_int_t lwork,
    #ifdef COMPLEX
    double *rwork, icla_int_t lrwork,
    #endif
    double result[4] );

#undef REAL

#ifdef __cplusplus
}
#endif

#endif

