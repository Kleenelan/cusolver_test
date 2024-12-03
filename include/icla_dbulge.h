
#ifndef ICLA_DBULGE_H
#define ICLA_DBULGE_H

#include "icla_types.h"
#define ICLA_REAL

#ifdef __cplusplus
extern "C" {
#endif

icla_int_t
icla_dbulge_applyQ_v2(
    icla_side_t side,
    icla_int_t NE, icla_int_t n,
    icla_int_t nb, icla_int_t Vblksiz,
    iclaDouble_ptr dE, icla_int_t ldde,
    double *V, icla_int_t ldv,
    double *T, icla_int_t ldt,
    icla_int_t *info);

icla_int_t
icla_dbulge_applyQ_v2_m(
    icla_int_t ngpu, icla_side_t side,
    icla_int_t NE, icla_int_t n,
    icla_int_t nb, icla_int_t Vblksiz,
    double *E, icla_int_t lde,
    double *V, icla_int_t ldv,
    double *T, icla_int_t ldt,
    icla_int_t *info);

icla_int_t
icla_dbulge_back(
    icla_uplo_t uplo,
    icla_int_t n, icla_int_t nb,
    icla_int_t ne, icla_int_t Vblksiz,
    double *Z, icla_int_t ldz,
    iclaDouble_ptr dZ, icla_int_t lddz,
    double *V, icla_int_t ldv,
    double *TAU,
    double *T, icla_int_t ldt,
    icla_int_t* info);

icla_int_t
icla_dbulge_back_m(
    icla_int_t ngpu, icla_uplo_t uplo,
    icla_int_t n, icla_int_t nb,
    icla_int_t ne, icla_int_t Vblksiz,
    double *Z, icla_int_t ldz,
    double *V, icla_int_t ldv,
    double *TAU,
    double *T, icla_int_t ldt,
    icla_int_t* info);

void
icla_dtrdtype1cbHLsym_withQ_v2(
    icla_int_t n, icla_int_t nb,
    double *A, icla_int_t lda,
    double *V, icla_int_t ldv,
    double *TAU,
    icla_int_t st, icla_int_t ed,
    icla_int_t sweep, icla_int_t Vblksiz,
    double *work);

void
icla_dtrdtype2cbHLsym_withQ_v2(
    icla_int_t n, icla_int_t nb,
    double *A, icla_int_t lda,
    double *V, icla_int_t ldv,
    double *TAU,
    icla_int_t st, icla_int_t ed,
    icla_int_t sweep, icla_int_t Vblksiz,
    double *work);

void
icla_dtrdtype3cbHLsym_withQ_v2(
    icla_int_t n, icla_int_t nb,
    double *A, icla_int_t lda,
    double *V, icla_int_t ldv,
    double *TAU,
    icla_int_t st, icla_int_t ed,
    icla_int_t sweep, icla_int_t Vblksiz,
    double *work);

void
icla_dlarfy(
    icla_int_t n,
    double *A, icla_int_t lda,
    const double *V, const double *TAU,
    double *work);

void
icla_dsbtype1cb(icla_int_t n, icla_int_t nb,
                double *A, icla_int_t lda,
                double *V, icla_int_t LDV,
                double *TAU,
                icla_int_t st, icla_int_t ed, icla_int_t sweep,
                icla_int_t Vblksiz, icla_int_t wantz,
                double *work);

void
icla_dsbtype2cb(icla_int_t n, icla_int_t nb,
                double *A, icla_int_t lda,
                double *V, icla_int_t ldv,
                double *TAU,
                icla_int_t st, icla_int_t ed, icla_int_t sweep,
                icla_int_t Vblksiz, icla_int_t wantz,
                double *work);
void
icla_dsbtype3cb(icla_int_t n, icla_int_t nb,
                double *A, icla_int_t lda,
                double *V, icla_int_t ldv,
                double *TAU,
                icla_int_t st, icla_int_t ed, icla_int_t sweep,
                icla_int_t Vblksiz, icla_int_t wantz,
                double *work);

icla_int_t
icla_dormqr_2stage_gpu(
    icla_side_t side, icla_trans_t trans, icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDouble_ptr dA, icla_int_t ldda,
    iclaDouble_ptr dC, icla_int_t lddc,
    iclaDouble_ptr dT, icla_int_t nb,
    icla_int_t *info);

icla_int_t
icla_get_dbulge_lq2( icla_int_t n, icla_int_t threads, icla_int_t wantz);

icla_int_t
icla_dbulge_getstg2size(icla_int_t n, icla_int_t nb, icla_int_t wantz,
                         icla_int_t Vblksiz, icla_int_t ldv, icla_int_t ldt,
                         icla_int_t *blkcnt, icla_int_t *sizTAU2,
                         icla_int_t *sizT2, icla_int_t *sizV2);

icla_int_t
icla_dbulge_getlwstg2(icla_int_t n, icla_int_t threads, icla_int_t wantz,
                       icla_int_t *Vblksiz, icla_int_t *ldv, icla_int_t *ldt,
                       icla_int_t *blkcnt, icla_int_t *sizTAU2,
                       icla_int_t *sizT2, icla_int_t *sizV2);

void
icla_bulge_get_VTsiz(icla_int_t n, icla_int_t nb, icla_int_t threads,
        icla_int_t *Vblksiz, icla_int_t *ldv, icla_int_t *ldt);
void
icla_dsyevdx_getworksize(icla_int_t n, icla_int_t threads,
        icla_int_t wantz,
        icla_int_t *lwmin,
        #ifdef ICLA_COMPLEX
        icla_int_t *lrwmin,
        #endif
        icla_int_t *liwmin);

icla_int_t
icla_dsytrd_bsy2trc_v5(
    icla_int_t threads, icla_int_t wantz, icla_uplo_t uplo,
    icla_int_t ne, icla_int_t n, icla_int_t nb,
    double *A, icla_int_t lda,
    double *D, double *E,
    iclaDouble_ptr dT1, icla_int_t ldt1);

icla_int_t
icla_dorgqr_2stage_gpu(
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDouble_ptr dA, icla_int_t ldda,
    double *tau,
    iclaDouble_ptr dT,
    icla_int_t nb,
    icla_int_t *info);

#ifdef __cplusplus
}
#endif
#undef ICLA_REAL
#endif

