/*
    -- ICLA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from include/icla_zbulge.h, normal z -> s, Fri Nov 29 12:16:14 2024
*/

#ifndef ICLA_SBULGE_H
#define ICLA_SBULGE_H

#include "icla_types.h"
#define ICLA_REAL

#ifdef __cplusplus
extern "C" {
#endif

icla_int_t
icla_sbulge_applyQ_v2(
    icla_side_t side,
    icla_int_t NE, icla_int_t n,
    icla_int_t nb, icla_int_t Vblksiz,
    iclaFloat_ptr dE, icla_int_t ldde,
    float *V, icla_int_t ldv,
    float *T, icla_int_t ldt,
    icla_int_t *info);

icla_int_t
icla_sbulge_applyQ_v2_m(
    icla_int_t ngpu, icla_side_t side,
    icla_int_t NE, icla_int_t n,
    icla_int_t nb, icla_int_t Vblksiz,
    float *E, icla_int_t lde,
    float *V, icla_int_t ldv,
    float *T, icla_int_t ldt,
    icla_int_t *info);

icla_int_t
icla_sbulge_back(
    icla_uplo_t uplo,
    icla_int_t n, icla_int_t nb,
    icla_int_t ne, icla_int_t Vblksiz,
    float *Z, icla_int_t ldz,
    iclaFloat_ptr dZ, icla_int_t lddz,
    float *V, icla_int_t ldv,
    float *TAU,
    float *T, icla_int_t ldt,
    icla_int_t* info);

icla_int_t
icla_sbulge_back_m(
    icla_int_t ngpu, icla_uplo_t uplo,
    icla_int_t n, icla_int_t nb,
    icla_int_t ne, icla_int_t Vblksiz,
    float *Z, icla_int_t ldz,
    float *V, icla_int_t ldv,
    float *TAU,
    float *T, icla_int_t ldt,
    icla_int_t* info);

void
icla_strdtype1cbHLsym_withQ_v2(
    icla_int_t n, icla_int_t nb,
    float *A, icla_int_t lda,
    float *V, icla_int_t ldv,
    float *TAU,
    icla_int_t st, icla_int_t ed,
    icla_int_t sweep, icla_int_t Vblksiz,
    float *work);


void
icla_strdtype2cbHLsym_withQ_v2(
    icla_int_t n, icla_int_t nb,
    float *A, icla_int_t lda,
    float *V, icla_int_t ldv,
    float *TAU,
    icla_int_t st, icla_int_t ed,
    icla_int_t sweep, icla_int_t Vblksiz,
    float *work);


void
icla_strdtype3cbHLsym_withQ_v2(
    icla_int_t n, icla_int_t nb,
    float *A, icla_int_t lda,
    float *V, icla_int_t ldv,
    float *TAU,
    icla_int_t st, icla_int_t ed,
    icla_int_t sweep, icla_int_t Vblksiz,
    float *work);

void
icla_slarfy(
    icla_int_t n,
    float *A, icla_int_t lda,
    const float *V, const float *TAU,
    float *work);

void
icla_ssbtype1cb(icla_int_t n, icla_int_t nb,
                float *A, icla_int_t lda,
                float *V, icla_int_t LDV,
                float *TAU,
                icla_int_t st, icla_int_t ed, icla_int_t sweep,
                icla_int_t Vblksiz, icla_int_t wantz,
                float *work);

void
icla_ssbtype2cb(icla_int_t n, icla_int_t nb,
                float *A, icla_int_t lda,
                float *V, icla_int_t ldv,
                float *TAU,
                icla_int_t st, icla_int_t ed, icla_int_t sweep,
                icla_int_t Vblksiz, icla_int_t wantz,
                float *work);
void
icla_ssbtype3cb(icla_int_t n, icla_int_t nb,
                float *A, icla_int_t lda,
                float *V, icla_int_t ldv,
                float *TAU,
                icla_int_t st, icla_int_t ed, icla_int_t sweep,
                icla_int_t Vblksiz, icla_int_t wantz,
                float *work);


icla_int_t
icla_sormqr_2stage_gpu(
    icla_side_t side, icla_trans_t trans, icla_int_t m, icla_int_t n, icla_int_t k,
    iclaFloat_ptr dA, icla_int_t ldda,
    iclaFloat_ptr dC, icla_int_t lddc,
    iclaFloat_ptr dT, icla_int_t nb,
    icla_int_t *info);

icla_int_t
icla_get_sbulge_lq2( icla_int_t n, icla_int_t threads, icla_int_t wantz);

icla_int_t
icla_sbulge_getstg2size(icla_int_t n, icla_int_t nb, icla_int_t wantz,
                         icla_int_t Vblksiz, icla_int_t ldv, icla_int_t ldt,
                         icla_int_t *blkcnt, icla_int_t *sizTAU2,
                         icla_int_t *sizT2, icla_int_t *sizV2);


icla_int_t
icla_sbulge_getlwstg2(icla_int_t n, icla_int_t threads, icla_int_t wantz,
                       icla_int_t *Vblksiz, icla_int_t *ldv, icla_int_t *ldt,
                       icla_int_t *blkcnt, icla_int_t *sizTAU2,
                       icla_int_t *sizT2, icla_int_t *sizV2);


void
icla_bulge_get_VTsiz(icla_int_t n, icla_int_t nb, icla_int_t threads,
        icla_int_t *Vblksiz, icla_int_t *ldv, icla_int_t *ldt);
void
icla_ssyevdx_getworksize(icla_int_t n, icla_int_t threads,
        icla_int_t wantz,
        icla_int_t *lwmin,
        #ifdef ICLA_COMPLEX
        icla_int_t *lrwmin,
        #endif
        icla_int_t *liwmin);


// used only for old version and internal
icla_int_t
icla_ssytrd_bsy2trc_v5(
    icla_int_t threads, icla_int_t wantz, icla_uplo_t uplo,
    icla_int_t ne, icla_int_t n, icla_int_t nb,
    float *A, icla_int_t lda,
    float *D, float *E,
    iclaFloat_ptr dT1, icla_int_t ldt1);

icla_int_t
icla_sorgqr_2stage_gpu(
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaFloat_ptr dA, icla_int_t ldda,
    float *tau,
    iclaFloat_ptr dT,
    icla_int_t nb,
    icla_int_t *info);


#ifdef __cplusplus
}
#endif
#undef ICLA_REAL
#endif /* ICLA_SBULGE_H */
