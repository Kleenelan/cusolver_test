

#ifndef ICLA_CBULGE_H
#define ICLA_CBULGE_H

#include "icla_types.h"
#define ICLA_COMPLEX

#ifdef __cplusplus
extern "C" {
#endif

icla_int_t
icla_cbulge_applyQ_v2(
    icla_side_t side,
    icla_int_t NE, icla_int_t n,
    icla_int_t nb, icla_int_t Vblksiz,
    iclaFloatComplex_ptr dE, icla_int_t ldde,
    iclaFloatComplex *V, icla_int_t ldv,
    iclaFloatComplex *T, icla_int_t ldt,
    icla_int_t *info);

icla_int_t
icla_cbulge_applyQ_v2_m(
    icla_int_t ngpu, icla_side_t side,
    icla_int_t NE, icla_int_t n,
    icla_int_t nb, icla_int_t Vblksiz,
    iclaFloatComplex *E, icla_int_t lde,
    iclaFloatComplex *V, icla_int_t ldv,
    iclaFloatComplex *T, icla_int_t ldt,
    icla_int_t *info);

icla_int_t
icla_cbulge_back(
    icla_uplo_t uplo,
    icla_int_t n, icla_int_t nb,
    icla_int_t ne, icla_int_t Vblksiz,
    iclaFloatComplex *Z, icla_int_t ldz,
    iclaFloatComplex_ptr dZ, icla_int_t lddz,
    iclaFloatComplex *V, icla_int_t ldv,
    iclaFloatComplex *TAU,
    iclaFloatComplex *T, icla_int_t ldt,
    icla_int_t* info);

icla_int_t
icla_cbulge_back_m(
    icla_int_t ngpu, icla_uplo_t uplo,
    icla_int_t n, icla_int_t nb,
    icla_int_t ne, icla_int_t Vblksiz,
    iclaFloatComplex *Z, icla_int_t ldz,
    iclaFloatComplex *V, icla_int_t ldv,
    iclaFloatComplex *TAU,
    iclaFloatComplex *T, icla_int_t ldt,
    icla_int_t* info);

void
icla_ctrdtype1cbHLsym_withQ_v2(
    icla_int_t n, icla_int_t nb,
    iclaFloatComplex *A, icla_int_t lda,
    iclaFloatComplex *V, icla_int_t ldv,
    iclaFloatComplex *TAU,
    icla_int_t st, icla_int_t ed,
    icla_int_t sweep, icla_int_t Vblksiz,
    iclaFloatComplex *work);


void
icla_ctrdtype2cbHLsym_withQ_v2(
    icla_int_t n, icla_int_t nb,
    iclaFloatComplex *A, icla_int_t lda,
    iclaFloatComplex *V, icla_int_t ldv,
    iclaFloatComplex *TAU,
    icla_int_t st, icla_int_t ed,
    icla_int_t sweep, icla_int_t Vblksiz,
    iclaFloatComplex *work);


void
icla_ctrdtype3cbHLsym_withQ_v2(
    icla_int_t n, icla_int_t nb,
    iclaFloatComplex *A, icla_int_t lda,
    iclaFloatComplex *V, icla_int_t ldv,
    iclaFloatComplex *TAU,
    icla_int_t st, icla_int_t ed,
    icla_int_t sweep, icla_int_t Vblksiz,
    iclaFloatComplex *work);

void
icla_clarfy(
    icla_int_t n,
    iclaFloatComplex *A, icla_int_t lda,
    const iclaFloatComplex *V, const iclaFloatComplex *TAU,
    iclaFloatComplex *work);

void
icla_chbtype1cb(icla_int_t n, icla_int_t nb,
                iclaFloatComplex *A, icla_int_t lda,
                iclaFloatComplex *V, icla_int_t LDV,
                iclaFloatComplex *TAU,
                icla_int_t st, icla_int_t ed, icla_int_t sweep,
                icla_int_t Vblksiz, icla_int_t wantz,
                iclaFloatComplex *work);

void
icla_chbtype2cb(icla_int_t n, icla_int_t nb,
                iclaFloatComplex *A, icla_int_t lda,
                iclaFloatComplex *V, icla_int_t ldv,
                iclaFloatComplex *TAU,
                icla_int_t st, icla_int_t ed, icla_int_t sweep,
                icla_int_t Vblksiz, icla_int_t wantz,
                iclaFloatComplex *work);
void
icla_chbtype3cb(icla_int_t n, icla_int_t nb,
                iclaFloatComplex *A, icla_int_t lda,
                iclaFloatComplex *V, icla_int_t ldv,
                iclaFloatComplex *TAU,
                icla_int_t st, icla_int_t ed, icla_int_t sweep,
                icla_int_t Vblksiz, icla_int_t wantz,
                iclaFloatComplex *work);


icla_int_t
icla_cunmqr_2stage_gpu(
    icla_side_t side, icla_trans_t trans, icla_int_t m, icla_int_t n, icla_int_t k,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr dC, icla_int_t lddc,
    iclaFloatComplex_ptr dT, icla_int_t nb,
    icla_int_t *info);

icla_int_t
icla_get_cbulge_lq2( icla_int_t n, icla_int_t threads, icla_int_t wantz);

icla_int_t
icla_cbulge_getstg2size(icla_int_t n, icla_int_t nb, icla_int_t wantz,
                         icla_int_t Vblksiz, icla_int_t ldv, icla_int_t ldt,
                         icla_int_t *blkcnt, icla_int_t *sizTAU2,
                         icla_int_t *sizT2, icla_int_t *sizV2);


icla_int_t
icla_cbulge_getlwstg2(icla_int_t n, icla_int_t threads, icla_int_t wantz,
                       icla_int_t *Vblksiz, icla_int_t *ldv, icla_int_t *ldt,
                       icla_int_t *blkcnt, icla_int_t *sizTAU2,
                       icla_int_t *sizT2, icla_int_t *sizV2);


void
icla_bulge_get_VTsiz(icla_int_t n, icla_int_t nb, icla_int_t threads,
        icla_int_t *Vblksiz, icla_int_t *ldv, icla_int_t *ldt);
void
icla_cheevdx_getworksize(icla_int_t n, icla_int_t threads,
        icla_int_t wantz,
        icla_int_t *lwmin,
        #ifdef ICLA_COMPLEX
        icla_int_t *lrwmin,
        #endif
        icla_int_t *liwmin);



icla_int_t
icla_chetrd_bhe2trc_v5(
    icla_int_t threads, icla_int_t wantz, icla_uplo_t uplo,
    icla_int_t ne, icla_int_t n, icla_int_t nb,
    iclaFloatComplex *A, icla_int_t lda,
    float *D, float *E,
    iclaFloatComplex_ptr dT1, icla_int_t ldt1);

icla_int_t
icla_cungqr_2stage_gpu(
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    iclaFloatComplex *tau,
    iclaFloatComplex_ptr dT,
    icla_int_t nb,
    icla_int_t *info);


#ifdef __cplusplus
}
#endif
#undef ICLA_COMPLEX
#endif

