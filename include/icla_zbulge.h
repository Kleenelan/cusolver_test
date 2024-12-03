/*
    -- ICLA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
*/

#ifndef ICLA_ZBULGE_H
#define ICLA_ZBULGE_H

#include "icla_types.h"
#define ICLA_COMPLEX

#ifdef __cplusplus
extern "C" {
#endif

icla_int_t
icla_zbulge_applyQ_v2(
    icla_side_t side,
    icla_int_t NE, icla_int_t n,
    icla_int_t nb, icla_int_t Vblksiz,
    iclaDoubleComplex_ptr dE, icla_int_t ldde,
    iclaDoubleComplex *V, icla_int_t ldv,
    iclaDoubleComplex *T, icla_int_t ldt,
    icla_int_t *info);

icla_int_t
icla_zbulge_applyQ_v2_m(
    icla_int_t ngpu, icla_side_t side,
    icla_int_t NE, icla_int_t n,
    icla_int_t nb, icla_int_t Vblksiz,
    iclaDoubleComplex *E, icla_int_t lde,
    iclaDoubleComplex *V, icla_int_t ldv,
    iclaDoubleComplex *T, icla_int_t ldt,
    icla_int_t *info);

icla_int_t
icla_zbulge_back(
    icla_uplo_t uplo,
    icla_int_t n, icla_int_t nb,
    icla_int_t ne, icla_int_t Vblksiz,
    iclaDoubleComplex *Z, icla_int_t ldz,
    iclaDoubleComplex_ptr dZ, icla_int_t lddz,
    iclaDoubleComplex *V, icla_int_t ldv,
    iclaDoubleComplex *TAU,
    iclaDoubleComplex *T, icla_int_t ldt,
    icla_int_t* info);

icla_int_t
icla_zbulge_back_m(
    icla_int_t ngpu, icla_uplo_t uplo,
    icla_int_t n, icla_int_t nb,
    icla_int_t ne, icla_int_t Vblksiz,
    iclaDoubleComplex *Z, icla_int_t ldz,
    iclaDoubleComplex *V, icla_int_t ldv,
    iclaDoubleComplex *TAU,
    iclaDoubleComplex *T, icla_int_t ldt,
    icla_int_t* info);

void
icla_ztrdtype1cbHLsym_withQ_v2(
    icla_int_t n, icla_int_t nb,
    iclaDoubleComplex *A, icla_int_t lda,
    iclaDoubleComplex *V, icla_int_t ldv,
    iclaDoubleComplex *TAU,
    icla_int_t st, icla_int_t ed,
    icla_int_t sweep, icla_int_t Vblksiz,
    iclaDoubleComplex *work);


void
icla_ztrdtype2cbHLsym_withQ_v2(
    icla_int_t n, icla_int_t nb,
    iclaDoubleComplex *A, icla_int_t lda,
    iclaDoubleComplex *V, icla_int_t ldv,
    iclaDoubleComplex *TAU,
    icla_int_t st, icla_int_t ed,
    icla_int_t sweep, icla_int_t Vblksiz,
    iclaDoubleComplex *work);


void
icla_ztrdtype3cbHLsym_withQ_v2(
    icla_int_t n, icla_int_t nb,
    iclaDoubleComplex *A, icla_int_t lda,
    iclaDoubleComplex *V, icla_int_t ldv,
    iclaDoubleComplex *TAU,
    icla_int_t st, icla_int_t ed,
    icla_int_t sweep, icla_int_t Vblksiz,
    iclaDoubleComplex *work);

void
icla_zlarfy(
    icla_int_t n,
    iclaDoubleComplex *A, icla_int_t lda,
    const iclaDoubleComplex *V, const iclaDoubleComplex *TAU,
    iclaDoubleComplex *work);

void
icla_zhbtype1cb(icla_int_t n, icla_int_t nb,
                iclaDoubleComplex *A, icla_int_t lda,
                iclaDoubleComplex *V, icla_int_t LDV,
                iclaDoubleComplex *TAU,
                icla_int_t st, icla_int_t ed, icla_int_t sweep,
                icla_int_t Vblksiz, icla_int_t wantz,
                iclaDoubleComplex *work);

void
icla_zhbtype2cb(icla_int_t n, icla_int_t nb,
                iclaDoubleComplex *A, icla_int_t lda,
                iclaDoubleComplex *V, icla_int_t ldv,
                iclaDoubleComplex *TAU,
                icla_int_t st, icla_int_t ed, icla_int_t sweep,
                icla_int_t Vblksiz, icla_int_t wantz,
                iclaDoubleComplex *work);
void
icla_zhbtype3cb(icla_int_t n, icla_int_t nb,
                iclaDoubleComplex *A, icla_int_t lda,
                iclaDoubleComplex *V, icla_int_t ldv,
                iclaDoubleComplex *TAU,
                icla_int_t st, icla_int_t ed, icla_int_t sweep,
                icla_int_t Vblksiz, icla_int_t wantz,
                iclaDoubleComplex *work);


icla_int_t
icla_zunmqr_2stage_gpu(
    icla_side_t side, icla_trans_t trans, icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr dC, icla_int_t lddc,
    iclaDoubleComplex_ptr dT, icla_int_t nb,
    icla_int_t *info);

icla_int_t
icla_get_zbulge_lq2( icla_int_t n, icla_int_t threads, icla_int_t wantz);

icla_int_t
icla_zbulge_getstg2size(icla_int_t n, icla_int_t nb, icla_int_t wantz,
                         icla_int_t Vblksiz, icla_int_t ldv, icla_int_t ldt,
                         icla_int_t *blkcnt, icla_int_t *sizTAU2,
                         icla_int_t *sizT2, icla_int_t *sizV2);


icla_int_t
icla_zbulge_getlwstg2(icla_int_t n, icla_int_t threads, icla_int_t wantz,
                       icla_int_t *Vblksiz, icla_int_t *ldv, icla_int_t *ldt,
                       icla_int_t *blkcnt, icla_int_t *sizTAU2,
                       icla_int_t *sizT2, icla_int_t *sizV2);


void
icla_bulge_get_VTsiz(icla_int_t n, icla_int_t nb, icla_int_t threads,
        icla_int_t *Vblksiz, icla_int_t *ldv, icla_int_t *ldt);
void
icla_zheevdx_getworksize(icla_int_t n, icla_int_t threads,
        icla_int_t wantz,
        icla_int_t *lwmin,
        #ifdef ICLA_COMPLEX
        icla_int_t *lrwmin,
        #endif
        icla_int_t *liwmin);


// used only for old version and internal
icla_int_t
icla_zhetrd_bhe2trc_v5(
    icla_int_t threads, icla_int_t wantz, icla_uplo_t uplo,
    icla_int_t ne, icla_int_t n, icla_int_t nb,
    iclaDoubleComplex *A, icla_int_t lda,
    double *D, double *E,
    iclaDoubleComplex_ptr dT1, icla_int_t ldt1);

icla_int_t
icla_zungqr_2stage_gpu(
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    iclaDoubleComplex *tau,
    iclaDoubleComplex_ptr dT,
    icla_int_t nb,
    icla_int_t *info);


#ifdef __cplusplus
}
#endif
#undef ICLA_COMPLEX
#endif /* ICLA_ZBULGE_H */
