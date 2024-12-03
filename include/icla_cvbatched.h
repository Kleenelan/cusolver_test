/*
    -- ICLA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Azzam Haidar
       @author Tingxing Dong
       @author Ahmad Abdelfattah

       @generated from include/icla_zvbatched.h, normal z -> c, Fri Nov 29 12:16:14 2024
*/

#ifndef ICLA_CVBATCHED_H
#define ICLA_CVBATCHED_H

#include "icla_types.h"

#define ICLA_COMPLEX

#ifdef __cplusplus
extern "C" {
#endif

  /*
   *  control and tuning
   */
void icla_get_cgetrf_vbatched_nbparam(icla_int_t max_m, icla_int_t max_n, icla_int_t *nb, icla_int_t *recnb);


  /*
   *  LAPACK vbatched routines
   */

icla_int_t
icla_cgetf2_fused_vbatched(
    icla_int_t max_M, icla_int_t max_N,
    icla_int_t max_minMN, icla_int_t max_MxN,
    icla_int_t* M, icla_int_t* N,
    iclaFloatComplex **dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t* ldda,
    icla_int_t **dipiv_array, icla_int_t ipiv_i,
    icla_int_t *info_array, icla_int_t batchCount,
    icla_queue_t queue);

icla_int_t
icla_cgetf2_fused_sm_vbatched(
    icla_int_t max_M, icla_int_t max_N, icla_int_t max_minMN, icla_int_t max_MxN,
    icla_int_t* m, icla_int_t* n,
    iclaFloatComplex** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t* ldda,
    icla_int_t** dipiv_array, icla_int_t ipiv_i,
    icla_int_t* info_array, icla_int_t gbstep,
    icla_int_t nthreads, icla_int_t check_launch_only,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_cgetrf_vbatched(
        icla_int_t* m, icla_int_t* n,
        iclaFloatComplex **dA_array, icla_int_t *ldda,
        icla_int_t **ipiv_array, icla_int_t *info_array,
        icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_cgetrf_vbatched_max_nocheck(
        icla_int_t* m, icla_int_t* n, icla_int_t* minmn,
        icla_int_t max_m, icla_int_t max_n, icla_int_t max_minmn, icla_int_t max_mxn,
        icla_int_t nb, icla_int_t recnb,
        iclaFloatComplex **dA_array, icla_int_t *ldda,
        icla_int_t **ipiv_array, icla_int_t** pivinfo_array,
        icla_int_t *info_array, icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_cgetrf_vbatched_max_nocheck_work(
        icla_int_t* m, icla_int_t* n,
        icla_int_t max_m, icla_int_t max_n, icla_int_t max_minmn, icla_int_t max_mxn,
        iclaFloatComplex **dA_array, icla_int_t *ldda,
        icla_int_t **dipiv_array, icla_int_t *info_array,
        void* work, icla_int_t* lwork,
        icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_icamax_vbatched(
        icla_int_t length, icla_int_t *M, icla_int_t *N,
        iclaFloatComplex **dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t* ldda,
        icla_int_t** ipiv_array, icla_int_t ipiv_i,
        icla_int_t *info_array, icla_int_t step, icla_int_t gbstep,
        icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_cswap_vbatched(
        icla_int_t max_n, icla_int_t *M, icla_int_t *N,
        iclaFloatComplex **dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t *ldda,
        icla_int_t** ipiv_array, icla_int_t piv_adjustment,
        icla_int_t batchCount, icla_queue_t queue);

icla_int_t icla_cscal_cgeru_vbatched(
    icla_int_t max_M, icla_int_t max_N,
    icla_int_t *M, icla_int_t *N,
    iclaFloatComplex **dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t *ldda,
    icla_int_t *info_array, icla_int_t step, icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_cgetf2_vbatched(
    icla_int_t *m, icla_int_t *n, icla_int_t *minmn,
    icla_int_t max_m, icla_int_t max_n, icla_int_t max_minmn, icla_int_t max_mxn,
    iclaFloatComplex **dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t *ldda,
    icla_int_t **ipiv_array, icla_int_t *info_array,
    icla_int_t gbstep, icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_cgetrf_recpanel_vbatched(
    icla_int_t* m, icla_int_t* n, icla_int_t* minmn,
    icla_int_t max_m, icla_int_t max_n, icla_int_t max_minmn,
    icla_int_t max_mxn, icla_int_t min_recpnb,
    iclaFloatComplex** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t* ldda,
    icla_int_t** dipiv_array, icla_int_t dipiv_i, icla_int_t** dpivinfo_array,
    icla_int_t *info_array, icla_int_t gbstep,
    icla_int_t batchCount,  icla_queue_t queue);

void
icla_claswp_left_rowserial_vbatched(
        icla_int_t n,
        icla_int_t *M, icla_int_t *N, iclaFloatComplex** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t *ldda,
        icla_int_t **ipiv_array, icla_int_t ipiv_offset,
        icla_int_t k1, icla_int_t k2,
        icla_int_t batchCount, icla_queue_t queue);

void
icla_claswp_right_rowserial_vbatched(
        icla_int_t n,
        icla_int_t *M, icla_int_t *N, iclaFloatComplex** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t *ldda,
        icla_int_t **ipiv_array, icla_int_t ipiv_offset,
        icla_int_t k1, icla_int_t k2,
        icla_int_t batchCount, icla_queue_t queue);

void
icla_claswp_left_rowparallel_vbatched(
        icla_int_t n,
        icla_int_t* M, icla_int_t* N,
        iclaFloatComplex** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t* ldda,
        icla_int_t k1, icla_int_t k2,
        icla_int_t **pivinfo_array, icla_int_t pivinfo_i,
        icla_int_t batchCount, icla_queue_t queue);

void
icla_claswp_right_rowparallel_vbatched(
        icla_int_t n,
        icla_int_t* M, icla_int_t* N,
        iclaFloatComplex** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t* ldda,
        icla_int_t k1, icla_int_t k2,
        icla_int_t **pivinfo_array, icla_int_t pivinfo_i,
        icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_cpotrf_lpout_vbatched(
    icla_uplo_t uplo, icla_int_t *n, icla_int_t max_n,
    iclaFloatComplex **dA_array, icla_int_t *lda, icla_int_t gbstep,
    icla_int_t *info_array, icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_cpotf2_vbatched(
    icla_uplo_t uplo, icla_int_t* n, icla_int_t max_n,
    iclaFloatComplex **dA_array, icla_int_t* lda,
    iclaFloatComplex **dA_displ,
    iclaFloatComplex **dW_displ,
    iclaFloatComplex **dB_displ,
    iclaFloatComplex **dC_displ,
    icla_int_t *info_array, icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_cpotrf_panel_vbatched(
    icla_uplo_t uplo, icla_int_t* n, icla_int_t max_n,
    icla_int_t *ibvec, icla_int_t nb,
    iclaFloatComplex** dA_array,    icla_int_t* ldda,
    iclaFloatComplex** dX_array,    icla_int_t* dX_length,
    iclaFloatComplex** dinvA_array, icla_int_t* dinvA_length,
    iclaFloatComplex** dW0_displ, iclaFloatComplex** dW1_displ,
    iclaFloatComplex** dW2_displ, iclaFloatComplex** dW3_displ,
    iclaFloatComplex** dW4_displ,
    icla_int_t *info_array, icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_cpotrf_vbatched_max_nocheck(
    icla_uplo_t uplo, icla_int_t *n,
    iclaFloatComplex **dA_array, icla_int_t *ldda,
    icla_int_t *info_array,  icla_int_t batchCount,
    icla_int_t max_n, icla_queue_t queue);

icla_int_t
icla_cpotrf_vbatched(
    icla_uplo_t uplo, icla_int_t *n,
    iclaFloatComplex **dA_array, icla_int_t *ldda,
    icla_int_t *info_array,  icla_int_t batchCount,
    icla_queue_t queue);
  /*
   *  BLAS vbatched routines
   */
/* Level 3 */
void
iclablas_cgemm_vbatched_core(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t max_m, icla_int_t max_n, icla_int_t max_k,
    icla_int_t* m, icla_int_t* n, icla_int_t* k,
    iclaFloatComplex alpha,
    iclaFloatComplex const * const * dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t* ldda,
    iclaFloatComplex const * const * dB_array, icla_int_t Bi, icla_int_t Bj, icla_int_t* lddb,
    iclaFloatComplex beta,
    iclaFloatComplex              ** dC_array, icla_int_t Ci, icla_int_t Cj, icla_int_t* lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_cgemm_vbatched_max_nocheck(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t* m, icla_int_t* n, icla_int_t* k,
    iclaFloatComplex alpha,
    iclaFloatComplex const * const * dA_array, icla_int_t* ldda,
    iclaFloatComplex const * const * dB_array, icla_int_t* lddb,
    iclaFloatComplex beta,
    iclaFloatComplex **dC_array, icla_int_t* lddc,
    icla_int_t max_m, icla_int_t max_n, icla_int_t max_k,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_cgemm_vbatched_max(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t* m, icla_int_t* n, icla_int_t* k,
    iclaFloatComplex alpha,
    iclaFloatComplex const * const * dA_array, icla_int_t* ldda,
    iclaFloatComplex const * const * dB_array, icla_int_t* lddb,
    iclaFloatComplex beta,
    iclaFloatComplex **dC_array, icla_int_t* lddc,
    icla_int_t max_m, icla_int_t max_n, icla_int_t max_k,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_cgemm_vbatched_nocheck(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t* m, icla_int_t* n, icla_int_t* k,
    iclaFloatComplex alpha,
    iclaFloatComplex const * const * dA_array, icla_int_t* ldda,
    iclaFloatComplex const * const * dB_array, icla_int_t* lddb,
    iclaFloatComplex beta,
    iclaFloatComplex **dC_array, icla_int_t* lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_cgemm_vbatched(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t* m, icla_int_t* n, icla_int_t* k,
    iclaFloatComplex alpha,
    iclaFloatComplex const * const * dA_array, icla_int_t* ldda,
    iclaFloatComplex const * const * dB_array, icla_int_t* lddb,
    iclaFloatComplex beta,
    iclaFloatComplex **dC_array, icla_int_t* lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_cherk_internal_vbatched(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t* n, icla_int_t* k,
    iclaFloatComplex alpha,
    iclaFloatComplex const * const * dA_array, icla_int_t* ldda,
    iclaFloatComplex const * const * dB_array, icla_int_t* lddb,
    iclaFloatComplex beta,
    iclaFloatComplex **dC_array, icla_int_t* lddc,
    icla_int_t max_n, icla_int_t max_k,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_csyrk_internal_vbatched(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t* n, icla_int_t* k,
    iclaFloatComplex alpha,
    iclaFloatComplex const * const * dA_array, icla_int_t* ldda,
    iclaFloatComplex const * const * dB_array, icla_int_t* lddb,
    iclaFloatComplex beta,
    iclaFloatComplex **dC_array, icla_int_t* lddc,
    icla_int_t max_n, icla_int_t max_k,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_cherk_vbatched_max_nocheck(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t* n, icla_int_t* k,
    float alpha,
    iclaFloatComplex const * const * dA_array, icla_int_t* ldda,
    float beta,
    iclaFloatComplex **dC_array, icla_int_t* lddc,
    icla_int_t batchCount,
    icla_int_t max_n, icla_int_t max_k, icla_queue_t queue );

void
iclablas_cherk_vbatched_max(
        icla_uplo_t uplo, icla_trans_t trans,
        icla_int_t* n, icla_int_t* k,
        float alpha,
        iclaFloatComplex const * const * dA_array, icla_int_t* ldda,
        float beta,
        iclaFloatComplex **dC_array, icla_int_t* lddc,
        icla_int_t batchCount,
        icla_int_t max_n, icla_int_t max_k, icla_queue_t queue );

void
iclablas_cherk_vbatched_nocheck(
        icla_uplo_t uplo, icla_trans_t trans,
        icla_int_t* n, icla_int_t* k,
        float alpha,
        iclaFloatComplex const * const * dA_array, icla_int_t* ldda,
        float beta,
        iclaFloatComplex **dC_array, icla_int_t* lddc,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_cherk_vbatched(
        icla_uplo_t uplo, icla_trans_t trans,
        icla_int_t* n, icla_int_t* k,
        float alpha,
        iclaFloatComplex const * const * dA_array, icla_int_t* ldda,
        float beta,
        iclaFloatComplex **dC_array, icla_int_t* lddc,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_csyrk_vbatched_max_nocheck(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t* n, icla_int_t* k,
    iclaFloatComplex alpha,
    iclaFloatComplex const * const * dA_array, icla_int_t* ldda,
    iclaFloatComplex beta,
    iclaFloatComplex **dC_array, icla_int_t* lddc,
    icla_int_t batchCount,
    icla_int_t max_n, icla_int_t max_k, icla_queue_t queue );

void
iclablas_csyrk_vbatched_max(
        icla_uplo_t uplo, icla_trans_t trans,
        icla_int_t* n, icla_int_t* k,
        iclaFloatComplex alpha,
        iclaFloatComplex const * const * dA_array, icla_int_t* ldda,
        iclaFloatComplex beta,
        iclaFloatComplex **dC_array, icla_int_t* lddc,
        icla_int_t batchCount,
        icla_int_t max_n, icla_int_t max_k, icla_queue_t queue );

void
iclablas_csyrk_vbatched_nocheck(
        icla_uplo_t uplo, icla_trans_t trans,
        icla_int_t* n, icla_int_t* k,
        iclaFloatComplex alpha,
        iclaFloatComplex const * const * dA_array, icla_int_t* ldda,
        iclaFloatComplex beta,
        iclaFloatComplex **dC_array, icla_int_t* lddc,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_csyrk_vbatched(
        icla_uplo_t uplo, icla_trans_t trans,
        icla_int_t* n, icla_int_t* k,
        iclaFloatComplex alpha,
        iclaFloatComplex const * const * dA_array, icla_int_t* ldda,
        iclaFloatComplex beta,
        iclaFloatComplex **dC_array, icla_int_t* lddc,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_cher2k_vbatched_max_nocheck(
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t* n, icla_int_t* k,
    iclaFloatComplex alpha,
    iclaFloatComplex const * const * dA_array, icla_int_t* ldda,
    iclaFloatComplex const * const * dB_array, icla_int_t* lddb,
    float beta, iclaFloatComplex **dC_array, icla_int_t* lddc,
    icla_int_t batchCount,
    icla_int_t max_n, icla_int_t max_k, icla_queue_t queue );

void
iclablas_cher2k_vbatched_max(
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t* n, icla_int_t* k,
    iclaFloatComplex alpha,
    iclaFloatComplex const * const * dA_array, icla_int_t* ldda,
    iclaFloatComplex const * const * dB_array, icla_int_t* lddb,
    float beta, iclaFloatComplex **dC_array, icla_int_t* lddc,
    icla_int_t batchCount,
    icla_int_t max_n, icla_int_t max_k, icla_queue_t queue );

void
iclablas_cher2k_vbatched_nocheck(
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t* n, icla_int_t* k,
    iclaFloatComplex alpha,
    iclaFloatComplex const * const * dA_array, icla_int_t* ldda,
    iclaFloatComplex const * const * dB_array, icla_int_t* lddb,
    float beta, iclaFloatComplex **dC_array, icla_int_t* lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_cher2k_vbatched(
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t* n, icla_int_t* k,
    iclaFloatComplex alpha,
    iclaFloatComplex const * const * dA_array, icla_int_t* ldda,
    iclaFloatComplex const * const * dB_array, icla_int_t* lddb,
    float beta, iclaFloatComplex **dC_array, icla_int_t* lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_csyr2k_vbatched_max_nocheck(
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t* n, icla_int_t* k,
    iclaFloatComplex alpha,
    iclaFloatComplex const * const * dA_array, icla_int_t* ldda,
    iclaFloatComplex const * const * dB_array, icla_int_t* lddb,
    iclaFloatComplex beta,
    iclaFloatComplex **dC_array, icla_int_t* lddc,
    icla_int_t batchCount,
    icla_int_t max_n, icla_int_t max_k, icla_queue_t queue );

void
iclablas_csyr2k_vbatched_max(
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t* n, icla_int_t* k,
    iclaFloatComplex alpha,
    iclaFloatComplex const * const * dA_array, icla_int_t* ldda,
    iclaFloatComplex const * const * dB_array, icla_int_t* lddb,
    iclaFloatComplex beta,
    iclaFloatComplex **dC_array, icla_int_t* lddc,
    icla_int_t batchCount,
    icla_int_t max_n, icla_int_t max_k, icla_queue_t queue );

void
iclablas_csyr2k_vbatched_nocheck(
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t* n, icla_int_t* k,
    iclaFloatComplex alpha,
    iclaFloatComplex const * const * dA_array, icla_int_t* ldda,
    iclaFloatComplex const * const * dB_array, icla_int_t* lddb,
    iclaFloatComplex beta,
    iclaFloatComplex **dC_array, icla_int_t* lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_csyr2k_vbatched(
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t* n, icla_int_t* k,
    iclaFloatComplex alpha,
    iclaFloatComplex const * const * dA_array, icla_int_t* ldda,
    iclaFloatComplex const * const * dB_array, icla_int_t* lddb,
    iclaFloatComplex beta,
    iclaFloatComplex **dC_array, icla_int_t* lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_ctrmm_vbatched_core(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t max_m, icla_int_t max_n, icla_int_t* m, icla_int_t* n,
        iclaFloatComplex alpha,
        iclaFloatComplex **dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t* ldda,
        iclaFloatComplex **dB_array, icla_int_t Bi, icla_int_t Bj, icla_int_t* lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_ctrmm_vbatched_max_nocheck(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t max_m, icla_int_t max_n, icla_int_t* m, icla_int_t* n,
        iclaFloatComplex alpha,
        iclaFloatComplex **dA_array, icla_int_t* ldda,
        iclaFloatComplex **dB_array, icla_int_t* lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_ctrmm_vbatched_max(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t max_m, icla_int_t max_n, icla_int_t* m, icla_int_t* n,
        iclaFloatComplex alpha,
        iclaFloatComplex **dA_array, icla_int_t* ldda,
        iclaFloatComplex **dB_array, icla_int_t* lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_ctrmm_vbatched_nocheck(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t* m, icla_int_t* n,
        iclaFloatComplex alpha,
        iclaFloatComplex **dA_array, icla_int_t* ldda,
        iclaFloatComplex **dB_array, icla_int_t* lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_ctrmm_vbatched(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t* m, icla_int_t* n,
        iclaFloatComplex alpha,
        iclaFloatComplex **dA_array, icla_int_t* ldda,
        iclaFloatComplex **dB_array, icla_int_t* lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_ctrsm_small_vbatched(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t max_m, icla_int_t max_n, icla_int_t* m, icla_int_t* n,
        iclaFloatComplex alpha,
        iclaFloatComplex **dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t* ldda,
        iclaFloatComplex **dB_array, icla_int_t Bi, icla_int_t Bj, icla_int_t* lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_ctrsm_vbatched_core(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t max_m, icla_int_t max_n, icla_int_t* m, icla_int_t* n,
        iclaFloatComplex alpha,
        iclaFloatComplex **dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t* ldda,
        iclaFloatComplex **dB_array, icla_int_t Bi, icla_int_t Bj, icla_int_t* lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_ctrsm_vbatched_max_nocheck(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t max_m, icla_int_t max_n, icla_int_t* m, icla_int_t* n,
        iclaFloatComplex alpha,
        iclaFloatComplex **dA_array, icla_int_t* ldda,
        iclaFloatComplex **dB_array, icla_int_t* lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_ctrsm_vbatched_max(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t max_m, icla_int_t max_n, icla_int_t* m, icla_int_t* n,
    iclaFloatComplex alpha,
    iclaFloatComplex** dA_array,    icla_int_t* ldda,
    iclaFloatComplex** dB_array,    icla_int_t* lddb,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_ctrsm_vbatched(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t* m, icla_int_t* n,
    iclaFloatComplex alpha,
    iclaFloatComplex** dA_array,    icla_int_t* ldda,
    iclaFloatComplex** dB_array,    icla_int_t* lddb,
    icla_int_t batchCount, icla_queue_t queue);

void iclablas_ctrsm_inv_outofplace_vbatched(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t flag,
    icla_int_t *m, icla_int_t* n,
    iclaFloatComplex alpha,
    iclaFloatComplex** dA_array,    icla_int_t* ldda,
    iclaFloatComplex** dB_array,    icla_int_t* lddb,
    iclaFloatComplex** dX_array,    icla_int_t* lddx,
    iclaFloatComplex** dinvA_array, icla_int_t* dinvA_length,
    iclaFloatComplex** dA_displ, iclaFloatComplex** dB_displ,
    iclaFloatComplex** dX_displ, iclaFloatComplex** dinvA_displ,
    icla_int_t resetozero,
    icla_int_t batchCount,
    icla_int_t max_m, icla_int_t max_n,
    icla_queue_t queue);

void iclablas_ctrsm_inv_work_vbatched(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t flag,
    icla_int_t* m, icla_int_t* n,
    iclaFloatComplex alpha,
    iclaFloatComplex** dA_array,    icla_int_t* ldda,
    iclaFloatComplex** dB_array,    icla_int_t* lddb,
    iclaFloatComplex** dX_array,    icla_int_t* lddx,
    iclaFloatComplex** dinvA_array, icla_int_t* dinvA_length,
    iclaFloatComplex** dA_displ, iclaFloatComplex** dB_displ,
    iclaFloatComplex** dX_displ, iclaFloatComplex** dinvA_displ,
    icla_int_t resetozero,
    icla_int_t batchCount,
    icla_int_t max_m, icla_int_t max_n,
    icla_queue_t queue);

void iclablas_ctrsm_inv_vbatched_max_nocheck(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t* m, icla_int_t* n,
    iclaFloatComplex alpha,
    iclaFloatComplex** dA_array,    icla_int_t* ldda,
    iclaFloatComplex** dB_array,    icla_int_t* lddb,
    icla_int_t batchCount,
    icla_int_t max_m, icla_int_t max_n,
    icla_queue_t queue);

void
iclablas_ctrsm_inv_vbatched_max(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t* m, icla_int_t* n,
    iclaFloatComplex alpha,
    iclaFloatComplex** dA_array,    icla_int_t* ldda,
    iclaFloatComplex** dB_array,    icla_int_t* lddb,
    icla_int_t batchCount,
    icla_int_t max_m, icla_int_t max_n,
    icla_queue_t queue);

void
iclablas_ctrsm_inv_vbatched_nocheck(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t* m, icla_int_t* n,
    iclaFloatComplex alpha,
    iclaFloatComplex** dA_array,    icla_int_t* ldda,
    iclaFloatComplex** dB_array,    icla_int_t* lddb,
    icla_int_t batchCount,
    icla_queue_t queue);

void
iclablas_ctrsm_inv_vbatched(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t* m, icla_int_t* n,
    iclaFloatComplex alpha,
    iclaFloatComplex** dA_array,    icla_int_t* ldda,
    iclaFloatComplex** dB_array,    icla_int_t* lddb,
    icla_int_t batchCount,
    icla_queue_t queue);

void
iclablas_ctrtri_diag_vbatched(
    icla_uplo_t uplo, icla_diag_t diag, icla_int_t nmax, icla_int_t *n,
    iclaFloatComplex const * const *dA_array, icla_int_t *ldda,
    iclaFloatComplex **dinvA_array,
    icla_int_t resetozero, icla_int_t batchCount, icla_queue_t queue);

void
iclablas_chemm_vbatched_core(
        icla_side_t side, icla_uplo_t uplo,
        icla_int_t *m, icla_int_t *n,
        iclaFloatComplex alpha,
        iclaFloatComplex **dA_array, icla_int_t *ldda,
        iclaFloatComplex **dB_array, icla_int_t *lddb,
        iclaFloatComplex beta,
        iclaFloatComplex **dC_array, icla_int_t *lddc,
        icla_int_t max_m, icla_int_t max_n,
        icla_int_t roffA, icla_int_t coffA, icla_int_t roffB, icla_int_t coffB, icla_int_t roffC, icla_int_t coffC,
        icla_int_t specM, icla_int_t specN,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_chemm_vbatched_max_nocheck(
        icla_side_t side, icla_uplo_t uplo,
        icla_int_t *m, icla_int_t *n,
        iclaFloatComplex alpha,
        iclaFloatComplex **dA_array, icla_int_t *ldda,
        iclaFloatComplex **dB_array, icla_int_t *lddb,
        iclaFloatComplex beta,
        iclaFloatComplex **dC_array, icla_int_t *lddc,
        icla_int_t batchCount, icla_int_t max_m, icla_int_t max_n,
        icla_queue_t queue );

void
iclablas_chemm_vbatched_max(
        icla_side_t side, icla_uplo_t uplo,
        icla_int_t *m, icla_int_t *n,
        iclaFloatComplex alpha,
        iclaFloatComplex **dA_array, icla_int_t *ldda,
        iclaFloatComplex **dB_array, icla_int_t *lddb,
        iclaFloatComplex beta,
        iclaFloatComplex **dC_array, icla_int_t *lddc,
        icla_int_t batchCount, icla_int_t max_m, icla_int_t max_n,
        icla_queue_t queue );

void
iclablas_chemm_vbatched_nocheck(
        icla_side_t side, icla_uplo_t uplo,
        icla_int_t *m, icla_int_t *n,
        iclaFloatComplex alpha,
        iclaFloatComplex **dA_array, icla_int_t *ldda,
        iclaFloatComplex **dB_array, icla_int_t *lddb,
        iclaFloatComplex beta,
        iclaFloatComplex **dC_array, icla_int_t *lddc,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_chemm_vbatched(
        icla_side_t side, icla_uplo_t uplo,
        icla_int_t *m, icla_int_t *n,
        iclaFloatComplex alpha,
        iclaFloatComplex **dA_array, icla_int_t *ldda,
        iclaFloatComplex **dB_array, icla_int_t *lddb,
        iclaFloatComplex beta,
        iclaFloatComplex **dC_array, icla_int_t *lddc,
        icla_int_t batchCount, icla_queue_t queue );

/* Level 2 */
void
iclablas_cgemv_vbatched_max_nocheck(
    icla_trans_t trans, icla_int_t* m, icla_int_t* n,
    iclaFloatComplex alpha,
    iclaFloatComplex_ptr dA_array[], icla_int_t* ldda,
    iclaFloatComplex_ptr dx_array[], icla_int_t* incx,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr dy_array[], icla_int_t* incy,
    icla_int_t batchCount,
    icla_int_t max_m, icla_int_t max_n, icla_queue_t queue);

void
iclablas_cgemv_vbatched_max(
    icla_trans_t trans, icla_int_t* m, icla_int_t* n,
    iclaFloatComplex alpha,
    iclaFloatComplex_ptr dA_array[], icla_int_t* ldda,
    iclaFloatComplex_ptr dx_array[], icla_int_t* incx,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr dy_array[], icla_int_t* incy,
    icla_int_t batchCount,
    icla_int_t max_m, icla_int_t max_n, icla_queue_t queue);

void
iclablas_cgemv_vbatched_nocheck(
    icla_trans_t trans, icla_int_t* m, icla_int_t* n,
    iclaFloatComplex alpha,
    iclaFloatComplex_ptr dA_array[], icla_int_t* ldda,
    iclaFloatComplex_ptr dx_array[], icla_int_t* incx,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr dy_array[], icla_int_t* incy,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_cgemv_vbatched(
    icla_trans_t trans, icla_int_t* m, icla_int_t* n,
    iclaFloatComplex alpha,
    iclaFloatComplex_ptr dA_array[], icla_int_t* ldda,
    iclaFloatComplex_ptr dx_array[], icla_int_t* incx,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr dy_array[], icla_int_t* incy,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_chemv_vbatched_max_nocheck(
    icla_uplo_t uplo, icla_int_t* n, iclaFloatComplex alpha,
    iclaFloatComplex **dA_array, icla_int_t* ldda,
    iclaFloatComplex **dX_array, icla_int_t* incx,
    iclaFloatComplex beta,
    iclaFloatComplex **dY_array, icla_int_t* incy,
    icla_int_t max_n, icla_int_t batchCount, icla_queue_t queue );

void
iclablas_chemv_vbatched_max(
    icla_uplo_t uplo, icla_int_t* n,
    iclaFloatComplex alpha,
    iclaFloatComplex_ptr dA_array[], icla_int_t* ldda,
    iclaFloatComplex_ptr dx_array[], icla_int_t* incx,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr dy_array[], icla_int_t* incy,
    icla_int_t batchCount,
    icla_int_t max_n, icla_queue_t queue);

void
iclablas_chemv_vbatched_nocheck(
    icla_uplo_t uplo, icla_int_t* n,
    iclaFloatComplex alpha,
    iclaFloatComplex_ptr dA_array[], icla_int_t* ldda,
    iclaFloatComplex_ptr dx_array[], icla_int_t* incx,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr dy_array[], icla_int_t* incy,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_chemv_vbatched(
    icla_uplo_t uplo, icla_int_t* n,
    iclaFloatComplex alpha,
    iclaFloatComplex_ptr dA_array[], icla_int_t* ldda,
    iclaFloatComplex_ptr dx_array[], icla_int_t* incx,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr dy_array[], icla_int_t* incy,
    icla_int_t batchCount, icla_queue_t queue);
/* Level 1 */
/* Auxiliary routines */
void icla_cset_pointer_var_cc(
    iclaFloatComplex **output_array,
    iclaFloatComplex *input,
    icla_int_t *lda,
    icla_int_t row, icla_int_t column,
    icla_int_t *batch_offset,
    icla_int_t batchCount,
    icla_queue_t queue);

void
icla_cdisplace_pointers_var_cc(iclaFloatComplex **output_array,
    iclaFloatComplex **input_array, icla_int_t* lda,
    icla_int_t row, icla_int_t column,
    icla_int_t batchCount, icla_queue_t queue);

void
icla_cdisplace_pointers_var_cv(iclaFloatComplex **output_array,
    iclaFloatComplex **input_array, icla_int_t* lda,
    icla_int_t row, icla_int_t* column,
    icla_int_t batchCount, icla_queue_t queue);

void
icla_cdisplace_pointers_var_vc(iclaFloatComplex **output_array,
    iclaFloatComplex **input_array, icla_int_t* lda,
    icla_int_t *row, icla_int_t column,
    icla_int_t batchCount, icla_queue_t queue);

void
icla_cdisplace_pointers_var_vv(iclaFloatComplex **output_array,
    iclaFloatComplex **input_array, icla_int_t* lda,
    icla_int_t* row, icla_int_t* column,
    icla_int_t batchCount, icla_queue_t queue);

void iclablas_claset_vbatched(
    icla_uplo_t uplo, icla_int_t max_m, icla_int_t max_n,
    icla_int_t* m, icla_int_t* n,
    iclaFloatComplex offdiag, iclaFloatComplex diag,
    iclaFloatComplex_ptr dAarray[], icla_int_t* ldda,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_clacpy_vbatched(
    icla_uplo_t uplo,
    icla_int_t max_m, icla_int_t max_n,
    icla_int_t* m, icla_int_t* n,
    iclaFloatComplex const * const * dAarray, icla_int_t* ldda,
    iclaFloatComplex**               dBarray, icla_int_t* lddb,
    icla_int_t batchCount, icla_queue_t queue );

  /*
   *  Aux. vbatched routines
   */
icla_int_t icla_get_cpotrf_vbatched_crossover();

#ifdef __cplusplus
}
#endif

#undef ICLA_COMPLEX

#endif  /* ICLA_CVBATCHED_H */
