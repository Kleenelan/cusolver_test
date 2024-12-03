/*
    -- ICLA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Azzam Haidar
       @author Tingxing Dong
       @author Ahmad Abdelfattah

       @precisions normal z -> s d c
*/

#ifndef ICLA_ZVBATCHED_H
#define ICLA_ZVBATCHED_H

#include "icla_types.h"

#define ICLA_COMPLEX

#ifdef __cplusplus
extern "C" {
#endif

  /*
   *  control and tuning
   */
void icla_get_zgetrf_vbatched_nbparam(icla_int_t max_m, icla_int_t max_n, icla_int_t *nb, icla_int_t *recnb);


  /*
   *  LAPACK vbatched routines
   */

icla_int_t
icla_zgetf2_fused_vbatched(
    icla_int_t max_M, icla_int_t max_N,
    icla_int_t max_minMN, icla_int_t max_MxN,
    icla_int_t* M, icla_int_t* N,
    iclaDoubleComplex **dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t* ldda,
    icla_int_t **dipiv_array, icla_int_t ipiv_i,
    icla_int_t *info_array, icla_int_t batchCount,
    icla_queue_t queue);

icla_int_t
icla_zgetf2_fused_sm_vbatched(
    icla_int_t max_M, icla_int_t max_N, icla_int_t max_minMN, icla_int_t max_MxN,
    icla_int_t* m, icla_int_t* n,
    iclaDoubleComplex** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t* ldda,
    icla_int_t** dipiv_array, icla_int_t ipiv_i,
    icla_int_t* info_array, icla_int_t gbstep,
    icla_int_t nthreads, icla_int_t check_launch_only,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_zgetrf_vbatched(
        icla_int_t* m, icla_int_t* n,
        iclaDoubleComplex **dA_array, icla_int_t *ldda,
        icla_int_t **ipiv_array, icla_int_t *info_array,
        icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zgetrf_vbatched_max_nocheck(
        icla_int_t* m, icla_int_t* n, icla_int_t* minmn,
        icla_int_t max_m, icla_int_t max_n, icla_int_t max_minmn, icla_int_t max_mxn,
        icla_int_t nb, icla_int_t recnb,
        iclaDoubleComplex **dA_array, icla_int_t *ldda,
        icla_int_t **ipiv_array, icla_int_t** pivinfo_array,
        icla_int_t *info_array, icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zgetrf_vbatched_max_nocheck_work(
        icla_int_t* m, icla_int_t* n,
        icla_int_t max_m, icla_int_t max_n, icla_int_t max_minmn, icla_int_t max_mxn,
        iclaDoubleComplex **dA_array, icla_int_t *ldda,
        icla_int_t **dipiv_array, icla_int_t *info_array,
        void* work, icla_int_t* lwork,
        icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_izamax_vbatched(
        icla_int_t length, icla_int_t *M, icla_int_t *N,
        iclaDoubleComplex **dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t* ldda,
        icla_int_t** ipiv_array, icla_int_t ipiv_i,
        icla_int_t *info_array, icla_int_t step, icla_int_t gbstep,
        icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zswap_vbatched(
        icla_int_t max_n, icla_int_t *M, icla_int_t *N,
        iclaDoubleComplex **dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t *ldda,
        icla_int_t** ipiv_array, icla_int_t piv_adjustment,
        icla_int_t batchCount, icla_queue_t queue);

icla_int_t icla_zscal_zgeru_vbatched(
    icla_int_t max_M, icla_int_t max_N,
    icla_int_t *M, icla_int_t *N,
    iclaDoubleComplex **dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t *ldda,
    icla_int_t *info_array, icla_int_t step, icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zgetf2_vbatched(
    icla_int_t *m, icla_int_t *n, icla_int_t *minmn,
    icla_int_t max_m, icla_int_t max_n, icla_int_t max_minmn, icla_int_t max_mxn,
    iclaDoubleComplex **dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t *ldda,
    icla_int_t **ipiv_array, icla_int_t *info_array,
    icla_int_t gbstep, icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zgetrf_recpanel_vbatched(
    icla_int_t* m, icla_int_t* n, icla_int_t* minmn,
    icla_int_t max_m, icla_int_t max_n, icla_int_t max_minmn,
    icla_int_t max_mxn, icla_int_t min_recpnb,
    iclaDoubleComplex** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t* ldda,
    icla_int_t** dipiv_array, icla_int_t dipiv_i, icla_int_t** dpivinfo_array,
    icla_int_t *info_array, icla_int_t gbstep,
    icla_int_t batchCount,  icla_queue_t queue);

void
icla_zlaswp_left_rowserial_vbatched(
        icla_int_t n,
        icla_int_t *M, icla_int_t *N, iclaDoubleComplex** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t *ldda,
        icla_int_t **ipiv_array, icla_int_t ipiv_offset,
        icla_int_t k1, icla_int_t k2,
        icla_int_t batchCount, icla_queue_t queue);

void
icla_zlaswp_right_rowserial_vbatched(
        icla_int_t n,
        icla_int_t *M, icla_int_t *N, iclaDoubleComplex** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t *ldda,
        icla_int_t **ipiv_array, icla_int_t ipiv_offset,
        icla_int_t k1, icla_int_t k2,
        icla_int_t batchCount, icla_queue_t queue);

void
icla_zlaswp_left_rowparallel_vbatched(
        icla_int_t n,
        icla_int_t* M, icla_int_t* N,
        iclaDoubleComplex** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t* ldda,
        icla_int_t k1, icla_int_t k2,
        icla_int_t **pivinfo_array, icla_int_t pivinfo_i,
        icla_int_t batchCount, icla_queue_t queue);

void
icla_zlaswp_right_rowparallel_vbatched(
        icla_int_t n,
        icla_int_t* M, icla_int_t* N,
        iclaDoubleComplex** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t* ldda,
        icla_int_t k1, icla_int_t k2,
        icla_int_t **pivinfo_array, icla_int_t pivinfo_i,
        icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zpotrf_lpout_vbatched(
    icla_uplo_t uplo, icla_int_t *n, icla_int_t max_n,
    iclaDoubleComplex **dA_array, icla_int_t *lda, icla_int_t gbstep,
    icla_int_t *info_array, icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zpotf2_vbatched(
    icla_uplo_t uplo, icla_int_t* n, icla_int_t max_n,
    iclaDoubleComplex **dA_array, icla_int_t* lda,
    iclaDoubleComplex **dA_displ,
    iclaDoubleComplex **dW_displ,
    iclaDoubleComplex **dB_displ,
    iclaDoubleComplex **dC_displ,
    icla_int_t *info_array, icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zpotrf_panel_vbatched(
    icla_uplo_t uplo, icla_int_t* n, icla_int_t max_n,
    icla_int_t *ibvec, icla_int_t nb,
    iclaDoubleComplex** dA_array,    icla_int_t* ldda,
    iclaDoubleComplex** dX_array,    icla_int_t* dX_length,
    iclaDoubleComplex** dinvA_array, icla_int_t* dinvA_length,
    iclaDoubleComplex** dW0_displ, iclaDoubleComplex** dW1_displ,
    iclaDoubleComplex** dW2_displ, iclaDoubleComplex** dW3_displ,
    iclaDoubleComplex** dW4_displ,
    icla_int_t *info_array, icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zpotrf_vbatched_max_nocheck(
    icla_uplo_t uplo, icla_int_t *n,
    iclaDoubleComplex **dA_array, icla_int_t *ldda,
    icla_int_t *info_array,  icla_int_t batchCount,
    icla_int_t max_n, icla_queue_t queue);

icla_int_t
icla_zpotrf_vbatched(
    icla_uplo_t uplo, icla_int_t *n,
    iclaDoubleComplex **dA_array, icla_int_t *ldda,
    icla_int_t *info_array,  icla_int_t batchCount,
    icla_queue_t queue);
  /*
   *  BLAS vbatched routines
   */
/* Level 3 */
void
iclablas_zgemm_vbatched_core(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t max_m, icla_int_t max_n, icla_int_t max_k,
    icla_int_t* m, icla_int_t* n, icla_int_t* k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex const * const * dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t* ldda,
    iclaDoubleComplex const * const * dB_array, icla_int_t Bi, icla_int_t Bj, icla_int_t* lddb,
    iclaDoubleComplex beta,
    iclaDoubleComplex              ** dC_array, icla_int_t Ci, icla_int_t Cj, icla_int_t* lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_zgemm_vbatched_max_nocheck(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t* m, icla_int_t* n, icla_int_t* k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex const * const * dA_array, icla_int_t* ldda,
    iclaDoubleComplex const * const * dB_array, icla_int_t* lddb,
    iclaDoubleComplex beta,
    iclaDoubleComplex **dC_array, icla_int_t* lddc,
    icla_int_t max_m, icla_int_t max_n, icla_int_t max_k,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_zgemm_vbatched_max(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t* m, icla_int_t* n, icla_int_t* k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex const * const * dA_array, icla_int_t* ldda,
    iclaDoubleComplex const * const * dB_array, icla_int_t* lddb,
    iclaDoubleComplex beta,
    iclaDoubleComplex **dC_array, icla_int_t* lddc,
    icla_int_t max_m, icla_int_t max_n, icla_int_t max_k,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_zgemm_vbatched_nocheck(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t* m, icla_int_t* n, icla_int_t* k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex const * const * dA_array, icla_int_t* ldda,
    iclaDoubleComplex const * const * dB_array, icla_int_t* lddb,
    iclaDoubleComplex beta,
    iclaDoubleComplex **dC_array, icla_int_t* lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_zgemm_vbatched(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t* m, icla_int_t* n, icla_int_t* k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex const * const * dA_array, icla_int_t* ldda,
    iclaDoubleComplex const * const * dB_array, icla_int_t* lddb,
    iclaDoubleComplex beta,
    iclaDoubleComplex **dC_array, icla_int_t* lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_zherk_internal_vbatched(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t* n, icla_int_t* k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex const * const * dA_array, icla_int_t* ldda,
    iclaDoubleComplex const * const * dB_array, icla_int_t* lddb,
    iclaDoubleComplex beta,
    iclaDoubleComplex **dC_array, icla_int_t* lddc,
    icla_int_t max_n, icla_int_t max_k,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_zsyrk_internal_vbatched(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t* n, icla_int_t* k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex const * const * dA_array, icla_int_t* ldda,
    iclaDoubleComplex const * const * dB_array, icla_int_t* lddb,
    iclaDoubleComplex beta,
    iclaDoubleComplex **dC_array, icla_int_t* lddc,
    icla_int_t max_n, icla_int_t max_k,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_zherk_vbatched_max_nocheck(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t* n, icla_int_t* k,
    double alpha,
    iclaDoubleComplex const * const * dA_array, icla_int_t* ldda,
    double beta,
    iclaDoubleComplex **dC_array, icla_int_t* lddc,
    icla_int_t batchCount,
    icla_int_t max_n, icla_int_t max_k, icla_queue_t queue );

void
iclablas_zherk_vbatched_max(
        icla_uplo_t uplo, icla_trans_t trans,
        icla_int_t* n, icla_int_t* k,
        double alpha,
        iclaDoubleComplex const * const * dA_array, icla_int_t* ldda,
        double beta,
        iclaDoubleComplex **dC_array, icla_int_t* lddc,
        icla_int_t batchCount,
        icla_int_t max_n, icla_int_t max_k, icla_queue_t queue );

void
iclablas_zherk_vbatched_nocheck(
        icla_uplo_t uplo, icla_trans_t trans,
        icla_int_t* n, icla_int_t* k,
        double alpha,
        iclaDoubleComplex const * const * dA_array, icla_int_t* ldda,
        double beta,
        iclaDoubleComplex **dC_array, icla_int_t* lddc,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_zherk_vbatched(
        icla_uplo_t uplo, icla_trans_t trans,
        icla_int_t* n, icla_int_t* k,
        double alpha,
        iclaDoubleComplex const * const * dA_array, icla_int_t* ldda,
        double beta,
        iclaDoubleComplex **dC_array, icla_int_t* lddc,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_zsyrk_vbatched_max_nocheck(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t* n, icla_int_t* k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex const * const * dA_array, icla_int_t* ldda,
    iclaDoubleComplex beta,
    iclaDoubleComplex **dC_array, icla_int_t* lddc,
    icla_int_t batchCount,
    icla_int_t max_n, icla_int_t max_k, icla_queue_t queue );

void
iclablas_zsyrk_vbatched_max(
        icla_uplo_t uplo, icla_trans_t trans,
        icla_int_t* n, icla_int_t* k,
        iclaDoubleComplex alpha,
        iclaDoubleComplex const * const * dA_array, icla_int_t* ldda,
        iclaDoubleComplex beta,
        iclaDoubleComplex **dC_array, icla_int_t* lddc,
        icla_int_t batchCount,
        icla_int_t max_n, icla_int_t max_k, icla_queue_t queue );

void
iclablas_zsyrk_vbatched_nocheck(
        icla_uplo_t uplo, icla_trans_t trans,
        icla_int_t* n, icla_int_t* k,
        iclaDoubleComplex alpha,
        iclaDoubleComplex const * const * dA_array, icla_int_t* ldda,
        iclaDoubleComplex beta,
        iclaDoubleComplex **dC_array, icla_int_t* lddc,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_zsyrk_vbatched(
        icla_uplo_t uplo, icla_trans_t trans,
        icla_int_t* n, icla_int_t* k,
        iclaDoubleComplex alpha,
        iclaDoubleComplex const * const * dA_array, icla_int_t* ldda,
        iclaDoubleComplex beta,
        iclaDoubleComplex **dC_array, icla_int_t* lddc,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_zher2k_vbatched_max_nocheck(
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t* n, icla_int_t* k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex const * const * dA_array, icla_int_t* ldda,
    iclaDoubleComplex const * const * dB_array, icla_int_t* lddb,
    double beta, iclaDoubleComplex **dC_array, icla_int_t* lddc,
    icla_int_t batchCount,
    icla_int_t max_n, icla_int_t max_k, icla_queue_t queue );

void
iclablas_zher2k_vbatched_max(
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t* n, icla_int_t* k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex const * const * dA_array, icla_int_t* ldda,
    iclaDoubleComplex const * const * dB_array, icla_int_t* lddb,
    double beta, iclaDoubleComplex **dC_array, icla_int_t* lddc,
    icla_int_t batchCount,
    icla_int_t max_n, icla_int_t max_k, icla_queue_t queue );

void
iclablas_zher2k_vbatched_nocheck(
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t* n, icla_int_t* k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex const * const * dA_array, icla_int_t* ldda,
    iclaDoubleComplex const * const * dB_array, icla_int_t* lddb,
    double beta, iclaDoubleComplex **dC_array, icla_int_t* lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_zher2k_vbatched(
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t* n, icla_int_t* k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex const * const * dA_array, icla_int_t* ldda,
    iclaDoubleComplex const * const * dB_array, icla_int_t* lddb,
    double beta, iclaDoubleComplex **dC_array, icla_int_t* lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_zsyr2k_vbatched_max_nocheck(
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t* n, icla_int_t* k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex const * const * dA_array, icla_int_t* ldda,
    iclaDoubleComplex const * const * dB_array, icla_int_t* lddb,
    iclaDoubleComplex beta,
    iclaDoubleComplex **dC_array, icla_int_t* lddc,
    icla_int_t batchCount,
    icla_int_t max_n, icla_int_t max_k, icla_queue_t queue );

void
iclablas_zsyr2k_vbatched_max(
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t* n, icla_int_t* k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex const * const * dA_array, icla_int_t* ldda,
    iclaDoubleComplex const * const * dB_array, icla_int_t* lddb,
    iclaDoubleComplex beta,
    iclaDoubleComplex **dC_array, icla_int_t* lddc,
    icla_int_t batchCount,
    icla_int_t max_n, icla_int_t max_k, icla_queue_t queue );

void
iclablas_zsyr2k_vbatched_nocheck(
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t* n, icla_int_t* k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex const * const * dA_array, icla_int_t* ldda,
    iclaDoubleComplex const * const * dB_array, icla_int_t* lddb,
    iclaDoubleComplex beta,
    iclaDoubleComplex **dC_array, icla_int_t* lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_zsyr2k_vbatched(
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t* n, icla_int_t* k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex const * const * dA_array, icla_int_t* ldda,
    iclaDoubleComplex const * const * dB_array, icla_int_t* lddb,
    iclaDoubleComplex beta,
    iclaDoubleComplex **dC_array, icla_int_t* lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_ztrmm_vbatched_core(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t max_m, icla_int_t max_n, icla_int_t* m, icla_int_t* n,
        iclaDoubleComplex alpha,
        iclaDoubleComplex **dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t* ldda,
        iclaDoubleComplex **dB_array, icla_int_t Bi, icla_int_t Bj, icla_int_t* lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_ztrmm_vbatched_max_nocheck(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t max_m, icla_int_t max_n, icla_int_t* m, icla_int_t* n,
        iclaDoubleComplex alpha,
        iclaDoubleComplex **dA_array, icla_int_t* ldda,
        iclaDoubleComplex **dB_array, icla_int_t* lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_ztrmm_vbatched_max(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t max_m, icla_int_t max_n, icla_int_t* m, icla_int_t* n,
        iclaDoubleComplex alpha,
        iclaDoubleComplex **dA_array, icla_int_t* ldda,
        iclaDoubleComplex **dB_array, icla_int_t* lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_ztrmm_vbatched_nocheck(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t* m, icla_int_t* n,
        iclaDoubleComplex alpha,
        iclaDoubleComplex **dA_array, icla_int_t* ldda,
        iclaDoubleComplex **dB_array, icla_int_t* lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_ztrmm_vbatched(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t* m, icla_int_t* n,
        iclaDoubleComplex alpha,
        iclaDoubleComplex **dA_array, icla_int_t* ldda,
        iclaDoubleComplex **dB_array, icla_int_t* lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_ztrsm_small_vbatched(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t max_m, icla_int_t max_n, icla_int_t* m, icla_int_t* n,
        iclaDoubleComplex alpha,
        iclaDoubleComplex **dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t* ldda,
        iclaDoubleComplex **dB_array, icla_int_t Bi, icla_int_t Bj, icla_int_t* lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_ztrsm_vbatched_core(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t max_m, icla_int_t max_n, icla_int_t* m, icla_int_t* n,
        iclaDoubleComplex alpha,
        iclaDoubleComplex **dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t* ldda,
        iclaDoubleComplex **dB_array, icla_int_t Bi, icla_int_t Bj, icla_int_t* lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_ztrsm_vbatched_max_nocheck(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t max_m, icla_int_t max_n, icla_int_t* m, icla_int_t* n,
        iclaDoubleComplex alpha,
        iclaDoubleComplex **dA_array, icla_int_t* ldda,
        iclaDoubleComplex **dB_array, icla_int_t* lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_ztrsm_vbatched_max(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t max_m, icla_int_t max_n, icla_int_t* m, icla_int_t* n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex** dA_array,    icla_int_t* ldda,
    iclaDoubleComplex** dB_array,    icla_int_t* lddb,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_ztrsm_vbatched(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t* m, icla_int_t* n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex** dA_array,    icla_int_t* ldda,
    iclaDoubleComplex** dB_array,    icla_int_t* lddb,
    icla_int_t batchCount, icla_queue_t queue);

void iclablas_ztrsm_inv_outofplace_vbatched(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t flag,
    icla_int_t *m, icla_int_t* n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex** dA_array,    icla_int_t* ldda,
    iclaDoubleComplex** dB_array,    icla_int_t* lddb,
    iclaDoubleComplex** dX_array,    icla_int_t* lddx,
    iclaDoubleComplex** dinvA_array, icla_int_t* dinvA_length,
    iclaDoubleComplex** dA_displ, iclaDoubleComplex** dB_displ,
    iclaDoubleComplex** dX_displ, iclaDoubleComplex** dinvA_displ,
    icla_int_t resetozero,
    icla_int_t batchCount,
    icla_int_t max_m, icla_int_t max_n,
    icla_queue_t queue);

void iclablas_ztrsm_inv_work_vbatched(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t flag,
    icla_int_t* m, icla_int_t* n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex** dA_array,    icla_int_t* ldda,
    iclaDoubleComplex** dB_array,    icla_int_t* lddb,
    iclaDoubleComplex** dX_array,    icla_int_t* lddx,
    iclaDoubleComplex** dinvA_array, icla_int_t* dinvA_length,
    iclaDoubleComplex** dA_displ, iclaDoubleComplex** dB_displ,
    iclaDoubleComplex** dX_displ, iclaDoubleComplex** dinvA_displ,
    icla_int_t resetozero,
    icla_int_t batchCount,
    icla_int_t max_m, icla_int_t max_n,
    icla_queue_t queue);

void iclablas_ztrsm_inv_vbatched_max_nocheck(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t* m, icla_int_t* n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex** dA_array,    icla_int_t* ldda,
    iclaDoubleComplex** dB_array,    icla_int_t* lddb,
    icla_int_t batchCount,
    icla_int_t max_m, icla_int_t max_n,
    icla_queue_t queue);

void
iclablas_ztrsm_inv_vbatched_max(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t* m, icla_int_t* n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex** dA_array,    icla_int_t* ldda,
    iclaDoubleComplex** dB_array,    icla_int_t* lddb,
    icla_int_t batchCount,
    icla_int_t max_m, icla_int_t max_n,
    icla_queue_t queue);

void
iclablas_ztrsm_inv_vbatched_nocheck(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t* m, icla_int_t* n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex** dA_array,    icla_int_t* ldda,
    iclaDoubleComplex** dB_array,    icla_int_t* lddb,
    icla_int_t batchCount,
    icla_queue_t queue);

void
iclablas_ztrsm_inv_vbatched(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t* m, icla_int_t* n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex** dA_array,    icla_int_t* ldda,
    iclaDoubleComplex** dB_array,    icla_int_t* lddb,
    icla_int_t batchCount,
    icla_queue_t queue);

void
iclablas_ztrtri_diag_vbatched(
    icla_uplo_t uplo, icla_diag_t diag, icla_int_t nmax, icla_int_t *n,
    iclaDoubleComplex const * const *dA_array, icla_int_t *ldda,
    iclaDoubleComplex **dinvA_array,
    icla_int_t resetozero, icla_int_t batchCount, icla_queue_t queue);

void
iclablas_zhemm_vbatched_core(
        icla_side_t side, icla_uplo_t uplo,
        icla_int_t *m, icla_int_t *n,
        iclaDoubleComplex alpha,
        iclaDoubleComplex **dA_array, icla_int_t *ldda,
        iclaDoubleComplex **dB_array, icla_int_t *lddb,
        iclaDoubleComplex beta,
        iclaDoubleComplex **dC_array, icla_int_t *lddc,
        icla_int_t max_m, icla_int_t max_n,
        icla_int_t roffA, icla_int_t coffA, icla_int_t roffB, icla_int_t coffB, icla_int_t roffC, icla_int_t coffC,
        icla_int_t specM, icla_int_t specN,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_zhemm_vbatched_max_nocheck(
        icla_side_t side, icla_uplo_t uplo,
        icla_int_t *m, icla_int_t *n,
        iclaDoubleComplex alpha,
        iclaDoubleComplex **dA_array, icla_int_t *ldda,
        iclaDoubleComplex **dB_array, icla_int_t *lddb,
        iclaDoubleComplex beta,
        iclaDoubleComplex **dC_array, icla_int_t *lddc,
        icla_int_t batchCount, icla_int_t max_m, icla_int_t max_n,
        icla_queue_t queue );

void
iclablas_zhemm_vbatched_max(
        icla_side_t side, icla_uplo_t uplo,
        icla_int_t *m, icla_int_t *n,
        iclaDoubleComplex alpha,
        iclaDoubleComplex **dA_array, icla_int_t *ldda,
        iclaDoubleComplex **dB_array, icla_int_t *lddb,
        iclaDoubleComplex beta,
        iclaDoubleComplex **dC_array, icla_int_t *lddc,
        icla_int_t batchCount, icla_int_t max_m, icla_int_t max_n,
        icla_queue_t queue );

void
iclablas_zhemm_vbatched_nocheck(
        icla_side_t side, icla_uplo_t uplo,
        icla_int_t *m, icla_int_t *n,
        iclaDoubleComplex alpha,
        iclaDoubleComplex **dA_array, icla_int_t *ldda,
        iclaDoubleComplex **dB_array, icla_int_t *lddb,
        iclaDoubleComplex beta,
        iclaDoubleComplex **dC_array, icla_int_t *lddc,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_zhemm_vbatched(
        icla_side_t side, icla_uplo_t uplo,
        icla_int_t *m, icla_int_t *n,
        iclaDoubleComplex alpha,
        iclaDoubleComplex **dA_array, icla_int_t *ldda,
        iclaDoubleComplex **dB_array, icla_int_t *lddb,
        iclaDoubleComplex beta,
        iclaDoubleComplex **dC_array, icla_int_t *lddc,
        icla_int_t batchCount, icla_queue_t queue );

/* Level 2 */
void
iclablas_zgemv_vbatched_max_nocheck(
    icla_trans_t trans, icla_int_t* m, icla_int_t* n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_ptr dA_array[], icla_int_t* ldda,
    iclaDoubleComplex_ptr dx_array[], icla_int_t* incx,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr dy_array[], icla_int_t* incy,
    icla_int_t batchCount,
    icla_int_t max_m, icla_int_t max_n, icla_queue_t queue);

void
iclablas_zgemv_vbatched_max(
    icla_trans_t trans, icla_int_t* m, icla_int_t* n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_ptr dA_array[], icla_int_t* ldda,
    iclaDoubleComplex_ptr dx_array[], icla_int_t* incx,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr dy_array[], icla_int_t* incy,
    icla_int_t batchCount,
    icla_int_t max_m, icla_int_t max_n, icla_queue_t queue);

void
iclablas_zgemv_vbatched_nocheck(
    icla_trans_t trans, icla_int_t* m, icla_int_t* n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_ptr dA_array[], icla_int_t* ldda,
    iclaDoubleComplex_ptr dx_array[], icla_int_t* incx,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr dy_array[], icla_int_t* incy,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_zgemv_vbatched(
    icla_trans_t trans, icla_int_t* m, icla_int_t* n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_ptr dA_array[], icla_int_t* ldda,
    iclaDoubleComplex_ptr dx_array[], icla_int_t* incx,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr dy_array[], icla_int_t* incy,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_zhemv_vbatched_max_nocheck(
    icla_uplo_t uplo, icla_int_t* n, iclaDoubleComplex alpha,
    iclaDoubleComplex **dA_array, icla_int_t* ldda,
    iclaDoubleComplex **dX_array, icla_int_t* incx,
    iclaDoubleComplex beta,
    iclaDoubleComplex **dY_array, icla_int_t* incy,
    icla_int_t max_n, icla_int_t batchCount, icla_queue_t queue );

void
iclablas_zhemv_vbatched_max(
    icla_uplo_t uplo, icla_int_t* n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_ptr dA_array[], icla_int_t* ldda,
    iclaDoubleComplex_ptr dx_array[], icla_int_t* incx,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr dy_array[], icla_int_t* incy,
    icla_int_t batchCount,
    icla_int_t max_n, icla_queue_t queue);

void
iclablas_zhemv_vbatched_nocheck(
    icla_uplo_t uplo, icla_int_t* n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_ptr dA_array[], icla_int_t* ldda,
    iclaDoubleComplex_ptr dx_array[], icla_int_t* incx,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr dy_array[], icla_int_t* incy,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_zhemv_vbatched(
    icla_uplo_t uplo, icla_int_t* n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_ptr dA_array[], icla_int_t* ldda,
    iclaDoubleComplex_ptr dx_array[], icla_int_t* incx,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr dy_array[], icla_int_t* incy,
    icla_int_t batchCount, icla_queue_t queue);
/* Level 1 */
/* Auxiliary routines */
void icla_zset_pointer_var_cc(
    iclaDoubleComplex **output_array,
    iclaDoubleComplex *input,
    icla_int_t *lda,
    icla_int_t row, icla_int_t column,
    icla_int_t *batch_offset,
    icla_int_t batchCount,
    icla_queue_t queue);

void
icla_zdisplace_pointers_var_cc(iclaDoubleComplex **output_array,
    iclaDoubleComplex **input_array, icla_int_t* lda,
    icla_int_t row, icla_int_t column,
    icla_int_t batchCount, icla_queue_t queue);

void
icla_zdisplace_pointers_var_cv(iclaDoubleComplex **output_array,
    iclaDoubleComplex **input_array, icla_int_t* lda,
    icla_int_t row, icla_int_t* column,
    icla_int_t batchCount, icla_queue_t queue);

void
icla_zdisplace_pointers_var_vc(iclaDoubleComplex **output_array,
    iclaDoubleComplex **input_array, icla_int_t* lda,
    icla_int_t *row, icla_int_t column,
    icla_int_t batchCount, icla_queue_t queue);

void
icla_zdisplace_pointers_var_vv(iclaDoubleComplex **output_array,
    iclaDoubleComplex **input_array, icla_int_t* lda,
    icla_int_t* row, icla_int_t* column,
    icla_int_t batchCount, icla_queue_t queue);

void iclablas_zlaset_vbatched(
    icla_uplo_t uplo, icla_int_t max_m, icla_int_t max_n,
    icla_int_t* m, icla_int_t* n,
    iclaDoubleComplex offdiag, iclaDoubleComplex diag,
    iclaDoubleComplex_ptr dAarray[], icla_int_t* ldda,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_zlacpy_vbatched(
    icla_uplo_t uplo,
    icla_int_t max_m, icla_int_t max_n,
    icla_int_t* m, icla_int_t* n,
    iclaDoubleComplex const * const * dAarray, icla_int_t* ldda,
    iclaDoubleComplex**               dBarray, icla_int_t* lddb,
    icla_int_t batchCount, icla_queue_t queue );

  /*
   *  Aux. vbatched routines
   */
icla_int_t icla_get_zpotrf_vbatched_crossover();

#ifdef __cplusplus
}
#endif

#undef ICLA_COMPLEX

#endif  /* ICLA_ZVBATCHED_H */
