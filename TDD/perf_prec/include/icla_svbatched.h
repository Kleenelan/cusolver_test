

#ifndef ICLA_SVBATCHED_H
#define ICLA_SVBATCHED_H

#include "icla_types.h"

#define ICLA_REAL

#ifdef __cplusplus
extern "C" {
#endif


void icla_get_sgetrf_vbatched_nbparam(icla_int_t max_m, icla_int_t max_n, icla_int_t *nb, icla_int_t *recnb);



icla_int_t
icla_sgetf2_fused_vbatched(
    icla_int_t max_M, icla_int_t max_N,
    icla_int_t max_minMN, icla_int_t max_MxN,
    icla_int_t* M, icla_int_t* N,
    float **dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t* ldda,
    icla_int_t **dipiv_array, icla_int_t ipiv_i,
    icla_int_t *info_array, icla_int_t batchCount,
    icla_queue_t queue);

icla_int_t
icla_sgetf2_fused_sm_vbatched(
    icla_int_t max_M, icla_int_t max_N, icla_int_t max_minMN, icla_int_t max_MxN,
    icla_int_t* m, icla_int_t* n,
    float** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t* ldda,
    icla_int_t** dipiv_array, icla_int_t ipiv_i,
    icla_int_t* info_array, icla_int_t gbstep,
    icla_int_t nthreads, icla_int_t check_launch_only,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_sgetrf_vbatched(
        icla_int_t* m, icla_int_t* n,
        float **dA_array, icla_int_t *ldda,
        icla_int_t **ipiv_array, icla_int_t *info_array,
        icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_sgetrf_vbatched_max_nocheck(
        icla_int_t* m, icla_int_t* n, icla_int_t* minmn,
        icla_int_t max_m, icla_int_t max_n, icla_int_t max_minmn, icla_int_t max_mxn,
        icla_int_t nb, icla_int_t recnb,
        float **dA_array, icla_int_t *ldda,
        icla_int_t **ipiv_array, icla_int_t** pivinfo_array,
        icla_int_t *info_array, icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_sgetrf_vbatched_max_nocheck_work(
        icla_int_t* m, icla_int_t* n,
        icla_int_t max_m, icla_int_t max_n, icla_int_t max_minmn, icla_int_t max_mxn,
        float **dA_array, icla_int_t *ldda,
        icla_int_t **dipiv_array, icla_int_t *info_array,
        void* work, icla_int_t* lwork,
        icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_isamax_vbatched(
        icla_int_t length, icla_int_t *M, icla_int_t *N,
        float **dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t* ldda,
        icla_int_t** ipiv_array, icla_int_t ipiv_i,
        icla_int_t *info_array, icla_int_t step, icla_int_t gbstep,
        icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_sswap_vbatched(
        icla_int_t max_n, icla_int_t *M, icla_int_t *N,
        float **dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t *ldda,
        icla_int_t** ipiv_array, icla_int_t piv_adjustment,
        icla_int_t batchCount, icla_queue_t queue);

icla_int_t icla_sscal_sger_vbatched(
    icla_int_t max_M, icla_int_t max_N,
    icla_int_t *M, icla_int_t *N,
    float **dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t *ldda,
    icla_int_t *info_array, icla_int_t step, icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_sgetf2_vbatched(
    icla_int_t *m, icla_int_t *n, icla_int_t *minmn,
    icla_int_t max_m, icla_int_t max_n, icla_int_t max_minmn, icla_int_t max_mxn,
    float **dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t *ldda,
    icla_int_t **ipiv_array, icla_int_t *info_array,
    icla_int_t gbstep, icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_sgetrf_recpanel_vbatched(
    icla_int_t* m, icla_int_t* n, icla_int_t* minmn,
    icla_int_t max_m, icla_int_t max_n, icla_int_t max_minmn,
    icla_int_t max_mxn, icla_int_t min_recpnb,
    float** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t* ldda,
    icla_int_t** dipiv_array, icla_int_t dipiv_i, icla_int_t** dpivinfo_array,
    icla_int_t *info_array, icla_int_t gbstep,
    icla_int_t batchCount,  icla_queue_t queue);

void
icla_slaswp_left_rowserial_vbatched(
        icla_int_t n,
        icla_int_t *M, icla_int_t *N, float** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t *ldda,
        icla_int_t **ipiv_array, icla_int_t ipiv_offset,
        icla_int_t k1, icla_int_t k2,
        icla_int_t batchCount, icla_queue_t queue);

void
icla_slaswp_right_rowserial_vbatched(
        icla_int_t n,
        icla_int_t *M, icla_int_t *N, float** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t *ldda,
        icla_int_t **ipiv_array, icla_int_t ipiv_offset,
        icla_int_t k1, icla_int_t k2,
        icla_int_t batchCount, icla_queue_t queue);

void
icla_slaswp_left_rowparallel_vbatched(
        icla_int_t n,
        icla_int_t* M, icla_int_t* N,
        float** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t* ldda,
        icla_int_t k1, icla_int_t k2,
        icla_int_t **pivinfo_array, icla_int_t pivinfo_i,
        icla_int_t batchCount, icla_queue_t queue);

void
icla_slaswp_right_rowparallel_vbatched(
        icla_int_t n,
        icla_int_t* M, icla_int_t* N,
        float** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t* ldda,
        icla_int_t k1, icla_int_t k2,
        icla_int_t **pivinfo_array, icla_int_t pivinfo_i,
        icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_spotrf_lpout_vbatched(
    icla_uplo_t uplo, icla_int_t *n, icla_int_t max_n,
    float **dA_array, icla_int_t *lda, icla_int_t gbstep,
    icla_int_t *info_array, icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_spotf2_vbatched(
    icla_uplo_t uplo, icla_int_t* n, icla_int_t max_n,
    float **dA_array, icla_int_t* lda,
    float **dA_displ,
    float **dW_displ,
    float **dB_displ,
    float **dC_displ,
    icla_int_t *info_array, icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_spotrf_panel_vbatched(
    icla_uplo_t uplo, icla_int_t* n, icla_int_t max_n,
    icla_int_t *ibvec, icla_int_t nb,
    float** dA_array,    icla_int_t* ldda,
    float** dX_array,    icla_int_t* dX_length,
    float** dinvA_array, icla_int_t* dinvA_length,
    float** dW0_displ, float** dW1_displ,
    float** dW2_displ, float** dW3_displ,
    float** dW4_displ,
    icla_int_t *info_array, icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_spotrf_vbatched_max_nocheck(
    icla_uplo_t uplo, icla_int_t *n,
    float **dA_array, icla_int_t *ldda,
    icla_int_t *info_array,  icla_int_t batchCount,
    icla_int_t max_n, icla_queue_t queue);

icla_int_t
icla_spotrf_vbatched(
    icla_uplo_t uplo, icla_int_t *n,
    float **dA_array, icla_int_t *ldda,
    icla_int_t *info_array,  icla_int_t batchCount,
    icla_queue_t queue);

void
iclablas_sgemm_vbatched_core(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t max_m, icla_int_t max_n, icla_int_t max_k,
    icla_int_t* m, icla_int_t* n, icla_int_t* k,
    float alpha,
    float const * const * dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t* ldda,
    float const * const * dB_array, icla_int_t Bi, icla_int_t Bj, icla_int_t* lddb,
    float beta,
    float              ** dC_array, icla_int_t Ci, icla_int_t Cj, icla_int_t* lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_sgemm_vbatched_max_nocheck(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t* m, icla_int_t* n, icla_int_t* k,
    float alpha,
    float const * const * dA_array, icla_int_t* ldda,
    float const * const * dB_array, icla_int_t* lddb,
    float beta,
    float **dC_array, icla_int_t* lddc,
    icla_int_t max_m, icla_int_t max_n, icla_int_t max_k,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_sgemm_vbatched_max(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t* m, icla_int_t* n, icla_int_t* k,
    float alpha,
    float const * const * dA_array, icla_int_t* ldda,
    float const * const * dB_array, icla_int_t* lddb,
    float beta,
    float **dC_array, icla_int_t* lddc,
    icla_int_t max_m, icla_int_t max_n, icla_int_t max_k,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_sgemm_vbatched_nocheck(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t* m, icla_int_t* n, icla_int_t* k,
    float alpha,
    float const * const * dA_array, icla_int_t* ldda,
    float const * const * dB_array, icla_int_t* lddb,
    float beta,
    float **dC_array, icla_int_t* lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_sgemm_vbatched(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t* m, icla_int_t* n, icla_int_t* k,
    float alpha,
    float const * const * dA_array, icla_int_t* ldda,
    float const * const * dB_array, icla_int_t* lddb,
    float beta,
    float **dC_array, icla_int_t* lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_ssyrk_internal_vbatched(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t* n, icla_int_t* k,
    float alpha,
    float const * const * dA_array, icla_int_t* ldda,
    float const * const * dB_array, icla_int_t* lddb,
    float beta,
    float **dC_array, icla_int_t* lddc,
    icla_int_t max_n, icla_int_t max_k,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_ssyrk_internal_vbatched(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t* n, icla_int_t* k,
    float alpha,
    float const * const * dA_array, icla_int_t* ldda,
    float const * const * dB_array, icla_int_t* lddb,
    float beta,
    float **dC_array, icla_int_t* lddc,
    icla_int_t max_n, icla_int_t max_k,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_ssyrk_vbatched_max_nocheck(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t* n, icla_int_t* k,
    float alpha,
    float const * const * dA_array, icla_int_t* ldda,
    float beta,
    float **dC_array, icla_int_t* lddc,
    icla_int_t batchCount,
    icla_int_t max_n, icla_int_t max_k, icla_queue_t queue );

void
iclablas_ssyrk_vbatched_max(
        icla_uplo_t uplo, icla_trans_t trans,
        icla_int_t* n, icla_int_t* k,
        float alpha,
        float const * const * dA_array, icla_int_t* ldda,
        float beta,
        float **dC_array, icla_int_t* lddc,
        icla_int_t batchCount,
        icla_int_t max_n, icla_int_t max_k, icla_queue_t queue );

void
iclablas_ssyrk_vbatched_nocheck(
        icla_uplo_t uplo, icla_trans_t trans,
        icla_int_t* n, icla_int_t* k,
        float alpha,
        float const * const * dA_array, icla_int_t* ldda,
        float beta,
        float **dC_array, icla_int_t* lddc,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_ssyrk_vbatched(
        icla_uplo_t uplo, icla_trans_t trans,
        icla_int_t* n, icla_int_t* k,
        float alpha,
        float const * const * dA_array, icla_int_t* ldda,
        float beta,
        float **dC_array, icla_int_t* lddc,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_ssyrk_vbatched_max_nocheck(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t* n, icla_int_t* k,
    float alpha,
    float const * const * dA_array, icla_int_t* ldda,
    float beta,
    float **dC_array, icla_int_t* lddc,
    icla_int_t batchCount,
    icla_int_t max_n, icla_int_t max_k, icla_queue_t queue );

void
iclablas_ssyrk_vbatched_max(
        icla_uplo_t uplo, icla_trans_t trans,
        icla_int_t* n, icla_int_t* k,
        float alpha,
        float const * const * dA_array, icla_int_t* ldda,
        float beta,
        float **dC_array, icla_int_t* lddc,
        icla_int_t batchCount,
        icla_int_t max_n, icla_int_t max_k, icla_queue_t queue );

void
iclablas_ssyrk_vbatched_nocheck(
        icla_uplo_t uplo, icla_trans_t trans,
        icla_int_t* n, icla_int_t* k,
        float alpha,
        float const * const * dA_array, icla_int_t* ldda,
        float beta,
        float **dC_array, icla_int_t* lddc,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_ssyrk_vbatched(
        icla_uplo_t uplo, icla_trans_t trans,
        icla_int_t* n, icla_int_t* k,
        float alpha,
        float const * const * dA_array, icla_int_t* ldda,
        float beta,
        float **dC_array, icla_int_t* lddc,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_ssyr2k_vbatched_max_nocheck(
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t* n, icla_int_t* k,
    float alpha,
    float const * const * dA_array, icla_int_t* ldda,
    float const * const * dB_array, icla_int_t* lddb,
    float beta, float **dC_array, icla_int_t* lddc,
    icla_int_t batchCount,
    icla_int_t max_n, icla_int_t max_k, icla_queue_t queue );

void
iclablas_ssyr2k_vbatched_max(
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t* n, icla_int_t* k,
    float alpha,
    float const * const * dA_array, icla_int_t* ldda,
    float const * const * dB_array, icla_int_t* lddb,
    float beta, float **dC_array, icla_int_t* lddc,
    icla_int_t batchCount,
    icla_int_t max_n, icla_int_t max_k, icla_queue_t queue );

void
iclablas_ssyr2k_vbatched_nocheck(
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t* n, icla_int_t* k,
    float alpha,
    float const * const * dA_array, icla_int_t* ldda,
    float const * const * dB_array, icla_int_t* lddb,
    float beta, float **dC_array, icla_int_t* lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_ssyr2k_vbatched(
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t* n, icla_int_t* k,
    float alpha,
    float const * const * dA_array, icla_int_t* ldda,
    float const * const * dB_array, icla_int_t* lddb,
    float beta, float **dC_array, icla_int_t* lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_ssyr2k_vbatched_max_nocheck(
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t* n, icla_int_t* k,
    float alpha,
    float const * const * dA_array, icla_int_t* ldda,
    float const * const * dB_array, icla_int_t* lddb,
    float beta,
    float **dC_array, icla_int_t* lddc,
    icla_int_t batchCount,
    icla_int_t max_n, icla_int_t max_k, icla_queue_t queue );

void
iclablas_ssyr2k_vbatched_max(
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t* n, icla_int_t* k,
    float alpha,
    float const * const * dA_array, icla_int_t* ldda,
    float const * const * dB_array, icla_int_t* lddb,
    float beta,
    float **dC_array, icla_int_t* lddc,
    icla_int_t batchCount,
    icla_int_t max_n, icla_int_t max_k, icla_queue_t queue );

void
iclablas_ssyr2k_vbatched_nocheck(
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t* n, icla_int_t* k,
    float alpha,
    float const * const * dA_array, icla_int_t* ldda,
    float const * const * dB_array, icla_int_t* lddb,
    float beta,
    float **dC_array, icla_int_t* lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_ssyr2k_vbatched(
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t* n, icla_int_t* k,
    float alpha,
    float const * const * dA_array, icla_int_t* ldda,
    float const * const * dB_array, icla_int_t* lddb,
    float beta,
    float **dC_array, icla_int_t* lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_strmm_vbatched_core(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t max_m, icla_int_t max_n, icla_int_t* m, icla_int_t* n,
        float alpha,
        float **dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t* ldda,
        float **dB_array, icla_int_t Bi, icla_int_t Bj, icla_int_t* lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_strmm_vbatched_max_nocheck(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t max_m, icla_int_t max_n, icla_int_t* m, icla_int_t* n,
        float alpha,
        float **dA_array, icla_int_t* ldda,
        float **dB_array, icla_int_t* lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_strmm_vbatched_max(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t max_m, icla_int_t max_n, icla_int_t* m, icla_int_t* n,
        float alpha,
        float **dA_array, icla_int_t* ldda,
        float **dB_array, icla_int_t* lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_strmm_vbatched_nocheck(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t* m, icla_int_t* n,
        float alpha,
        float **dA_array, icla_int_t* ldda,
        float **dB_array, icla_int_t* lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_strmm_vbatched(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t* m, icla_int_t* n,
        float alpha,
        float **dA_array, icla_int_t* ldda,
        float **dB_array, icla_int_t* lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_strsm_small_vbatched(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t max_m, icla_int_t max_n, icla_int_t* m, icla_int_t* n,
        float alpha,
        float **dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t* ldda,
        float **dB_array, icla_int_t Bi, icla_int_t Bj, icla_int_t* lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_strsm_vbatched_core(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t max_m, icla_int_t max_n, icla_int_t* m, icla_int_t* n,
        float alpha,
        float **dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t* ldda,
        float **dB_array, icla_int_t Bi, icla_int_t Bj, icla_int_t* lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_strsm_vbatched_max_nocheck(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t max_m, icla_int_t max_n, icla_int_t* m, icla_int_t* n,
        float alpha,
        float **dA_array, icla_int_t* ldda,
        float **dB_array, icla_int_t* lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_strsm_vbatched_max(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t max_m, icla_int_t max_n, icla_int_t* m, icla_int_t* n,
    float alpha,
    float** dA_array,    icla_int_t* ldda,
    float** dB_array,    icla_int_t* lddb,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_strsm_vbatched(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t* m, icla_int_t* n,
    float alpha,
    float** dA_array,    icla_int_t* ldda,
    float** dB_array,    icla_int_t* lddb,
    icla_int_t batchCount, icla_queue_t queue);

void iclablas_strsm_inv_outofplace_vbatched(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t flag,
    icla_int_t *m, icla_int_t* n,
    float alpha,
    float** dA_array,    icla_int_t* ldda,
    float** dB_array,    icla_int_t* lddb,
    float** dX_array,    icla_int_t* lddx,
    float** dinvA_array, icla_int_t* dinvA_length,
    float** dA_displ, float** dB_displ,
    float** dX_displ, float** dinvA_displ,
    icla_int_t resetozero,
    icla_int_t batchCount,
    icla_int_t max_m, icla_int_t max_n,
    icla_queue_t queue);

void iclablas_strsm_inv_work_vbatched(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t flag,
    icla_int_t* m, icla_int_t* n,
    float alpha,
    float** dA_array,    icla_int_t* ldda,
    float** dB_array,    icla_int_t* lddb,
    float** dX_array,    icla_int_t* lddx,
    float** dinvA_array, icla_int_t* dinvA_length,
    float** dA_displ, float** dB_displ,
    float** dX_displ, float** dinvA_displ,
    icla_int_t resetozero,
    icla_int_t batchCount,
    icla_int_t max_m, icla_int_t max_n,
    icla_queue_t queue);

void iclablas_strsm_inv_vbatched_max_nocheck(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t* m, icla_int_t* n,
    float alpha,
    float** dA_array,    icla_int_t* ldda,
    float** dB_array,    icla_int_t* lddb,
    icla_int_t batchCount,
    icla_int_t max_m, icla_int_t max_n,
    icla_queue_t queue);

void
iclablas_strsm_inv_vbatched_max(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t* m, icla_int_t* n,
    float alpha,
    float** dA_array,    icla_int_t* ldda,
    float** dB_array,    icla_int_t* lddb,
    icla_int_t batchCount,
    icla_int_t max_m, icla_int_t max_n,
    icla_queue_t queue);

void
iclablas_strsm_inv_vbatched_nocheck(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t* m, icla_int_t* n,
    float alpha,
    float** dA_array,    icla_int_t* ldda,
    float** dB_array,    icla_int_t* lddb,
    icla_int_t batchCount,
    icla_queue_t queue);

void
iclablas_strsm_inv_vbatched(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t* m, icla_int_t* n,
    float alpha,
    float** dA_array,    icla_int_t* ldda,
    float** dB_array,    icla_int_t* lddb,
    icla_int_t batchCount,
    icla_queue_t queue);

void
iclablas_strtri_diag_vbatched(
    icla_uplo_t uplo, icla_diag_t diag, icla_int_t nmax, icla_int_t *n,
    float const * const *dA_array, icla_int_t *ldda,
    float **dinvA_array,
    icla_int_t resetozero, icla_int_t batchCount, icla_queue_t queue);

void
iclablas_ssymm_vbatched_core(
        icla_side_t side, icla_uplo_t uplo,
        icla_int_t *m, icla_int_t *n,
        float alpha,
        float **dA_array, icla_int_t *ldda,
        float **dB_array, icla_int_t *lddb,
        float beta,
        float **dC_array, icla_int_t *lddc,
        icla_int_t max_m, icla_int_t max_n,
        icla_int_t roffA, icla_int_t coffA, icla_int_t roffB, icla_int_t coffB, icla_int_t roffC, icla_int_t coffC,
        icla_int_t specM, icla_int_t specN,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_ssymm_vbatched_max_nocheck(
        icla_side_t side, icla_uplo_t uplo,
        icla_int_t *m, icla_int_t *n,
        float alpha,
        float **dA_array, icla_int_t *ldda,
        float **dB_array, icla_int_t *lddb,
        float beta,
        float **dC_array, icla_int_t *lddc,
        icla_int_t batchCount, icla_int_t max_m, icla_int_t max_n,
        icla_queue_t queue );

void
iclablas_ssymm_vbatched_max(
        icla_side_t side, icla_uplo_t uplo,
        icla_int_t *m, icla_int_t *n,
        float alpha,
        float **dA_array, icla_int_t *ldda,
        float **dB_array, icla_int_t *lddb,
        float beta,
        float **dC_array, icla_int_t *lddc,
        icla_int_t batchCount, icla_int_t max_m, icla_int_t max_n,
        icla_queue_t queue );

void
iclablas_ssymm_vbatched_nocheck(
        icla_side_t side, icla_uplo_t uplo,
        icla_int_t *m, icla_int_t *n,
        float alpha,
        float **dA_array, icla_int_t *ldda,
        float **dB_array, icla_int_t *lddb,
        float beta,
        float **dC_array, icla_int_t *lddc,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_ssymm_vbatched(
        icla_side_t side, icla_uplo_t uplo,
        icla_int_t *m, icla_int_t *n,
        float alpha,
        float **dA_array, icla_int_t *ldda,
        float **dB_array, icla_int_t *lddb,
        float beta,
        float **dC_array, icla_int_t *lddc,
        icla_int_t batchCount, icla_queue_t queue );



void
iclablas_sgemv_vbatched_max_nocheck(
    icla_trans_t trans, icla_int_t* m, icla_int_t* n,
    float alpha,
    iclaFloat_ptr dA_array[], icla_int_t* ldda,
    iclaFloat_ptr dx_array[], icla_int_t* incx,
    float beta,
    iclaFloat_ptr dy_array[], icla_int_t* incy,
    icla_int_t batchCount,
    icla_int_t max_m, icla_int_t max_n, icla_queue_t queue);

void
iclablas_sgemv_vbatched_max(
    icla_trans_t trans, icla_int_t* m, icla_int_t* n,
    float alpha,
    iclaFloat_ptr dA_array[], icla_int_t* ldda,
    iclaFloat_ptr dx_array[], icla_int_t* incx,
    float beta,
    iclaFloat_ptr dy_array[], icla_int_t* incy,
    icla_int_t batchCount,
    icla_int_t max_m, icla_int_t max_n, icla_queue_t queue);

void
iclablas_sgemv_vbatched_nocheck(
    icla_trans_t trans, icla_int_t* m, icla_int_t* n,
    float alpha,
    iclaFloat_ptr dA_array[], icla_int_t* ldda,
    iclaFloat_ptr dx_array[], icla_int_t* incx,
    float beta,
    iclaFloat_ptr dy_array[], icla_int_t* incy,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_sgemv_vbatched(
    icla_trans_t trans, icla_int_t* m, icla_int_t* n,
    float alpha,
    iclaFloat_ptr dA_array[], icla_int_t* ldda,
    iclaFloat_ptr dx_array[], icla_int_t* incx,
    float beta,
    iclaFloat_ptr dy_array[], icla_int_t* incy,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_ssymv_vbatched_max_nocheck(
    icla_uplo_t uplo, icla_int_t* n, float alpha,
    float **dA_array, icla_int_t* ldda,
    float **dX_array, icla_int_t* incx,
    float beta,
    float **dY_array, icla_int_t* incy,
    icla_int_t max_n, icla_int_t batchCount, icla_queue_t queue );

void
iclablas_ssymv_vbatched_max(
    icla_uplo_t uplo, icla_int_t* n,
    float alpha,
    iclaFloat_ptr dA_array[], icla_int_t* ldda,
    iclaFloat_ptr dx_array[], icla_int_t* incx,
    float beta,
    iclaFloat_ptr dy_array[], icla_int_t* incy,
    icla_int_t batchCount,
    icla_int_t max_n, icla_queue_t queue);

void
iclablas_ssymv_vbatched_nocheck(
    icla_uplo_t uplo, icla_int_t* n,
    float alpha,
    iclaFloat_ptr dA_array[], icla_int_t* ldda,
    iclaFloat_ptr dx_array[], icla_int_t* incx,
    float beta,
    iclaFloat_ptr dy_array[], icla_int_t* incy,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_ssymv_vbatched(
    icla_uplo_t uplo, icla_int_t* n,
    float alpha,
    iclaFloat_ptr dA_array[], icla_int_t* ldda,
    iclaFloat_ptr dx_array[], icla_int_t* incx,
    float beta,
    iclaFloat_ptr dy_array[], icla_int_t* incy,
    icla_int_t batchCount, icla_queue_t queue);




void icla_sset_pointer_var_cc(
    float **output_array,
    float *input,
    icla_int_t *lda,
    icla_int_t row, icla_int_t column,
    icla_int_t *batch_offset,
    icla_int_t batchCount,
    icla_queue_t queue);

void
icla_sdisplace_pointers_var_cc(float **output_array,
    float **input_array, icla_int_t* lda,
    icla_int_t row, icla_int_t column,
    icla_int_t batchCount, icla_queue_t queue);

void
icla_sdisplace_pointers_var_cv(float **output_array,
    float **input_array, icla_int_t* lda,
    icla_int_t row, icla_int_t* column,
    icla_int_t batchCount, icla_queue_t queue);

void
icla_sdisplace_pointers_var_vc(float **output_array,
    float **input_array, icla_int_t* lda,
    icla_int_t *row, icla_int_t column,
    icla_int_t batchCount, icla_queue_t queue);

void
icla_sdisplace_pointers_var_vv(float **output_array,
    float **input_array, icla_int_t* lda,
    icla_int_t* row, icla_int_t* column,
    icla_int_t batchCount, icla_queue_t queue);

void iclablas_slaset_vbatched(
    icla_uplo_t uplo, icla_int_t max_m, icla_int_t max_n,
    icla_int_t* m, icla_int_t* n,
    float offdiag, float diag,
    iclaFloat_ptr dAarray[], icla_int_t* ldda,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_slacpy_vbatched(
    icla_uplo_t uplo,
    icla_int_t max_m, icla_int_t max_n,
    icla_int_t* m, icla_int_t* n,
    float const * const * dAarray, icla_int_t* ldda,
    float**               dBarray, icla_int_t* lddb,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t icla_get_spotrf_vbatched_crossover();

#ifdef __cplusplus
}
#endif

#undef ICLA_REAL

#endif

