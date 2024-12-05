

#ifndef ICLA_DVBATCHED_H
#define ICLA_DVBATCHED_H

#include "icla_types.h"

#define ICLA_REAL

#ifdef __cplusplus
extern "C" {
#endif


void icla_get_dgetrf_vbatched_nbparam(icla_int_t max_m, icla_int_t max_n, icla_int_t *nb, icla_int_t *recnb);


icla_int_t
icla_dgetf2_fused_vbatched(
    icla_int_t max_M, icla_int_t max_N,
    icla_int_t max_minMN, icla_int_t max_MxN,
    icla_int_t* M, icla_int_t* N,
    double **dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t* ldda,
    icla_int_t **dipiv_array, icla_int_t ipiv_i,
    icla_int_t *info_array, icla_int_t batchCount,
    icla_queue_t queue);

icla_int_t
icla_dgetf2_fused_sm_vbatched(
    icla_int_t max_M, icla_int_t max_N, icla_int_t max_minMN, icla_int_t max_MxN,
    icla_int_t* m, icla_int_t* n,
    double** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t* ldda,
    icla_int_t** dipiv_array, icla_int_t ipiv_i,
    icla_int_t* info_array, icla_int_t gbstep,
    icla_int_t nthreads, icla_int_t check_launch_only,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_dgetrf_vbatched(
        icla_int_t* m, icla_int_t* n,
        double **dA_array, icla_int_t *ldda,
        icla_int_t **ipiv_array, icla_int_t *info_array,
        icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dgetrf_vbatched_max_nocheck(
        icla_int_t* m, icla_int_t* n, icla_int_t* minmn,
        icla_int_t max_m, icla_int_t max_n, icla_int_t max_minmn, icla_int_t max_mxn,
        icla_int_t nb, icla_int_t recnb,
        double **dA_array, icla_int_t *ldda,
        icla_int_t **ipiv_array, icla_int_t** pivinfo_array,
        icla_int_t *info_array, icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dgetrf_vbatched_max_nocheck_work(
        icla_int_t* m, icla_int_t* n,
        icla_int_t max_m, icla_int_t max_n, icla_int_t max_minmn, icla_int_t max_mxn,
        double **dA_array, icla_int_t *ldda,
        icla_int_t **dipiv_array, icla_int_t *info_array,
        void* work, icla_int_t* lwork,
        icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_idamax_vbatched(
        icla_int_t length, icla_int_t *M, icla_int_t *N,
        double **dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t* ldda,
        icla_int_t** ipiv_array, icla_int_t ipiv_i,
        icla_int_t *info_array, icla_int_t step, icla_int_t gbstep,
        icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dswap_vbatched(
        icla_int_t max_n, icla_int_t *M, icla_int_t *N,
        double **dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t *ldda,
        icla_int_t** ipiv_array, icla_int_t piv_adjustment,
        icla_int_t batchCount, icla_queue_t queue);

icla_int_t icla_dscal_dger_vbatched(
    icla_int_t max_M, icla_int_t max_N,
    icla_int_t *M, icla_int_t *N,
    double **dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t *ldda,
    icla_int_t *info_array, icla_int_t step, icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dgetf2_vbatched(
    icla_int_t *m, icla_int_t *n, icla_int_t *minmn,
    icla_int_t max_m, icla_int_t max_n, icla_int_t max_minmn, icla_int_t max_mxn,
    double **dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t *ldda,
    icla_int_t **ipiv_array, icla_int_t *info_array,
    icla_int_t gbstep, icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dgetrf_recpanel_vbatched(
    icla_int_t* m, icla_int_t* n, icla_int_t* minmn,
    icla_int_t max_m, icla_int_t max_n, icla_int_t max_minmn,
    icla_int_t max_mxn, icla_int_t min_recpnb,
    double** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t* ldda,
    icla_int_t** dipiv_array, icla_int_t dipiv_i, icla_int_t** dpivinfo_array,
    icla_int_t *info_array, icla_int_t gbstep,
    icla_int_t batchCount,  icla_queue_t queue);

void
icla_dlaswp_left_rowserial_vbatched(
        icla_int_t n,
        icla_int_t *M, icla_int_t *N, double** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t *ldda,
        icla_int_t **ipiv_array, icla_int_t ipiv_offset,
        icla_int_t k1, icla_int_t k2,
        icla_int_t batchCount, icla_queue_t queue);

void
icla_dlaswp_right_rowserial_vbatched(
        icla_int_t n,
        icla_int_t *M, icla_int_t *N, double** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t *ldda,
        icla_int_t **ipiv_array, icla_int_t ipiv_offset,
        icla_int_t k1, icla_int_t k2,
        icla_int_t batchCount, icla_queue_t queue);

void
icla_dlaswp_left_rowparallel_vbatched(
        icla_int_t n,
        icla_int_t* M, icla_int_t* N,
        double** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t* ldda,
        icla_int_t k1, icla_int_t k2,
        icla_int_t **pivinfo_array, icla_int_t pivinfo_i,
        icla_int_t batchCount, icla_queue_t queue);

void
icla_dlaswp_right_rowparallel_vbatched(
        icla_int_t n,
        icla_int_t* M, icla_int_t* N,
        double** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t* ldda,
        icla_int_t k1, icla_int_t k2,
        icla_int_t **pivinfo_array, icla_int_t pivinfo_i,
        icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dpotrf_lpout_vbatched(
    icla_uplo_t uplo, icla_int_t *n, icla_int_t max_n,
    double **dA_array, icla_int_t *lda, icla_int_t gbstep,
    icla_int_t *info_array, icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dpotf2_vbatched(
    icla_uplo_t uplo, icla_int_t* n, icla_int_t max_n,
    double **dA_array, icla_int_t* lda,
    double **dA_displ,
    double **dW_displ,
    double **dB_displ,
    double **dC_displ,
    icla_int_t *info_array, icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dpotrf_panel_vbatched(
    icla_uplo_t uplo, icla_int_t* n, icla_int_t max_n,
    icla_int_t *ibvec, icla_int_t nb,
    double** dA_array,    icla_int_t* ldda,
    double** dX_array,    icla_int_t* dX_length,
    double** dinvA_array, icla_int_t* dinvA_length,
    double** dW0_displ, double** dW1_displ,
    double** dW2_displ, double** dW3_displ,
    double** dW4_displ,
    icla_int_t *info_array, icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dpotrf_vbatched_max_nocheck(
    icla_uplo_t uplo, icla_int_t *n,
    double **dA_array, icla_int_t *ldda,
    icla_int_t *info_array,  icla_int_t batchCount,
    icla_int_t max_n, icla_queue_t queue);

icla_int_t
icla_dpotrf_vbatched(
    icla_uplo_t uplo, icla_int_t *n,
    double **dA_array, icla_int_t *ldda,
    icla_int_t *info_array,  icla_int_t batchCount,
    icla_queue_t queue);

void
iclablas_dgemm_vbatched_core(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t max_m, icla_int_t max_n, icla_int_t max_k,
    icla_int_t* m, icla_int_t* n, icla_int_t* k,
    double alpha,
    double const * const * dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t* ldda,
    double const * const * dB_array, icla_int_t Bi, icla_int_t Bj, icla_int_t* lddb,
    double beta,
    double              ** dC_array, icla_int_t Ci, icla_int_t Cj, icla_int_t* lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_dgemm_vbatched_max_nocheck(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t* m, icla_int_t* n, icla_int_t* k,
    double alpha,
    double const * const * dA_array, icla_int_t* ldda,
    double const * const * dB_array, icla_int_t* lddb,
    double beta,
    double **dC_array, icla_int_t* lddc,
    icla_int_t max_m, icla_int_t max_n, icla_int_t max_k,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_dgemm_vbatched_max(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t* m, icla_int_t* n, icla_int_t* k,
    double alpha,
    double const * const * dA_array, icla_int_t* ldda,
    double const * const * dB_array, icla_int_t* lddb,
    double beta,
    double **dC_array, icla_int_t* lddc,
    icla_int_t max_m, icla_int_t max_n, icla_int_t max_k,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_dgemm_vbatched_nocheck(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t* m, icla_int_t* n, icla_int_t* k,
    double alpha,
    double const * const * dA_array, icla_int_t* ldda,
    double const * const * dB_array, icla_int_t* lddb,
    double beta,
    double **dC_array, icla_int_t* lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_dgemm_vbatched(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t* m, icla_int_t* n, icla_int_t* k,
    double alpha,
    double const * const * dA_array, icla_int_t* ldda,
    double const * const * dB_array, icla_int_t* lddb,
    double beta,
    double **dC_array, icla_int_t* lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_dsyrk_internal_vbatched(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t* n, icla_int_t* k,
    double alpha,
    double const * const * dA_array, icla_int_t* ldda,
    double const * const * dB_array, icla_int_t* lddb,
    double beta,
    double **dC_array, icla_int_t* lddc,
    icla_int_t max_n, icla_int_t max_k,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_dsyrk_internal_vbatched(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t* n, icla_int_t* k,
    double alpha,
    double const * const * dA_array, icla_int_t* ldda,
    double const * const * dB_array, icla_int_t* lddb,
    double beta,
    double **dC_array, icla_int_t* lddc,
    icla_int_t max_n, icla_int_t max_k,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_dsyrk_vbatched_max_nocheck(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t* n, icla_int_t* k,
    double alpha,
    double const * const * dA_array, icla_int_t* ldda,
    double beta,
    double **dC_array, icla_int_t* lddc,
    icla_int_t batchCount,
    icla_int_t max_n, icla_int_t max_k, icla_queue_t queue );

void
iclablas_dsyrk_vbatched_max(
        icla_uplo_t uplo, icla_trans_t trans,
        icla_int_t* n, icla_int_t* k,
        double alpha,
        double const * const * dA_array, icla_int_t* ldda,
        double beta,
        double **dC_array, icla_int_t* lddc,
        icla_int_t batchCount,
        icla_int_t max_n, icla_int_t max_k, icla_queue_t queue );

void
iclablas_dsyrk_vbatched_nocheck(
        icla_uplo_t uplo, icla_trans_t trans,
        icla_int_t* n, icla_int_t* k,
        double alpha,
        double const * const * dA_array, icla_int_t* ldda,
        double beta,
        double **dC_array, icla_int_t* lddc,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_dsyrk_vbatched(
        icla_uplo_t uplo, icla_trans_t trans,
        icla_int_t* n, icla_int_t* k,
        double alpha,
        double const * const * dA_array, icla_int_t* ldda,
        double beta,
        double **dC_array, icla_int_t* lddc,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_dsyrk_vbatched_max_nocheck(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t* n, icla_int_t* k,
    double alpha,
    double const * const * dA_array, icla_int_t* ldda,
    double beta,
    double **dC_array, icla_int_t* lddc,
    icla_int_t batchCount,
    icla_int_t max_n, icla_int_t max_k, icla_queue_t queue );

void
iclablas_dsyrk_vbatched_max(
        icla_uplo_t uplo, icla_trans_t trans,
        icla_int_t* n, icla_int_t* k,
        double alpha,
        double const * const * dA_array, icla_int_t* ldda,
        double beta,
        double **dC_array, icla_int_t* lddc,
        icla_int_t batchCount,
        icla_int_t max_n, icla_int_t max_k, icla_queue_t queue );

void
iclablas_dsyrk_vbatched_nocheck(
        icla_uplo_t uplo, icla_trans_t trans,
        icla_int_t* n, icla_int_t* k,
        double alpha,
        double const * const * dA_array, icla_int_t* ldda,
        double beta,
        double **dC_array, icla_int_t* lddc,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_dsyrk_vbatched(
        icla_uplo_t uplo, icla_trans_t trans,
        icla_int_t* n, icla_int_t* k,
        double alpha,
        double const * const * dA_array, icla_int_t* ldda,
        double beta,
        double **dC_array, icla_int_t* lddc,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_dsyr2k_vbatched_max_nocheck(
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t* n, icla_int_t* k,
    double alpha,
    double const * const * dA_array, icla_int_t* ldda,
    double const * const * dB_array, icla_int_t* lddb,
    double beta, double **dC_array, icla_int_t* lddc,
    icla_int_t batchCount,
    icla_int_t max_n, icla_int_t max_k, icla_queue_t queue );

void
iclablas_dsyr2k_vbatched_max(
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t* n, icla_int_t* k,
    double alpha,
    double const * const * dA_array, icla_int_t* ldda,
    double const * const * dB_array, icla_int_t* lddb,
    double beta, double **dC_array, icla_int_t* lddc,
    icla_int_t batchCount,
    icla_int_t max_n, icla_int_t max_k, icla_queue_t queue );

void
iclablas_dsyr2k_vbatched_nocheck(
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t* n, icla_int_t* k,
    double alpha,
    double const * const * dA_array, icla_int_t* ldda,
    double const * const * dB_array, icla_int_t* lddb,
    double beta, double **dC_array, icla_int_t* lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_dsyr2k_vbatched(
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t* n, icla_int_t* k,
    double alpha,
    double const * const * dA_array, icla_int_t* ldda,
    double const * const * dB_array, icla_int_t* lddb,
    double beta, double **dC_array, icla_int_t* lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_dsyr2k_vbatched_max_nocheck(
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t* n, icla_int_t* k,
    double alpha,
    double const * const * dA_array, icla_int_t* ldda,
    double const * const * dB_array, icla_int_t* lddb,
    double beta,
    double **dC_array, icla_int_t* lddc,
    icla_int_t batchCount,
    icla_int_t max_n, icla_int_t max_k, icla_queue_t queue );

void
iclablas_dsyr2k_vbatched_max(
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t* n, icla_int_t* k,
    double alpha,
    double const * const * dA_array, icla_int_t* ldda,
    double const * const * dB_array, icla_int_t* lddb,
    double beta,
    double **dC_array, icla_int_t* lddc,
    icla_int_t batchCount,
    icla_int_t max_n, icla_int_t max_k, icla_queue_t queue );

void
iclablas_dsyr2k_vbatched_nocheck(
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t* n, icla_int_t* k,
    double alpha,
    double const * const * dA_array, icla_int_t* ldda,
    double const * const * dB_array, icla_int_t* lddb,
    double beta,
    double **dC_array, icla_int_t* lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_dsyr2k_vbatched(
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t* n, icla_int_t* k,
    double alpha,
    double const * const * dA_array, icla_int_t* ldda,
    double const * const * dB_array, icla_int_t* lddb,
    double beta,
    double **dC_array, icla_int_t* lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_dtrmm_vbatched_core(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t max_m, icla_int_t max_n, icla_int_t* m, icla_int_t* n,
        double alpha,
        double **dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t* ldda,
        double **dB_array, icla_int_t Bi, icla_int_t Bj, icla_int_t* lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_dtrmm_vbatched_max_nocheck(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t max_m, icla_int_t max_n, icla_int_t* m, icla_int_t* n,
        double alpha,
        double **dA_array, icla_int_t* ldda,
        double **dB_array, icla_int_t* lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_dtrmm_vbatched_max(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t max_m, icla_int_t max_n, icla_int_t* m, icla_int_t* n,
        double alpha,
        double **dA_array, icla_int_t* ldda,
        double **dB_array, icla_int_t* lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_dtrmm_vbatched_nocheck(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t* m, icla_int_t* n,
        double alpha,
        double **dA_array, icla_int_t* ldda,
        double **dB_array, icla_int_t* lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_dtrmm_vbatched(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t* m, icla_int_t* n,
        double alpha,
        double **dA_array, icla_int_t* ldda,
        double **dB_array, icla_int_t* lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_dtrsm_small_vbatched(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t max_m, icla_int_t max_n, icla_int_t* m, icla_int_t* n,
        double alpha,
        double **dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t* ldda,
        double **dB_array, icla_int_t Bi, icla_int_t Bj, icla_int_t* lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_dtrsm_vbatched_core(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t max_m, icla_int_t max_n, icla_int_t* m, icla_int_t* n,
        double alpha,
        double **dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t* ldda,
        double **dB_array, icla_int_t Bi, icla_int_t Bj, icla_int_t* lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_dtrsm_vbatched_max_nocheck(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t max_m, icla_int_t max_n, icla_int_t* m, icla_int_t* n,
        double alpha,
        double **dA_array, icla_int_t* ldda,
        double **dB_array, icla_int_t* lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_dtrsm_vbatched_max(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t max_m, icla_int_t max_n, icla_int_t* m, icla_int_t* n,
    double alpha,
    double** dA_array,    icla_int_t* ldda,
    double** dB_array,    icla_int_t* lddb,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_dtrsm_vbatched(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t* m, icla_int_t* n,
    double alpha,
    double** dA_array,    icla_int_t* ldda,
    double** dB_array,    icla_int_t* lddb,
    icla_int_t batchCount, icla_queue_t queue);

void iclablas_dtrsm_inv_outofplace_vbatched(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t flag,
    icla_int_t *m, icla_int_t* n,
    double alpha,
    double** dA_array,    icla_int_t* ldda,
    double** dB_array,    icla_int_t* lddb,
    double** dX_array,    icla_int_t* lddx,
    double** dinvA_array, icla_int_t* dinvA_length,
    double** dA_displ, double** dB_displ,
    double** dX_displ, double** dinvA_displ,
    icla_int_t resetozero,
    icla_int_t batchCount,
    icla_int_t max_m, icla_int_t max_n,
    icla_queue_t queue);

void iclablas_dtrsm_inv_work_vbatched(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t flag,
    icla_int_t* m, icla_int_t* n,
    double alpha,
    double** dA_array,    icla_int_t* ldda,
    double** dB_array,    icla_int_t* lddb,
    double** dX_array,    icla_int_t* lddx,
    double** dinvA_array, icla_int_t* dinvA_length,
    double** dA_displ, double** dB_displ,
    double** dX_displ, double** dinvA_displ,
    icla_int_t resetozero,
    icla_int_t batchCount,
    icla_int_t max_m, icla_int_t max_n,
    icla_queue_t queue);

void iclablas_dtrsm_inv_vbatched_max_nocheck(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t* m, icla_int_t* n,
    double alpha,
    double** dA_array,    icla_int_t* ldda,
    double** dB_array,    icla_int_t* lddb,
    icla_int_t batchCount,
    icla_int_t max_m, icla_int_t max_n,
    icla_queue_t queue);

void
iclablas_dtrsm_inv_vbatched_max(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t* m, icla_int_t* n,
    double alpha,
    double** dA_array,    icla_int_t* ldda,
    double** dB_array,    icla_int_t* lddb,
    icla_int_t batchCount,
    icla_int_t max_m, icla_int_t max_n,
    icla_queue_t queue);

void
iclablas_dtrsm_inv_vbatched_nocheck(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t* m, icla_int_t* n,
    double alpha,
    double** dA_array,    icla_int_t* ldda,
    double** dB_array,    icla_int_t* lddb,
    icla_int_t batchCount,
    icla_queue_t queue);

void
iclablas_dtrsm_inv_vbatched(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t* m, icla_int_t* n,
    double alpha,
    double** dA_array,    icla_int_t* ldda,
    double** dB_array,    icla_int_t* lddb,
    icla_int_t batchCount,
    icla_queue_t queue);

void
iclablas_dtrtri_diag_vbatched(
    icla_uplo_t uplo, icla_diag_t diag, icla_int_t nmax, icla_int_t *n,
    double const * const *dA_array, icla_int_t *ldda,
    double **dinvA_array,
    icla_int_t resetozero, icla_int_t batchCount, icla_queue_t queue);

void
iclablas_dsymm_vbatched_core(
        icla_side_t side, icla_uplo_t uplo,
        icla_int_t *m, icla_int_t *n,
        double alpha,
        double **dA_array, icla_int_t *ldda,
        double **dB_array, icla_int_t *lddb,
        double beta,
        double **dC_array, icla_int_t *lddc,
        icla_int_t max_m, icla_int_t max_n,
        icla_int_t roffA, icla_int_t coffA, icla_int_t roffB, icla_int_t coffB, icla_int_t roffC, icla_int_t coffC,
        icla_int_t specM, icla_int_t specN,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_dsymm_vbatched_max_nocheck(
        icla_side_t side, icla_uplo_t uplo,
        icla_int_t *m, icla_int_t *n,
        double alpha,
        double **dA_array, icla_int_t *ldda,
        double **dB_array, icla_int_t *lddb,
        double beta,
        double **dC_array, icla_int_t *lddc,
        icla_int_t batchCount, icla_int_t max_m, icla_int_t max_n,
        icla_queue_t queue );

void
iclablas_dsymm_vbatched_max(
        icla_side_t side, icla_uplo_t uplo,
        icla_int_t *m, icla_int_t *n,
        double alpha,
        double **dA_array, icla_int_t *ldda,
        double **dB_array, icla_int_t *lddb,
        double beta,
        double **dC_array, icla_int_t *lddc,
        icla_int_t batchCount, icla_int_t max_m, icla_int_t max_n,
        icla_queue_t queue );

void
iclablas_dsymm_vbatched_nocheck(
        icla_side_t side, icla_uplo_t uplo,
        icla_int_t *m, icla_int_t *n,
        double alpha,
        double **dA_array, icla_int_t *ldda,
        double **dB_array, icla_int_t *lddb,
        double beta,
        double **dC_array, icla_int_t *lddc,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_dsymm_vbatched(
        icla_side_t side, icla_uplo_t uplo,
        icla_int_t *m, icla_int_t *n,
        double alpha,
        double **dA_array, icla_int_t *ldda,
        double **dB_array, icla_int_t *lddb,
        double beta,
        double **dC_array, icla_int_t *lddc,
        icla_int_t batchCount, icla_queue_t queue );



void
iclablas_dgemv_vbatched_max_nocheck(
    icla_trans_t trans, icla_int_t* m, icla_int_t* n,
    double alpha,
    iclaDouble_ptr dA_array[], icla_int_t* ldda,
    iclaDouble_ptr dx_array[], icla_int_t* incx,
    double beta,
    iclaDouble_ptr dy_array[], icla_int_t* incy,
    icla_int_t batchCount,
    icla_int_t max_m, icla_int_t max_n, icla_queue_t queue);

void
iclablas_dgemv_vbatched_max(
    icla_trans_t trans, icla_int_t* m, icla_int_t* n,
    double alpha,
    iclaDouble_ptr dA_array[], icla_int_t* ldda,
    iclaDouble_ptr dx_array[], icla_int_t* incx,
    double beta,
    iclaDouble_ptr dy_array[], icla_int_t* incy,
    icla_int_t batchCount,
    icla_int_t max_m, icla_int_t max_n, icla_queue_t queue);

void
iclablas_dgemv_vbatched_nocheck(
    icla_trans_t trans, icla_int_t* m, icla_int_t* n,
    double alpha,
    iclaDouble_ptr dA_array[], icla_int_t* ldda,
    iclaDouble_ptr dx_array[], icla_int_t* incx,
    double beta,
    iclaDouble_ptr dy_array[], icla_int_t* incy,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_dgemv_vbatched(
    icla_trans_t trans, icla_int_t* m, icla_int_t* n,
    double alpha,
    iclaDouble_ptr dA_array[], icla_int_t* ldda,
    iclaDouble_ptr dx_array[], icla_int_t* incx,
    double beta,
    iclaDouble_ptr dy_array[], icla_int_t* incy,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_dsymv_vbatched_max_nocheck(
    icla_uplo_t uplo, icla_int_t* n, double alpha,
    double **dA_array, icla_int_t* ldda,
    double **dX_array, icla_int_t* incx,
    double beta,
    double **dY_array, icla_int_t* incy,
    icla_int_t max_n, icla_int_t batchCount, icla_queue_t queue );

void
iclablas_dsymv_vbatched_max(
    icla_uplo_t uplo, icla_int_t* n,
    double alpha,
    iclaDouble_ptr dA_array[], icla_int_t* ldda,
    iclaDouble_ptr dx_array[], icla_int_t* incx,
    double beta,
    iclaDouble_ptr dy_array[], icla_int_t* incy,
    icla_int_t batchCount,
    icla_int_t max_n, icla_queue_t queue);

void
iclablas_dsymv_vbatched_nocheck(
    icla_uplo_t uplo, icla_int_t* n,
    double alpha,
    iclaDouble_ptr dA_array[], icla_int_t* ldda,
    iclaDouble_ptr dx_array[], icla_int_t* incx,
    double beta,
    iclaDouble_ptr dy_array[], icla_int_t* incy,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_dsymv_vbatched(
    icla_uplo_t uplo, icla_int_t* n,
    double alpha,
    iclaDouble_ptr dA_array[], icla_int_t* ldda,
    iclaDouble_ptr dx_array[], icla_int_t* incx,
    double beta,
    iclaDouble_ptr dy_array[], icla_int_t* incy,
    icla_int_t batchCount, icla_queue_t queue);




void icla_dset_pointer_var_cc(
    double **output_array,
    double *input,
    icla_int_t *lda,
    icla_int_t row, icla_int_t column,
    icla_int_t *batch_offset,
    icla_int_t batchCount,
    icla_queue_t queue);

void
icla_ddisplace_pointers_var_cc(double **output_array,
    double **input_array, icla_int_t* lda,
    icla_int_t row, icla_int_t column,
    icla_int_t batchCount, icla_queue_t queue);

void
icla_ddisplace_pointers_var_cv(double **output_array,
    double **input_array, icla_int_t* lda,
    icla_int_t row, icla_int_t* column,
    icla_int_t batchCount, icla_queue_t queue);

void
icla_ddisplace_pointers_var_vc(double **output_array,
    double **input_array, icla_int_t* lda,
    icla_int_t *row, icla_int_t column,
    icla_int_t batchCount, icla_queue_t queue);

void
icla_ddisplace_pointers_var_vv(double **output_array,
    double **input_array, icla_int_t* lda,
    icla_int_t* row, icla_int_t* column,
    icla_int_t batchCount, icla_queue_t queue);

void iclablas_dlaset_vbatched(
    icla_uplo_t uplo, icla_int_t max_m, icla_int_t max_n,
    icla_int_t* m, icla_int_t* n,
    double offdiag, double diag,
    iclaDouble_ptr dAarray[], icla_int_t* ldda,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_dlacpy_vbatched(
    icla_uplo_t uplo,
    icla_int_t max_m, icla_int_t max_n,
    icla_int_t* m, icla_int_t* n,
    double const * const * dAarray, icla_int_t* ldda,
    double**               dBarray, icla_int_t* lddb,
    icla_int_t batchCount, icla_queue_t queue );


icla_int_t icla_get_dpotrf_vbatched_crossover();

#ifdef __cplusplus
}
#endif

#undef ICLA_REAL

#endif

