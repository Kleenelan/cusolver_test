
#ifndef ICLA_CBATCHED_H
#define ICLA_CBATCHED_H

#include "icla_types.h"

#define ICLA_COMPLEX

#ifdef __cplusplus
extern "C" {
#endif

void
icla_cset_pointer(
    iclaFloatComplex **output_array,
    iclaFloatComplex *input,
    icla_int_t lda,
    icla_int_t row, icla_int_t column,
    icla_int_t batch_offset,
    icla_int_t batchCount, icla_queue_t queue);

void
icla_cdisplace_pointers(
    iclaFloatComplex **output_array,
    iclaFloatComplex **input_array, icla_int_t lda,
    icla_int_t row, icla_int_t column,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_crecommend_cublas_gemm_batched(
    icla_trans_t transa, icla_trans_t transb,
    icla_int_t m, icla_int_t n, icla_int_t k);

icla_int_t
icla_crecommend_cublas_gemm_stream(
    icla_trans_t transa, icla_trans_t transb,
    icla_int_t m, icla_int_t n, icla_int_t k);

void icla_get_cpotrf_batched_nbparam(icla_int_t n, icla_int_t *nb, icla_int_t *recnb);

icla_int_t icla_get_cpotrf_batched_crossover();

void icla_get_cgetrf_batched_nbparam(icla_int_t n, icla_int_t *nb, icla_int_t *recnb);
icla_int_t icla_get_cgetrf_batched_ntcol(icla_int_t m, icla_int_t n);
icla_int_t icla_get_cgemm_batched_ntcol(icla_int_t n);
icla_int_t icla_get_cgemm_batched_smallsq_limit(icla_int_t n);
icla_int_t icla_get_cgeqrf_batched_nb(icla_int_t m);
icla_int_t icla_use_cgeqrf_batched_fused_update(icla_int_t m, icla_int_t n, icla_int_t batchCount);
icla_int_t icla_get_cgeqr2_fused_sm_batched_nthreads(icla_int_t m, icla_int_t n);
icla_int_t icla_get_cgeqrf_batched_ntcol(icla_int_t m, icla_int_t n);
icla_int_t icla_get_cgetri_batched_ntcol(icla_int_t m, icla_int_t n);
icla_int_t icla_get_ctrsm_batched_stop_nb(icla_side_t side, icla_int_t m, icla_int_t n);
void icla_get_cgbtrf_batched_params(icla_int_t m, icla_int_t n, icla_int_t kl, icla_int_t ku, icla_int_t *nb, icla_int_t *threads);

void
iclablas_cswapdblk_batched(
    icla_int_t n, icla_int_t nb,
    iclaFloatComplex **dA, icla_int_t ldda, icla_int_t inca,
    iclaFloatComplex **dB, icla_int_t lddb, icla_int_t incb,
    icla_int_t batchCount, icla_queue_t queue );


void
iclablas_cgemm_batched_core(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaFloatComplex alpha,
    iclaFloatComplex const * const * dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    iclaFloatComplex const * const * dB_array, icla_int_t Bi, icla_int_t Bj, icla_int_t lddb,
    iclaFloatComplex beta,
    iclaFloatComplex **dC_array, icla_int_t Ci, icla_int_t Cj, icla_int_t lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
icla_cgemm_batched_core(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaFloatComplex alpha,
    iclaFloatComplex const * const * dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    iclaFloatComplex const * const * dB_array, icla_int_t Bi, icla_int_t Bj, icla_int_t lddb,
    iclaFloatComplex beta,
    iclaFloatComplex **dC_array, icla_int_t Ci, icla_int_t Cj, icla_int_t lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
icla_cgemm_batched(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaFloatComplex alpha,
    iclaFloatComplex const * const * dA_array, icla_int_t ldda,
    iclaFloatComplex const * const * dB_array, icla_int_t lddb,
    iclaFloatComplex beta,
    iclaFloatComplex **dC_array, icla_int_t lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_cgemm_batched(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaFloatComplex alpha,
    iclaFloatComplex const * const * dA_array, icla_int_t ldda,
    iclaFloatComplex const * const * dB_array, icla_int_t lddb,
    iclaFloatComplex beta,
    iclaFloatComplex **dC_array, icla_int_t lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_cgemm_batched_strided(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaFloatComplex alpha,
    iclaFloatComplex const * dA, icla_int_t ldda, icla_int_t strideA,
    iclaFloatComplex const * dB, icla_int_t lddb, icla_int_t strideB,
    iclaFloatComplex beta,
    iclaFloatComplex       * dC, icla_int_t lddc, icla_int_t strideC,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_cgemm_batched_smallsq(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaFloatComplex alpha,
    iclaFloatComplex const * const * dA_array, icla_int_t ai, icla_int_t aj, icla_int_t ldda,
    iclaFloatComplex const * const * dB_array, icla_int_t bi, icla_int_t bj, icla_int_t lddb,
    iclaFloatComplex beta,
    iclaFloatComplex **dC_array, icla_int_t ci, icla_int_t cj, icla_int_t lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_csyrk_batched_core(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    iclaFloatComplex alpha,
    iclaFloatComplex const * const * dA_array, icla_int_t ai, icla_int_t aj, icla_int_t ldda,
    iclaFloatComplex const * const * dB_array, icla_int_t bi, icla_int_t bj, icla_int_t lddb,
    iclaFloatComplex beta,
    iclaFloatComplex **dC_array, icla_int_t ci, icla_int_t cj, icla_int_t lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_cherk_batched_core(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    iclaFloatComplex alpha,
    iclaFloatComplex const * const * dA_array, icla_int_t ai, icla_int_t aj, icla_int_t ldda,
    iclaFloatComplex const * const * dB_array, icla_int_t bi, icla_int_t bj, icla_int_t lddb,
    iclaFloatComplex beta,
    iclaFloatComplex **dC_array, icla_int_t ci, icla_int_t cj, icla_int_t lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_csyrk_batched(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    iclaFloatComplex alpha,
    iclaFloatComplex const * const * dA_array, icla_int_t ldda,
    iclaFloatComplex beta,
    iclaFloatComplex **dC_array, icla_int_t lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
icla_cherk_batched(
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t n, icla_int_t k,
    float alpha,
    iclaFloatComplex const * const * dA_array, icla_int_t ldda,
    float beta,
    iclaFloatComplex **dC_array, icla_int_t lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_cherk_batched(
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t n, icla_int_t k,
    float alpha,
    iclaFloatComplex const * const * dA_array, icla_int_t ldda,
    float beta,
    iclaFloatComplex **dC_array, icla_int_t lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_cher2k_batched(
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t n, icla_int_t k,
    iclaFloatComplex alpha,
    iclaFloatComplex const * const * dA_array, icla_int_t ldda,
    iclaFloatComplex const * const * dB_array, icla_int_t lddb,
    float beta, iclaFloatComplex **dC_array, icla_int_t lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_csyr2k_batched(
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t n, icla_int_t k,
    iclaFloatComplex alpha,
    iclaFloatComplex const * const * dA_array, icla_int_t ldda,
    iclaFloatComplex const * const * dB_array, icla_int_t lddb,
    iclaFloatComplex beta, iclaFloatComplex **dC_array, icla_int_t lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_ctrtri_diag_batched(
    icla_uplo_t uplo, icla_diag_t diag, icla_int_t n,
    iclaFloatComplex const * const *dA_array, icla_int_t ldda,
    iclaFloatComplex **dinvA_array,
    icla_int_t resetozero,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_ctrsm_small_batched(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t m, icla_int_t n,
        iclaFloatComplex alpha,
        iclaFloatComplex **dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
        iclaFloatComplex **dB_array, icla_int_t Bi, icla_int_t Bj, icla_int_t lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_ctrsm_recursive_batched(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t m, icla_int_t n,
        iclaFloatComplex alpha,
        iclaFloatComplex **dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
        iclaFloatComplex **dB_array, icla_int_t Bi, icla_int_t Bj, icla_int_t lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_ctrsm_batched(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t m, icla_int_t n,
        iclaFloatComplex alpha,
        iclaFloatComplex **dA_array, icla_int_t ldda,
        iclaFloatComplex **dB_array, icla_int_t lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_ctrsm_inv_batched(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex** dA_array,    icla_int_t ldda,
    iclaFloatComplex** dB_array,    icla_int_t lddb,
    icla_int_t batchCount, icla_queue_t queue);

void iclablas_ctrsm_inv_work_batched(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t flag, icla_int_t m, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex** dA_array,    icla_int_t ldda,
    iclaFloatComplex** dB_array,    icla_int_t lddb,
    iclaFloatComplex** dX_array,    icla_int_t lddx,
    iclaFloatComplex** dinvA_array, icla_int_t dinvA_length,
    iclaFloatComplex** dA_displ, iclaFloatComplex** dB_displ,
    iclaFloatComplex** dX_displ, iclaFloatComplex** dinvA_displ,
    icla_int_t resetozero,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_ctrsm_inv_outofplace_batched(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t flag, icla_int_t m, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex** dA_array,    icla_int_t ldda,
    iclaFloatComplex** dB_array,    icla_int_t lddb,
    iclaFloatComplex** dX_array,    icla_int_t lddx,
    iclaFloatComplex** dinvA_array, icla_int_t dinvA_length,
    iclaFloatComplex** dA_displ, iclaFloatComplex** dB_displ,
    iclaFloatComplex** dX_displ, iclaFloatComplex** dinvA_displ,
    icla_int_t resetozero,
    icla_int_t batchCount, icla_queue_t queue);

void iclablas_ctrsv_batched(
    icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t n,
    iclaFloatComplex** dA_array,    icla_int_t ldda,
    iclaFloatComplex** dB_array,    icla_int_t incb,
    icla_int_t batchCount, icla_queue_t queue);

void iclablas_ctrsv_work_batched(
    icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t n,
    iclaFloatComplex** dA_array,    icla_int_t ldda,
    iclaFloatComplex** dB_array,    icla_int_t incb,
    iclaFloatComplex** dX_array,
    icla_int_t batchCount, icla_queue_t queue);

void iclablas_ctrsv_outofplace_batched(
    icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t n,
    iclaFloatComplex ** A_array, icla_int_t lda,
    iclaFloatComplex **b_array, icla_int_t incb,
    iclaFloatComplex **x_array,
    icla_int_t batchCount, icla_queue_t queue, icla_int_t flag);

void
iclablas_ctrmm_batched_core(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t m, icla_int_t n,
        iclaFloatComplex alpha,
        iclaFloatComplex **dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
        iclaFloatComplex **dB_array, icla_int_t Bi, icla_int_t Bj, icla_int_t lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_ctrmm_batched(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t m, icla_int_t n,
        iclaFloatComplex alpha,
        iclaFloatComplex **dA_array, icla_int_t ldda,
        iclaFloatComplex **dB_array, icla_int_t lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_chemm_batched_core(
        icla_side_t side, icla_uplo_t uplo,
        icla_int_t m, icla_int_t n,
        iclaFloatComplex alpha,
        iclaFloatComplex **dA_array, icla_int_t ldda,
        iclaFloatComplex **dB_array, icla_int_t lddb,
        iclaFloatComplex beta,
        iclaFloatComplex **dC_array, icla_int_t lddc,
        icla_int_t roffA, icla_int_t coffA, icla_int_t roffB, icla_int_t coffB, icla_int_t roffC, icla_int_t coffC,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_chemm_batched(
        icla_side_t side, icla_uplo_t uplo,
        icla_int_t m, icla_int_t n,
        iclaFloatComplex alpha,
        iclaFloatComplex **dA_array, icla_int_t ldda,
        iclaFloatComplex **dB_array, icla_int_t lddb,
        iclaFloatComplex beta,
        iclaFloatComplex **dC_array, icla_int_t lddc,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_chemv_batched_core(
        icla_uplo_t uplo, icla_int_t n,
        iclaFloatComplex alpha,
        iclaFloatComplex **dA_array, icla_int_t ldda,
        iclaFloatComplex **dX_array, icla_int_t incx,
        iclaFloatComplex beta,
        iclaFloatComplex **dY_array, icla_int_t incy,
        icla_int_t offA, icla_int_t offX, icla_int_t offY,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_chemv_batched(
        icla_uplo_t uplo, icla_int_t n,
        iclaFloatComplex alpha,
        iclaFloatComplex **dA_array, icla_int_t ldda,
        iclaFloatComplex **dX_array, icla_int_t incx,
        iclaFloatComplex beta,
        iclaFloatComplex **dY_array, icla_int_t incy,
        icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_cpotrf_batched(
    icla_uplo_t uplo, icla_int_t n,
    iclaFloatComplex **dA_array, icla_int_t lda,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_cpotf2_batched(
    icla_uplo_t uplo, icla_int_t n,
    iclaFloatComplex **dA_array, icla_int_t ai, icla_int_t aj, icla_int_t lda,
    icla_int_t *info_array, icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_cpotrf_panel_batched(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nb,
    iclaFloatComplex** dA_array, icla_int_t ai, icla_int_t aj, icla_int_t ldda,
    icla_int_t *info_array, icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_cpotrf_recpanel_batched(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n, icla_int_t min_recpnb,
    iclaFloatComplex** dA_array, icla_int_t ai, icla_int_t aj, icla_int_t ldda,
    icla_int_t *info_array, icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_cpotrf_rectile_batched(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n, icla_int_t min_recpnb,
    iclaFloatComplex** dA_array, icla_int_t ai, icla_int_t aj, icla_int_t ldda,
    icla_int_t *info_array, icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_cpotrs_batched(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nrhs,
    iclaFloatComplex **dA_array, icla_int_t ldda,
    iclaFloatComplex **dB_array, icla_int_t lddb,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_cposv_batched(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nrhs,
    iclaFloatComplex **dA_array, icla_int_t ldda,
    iclaFloatComplex **dB_array, icla_int_t lddb,
    icla_int_t *dinfo_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_cgetrs_batched(
    icla_trans_t trans, icla_int_t n, icla_int_t nrhs,
    iclaFloatComplex **dA_array, icla_int_t ldda,
    icla_int_t **dipiv_array,
    iclaFloatComplex **dB_array, icla_int_t lddb,
    icla_int_t batchCount, icla_queue_t queue);

void
icla_claswp_rowparallel_batched(
    icla_int_t n,
    iclaFloatComplex**  input_array, icla_int_t  input_i, icla_int_t  input_j, icla_int_t ldi,
    iclaFloatComplex** output_array, icla_int_t output_i, icla_int_t output_j, icla_int_t ldo,
    icla_int_t k1, icla_int_t k2,
    icla_int_t **pivinfo_array,
    icla_int_t batchCount, icla_queue_t queue);

void
icla_claswp_rowserial_batched(
    icla_int_t n, iclaFloatComplex** dA_array, icla_int_t lda,
    icla_int_t k1, icla_int_t k2,
    icla_int_t **ipiv_array,
    icla_int_t batchCount, icla_queue_t queue);

void
icla_claswp_columnserial_batched(
    icla_int_t n, iclaFloatComplex** dA_array, icla_int_t lda,
    icla_int_t k1, icla_int_t k2,
    icla_int_t **ipiv_array,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_ctranspose_batched(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex **dA_array,  icla_int_t ldda,
    iclaFloatComplex **dAT_array, icla_int_t lddat,
    icla_int_t batchCount, icla_queue_t queue);

void iclablas_claset_internal_batched(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    iclaFloatComplex offdiag, iclaFloatComplex diag,
    iclaFloatComplex_ptr dAarray[], icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_claset_batched(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    iclaFloatComplex offdiag, iclaFloatComplex diag,
    iclaFloatComplex_ptr dAarray[], icla_int_t ldda,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_cgbsv_batched(
    icla_int_t n, icla_int_t kl, icla_int_t ku, icla_int_t nrhs,
    iclaFloatComplex **dA_array, icla_int_t ldda, icla_int_t **dipiv_array,
    iclaFloatComplex** dB_array, icla_int_t lddb,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_cgbsv_batched_fused_sm(
    icla_int_t n, icla_int_t kl, icla_int_t ku, icla_int_t nrhs,
    iclaFloatComplex** dA_array, icla_int_t ldda, icla_int_t** ipiv_array,
    iclaFloatComplex** dB_array, icla_int_t lddb, icla_int_t* info_array,
    icla_int_t nthreads, icla_int_t ntcol,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_cgbsv_batched_strided_work(
    icla_int_t n, icla_int_t kl, icla_int_t ku, icla_int_t nrhs,
    iclaFloatComplex* dA, icla_int_t ldda, icla_int_t strideA,
    icla_int_t* dipiv, icla_int_t stride_piv,
    iclaFloatComplex* dB, icla_int_t lddb, icla_int_t strideB,
    icla_int_t *info_array,
    void* device_work, icla_int_t *lwork,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_cgbsv_batched_strided(
    icla_int_t n, icla_int_t kl, icla_int_t ku, icla_int_t nrhs,
    iclaFloatComplex* dA, icla_int_t ldda, icla_int_t strideA,
    icla_int_t* dipiv, icla_int_t stride_piv,
    iclaFloatComplex* dB, icla_int_t lddb, icla_int_t strideB,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_cgbsv_batched_work(
    icla_int_t n, icla_int_t kl, icla_int_t ku, icla_int_t nrhs,
    iclaFloatComplex** dA_array, icla_int_t ldda, icla_int_t **dipiv_array,
    iclaFloatComplex** dB_array, icla_int_t lddb,
    icla_int_t *info_array,
    void* device_work, icla_int_t *lwork,
    icla_int_t batchCount, icla_queue_t queue);

void
icla_cgbtrf_set_fillin(
    icla_int_t n, icla_int_t kl, icla_int_t ku,
    iclaFloatComplex** dAB_array, icla_int_t lddab,
    icla_int_t** dipiv_array, int* ju_array, icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_cgbtf2_cswap_batched(
    icla_int_t kl, icla_int_t ku,
    iclaFloatComplex **dAB_array, icla_int_t ai, icla_int_t aj, icla_int_t lddab,
    icla_int_t** dipiv_array, icla_int_t ipiv_offset,
    int* ju_array, icla_int_t gbstep, icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_cgbtf2_scal_ger_batched(
    icla_int_t m, icla_int_t n, icla_int_t kl, icla_int_t ku,
    iclaFloatComplex **dAB_array, icla_int_t ai, icla_int_t aj, icla_int_t lddab,
    int* ju_array, icla_int_t gbstep, icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_cgbtrf_batched_fused_sm(
    icla_int_t m,  icla_int_t n,
    icla_int_t kl, icla_int_t ku,
    iclaFloatComplex** dAB_array, icla_int_t lddab,
    icla_int_t** ipiv_array, icla_int_t* info_array,
    icla_int_t nthreads, icla_int_t ntcol,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_cgbtrf_batched_sliding_window_loopout(
    icla_int_t m,  icla_int_t n,
    icla_int_t kl, icla_int_t ku,
    iclaFloatComplex** dAB_array, icla_int_t lddab,
    icla_int_t** ipiv_array, icla_int_t* info_array,
    void* device_work, icla_int_t *lwork,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_cgbtrf_batched_sliding_window_loopin(
    icla_int_t m,  icla_int_t n,
    icla_int_t kl, icla_int_t ku,
    iclaFloatComplex** dAB_array, icla_int_t lddab,
    icla_int_t** ipiv_array, icla_int_t* info_array,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_cgbtrf_batched_strided(
        icla_int_t m, icla_int_t n,
        icla_int_t kl, icla_int_t ku,
        iclaFloatComplex* dAB, icla_int_t lddab, icla_int_t strideAB,
        icla_int_t* dipiv, icla_int_t stride_piv,
        icla_int_t *info_array,
        icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_cgbtrf_batched_strided_work(
        icla_int_t m, icla_int_t n,
        icla_int_t kl, icla_int_t ku,
        iclaFloatComplex* dAB, icla_int_t lddab, icla_int_t strideAB,
        icla_int_t* dipiv, icla_int_t stride_piv,
        icla_int_t *info_array,
        void* device_work, icla_int_t *lwork,
        icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_cgbtrf_batched_work(
        icla_int_t m, icla_int_t n,
        icla_int_t kl, icla_int_t ku,
        iclaFloatComplex **dAB_array, icla_int_t lddab,
        icla_int_t **dipiv_array, icla_int_t *info_array,
        void* device_work, icla_int_t *lwork,
        icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_cgbtrf_batched(
    icla_int_t m, icla_int_t n,
    icla_int_t kl, icla_int_t ku,
    iclaFloatComplex **dAB_array, icla_int_t lddab,
    icla_int_t **dipiv_array, icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_cgbtrs_batched(
    icla_trans_t transA,
    icla_int_t n, icla_int_t kl, icla_int_t ku, icla_int_t nrhs,
    iclaFloatComplex **dA_array, icla_int_t ldda, icla_int_t **dipiv_array,
    iclaFloatComplex** dB_array, icla_int_t lddb,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_cgbtrs_lower_batched(
    icla_trans_t transA,
    icla_int_t n, icla_int_t kl, icla_int_t ku, icla_int_t nrhs,
    iclaFloatComplex** dA_array, icla_int_t ldda, icla_int_t **dipiv_array,
    iclaFloatComplex** dB_array, icla_int_t lddb,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
iclablas_cgbtrs_lower_blocked_batched(
        icla_int_t n, icla_int_t kl, icla_int_t ku, icla_int_t nrhs,
        iclaFloatComplex** dA_array, icla_int_t ldda, icla_int_t** dipiv_array,
        iclaFloatComplex** dB_array, icla_int_t lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_cgbtrs_swap_batched(
    icla_int_t n, iclaFloatComplex** dA_array, icla_int_t ldda,
    icla_int_t** dipiv_array, icla_int_t j,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_cgbtrs_upper_batched(
    icla_trans_t transA,
    icla_int_t n, icla_int_t kl, icla_int_t ku, icla_int_t nrhs,
    iclaFloatComplex** dA_array, icla_int_t ldda, icla_int_t **dipiv_array,
    iclaFloatComplex** dB_array, icla_int_t lddb,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
iclablas_cgbtrs_upper_blocked_batched(
        icla_int_t n, icla_int_t kl, icla_int_t ku, icla_int_t nrhs,
        iclaFloatComplex** dA_array, icla_int_t ldda,
        iclaFloatComplex** dB_array, icla_int_t lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_cgbtrs_upper_columnwise_batched(
    icla_int_t n, icla_int_t kl, icla_int_t ku,
    icla_int_t nrhs, icla_int_t j,
    iclaFloatComplex** dA_array, icla_int_t ldda,
    iclaFloatComplex** dB_array, icla_int_t lddb,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_cgeru_batched_core(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex** dX_array, icla_int_t xi, icla_int_t xj, icla_int_t lddx, icla_int_t incx,
    iclaFloatComplex** dY_array, icla_int_t yi, icla_int_t yj, icla_int_t lddy, icla_int_t incy,
    iclaFloatComplex** dA_array, icla_int_t ai, icla_int_t aj, icla_int_t ldda,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_cgesv_batched_small(
    icla_int_t n, icla_int_t nrhs,
    iclaFloatComplex** dA_array, icla_int_t ldda,
    icla_int_t** dipiv_array,
    iclaFloatComplex **dB_array, icla_int_t lddb,
    icla_int_t* dinfo_array,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_cgetf2_batched(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex **dA_array, icla_int_t ai, icla_int_t aj, icla_int_t lda,
    icla_int_t **ipiv_array,
    icla_int_t **dpivinfo_array,
    icla_int_t *info_array,
    icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_cgetrf_recpanel_batched(
    icla_int_t m, icla_int_t n, icla_int_t min_recpnb,
    iclaFloatComplex** dA_array, icla_int_t ai, icla_int_t aj, icla_int_t ldda,
    icla_int_t** dipiv_array, icla_int_t** dpivinfo_array,
    icla_int_t *info_array, icla_int_t gbstep,
    icla_int_t batchCount,  icla_queue_t queue);

icla_int_t
icla_cgetrf_batched(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex **dA_array,
    icla_int_t lda,
    icla_int_t **ipiv_array,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_cgetf2_fused_batched(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex **dA_array, icla_int_t ai, icla_int_t aj, icla_int_t ldda,
    icla_int_t **dipiv_array,
    icla_int_t *info_array, icla_int_t batchCount,
    icla_queue_t queue);

icla_int_t
icla_cgetrf_batched_smallsq_noshfl(
    icla_int_t n,
    iclaFloatComplex** dA_array, icla_int_t ldda,
    icla_int_t** ipiv_array, icla_int_t* info_array,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_cgetri_outofplace_batched(
    icla_int_t n,
    iclaFloatComplex **dA_array, icla_int_t ldda,
    icla_int_t **dipiv_array,
    iclaFloatComplex **dinvA_array, icla_int_t lddia,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

void
icla_cdisplace_intpointers(
    icla_int_t **output_array,
    icla_int_t **input_array, icla_int_t lda,
    icla_int_t row, icla_int_t column,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_icamax_atomic_batched(
    icla_int_t n,
    iclaFloatComplex** x_array, icla_int_t incx,
    icla_int_t **max_id_array,
    icla_int_t batchCount);

void
iclablas_icamax_tree_batched(
    icla_int_t n,
    iclaFloatComplex** x_array, icla_int_t incx,
    icla_int_t **max_id_array,
    icla_int_t batchCount);

void
iclablas_icamax_batched(
    icla_int_t n,
    iclaFloatComplex** x_array, icla_int_t incx,
    icla_int_t **max_id_array,
    icla_int_t batchCount);

void
iclablas_icamax(
    icla_int_t n,
    iclaFloatComplex* x, icla_int_t incx,
    icla_int_t *max_id);

icla_int_t
icla_icamax_batched(
        icla_int_t length,
        iclaFloatComplex **x_array, icla_int_t xi, icla_int_t xj, icla_int_t lda, icla_int_t incx,
        icla_int_t** ipiv_array, icla_int_t ipiv_i,
        icla_int_t step, icla_int_t gbstep, icla_int_t *info_array,
        icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_cswap_batched(
    icla_int_t n, iclaFloatComplex **x_array, icla_int_t xi, icla_int_t xj, icla_int_t incx,
    icla_int_t step, icla_int_t** ipiv_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_cscal_cgeru_batched(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex **dA_array, icla_int_t ai, icla_int_t aj, icla_int_t lda,
    icla_int_t *info_array, icla_int_t step, icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_ccomputecolumn_batched(
    icla_int_t m, icla_int_t paneloffset, icla_int_t step,
    iclaFloatComplex **dA_array,  icla_int_t lda,
    icla_int_t ai, icla_int_t aj,
    icla_int_t **ipiv_array,
    icla_int_t *info_array, icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue);

void
icla_cgetf2trsm_batched(
    icla_int_t ib, icla_int_t n,
    iclaFloatComplex **dA_array,  icla_int_t j, icla_int_t lda,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_cgetf2_nopiv_internal_batched(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex** dA_array, icla_int_t ai, icla_int_t aj, icla_int_t ldda,
    icla_int_t* info_array, icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_cgetf2_nopiv_batched(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex **dA_array, icla_int_t ai, icla_int_t aj, icla_int_t ldda,
    icla_int_t *info_array, icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_cgetrf_recpanel_nopiv_batched(
    icla_int_t m, icla_int_t n, icla_int_t min_recpnb,
    iclaFloatComplex** dA_array,    icla_int_t ldda,
    iclaFloatComplex** dX_array,    icla_int_t dX_length,
    iclaFloatComplex** dinvA_array, icla_int_t dinvA_length,
    iclaFloatComplex** dW1_displ, iclaFloatComplex** dW2_displ,
    iclaFloatComplex** dW3_displ, iclaFloatComplex** dW4_displ,
    iclaFloatComplex** dW5_displ,
    icla_int_t *info_array, icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_cgetrf_nopiv_batched(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex **dA_array,
    icla_int_t lda,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_cgetrs_nopiv_batched(
    icla_trans_t trans, icla_int_t n, icla_int_t nrhs,
    iclaFloatComplex **dA_array, icla_int_t ldda,
    iclaFloatComplex **dB_array, icla_int_t lddb,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_cgesv_nopiv_batched(
    icla_int_t n, icla_int_t nrhs,
    iclaFloatComplex **dA_array, icla_int_t ldda,
    iclaFloatComplex **dB_array, icla_int_t lddb,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_cgesv_rbt_batched(
    icla_int_t n, icla_int_t nrhs,
    iclaFloatComplex **dA_array, icla_int_t ldda,
    iclaFloatComplex **dB_array, icla_int_t lddb,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_cgesv_batched(
    icla_int_t n, icla_int_t nrhs,
    iclaFloatComplex **dA_array, icla_int_t ldda,
    icla_int_t **dipiv_array,
    iclaFloatComplex **dB_array, icla_int_t lddb,
    icla_int_t *dinfo_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_cgerbt_batched(
    icla_bool_t gen, icla_int_t n, icla_int_t nrhs,
    iclaFloatComplex **dA_array, icla_int_t ldda,
    iclaFloatComplex **dB_array, icla_int_t lddb,
    iclaFloatComplex *U, iclaFloatComplex *V,
    icla_int_t *info,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_cprbt_batched(
    icla_int_t n,
    iclaFloatComplex **dA_array, icla_int_t ldda,
    iclaFloatComplex *du, iclaFloatComplex *dv,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_cprbt_mv_batched(
    icla_int_t n, icla_int_t nrhs,
    iclaFloatComplex *dv, iclaFloatComplex **db_array, icla_int_t lddb,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_cprbt_mtv_batched(
    icla_int_t n, icla_int_t nrhs,
    iclaFloatComplex *du, iclaFloatComplex **db_array, icla_int_t lddb,
    icla_int_t batchCount, icla_queue_t queue);

void
setup_pivinfo(
    icla_int_t *pivinfo, icla_int_t *ipiv,
    icla_int_t m, icla_int_t nb,
    icla_queue_t queue);

void
iclablas_cgeadd_batched(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr  const dAarray[], icla_int_t ldda,
    iclaFloatComplex_ptr              dBarray[], icla_int_t lddb,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_clacpy_internal_batched(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    iclaFloatComplex_const_ptr const dAarray[], icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    iclaFloatComplex_ptr             dBarray[], icla_int_t Bi, icla_int_t Bj, icla_int_t lddb,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_clacpy_batched(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    iclaFloatComplex_const_ptr  const dAarray[], icla_int_t ldda,
    iclaFloatComplex_ptr              dBarray[], icla_int_t lddb,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_cgemv_batched_core(
    icla_trans_t trans, icla_int_t m, icla_int_t n,
    const iclaFloatComplex alpha,
    iclaFloatComplex const * const * dA_array, const iclaFloatComplex* dA, icla_int_t ldda, icla_int_t strideA,
    iclaFloatComplex const * const * dx_array, const iclaFloatComplex* dx, icla_int_t incx, icla_int_t stridex,
    const iclaFloatComplex beta,
    iclaFloatComplex** dy_array, iclaFloatComplex_ptr dy, icla_int_t incy, icla_int_t stridey,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_cgemv_batched(
    icla_trans_t trans, icla_int_t m, icla_int_t n,
    const iclaFloatComplex alpha,
    iclaFloatComplex const * const * dA_array, icla_int_t ldda,
    iclaFloatComplex const * const * dx_array, icla_int_t incx,
    const iclaFloatComplex beta,
    iclaFloatComplex** dy_array, icla_int_t incy,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_cgemv_batched_strided(
    icla_trans_t trans, icla_int_t m, icla_int_t n,
    const iclaFloatComplex alpha,
    const iclaFloatComplex* dA, icla_int_t ldda, icla_int_t strideA,
    const iclaFloatComplex* dx, icla_int_t incx, icla_int_t stridex,
    const iclaFloatComplex beta,
    iclaFloatComplex* dy, icla_int_t incy, icla_int_t stridey,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
iclablas_cgemv_batched_smallsq(
    icla_trans_t trans, icla_int_t n,
    const iclaFloatComplex alpha,
    iclaFloatComplex const * const * dA_array, icla_int_t ldda,
    iclaFloatComplex const * const * dx_array, icla_int_t incx,
    const iclaFloatComplex beta,
    iclaFloatComplex** dy_array, icla_int_t incy,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
iclablas_cgemv_batched_strided_smallsq(
    icla_trans_t transA, icla_int_t n,
    const iclaFloatComplex alpha,
    const iclaFloatComplex* dA, icla_int_t ldda, icla_int_t strideA,
    const iclaFloatComplex* dx, icla_int_t incx, icla_int_t stridex,
    const iclaFloatComplex beta,
    iclaFloatComplex* dy, icla_int_t incy, icla_int_t stridey,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_cgeqrf_batched_smallsq(
    icla_int_t n,
    iclaFloatComplex** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    iclaFloatComplex **dtau_array, icla_int_t taui, icla_int_t* info_array,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_cgeqrf_batched(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex **dA_array,
    icla_int_t lda,
    iclaFloatComplex **dtau_array,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_cgeqrf_batched_work(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex **dA_array, icla_int_t ldda,
    iclaFloatComplex **dtau_array,
    icla_int_t *info_array,
    void* device_work, icla_int_t* device_lwork,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_cgeqrf_expert_batched(
    icla_int_t m, icla_int_t n, icla_int_t nb,
    iclaFloatComplex **dA_array, icla_int_t ldda,
    iclaFloatComplex **dR_array, icla_int_t lddr,
    iclaFloatComplex **dT_array, icla_int_t lddt,
    iclaFloatComplex **dtau_array, icla_int_t provide_RT,
    iclaFloatComplex **dW_array,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_cgeqrf_batched_v4(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex **dA_array,
    icla_int_t lda,
    iclaFloatComplex **tau_array,
    icla_int_t *info_array,
    icla_int_t batchCount);

icla_int_t
icla_cgeqrf_panel_fused_update_batched(
        icla_int_t m, icla_int_t n, icla_int_t nb,
        iclaFloatComplex** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
        iclaFloatComplex** tau_array, icla_int_t taui,
        iclaFloatComplex** dR_array, icla_int_t Ri, icla_int_t Rj, icla_int_t lddr,
        icla_int_t *info_array, icla_int_t separate_R_V,
        icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_cgeqrf_panel_internal_batched(
        icla_int_t m, icla_int_t n, icla_int_t nb,
        iclaFloatComplex** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
        iclaFloatComplex** tau_array, icla_int_t taui,
        iclaFloatComplex** dT_array, icla_int_t Ti, icla_int_t Tj, icla_int_t lddt,
        iclaFloatComplex** dR_array, icla_int_t Ri, icla_int_t Rj, icla_int_t lddr,
        iclaFloatComplex** dwork_array,
        icla_int_t *info_array,
        icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_cgeqrf_panel_batched(
    icla_int_t m, icla_int_t n, icla_int_t nb,
    iclaFloatComplex** dA_array,    icla_int_t ldda,
    iclaFloatComplex** tau_array,
    iclaFloatComplex** dT_array, icla_int_t ldt,
    iclaFloatComplex** dR_array, icla_int_t ldr,
    iclaFloatComplex** dwork_array,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_cgels_batched(
    icla_trans_t trans, icla_int_t m, icla_int_t n, icla_int_t nrhs,
    iclaFloatComplex **dA_array, icla_int_t ldda,
    iclaFloatComplex **dB_array, icla_int_t lddb,
    iclaFloatComplex *hwork, icla_int_t lwork,
    icla_int_t *info,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_cgeqr2_fused_reg_tall_batched(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    iclaFloatComplex **dtau_array, icla_int_t taui,
    icla_int_t* info_array, icla_int_t check_launch_only,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_cgeqr2_fused_reg_medium_batched(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    iclaFloatComplex **dtau_array, icla_int_t taui,
    icla_int_t* info_array, icla_int_t check_launch_only,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_cgeqr2_fused_reg_batched(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    iclaFloatComplex **dtau_array, icla_int_t taui,
    icla_int_t* info_array, icla_int_t check_launch_only,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_cgeqr2_fused_sm_batched(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    iclaFloatComplex **dtau_array, icla_int_t taui,
    icla_int_t* info_array, icla_int_t nthreads, icla_int_t check_launch_only,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_cgeqr2_batched(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex **dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    iclaFloatComplex **dtau_array, icla_int_t taui,
    icla_int_t *info_array, icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_clarf_fused_reg_tall_batched(
    icla_int_t m, icla_int_t n, icla_int_t nb, icla_int_t ib,
    iclaFloatComplex** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    iclaFloatComplex** dV_array, icla_int_t Vi, icla_int_t Vj, icla_int_t lddv,
    iclaFloatComplex **dtau_array, icla_int_t taui,
    icla_int_t check_launch_only,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_clarf_fused_reg_medium_batched(
    icla_int_t m, icla_int_t n, icla_int_t nb, icla_int_t ib,
    iclaFloatComplex** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    iclaFloatComplex** dV_array, icla_int_t Vi, icla_int_t Vj, icla_int_t lddv,
    iclaFloatComplex **dtau_array, icla_int_t taui,
    icla_int_t check_launch_only,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_clarf_fused_reg_batched(
    icla_int_t m, icla_int_t n, icla_int_t nb, icla_int_t ib,
    iclaFloatComplex** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    iclaFloatComplex** dV_array, icla_int_t Vi, icla_int_t Vj, icla_int_t lddv,
    iclaFloatComplex **dtau_array, icla_int_t taui,
    icla_int_t check_launch_only,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_clarf_fused_sm_batched(
    icla_int_t m, icla_int_t n, icla_int_t nb, icla_int_t ib,
    iclaFloatComplex** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    iclaFloatComplex** dV_array, icla_int_t Vi, icla_int_t Vj, icla_int_t lddv,
    iclaFloatComplex **dtau_array, icla_int_t taui,
    icla_int_t nthreads, icla_int_t check_launch_only,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_clarfb_gemm_internal_batched(
    icla_side_t side, icla_trans_t trans, icla_direct_t direct, icla_storev_t storev,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaFloatComplex_const_ptr dV_array[],    icla_int_t vi, icla_int_t vj, icla_int_t lddv,
    iclaFloatComplex_const_ptr dT_array[],    icla_int_t Ti, icla_int_t Tj, icla_int_t lddt,
    iclaFloatComplex_ptr dC_array[],          icla_int_t Ci, icla_int_t Cj, icla_int_t lddc,
    iclaFloatComplex_ptr dwork_array[],       icla_int_t ldwork,
    iclaFloatComplex_ptr dworkvt_array[],     icla_int_t ldworkvt,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_clarfb_gemm_batched(
    icla_side_t side, icla_trans_t trans, icla_direct_t direct, icla_storev_t storev,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaFloatComplex_const_ptr dV_array[],    icla_int_t lddv,
    iclaFloatComplex_const_ptr dT_array[],    icla_int_t lddt,
    iclaFloatComplex_ptr dC_array[],          icla_int_t lddc,
    iclaFloatComplex_ptr dwork_array[],       icla_int_t ldwork,
    iclaFloatComplex_ptr dworkvt_array[],     icla_int_t ldworkvt,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_clarft_internal_batched(
        icla_int_t n, icla_int_t k, icla_int_t stair_T,
        iclaFloatComplex **v_array,   icla_int_t vi, icla_int_t vj, icla_int_t ldv,
        iclaFloatComplex **tau_array, icla_int_t taui,
        iclaFloatComplex **T_array,   icla_int_t Ti, icla_int_t Tj, icla_int_t ldt,
        iclaFloatComplex **work_array, icla_int_t lwork,
        icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_clarft_batched(
    icla_int_t n, icla_int_t k, icla_int_t stair_T,
    iclaFloatComplex **v_array, icla_int_t ldv,
    iclaFloatComplex **tau_array,
    iclaFloatComplex **T_array, icla_int_t ldt,
    iclaFloatComplex **work_array, icla_int_t lwork,
    icla_int_t batchCount, icla_queue_t queue);

void
icla_clarft_sm32x32_batched(
    icla_int_t n, icla_int_t k,
    iclaFloatComplex **v_array, icla_int_t vi, icla_int_t vj, icla_int_t ldv,
    iclaFloatComplex **tau_array, icla_int_t taui,
    iclaFloatComplex **T_array, icla_int_t Ti, icla_int_t Tj, icla_int_t ldt,
    icla_int_t batchCount, icla_queue_t queue);

void iclablas_clarft_recctrmv_sm32x32(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex *tau,
    iclaFloatComplex *Trec, icla_int_t ldtrec,
    iclaFloatComplex *Ttri, icla_int_t ldttri,
    icla_queue_t queue);

void iclablas_clarft_recctrmv_sm32x32_batched(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex **tau_array,  icla_int_t taui,
    iclaFloatComplex **Trec_array, icla_int_t Treci, icla_int_t Trecj, icla_int_t ldtrec,
    iclaFloatComplex **Ttri_array, icla_int_t Ttrii, icla_int_t Ttrij, icla_int_t ldttri,
    icla_int_t batchCount, icla_queue_t queue);

void iclablas_clarft_ctrmv_sm32x32(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex *tau,
    iclaFloatComplex *Tin, icla_int_t ldtin,
    iclaFloatComplex *Tout, icla_int_t ldtout,
    icla_queue_t queue);

void iclablas_clarft_ctrmv_sm32x32_batched(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex **tau_array, icla_int_t taui,
    iclaFloatComplex **Tin_array, icla_int_t Tini, icla_int_t Tinj, icla_int_t ldtin,
    iclaFloatComplex **Tout_array, icla_int_t Touti, icla_int_t Toutj, icla_int_t ldtout,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_scnrm2_cols_batched(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex **dA_array, icla_int_t lda,
    float **dxnorm_array,
    icla_int_t batchCount);

void
icla_clarfgx_batched(
    icla_int_t n, iclaFloatComplex **dx0_array, iclaFloatComplex **dx_array,
    iclaFloatComplex **dtau_array, float **dxnorm_array,
    iclaFloatComplex **dR_array, icla_int_t it,
    icla_int_t batchCount);

void
icla_clarfx_batched_v4(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex **v_array,
    iclaFloatComplex **tau_array,
    iclaFloatComplex **C_array, icla_int_t ldc, float **xnorm_array,
    icla_int_t step,
    icla_int_t batchCount);

void
iclablas_clarfg_batched(
    icla_int_t n,
    iclaFloatComplex** dalpha_array,
    iclaFloatComplex** dx_array, icla_int_t incx,
    iclaFloatComplex** dtau_array,
    icla_int_t batchCount );

icla_int_t
icla_cpotrf_lpout_batched(
    icla_uplo_t uplo, icla_int_t n,
    iclaFloatComplex **dA_array, icla_int_t ai, icla_int_t aj, icla_int_t lda, icla_int_t gbstep,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_cpotrf_lpin_batched(
    icla_uplo_t uplo, icla_int_t n,
    iclaFloatComplex **dA_array, icla_int_t ai, icla_int_t aj, icla_int_t lda, icla_int_t gbstep,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_cpotrf_v33_batched(
    icla_uplo_t uplo, icla_int_t n,
    iclaFloatComplex **dA_array, icla_int_t lda,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);


void
blas_clacpy_batched(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    iclaFloatComplex const * const * hA_array, icla_int_t lda,
    iclaFloatComplex               **hB_array, icla_int_t ldb,
    icla_int_t batchCount );

void
blas_cgemm_batched(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaFloatComplex alpha,
    iclaFloatComplex const * const * hA_array, icla_int_t lda,
    iclaFloatComplex const * const * hB_array, icla_int_t ldb,
    iclaFloatComplex beta,
    iclaFloatComplex **hC_array, icla_int_t ldc,
    icla_int_t batchCount );

void
blas_ctrsm_batched(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t m, icla_int_t n,
        iclaFloatComplex alpha,
        iclaFloatComplex **hA_array, icla_int_t lda,
        iclaFloatComplex **hB_array, icla_int_t ldb,
        icla_int_t batchCount );

void
blas_ctrmm_batched(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t m, icla_int_t n,
        iclaFloatComplex alpha,
        iclaFloatComplex **hA_array, icla_int_t lda,
        iclaFloatComplex **hB_array, icla_int_t ldb,
        icla_int_t batchCount );

void
blas_chemm_batched(
        icla_side_t side, icla_uplo_t uplo,
        icla_int_t m, icla_int_t n,
        iclaFloatComplex alpha,
        iclaFloatComplex **hA_array, icla_int_t lda,
        iclaFloatComplex **hB_array, icla_int_t ldb,
        iclaFloatComplex beta,
        iclaFloatComplex **hC_array, icla_int_t ldc,
        icla_int_t batchCount );

void
blas_cherk_batched(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    float alpha, iclaFloatComplex const * const * hA_array, icla_int_t lda,
    float beta,  iclaFloatComplex               **hC_array, icla_int_t ldc,
    icla_int_t batchCount );

void
blas_cher2k_batched(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    iclaFloatComplex alpha, iclaFloatComplex const * const * hA_array, icla_int_t lda,
                              iclaFloatComplex const * const * hB_array, icla_int_t ldb,
    float beta,              iclaFloatComplex               **hC_array, icla_int_t ldc,
    icla_int_t batchCount );


void
cset_stepinit_ipiv(
    icla_int_t **ipiv_array,
    icla_int_t pm,
    icla_int_t batchCount);

#ifdef __cplusplus
}
#endif

#undef ICLA_COMPLEX

#endif

