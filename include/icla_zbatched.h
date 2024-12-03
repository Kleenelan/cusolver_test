/*
    -- ICLA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Azzam Haidar
       @author Tingxing Dong

       @precisions normal z -> s d c
*/

#ifndef ICLA_ZBATCHED_H
#define ICLA_ZBATCHED_H

#include "icla_types.h"

#define ICLA_COMPLEX

#ifdef __cplusplus
extern "C" {
#endif
  /*
   *  local auxiliary routines
   */
void
icla_zset_pointer(
    iclaDoubleComplex **output_array,
    iclaDoubleComplex *input,
    icla_int_t lda,
    icla_int_t row, icla_int_t column,
    icla_int_t batch_offset,
    icla_int_t batchCount, icla_queue_t queue);

void
icla_zdisplace_pointers(
    iclaDoubleComplex **output_array,
    iclaDoubleComplex **input_array, icla_int_t lda,
    icla_int_t row, icla_int_t column,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zrecommend_cublas_gemm_batched(
    icla_trans_t transa, icla_trans_t transb,
    icla_int_t m, icla_int_t n, icla_int_t k);

icla_int_t
icla_zrecommend_cublas_gemm_stream(
    icla_trans_t transa, icla_trans_t transb,
    icla_int_t m, icla_int_t n, icla_int_t k);

void icla_get_zpotrf_batched_nbparam(icla_int_t n, icla_int_t *nb, icla_int_t *recnb);

icla_int_t icla_get_zpotrf_batched_crossover();

void icla_get_zgetrf_batched_nbparam(icla_int_t n, icla_int_t *nb, icla_int_t *recnb);
icla_int_t icla_get_zgetrf_batched_ntcol(icla_int_t m, icla_int_t n);
icla_int_t icla_get_zgemm_batched_ntcol(icla_int_t n);
icla_int_t icla_get_zgemm_batched_smallsq_limit(icla_int_t n);
icla_int_t icla_get_zgeqrf_batched_nb(icla_int_t m);
icla_int_t icla_use_zgeqrf_batched_fused_update(icla_int_t m, icla_int_t n, icla_int_t batchCount);
icla_int_t icla_get_zgeqr2_fused_sm_batched_nthreads(icla_int_t m, icla_int_t n);
icla_int_t icla_get_zgeqrf_batched_ntcol(icla_int_t m, icla_int_t n);
icla_int_t icla_get_zgetri_batched_ntcol(icla_int_t m, icla_int_t n);
icla_int_t icla_get_ztrsm_batched_stop_nb(icla_side_t side, icla_int_t m, icla_int_t n);
void icla_get_zgbtrf_batched_params(icla_int_t m, icla_int_t n, icla_int_t kl, icla_int_t ku, icla_int_t *nb, icla_int_t *threads);

void
iclablas_zswapdblk_batched(
    icla_int_t n, icla_int_t nb,
    iclaDoubleComplex **dA, icla_int_t ldda, icla_int_t inca,
    iclaDoubleComplex **dB, icla_int_t lddb, icla_int_t incb,
    icla_int_t batchCount, icla_queue_t queue );

  /*
   *  LAPACK batched routines
   */

  /*
   *  BLAS batched routines
   */
void
iclablas_zgemm_batched_core(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex const * const * dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    iclaDoubleComplex const * const * dB_array, icla_int_t Bi, icla_int_t Bj, icla_int_t lddb,
    iclaDoubleComplex beta,
    iclaDoubleComplex **dC_array, icla_int_t Ci, icla_int_t Cj, icla_int_t lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
icla_zgemm_batched_core(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex const * const * dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    iclaDoubleComplex const * const * dB_array, icla_int_t Bi, icla_int_t Bj, icla_int_t lddb,
    iclaDoubleComplex beta,
    iclaDoubleComplex **dC_array, icla_int_t Ci, icla_int_t Cj, icla_int_t lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
icla_zgemm_batched(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex const * const * dA_array, icla_int_t ldda,
    iclaDoubleComplex const * const * dB_array, icla_int_t lddb,
    iclaDoubleComplex beta,
    iclaDoubleComplex **dC_array, icla_int_t lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_zgemm_batched(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex const * const * dA_array, icla_int_t ldda,
    iclaDoubleComplex const * const * dB_array, icla_int_t lddb,
    iclaDoubleComplex beta,
    iclaDoubleComplex **dC_array, icla_int_t lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_zgemm_batched_strided(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex const * dA, icla_int_t ldda, icla_int_t strideA,
    iclaDoubleComplex const * dB, icla_int_t lddb, icla_int_t strideB,
    iclaDoubleComplex beta,
    iclaDoubleComplex       * dC, icla_int_t lddc, icla_int_t strideC,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_zgemm_batched_smallsq(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex const * const * dA_array, icla_int_t ai, icla_int_t aj, icla_int_t ldda,
    iclaDoubleComplex const * const * dB_array, icla_int_t bi, icla_int_t bj, icla_int_t lddb,
    iclaDoubleComplex beta,
    iclaDoubleComplex **dC_array, icla_int_t ci, icla_int_t cj, icla_int_t lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_zsyrk_batched_core(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex const * const * dA_array, icla_int_t ai, icla_int_t aj, icla_int_t ldda,
    iclaDoubleComplex const * const * dB_array, icla_int_t bi, icla_int_t bj, icla_int_t lddb,
    iclaDoubleComplex beta,
    iclaDoubleComplex **dC_array, icla_int_t ci, icla_int_t cj, icla_int_t lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_zherk_batched_core(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex const * const * dA_array, icla_int_t ai, icla_int_t aj, icla_int_t ldda,
    iclaDoubleComplex const * const * dB_array, icla_int_t bi, icla_int_t bj, icla_int_t lddb,
    iclaDoubleComplex beta,
    iclaDoubleComplex **dC_array, icla_int_t ci, icla_int_t cj, icla_int_t lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_zsyrk_batched(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex const * const * dA_array, icla_int_t ldda,
    iclaDoubleComplex beta,
    iclaDoubleComplex **dC_array, icla_int_t lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
icla_zherk_batched(
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t n, icla_int_t k,
    double alpha,
    iclaDoubleComplex const * const * dA_array, icla_int_t ldda,
    double beta,
    iclaDoubleComplex **dC_array, icla_int_t lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_zherk_batched(
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t n, icla_int_t k,
    double alpha,
    iclaDoubleComplex const * const * dA_array, icla_int_t ldda,
    double beta,
    iclaDoubleComplex **dC_array, icla_int_t lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_zher2k_batched(
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t n, icla_int_t k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex const * const * dA_array, icla_int_t ldda,
    iclaDoubleComplex const * const * dB_array, icla_int_t lddb,
    double beta, iclaDoubleComplex **dC_array, icla_int_t lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_zsyr2k_batched(
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t n, icla_int_t k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex const * const * dA_array, icla_int_t ldda,
    iclaDoubleComplex const * const * dB_array, icla_int_t lddb,
    iclaDoubleComplex beta, iclaDoubleComplex **dC_array, icla_int_t lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_ztrtri_diag_batched(
    icla_uplo_t uplo, icla_diag_t diag, icla_int_t n,
    iclaDoubleComplex const * const *dA_array, icla_int_t ldda,
    iclaDoubleComplex **dinvA_array,
    icla_int_t resetozero,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_ztrsm_small_batched(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t m, icla_int_t n,
        iclaDoubleComplex alpha,
        iclaDoubleComplex **dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
        iclaDoubleComplex **dB_array, icla_int_t Bi, icla_int_t Bj, icla_int_t lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_ztrsm_recursive_batched(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t m, icla_int_t n,
        iclaDoubleComplex alpha,
        iclaDoubleComplex **dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
        iclaDoubleComplex **dB_array, icla_int_t Bi, icla_int_t Bj, icla_int_t lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_ztrsm_batched(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t m, icla_int_t n,
        iclaDoubleComplex alpha,
        iclaDoubleComplex **dA_array, icla_int_t ldda,
        iclaDoubleComplex **dB_array, icla_int_t lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_ztrsm_inv_batched(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex** dA_array,    icla_int_t ldda,
    iclaDoubleComplex** dB_array,    icla_int_t lddb,
    icla_int_t batchCount, icla_queue_t queue);

void iclablas_ztrsm_inv_work_batched(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t flag, icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex** dA_array,    icla_int_t ldda,
    iclaDoubleComplex** dB_array,    icla_int_t lddb,
    iclaDoubleComplex** dX_array,    icla_int_t lddx,
    iclaDoubleComplex** dinvA_array, icla_int_t dinvA_length,
    iclaDoubleComplex** dA_displ, iclaDoubleComplex** dB_displ,
    iclaDoubleComplex** dX_displ, iclaDoubleComplex** dinvA_displ,
    icla_int_t resetozero,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_ztrsm_inv_outofplace_batched(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t flag, icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex** dA_array,    icla_int_t ldda,
    iclaDoubleComplex** dB_array,    icla_int_t lddb,
    iclaDoubleComplex** dX_array,    icla_int_t lddx,
    iclaDoubleComplex** dinvA_array, icla_int_t dinvA_length,
    iclaDoubleComplex** dA_displ, iclaDoubleComplex** dB_displ,
    iclaDoubleComplex** dX_displ, iclaDoubleComplex** dinvA_displ,
    icla_int_t resetozero,
    icla_int_t batchCount, icla_queue_t queue);

void iclablas_ztrsv_batched(
    icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t n,
    iclaDoubleComplex** dA_array,    icla_int_t ldda,
    iclaDoubleComplex** dB_array,    icla_int_t incb,
    icla_int_t batchCount, icla_queue_t queue);

void iclablas_ztrsv_work_batched(
    icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t n,
    iclaDoubleComplex** dA_array,    icla_int_t ldda,
    iclaDoubleComplex** dB_array,    icla_int_t incb,
    iclaDoubleComplex** dX_array,
    icla_int_t batchCount, icla_queue_t queue);

void iclablas_ztrsv_outofplace_batched(
    icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t n,
    iclaDoubleComplex ** A_array, icla_int_t lda,
    iclaDoubleComplex **b_array, icla_int_t incb,
    iclaDoubleComplex **x_array,
    icla_int_t batchCount, icla_queue_t queue, icla_int_t flag);

void
iclablas_ztrmm_batched_core(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t m, icla_int_t n,
        iclaDoubleComplex alpha,
        iclaDoubleComplex **dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
        iclaDoubleComplex **dB_array, icla_int_t Bi, icla_int_t Bj, icla_int_t lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_ztrmm_batched(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t m, icla_int_t n,
        iclaDoubleComplex alpha,
        iclaDoubleComplex **dA_array, icla_int_t ldda,
        iclaDoubleComplex **dB_array, icla_int_t lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_zhemm_batched_core(
        icla_side_t side, icla_uplo_t uplo,
        icla_int_t m, icla_int_t n,
        iclaDoubleComplex alpha,
        iclaDoubleComplex **dA_array, icla_int_t ldda,
        iclaDoubleComplex **dB_array, icla_int_t lddb,
        iclaDoubleComplex beta,
        iclaDoubleComplex **dC_array, icla_int_t lddc,
        icla_int_t roffA, icla_int_t coffA, icla_int_t roffB, icla_int_t coffB, icla_int_t roffC, icla_int_t coffC,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_zhemm_batched(
        icla_side_t side, icla_uplo_t uplo,
        icla_int_t m, icla_int_t n,
        iclaDoubleComplex alpha,
        iclaDoubleComplex **dA_array, icla_int_t ldda,
        iclaDoubleComplex **dB_array, icla_int_t lddb,
        iclaDoubleComplex beta,
        iclaDoubleComplex **dC_array, icla_int_t lddc,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_zhemv_batched_core(
        icla_uplo_t uplo, icla_int_t n,
        iclaDoubleComplex alpha,
        iclaDoubleComplex **dA_array, icla_int_t ldda,
        iclaDoubleComplex **dX_array, icla_int_t incx,
        iclaDoubleComplex beta,
        iclaDoubleComplex **dY_array, icla_int_t incy,
        icla_int_t offA, icla_int_t offX, icla_int_t offY,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_zhemv_batched(
        icla_uplo_t uplo, icla_int_t n,
        iclaDoubleComplex alpha,
        iclaDoubleComplex **dA_array, icla_int_t ldda,
        iclaDoubleComplex **dX_array, icla_int_t incx,
        iclaDoubleComplex beta,
        iclaDoubleComplex **dY_array, icla_int_t incy,
        icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_zpotrf_batched(
    icla_uplo_t uplo, icla_int_t n,
    iclaDoubleComplex **dA_array, icla_int_t lda,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zpotf2_batched(
    icla_uplo_t uplo, icla_int_t n,
    iclaDoubleComplex **dA_array, icla_int_t ai, icla_int_t aj, icla_int_t lda,
    icla_int_t *info_array, icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zpotrf_panel_batched(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nb,
    iclaDoubleComplex** dA_array, icla_int_t ai, icla_int_t aj, icla_int_t ldda,
    icla_int_t *info_array, icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zpotrf_recpanel_batched(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n, icla_int_t min_recpnb,
    iclaDoubleComplex** dA_array, icla_int_t ai, icla_int_t aj, icla_int_t ldda,
    icla_int_t *info_array, icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zpotrf_rectile_batched(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n, icla_int_t min_recpnb,
    iclaDoubleComplex** dA_array, icla_int_t ai, icla_int_t aj, icla_int_t ldda,
    icla_int_t *info_array, icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zpotrs_batched(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nrhs,
    iclaDoubleComplex **dA_array, icla_int_t ldda,
    iclaDoubleComplex **dB_array, icla_int_t lddb,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zposv_batched(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nrhs,
    iclaDoubleComplex **dA_array, icla_int_t ldda,
    iclaDoubleComplex **dB_array, icla_int_t lddb,
    icla_int_t *dinfo_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zgetrs_batched(
    icla_trans_t trans, icla_int_t n, icla_int_t nrhs,
    iclaDoubleComplex **dA_array, icla_int_t ldda,
    icla_int_t **dipiv_array,
    iclaDoubleComplex **dB_array, icla_int_t lddb,
    icla_int_t batchCount, icla_queue_t queue);

void
icla_zlaswp_rowparallel_batched(
    icla_int_t n,
    iclaDoubleComplex**  input_array, icla_int_t  input_i, icla_int_t  input_j, icla_int_t ldi,
    iclaDoubleComplex** output_array, icla_int_t output_i, icla_int_t output_j, icla_int_t ldo,
    icla_int_t k1, icla_int_t k2,
    icla_int_t **pivinfo_array,
    icla_int_t batchCount, icla_queue_t queue);

void
icla_zlaswp_rowserial_batched(
    icla_int_t n, iclaDoubleComplex** dA_array, icla_int_t lda,
    icla_int_t k1, icla_int_t k2,
    icla_int_t **ipiv_array,
    icla_int_t batchCount, icla_queue_t queue);

void
icla_zlaswp_columnserial_batched(
    icla_int_t n, iclaDoubleComplex** dA_array, icla_int_t lda,
    icla_int_t k1, icla_int_t k2,
    icla_int_t **ipiv_array,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_ztranspose_batched(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex **dA_array,  icla_int_t ldda,
    iclaDoubleComplex **dAT_array, icla_int_t lddat,
    icla_int_t batchCount, icla_queue_t queue);

void iclablas_zlaset_internal_batched(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    iclaDoubleComplex offdiag, iclaDoubleComplex diag,
    iclaDoubleComplex_ptr dAarray[], icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_zlaset_batched(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    iclaDoubleComplex offdiag, iclaDoubleComplex diag,
    iclaDoubleComplex_ptr dAarray[], icla_int_t ldda,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zgbsv_batched(
    icla_int_t n, icla_int_t kl, icla_int_t ku, icla_int_t nrhs,
    iclaDoubleComplex **dA_array, icla_int_t ldda, icla_int_t **dipiv_array,
    iclaDoubleComplex** dB_array, icla_int_t lddb,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zgbsv_batched_fused_sm(
    icla_int_t n, icla_int_t kl, icla_int_t ku, icla_int_t nrhs,
    iclaDoubleComplex** dA_array, icla_int_t ldda, icla_int_t** ipiv_array,
    iclaDoubleComplex** dB_array, icla_int_t lddb, icla_int_t* info_array,
    icla_int_t nthreads, icla_int_t ntcol,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_zgbsv_batched_strided_work(
    icla_int_t n, icla_int_t kl, icla_int_t ku, icla_int_t nrhs,
    iclaDoubleComplex* dA, icla_int_t ldda, icla_int_t strideA,
    icla_int_t* dipiv, icla_int_t stride_piv,
    iclaDoubleComplex* dB, icla_int_t lddb, icla_int_t strideB,
    icla_int_t *info_array,
    void* device_work, icla_int_t *lwork,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zgbsv_batched_strided(
    icla_int_t n, icla_int_t kl, icla_int_t ku, icla_int_t nrhs,
    iclaDoubleComplex* dA, icla_int_t ldda, icla_int_t strideA,
    icla_int_t* dipiv, icla_int_t stride_piv,
    iclaDoubleComplex* dB, icla_int_t lddb, icla_int_t strideB,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zgbsv_batched_work(
    icla_int_t n, icla_int_t kl, icla_int_t ku, icla_int_t nrhs,
    iclaDoubleComplex** dA_array, icla_int_t ldda, icla_int_t **dipiv_array,
    iclaDoubleComplex** dB_array, icla_int_t lddb,
    icla_int_t *info_array,
    void* device_work, icla_int_t *lwork,
    icla_int_t batchCount, icla_queue_t queue);

void
icla_zgbtrf_set_fillin(
    icla_int_t n, icla_int_t kl, icla_int_t ku,
    iclaDoubleComplex** dAB_array, icla_int_t lddab,
    icla_int_t** dipiv_array, int* ju_array, icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zgbtf2_zswap_batched(
    icla_int_t kl, icla_int_t ku,
    iclaDoubleComplex **dAB_array, icla_int_t ai, icla_int_t aj, icla_int_t lddab,
    icla_int_t** dipiv_array, icla_int_t ipiv_offset,
    int* ju_array, icla_int_t gbstep, icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zgbtf2_scal_ger_batched(
    icla_int_t m, icla_int_t n, icla_int_t kl, icla_int_t ku,
    iclaDoubleComplex **dAB_array, icla_int_t ai, icla_int_t aj, icla_int_t lddab,
    int* ju_array, icla_int_t gbstep, icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zgbtrf_batched_fused_sm(
    icla_int_t m,  icla_int_t n,
    icla_int_t kl, icla_int_t ku,
    iclaDoubleComplex** dAB_array, icla_int_t lddab,
    icla_int_t** ipiv_array, icla_int_t* info_array,
    icla_int_t nthreads, icla_int_t ntcol,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_zgbtrf_batched_sliding_window_loopout(
    icla_int_t m,  icla_int_t n,
    icla_int_t kl, icla_int_t ku,
    iclaDoubleComplex** dAB_array, icla_int_t lddab,
    icla_int_t** ipiv_array, icla_int_t* info_array,
    void* device_work, icla_int_t *lwork,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_zgbtrf_batched_sliding_window_loopin(
    icla_int_t m,  icla_int_t n,
    icla_int_t kl, icla_int_t ku,
    iclaDoubleComplex** dAB_array, icla_int_t lddab,
    icla_int_t** ipiv_array, icla_int_t* info_array,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_zgbtrf_batched_strided(
        icla_int_t m, icla_int_t n,
        icla_int_t kl, icla_int_t ku,
        iclaDoubleComplex* dAB, icla_int_t lddab, icla_int_t strideAB,
        icla_int_t* dipiv, icla_int_t stride_piv,
        icla_int_t *info_array,
        icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zgbtrf_batched_strided_work(
        icla_int_t m, icla_int_t n,
        icla_int_t kl, icla_int_t ku,
        iclaDoubleComplex* dAB, icla_int_t lddab, icla_int_t strideAB,
        icla_int_t* dipiv, icla_int_t stride_piv,
        icla_int_t *info_array,
        void* device_work, icla_int_t *lwork,
        icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zgbtrf_batched_work(
        icla_int_t m, icla_int_t n,
        icla_int_t kl, icla_int_t ku,
        iclaDoubleComplex **dAB_array, icla_int_t lddab,
        icla_int_t **dipiv_array, icla_int_t *info_array,
        void* device_work, icla_int_t *lwork,
        icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zgbtrf_batched(
    icla_int_t m, icla_int_t n,
    icla_int_t kl, icla_int_t ku,
    iclaDoubleComplex **dAB_array, icla_int_t lddab,
    icla_int_t **dipiv_array, icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zgbtrs_batched(
    icla_trans_t transA,
    icla_int_t n, icla_int_t kl, icla_int_t ku, icla_int_t nrhs,
    iclaDoubleComplex **dA_array, icla_int_t ldda, icla_int_t **dipiv_array,
    iclaDoubleComplex** dB_array, icla_int_t lddb,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zgbtrs_lower_batched(
    icla_trans_t transA,
    icla_int_t n, icla_int_t kl, icla_int_t ku, icla_int_t nrhs,
    iclaDoubleComplex** dA_array, icla_int_t ldda, icla_int_t **dipiv_array,
    iclaDoubleComplex** dB_array, icla_int_t lddb,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
iclablas_zgbtrs_lower_blocked_batched(
        icla_int_t n, icla_int_t kl, icla_int_t ku, icla_int_t nrhs,
        iclaDoubleComplex** dA_array, icla_int_t ldda, icla_int_t** dipiv_array,
        iclaDoubleComplex** dB_array, icla_int_t lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_zgbtrs_swap_batched(
    icla_int_t n, iclaDoubleComplex** dA_array, icla_int_t ldda,
    icla_int_t** dipiv_array, icla_int_t j,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zgbtrs_upper_batched(
    icla_trans_t transA,
    icla_int_t n, icla_int_t kl, icla_int_t ku, icla_int_t nrhs,
    iclaDoubleComplex** dA_array, icla_int_t ldda, icla_int_t **dipiv_array,
    iclaDoubleComplex** dB_array, icla_int_t lddb,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
iclablas_zgbtrs_upper_blocked_batched(
        icla_int_t n, icla_int_t kl, icla_int_t ku, icla_int_t nrhs,
        iclaDoubleComplex** dA_array, icla_int_t ldda,
        iclaDoubleComplex** dB_array, icla_int_t lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_zgbtrs_upper_columnwise_batched(
    icla_int_t n, icla_int_t kl, icla_int_t ku,
    icla_int_t nrhs, icla_int_t j,
    iclaDoubleComplex** dA_array, icla_int_t ldda,
    iclaDoubleComplex** dB_array, icla_int_t lddb,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_zgeru_batched_core(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex** dX_array, icla_int_t xi, icla_int_t xj, icla_int_t lddx, icla_int_t incx,
    iclaDoubleComplex** dY_array, icla_int_t yi, icla_int_t yj, icla_int_t lddy, icla_int_t incy,
    iclaDoubleComplex** dA_array, icla_int_t ai, icla_int_t aj, icla_int_t ldda,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_zgesv_batched_small(
    icla_int_t n, icla_int_t nrhs,
    iclaDoubleComplex** dA_array, icla_int_t ldda,
    icla_int_t** dipiv_array,
    iclaDoubleComplex **dB_array, icla_int_t lddb,
    icla_int_t* dinfo_array,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_zgetf2_batched(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex **dA_array, icla_int_t ai, icla_int_t aj, icla_int_t lda,
    icla_int_t **ipiv_array,
    icla_int_t **dpivinfo_array,
    icla_int_t *info_array,
    icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zgetrf_recpanel_batched(
    icla_int_t m, icla_int_t n, icla_int_t min_recpnb,
    iclaDoubleComplex** dA_array, icla_int_t ai, icla_int_t aj, icla_int_t ldda,
    icla_int_t** dipiv_array, icla_int_t** dpivinfo_array,
    icla_int_t *info_array, icla_int_t gbstep,
    icla_int_t batchCount,  icla_queue_t queue);

icla_int_t
icla_zgetrf_batched(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex **dA_array,
    icla_int_t lda,
    icla_int_t **ipiv_array,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zgetf2_fused_batched(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex **dA_array, icla_int_t ai, icla_int_t aj, icla_int_t ldda,
    icla_int_t **dipiv_array,
    icla_int_t *info_array, icla_int_t batchCount,
    icla_queue_t queue);

icla_int_t
icla_zgetrf_batched_smallsq_noshfl(
    icla_int_t n,
    iclaDoubleComplex** dA_array, icla_int_t ldda,
    icla_int_t** ipiv_array, icla_int_t* info_array,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_zgetri_outofplace_batched(
    icla_int_t n,
    iclaDoubleComplex **dA_array, icla_int_t ldda,
    icla_int_t **dipiv_array,
    iclaDoubleComplex **dinvA_array, icla_int_t lddia,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

void
icla_zdisplace_intpointers(
    icla_int_t **output_array,
    icla_int_t **input_array, icla_int_t lda,
    icla_int_t row, icla_int_t column,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_izamax_atomic_batched(
    icla_int_t n,
    iclaDoubleComplex** x_array, icla_int_t incx,
    icla_int_t **max_id_array,
    icla_int_t batchCount);

void
iclablas_izamax_tree_batched(
    icla_int_t n,
    iclaDoubleComplex** x_array, icla_int_t incx,
    icla_int_t **max_id_array,
    icla_int_t batchCount);

void
iclablas_izamax_batched(
    icla_int_t n,
    iclaDoubleComplex** x_array, icla_int_t incx,
    icla_int_t **max_id_array,
    icla_int_t batchCount);

void
iclablas_izamax(
    icla_int_t n,
    iclaDoubleComplex* x, icla_int_t incx,
    icla_int_t *max_id);

icla_int_t
icla_izamax_batched(
        icla_int_t length,
        iclaDoubleComplex **x_array, icla_int_t xi, icla_int_t xj, icla_int_t lda, icla_int_t incx,
        icla_int_t** ipiv_array, icla_int_t ipiv_i,
        icla_int_t step, icla_int_t gbstep, icla_int_t *info_array,
        icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zswap_batched(
    icla_int_t n, iclaDoubleComplex **x_array, icla_int_t xi, icla_int_t xj, icla_int_t incx,
    icla_int_t step, icla_int_t** ipiv_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zscal_zgeru_batched(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex **dA_array, icla_int_t ai, icla_int_t aj, icla_int_t lda,
    icla_int_t *info_array, icla_int_t step, icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zcomputecolumn_batched(
    icla_int_t m, icla_int_t paneloffset, icla_int_t step,
    iclaDoubleComplex **dA_array,  icla_int_t lda,
    icla_int_t ai, icla_int_t aj,
    icla_int_t **ipiv_array,
    icla_int_t *info_array, icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue);

void
icla_zgetf2trsm_batched(
    icla_int_t ib, icla_int_t n,
    iclaDoubleComplex **dA_array,  icla_int_t j, icla_int_t lda,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zgetf2_nopiv_internal_batched(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex** dA_array, icla_int_t ai, icla_int_t aj, icla_int_t ldda,
    icla_int_t* info_array, icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_zgetf2_nopiv_batched(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex **dA_array, icla_int_t ai, icla_int_t aj, icla_int_t ldda,
    icla_int_t *info_array, icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zgetrf_recpanel_nopiv_batched(
    icla_int_t m, icla_int_t n, icla_int_t min_recpnb,
    iclaDoubleComplex** dA_array,    icla_int_t ldda,
    iclaDoubleComplex** dX_array,    icla_int_t dX_length,
    iclaDoubleComplex** dinvA_array, icla_int_t dinvA_length,
    iclaDoubleComplex** dW1_displ, iclaDoubleComplex** dW2_displ,
    iclaDoubleComplex** dW3_displ, iclaDoubleComplex** dW4_displ,
    iclaDoubleComplex** dW5_displ,
    icla_int_t *info_array, icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zgetrf_nopiv_batched(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex **dA_array,
    icla_int_t lda,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zgetrs_nopiv_batched(
    icla_trans_t trans, icla_int_t n, icla_int_t nrhs,
    iclaDoubleComplex **dA_array, icla_int_t ldda,
    iclaDoubleComplex **dB_array, icla_int_t lddb,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zgesv_nopiv_batched(
    icla_int_t n, icla_int_t nrhs,
    iclaDoubleComplex **dA_array, icla_int_t ldda,
    iclaDoubleComplex **dB_array, icla_int_t lddb,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zgesv_rbt_batched(
    icla_int_t n, icla_int_t nrhs,
    iclaDoubleComplex **dA_array, icla_int_t ldda,
    iclaDoubleComplex **dB_array, icla_int_t lddb,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zgesv_batched(
    icla_int_t n, icla_int_t nrhs,
    iclaDoubleComplex **dA_array, icla_int_t ldda,
    icla_int_t **dipiv_array,
    iclaDoubleComplex **dB_array, icla_int_t lddb,
    icla_int_t *dinfo_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zgerbt_batched(
    icla_bool_t gen, icla_int_t n, icla_int_t nrhs,
    iclaDoubleComplex **dA_array, icla_int_t ldda,
    iclaDoubleComplex **dB_array, icla_int_t lddb,
    iclaDoubleComplex *U, iclaDoubleComplex *V,
    icla_int_t *info,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_zprbt_batched(
    icla_int_t n,
    iclaDoubleComplex **dA_array, icla_int_t ldda,
    iclaDoubleComplex *du, iclaDoubleComplex *dv,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_zprbt_mv_batched(
    icla_int_t n, icla_int_t nrhs,
    iclaDoubleComplex *dv, iclaDoubleComplex **db_array, icla_int_t lddb,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_zprbt_mtv_batched(
    icla_int_t n, icla_int_t nrhs,
    iclaDoubleComplex *du, iclaDoubleComplex **db_array, icla_int_t lddb,
    icla_int_t batchCount, icla_queue_t queue);

void
setup_pivinfo(
    icla_int_t *pivinfo, icla_int_t *ipiv,
    icla_int_t m, icla_int_t nb,
    icla_queue_t queue);

void
iclablas_zgeadd_batched(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr  const dAarray[], icla_int_t ldda,
    iclaDoubleComplex_ptr              dBarray[], icla_int_t lddb,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_zlacpy_internal_batched(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    iclaDoubleComplex_const_ptr const dAarray[], icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    iclaDoubleComplex_ptr             dBarray[], icla_int_t Bi, icla_int_t Bj, icla_int_t lddb,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_zlacpy_batched(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    iclaDoubleComplex_const_ptr  const dAarray[], icla_int_t ldda,
    iclaDoubleComplex_ptr              dBarray[], icla_int_t lddb,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_zgemv_batched_core(
    icla_trans_t trans, icla_int_t m, icla_int_t n,
    const iclaDoubleComplex alpha,
    iclaDoubleComplex const * const * dA_array, const iclaDoubleComplex* dA, icla_int_t ldda, icla_int_t strideA,
    iclaDoubleComplex const * const * dx_array, const iclaDoubleComplex* dx, icla_int_t incx, icla_int_t stridex,
    const iclaDoubleComplex beta,
    iclaDoubleComplex** dy_array, iclaDoubleComplex_ptr dy, icla_int_t incy, icla_int_t stridey,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_zgemv_batched(
    icla_trans_t trans, icla_int_t m, icla_int_t n,
    const iclaDoubleComplex alpha,
    iclaDoubleComplex const * const * dA_array, icla_int_t ldda,
    iclaDoubleComplex const * const * dx_array, icla_int_t incx,
    const iclaDoubleComplex beta,
    iclaDoubleComplex** dy_array, icla_int_t incy,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_zgemv_batched_strided(
    icla_trans_t trans, icla_int_t m, icla_int_t n,
    const iclaDoubleComplex alpha,
    const iclaDoubleComplex* dA, icla_int_t ldda, icla_int_t strideA,
    const iclaDoubleComplex* dx, icla_int_t incx, icla_int_t stridex,
    const iclaDoubleComplex beta,
    iclaDoubleComplex* dy, icla_int_t incy, icla_int_t stridey,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
iclablas_zgemv_batched_smallsq(
    icla_trans_t trans, icla_int_t n,
    const iclaDoubleComplex alpha,
    iclaDoubleComplex const * const * dA_array, icla_int_t ldda,
    iclaDoubleComplex const * const * dx_array, icla_int_t incx,
    const iclaDoubleComplex beta,
    iclaDoubleComplex** dy_array, icla_int_t incy,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
iclablas_zgemv_batched_strided_smallsq(
    icla_trans_t transA, icla_int_t n,
    const iclaDoubleComplex alpha,
    const iclaDoubleComplex* dA, icla_int_t ldda, icla_int_t strideA,
    const iclaDoubleComplex* dx, icla_int_t incx, icla_int_t stridex,
    const iclaDoubleComplex beta,
    iclaDoubleComplex* dy, icla_int_t incy, icla_int_t stridey,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zgeqrf_batched_smallsq(
    icla_int_t n,
    iclaDoubleComplex** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    iclaDoubleComplex **dtau_array, icla_int_t taui, icla_int_t* info_array,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_zgeqrf_batched(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex **dA_array,
    icla_int_t lda,
    iclaDoubleComplex **dtau_array,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zgeqrf_batched_work(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex **dA_array, icla_int_t ldda,
    iclaDoubleComplex **dtau_array,
    icla_int_t *info_array,
    void* device_work, icla_int_t* device_lwork,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zgeqrf_expert_batched(
    icla_int_t m, icla_int_t n, icla_int_t nb,
    iclaDoubleComplex **dA_array, icla_int_t ldda,
    iclaDoubleComplex **dR_array, icla_int_t lddr,
    iclaDoubleComplex **dT_array, icla_int_t lddt,
    iclaDoubleComplex **dtau_array, icla_int_t provide_RT,
    iclaDoubleComplex **dW_array,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zgeqrf_batched_v4(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex **dA_array,
    icla_int_t lda,
    iclaDoubleComplex **tau_array,
    icla_int_t *info_array,
    icla_int_t batchCount);

icla_int_t
icla_zgeqrf_panel_fused_update_batched(
        icla_int_t m, icla_int_t n, icla_int_t nb,
        iclaDoubleComplex** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
        iclaDoubleComplex** tau_array, icla_int_t taui,
        iclaDoubleComplex** dR_array, icla_int_t Ri, icla_int_t Rj, icla_int_t lddr,
        icla_int_t *info_array, icla_int_t separate_R_V,
        icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zgeqrf_panel_internal_batched(
        icla_int_t m, icla_int_t n, icla_int_t nb,
        iclaDoubleComplex** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
        iclaDoubleComplex** tau_array, icla_int_t taui,
        iclaDoubleComplex** dT_array, icla_int_t Ti, icla_int_t Tj, icla_int_t lddt,
        iclaDoubleComplex** dR_array, icla_int_t Ri, icla_int_t Rj, icla_int_t lddr,
        iclaDoubleComplex** dwork_array,
        icla_int_t *info_array,
        icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zgeqrf_panel_batched(
    icla_int_t m, icla_int_t n, icla_int_t nb,
    iclaDoubleComplex** dA_array,    icla_int_t ldda,
    iclaDoubleComplex** tau_array,
    iclaDoubleComplex** dT_array, icla_int_t ldt,
    iclaDoubleComplex** dR_array, icla_int_t ldr,
    iclaDoubleComplex** dwork_array,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zgels_batched(
    icla_trans_t trans, icla_int_t m, icla_int_t n, icla_int_t nrhs,
    iclaDoubleComplex **dA_array, icla_int_t ldda,
    iclaDoubleComplex **dB_array, icla_int_t lddb,
    iclaDoubleComplex *hwork, icla_int_t lwork,
    icla_int_t *info,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zgeqr2_fused_reg_tall_batched(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    iclaDoubleComplex **dtau_array, icla_int_t taui,
    icla_int_t* info_array, icla_int_t check_launch_only,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_zgeqr2_fused_reg_medium_batched(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    iclaDoubleComplex **dtau_array, icla_int_t taui,
    icla_int_t* info_array, icla_int_t check_launch_only,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_zgeqr2_fused_reg_batched(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    iclaDoubleComplex **dtau_array, icla_int_t taui,
    icla_int_t* info_array, icla_int_t check_launch_only,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_zgeqr2_fused_sm_batched(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    iclaDoubleComplex **dtau_array, icla_int_t taui,
    icla_int_t* info_array, icla_int_t nthreads, icla_int_t check_launch_only,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_zgeqr2_batched(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex **dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    iclaDoubleComplex **dtau_array, icla_int_t taui,
    icla_int_t *info_array, icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zlarf_fused_reg_tall_batched(
    icla_int_t m, icla_int_t n, icla_int_t nb, icla_int_t ib,
    iclaDoubleComplex** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    iclaDoubleComplex** dV_array, icla_int_t Vi, icla_int_t Vj, icla_int_t lddv,
    iclaDoubleComplex **dtau_array, icla_int_t taui,
    icla_int_t check_launch_only,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_zlarf_fused_reg_medium_batched(
    icla_int_t m, icla_int_t n, icla_int_t nb, icla_int_t ib,
    iclaDoubleComplex** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    iclaDoubleComplex** dV_array, icla_int_t Vi, icla_int_t Vj, icla_int_t lddv,
    iclaDoubleComplex **dtau_array, icla_int_t taui,
    icla_int_t check_launch_only,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_zlarf_fused_reg_batched(
    icla_int_t m, icla_int_t n, icla_int_t nb, icla_int_t ib,
    iclaDoubleComplex** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    iclaDoubleComplex** dV_array, icla_int_t Vi, icla_int_t Vj, icla_int_t lddv,
    iclaDoubleComplex **dtau_array, icla_int_t taui,
    icla_int_t check_launch_only,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_zlarf_fused_sm_batched(
    icla_int_t m, icla_int_t n, icla_int_t nb, icla_int_t ib,
    iclaDoubleComplex** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    iclaDoubleComplex** dV_array, icla_int_t Vi, icla_int_t Vj, icla_int_t lddv,
    iclaDoubleComplex **dtau_array, icla_int_t taui,
    icla_int_t nthreads, icla_int_t check_launch_only,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_zlarfb_gemm_internal_batched(
    icla_side_t side, icla_trans_t trans, icla_direct_t direct, icla_storev_t storev,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDoubleComplex_const_ptr dV_array[],    icla_int_t vi, icla_int_t vj, icla_int_t lddv,
    iclaDoubleComplex_const_ptr dT_array[],    icla_int_t Ti, icla_int_t Tj, icla_int_t lddt,
    iclaDoubleComplex_ptr dC_array[],          icla_int_t Ci, icla_int_t Cj, icla_int_t lddc,
    iclaDoubleComplex_ptr dwork_array[],       icla_int_t ldwork,
    iclaDoubleComplex_ptr dworkvt_array[],     icla_int_t ldworkvt,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zlarfb_gemm_batched(
    icla_side_t side, icla_trans_t trans, icla_direct_t direct, icla_storev_t storev,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDoubleComplex_const_ptr dV_array[],    icla_int_t lddv,
    iclaDoubleComplex_const_ptr dT_array[],    icla_int_t lddt,
    iclaDoubleComplex_ptr dC_array[],          icla_int_t lddc,
    iclaDoubleComplex_ptr dwork_array[],       icla_int_t ldwork,
    iclaDoubleComplex_ptr dworkvt_array[],     icla_int_t ldworkvt,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zlarft_internal_batched(
        icla_int_t n, icla_int_t k, icla_int_t stair_T,
        iclaDoubleComplex **v_array,   icla_int_t vi, icla_int_t vj, icla_int_t ldv,
        iclaDoubleComplex **tau_array, icla_int_t taui,
        iclaDoubleComplex **T_array,   icla_int_t Ti, icla_int_t Tj, icla_int_t ldt,
        iclaDoubleComplex **work_array, icla_int_t lwork,
        icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zlarft_batched(
    icla_int_t n, icla_int_t k, icla_int_t stair_T,
    iclaDoubleComplex **v_array, icla_int_t ldv,
    iclaDoubleComplex **tau_array,
    iclaDoubleComplex **T_array, icla_int_t ldt,
    iclaDoubleComplex **work_array, icla_int_t lwork,
    icla_int_t batchCount, icla_queue_t queue);

void
icla_zlarft_sm32x32_batched(
    icla_int_t n, icla_int_t k,
    iclaDoubleComplex **v_array, icla_int_t vi, icla_int_t vj, icla_int_t ldv,
    iclaDoubleComplex **tau_array, icla_int_t taui,
    iclaDoubleComplex **T_array, icla_int_t Ti, icla_int_t Tj, icla_int_t ldt,
    icla_int_t batchCount, icla_queue_t queue);

void iclablas_zlarft_recztrmv_sm32x32(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex *tau,
    iclaDoubleComplex *Trec, icla_int_t ldtrec,
    iclaDoubleComplex *Ttri, icla_int_t ldttri,
    icla_queue_t queue);

void iclablas_zlarft_recztrmv_sm32x32_batched(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex **tau_array,  icla_int_t taui,
    iclaDoubleComplex **Trec_array, icla_int_t Treci, icla_int_t Trecj, icla_int_t ldtrec,
    iclaDoubleComplex **Ttri_array, icla_int_t Ttrii, icla_int_t Ttrij, icla_int_t ldttri,
    icla_int_t batchCount, icla_queue_t queue);

void iclablas_zlarft_ztrmv_sm32x32(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex *tau,
    iclaDoubleComplex *Tin, icla_int_t ldtin,
    iclaDoubleComplex *Tout, icla_int_t ldtout,
    icla_queue_t queue);

void iclablas_zlarft_ztrmv_sm32x32_batched(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex **tau_array, icla_int_t taui,
    iclaDoubleComplex **Tin_array, icla_int_t Tini, icla_int_t Tinj, icla_int_t ldtin,
    iclaDoubleComplex **Tout_array, icla_int_t Touti, icla_int_t Toutj, icla_int_t ldtout,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_dznrm2_cols_batched(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex **dA_array, icla_int_t lda,
    double **dxnorm_array,
    icla_int_t batchCount);

void
icla_zlarfgx_batched(
    icla_int_t n, iclaDoubleComplex **dx0_array, iclaDoubleComplex **dx_array,
    iclaDoubleComplex **dtau_array, double **dxnorm_array,
    iclaDoubleComplex **dR_array, icla_int_t it,
    icla_int_t batchCount);

void
icla_zlarfx_batched_v4(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex **v_array,
    iclaDoubleComplex **tau_array,
    iclaDoubleComplex **C_array, icla_int_t ldc, double **xnorm_array,
    icla_int_t step,
    icla_int_t batchCount);

void
iclablas_zlarfg_batched(
    icla_int_t n,
    iclaDoubleComplex** dalpha_array,
    iclaDoubleComplex** dx_array, icla_int_t incx,
    iclaDoubleComplex** dtau_array,
    icla_int_t batchCount );

icla_int_t
icla_zpotrf_lpout_batched(
    icla_uplo_t uplo, icla_int_t n,
    iclaDoubleComplex **dA_array, icla_int_t ai, icla_int_t aj, icla_int_t lda, icla_int_t gbstep,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zpotrf_lpin_batched(
    icla_uplo_t uplo, icla_int_t n,
    iclaDoubleComplex **dA_array, icla_int_t ai, icla_int_t aj, icla_int_t lda, icla_int_t gbstep,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_zpotrf_v33_batched(
    icla_uplo_t uplo, icla_int_t n,
    iclaDoubleComplex **dA_array, icla_int_t lda,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

// host interface
void
blas_zlacpy_batched(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    iclaDoubleComplex const * const * hA_array, icla_int_t lda,
    iclaDoubleComplex               **hB_array, icla_int_t ldb,
    icla_int_t batchCount );

void
blas_zgemm_batched(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex const * const * hA_array, icla_int_t lda,
    iclaDoubleComplex const * const * hB_array, icla_int_t ldb,
    iclaDoubleComplex beta,
    iclaDoubleComplex **hC_array, icla_int_t ldc,
    icla_int_t batchCount );

void
blas_ztrsm_batched(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t m, icla_int_t n,
        iclaDoubleComplex alpha,
        iclaDoubleComplex **hA_array, icla_int_t lda,
        iclaDoubleComplex **hB_array, icla_int_t ldb,
        icla_int_t batchCount );

void
blas_ztrmm_batched(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t m, icla_int_t n,
        iclaDoubleComplex alpha,
        iclaDoubleComplex **hA_array, icla_int_t lda,
        iclaDoubleComplex **hB_array, icla_int_t ldb,
        icla_int_t batchCount );

void
blas_zhemm_batched(
        icla_side_t side, icla_uplo_t uplo,
        icla_int_t m, icla_int_t n,
        iclaDoubleComplex alpha,
        iclaDoubleComplex **hA_array, icla_int_t lda,
        iclaDoubleComplex **hB_array, icla_int_t ldb,
        iclaDoubleComplex beta,
        iclaDoubleComplex **hC_array, icla_int_t ldc,
        icla_int_t batchCount );

void
blas_zherk_batched(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    double alpha, iclaDoubleComplex const * const * hA_array, icla_int_t lda,
    double beta,  iclaDoubleComplex               **hC_array, icla_int_t ldc,
    icla_int_t batchCount );

void
blas_zher2k_batched(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    iclaDoubleComplex alpha, iclaDoubleComplex const * const * hA_array, icla_int_t lda,
                              iclaDoubleComplex const * const * hB_array, icla_int_t ldb,
    double beta,              iclaDoubleComplex               **hC_array, icla_int_t ldc,
    icla_int_t batchCount );

// for debugging purpose
void
zset_stepinit_ipiv(
    icla_int_t **ipiv_array,
    icla_int_t pm,
    icla_int_t batchCount);

#ifdef __cplusplus
}
#endif

#undef ICLA_COMPLEX

#endif  /* ICLA_ZBATCHED_H */
