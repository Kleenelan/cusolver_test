/*
    -- ICLA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Azzam Haidar
       @author Tingxing Dong

       @generated from include/icla_zbatched.h, normal z -> d, Fri Nov 29 12:16:14 2024
*/

#ifndef ICLA_DBATCHED_H
#define ICLA_DBATCHED_H

#include "icla_types.h"

#define ICLA_REAL

#ifdef __cplusplus
extern "C" {
#endif
  /*
   *  local auxiliary routines
   */
void
icla_dset_pointer(
    double **output_array,
    double *input,
    icla_int_t lda,
    icla_int_t row, icla_int_t column,
    icla_int_t batch_offset,
    icla_int_t batchCount, icla_queue_t queue);

void
icla_ddisplace_pointers(
    double **output_array,
    double **input_array, icla_int_t lda,
    icla_int_t row, icla_int_t column,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_drecommend_cublas_gemm_batched(
    icla_trans_t transa, icla_trans_t transb,
    icla_int_t m, icla_int_t n, icla_int_t k);

icla_int_t
icla_drecommend_cublas_gemm_stream(
    icla_trans_t transa, icla_trans_t transb,
    icla_int_t m, icla_int_t n, icla_int_t k);

void icla_get_dpotrf_batched_nbparam(icla_int_t n, icla_int_t *nb, icla_int_t *recnb);

icla_int_t icla_get_dpotrf_batched_crossover();

void icla_get_dgetrf_batched_nbparam(icla_int_t n, icla_int_t *nb, icla_int_t *recnb);
icla_int_t icla_get_dgetrf_batched_ntcol(icla_int_t m, icla_int_t n);
icla_int_t icla_get_dgemm_batched_ntcol(icla_int_t n);
icla_int_t icla_get_dgemm_batched_smallsq_limit(icla_int_t n);
icla_int_t icla_get_dgeqrf_batched_nb(icla_int_t m);
icla_int_t icla_use_dgeqrf_batched_fused_update(icla_int_t m, icla_int_t n, icla_int_t batchCount);
icla_int_t icla_get_dgeqr2_fused_sm_batched_nthreads(icla_int_t m, icla_int_t n);
icla_int_t icla_get_dgeqrf_batched_ntcol(icla_int_t m, icla_int_t n);
icla_int_t icla_get_dgetri_batched_ntcol(icla_int_t m, icla_int_t n);
icla_int_t icla_get_dtrsm_batched_stop_nb(icla_side_t side, icla_int_t m, icla_int_t n);
void icla_get_dgbtrf_batched_params(icla_int_t m, icla_int_t n, icla_int_t kl, icla_int_t ku, icla_int_t *nb, icla_int_t *threads);

void
iclablas_dswapdblk_batched(
    icla_int_t n, icla_int_t nb,
    double **dA, icla_int_t ldda, icla_int_t inca,
    double **dB, icla_int_t lddb, icla_int_t incb,
    icla_int_t batchCount, icla_queue_t queue );

  /*
   *  LAPACK batched routines
   */

  /*
   *  BLAS batched routines
   */
void
iclablas_dgemm_batched_core(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    double alpha,
    double const * const * dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    double const * const * dB_array, icla_int_t Bi, icla_int_t Bj, icla_int_t lddb,
    double beta,
    double **dC_array, icla_int_t Ci, icla_int_t Cj, icla_int_t lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
icla_dgemm_batched_core(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    double alpha,
    double const * const * dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    double const * const * dB_array, icla_int_t Bi, icla_int_t Bj, icla_int_t lddb,
    double beta,
    double **dC_array, icla_int_t Ci, icla_int_t Cj, icla_int_t lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
icla_dgemm_batched(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    double alpha,
    double const * const * dA_array, icla_int_t ldda,
    double const * const * dB_array, icla_int_t lddb,
    double beta,
    double **dC_array, icla_int_t lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_dgemm_batched(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    double alpha,
    double const * const * dA_array, icla_int_t ldda,
    double const * const * dB_array, icla_int_t lddb,
    double beta,
    double **dC_array, icla_int_t lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_dgemm_batched_strided(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    double alpha,
    double const * dA, icla_int_t ldda, icla_int_t strideA,
    double const * dB, icla_int_t lddb, icla_int_t strideB,
    double beta,
    double       * dC, icla_int_t lddc, icla_int_t strideC,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_dgemm_batched_smallsq(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    double alpha,
    double const * const * dA_array, icla_int_t ai, icla_int_t aj, icla_int_t ldda,
    double const * const * dB_array, icla_int_t bi, icla_int_t bj, icla_int_t lddb,
    double beta,
    double **dC_array, icla_int_t ci, icla_int_t cj, icla_int_t lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_dsyrk_batched_core(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    double alpha,
    double const * const * dA_array, icla_int_t ai, icla_int_t aj, icla_int_t ldda,
    double const * const * dB_array, icla_int_t bi, icla_int_t bj, icla_int_t lddb,
    double beta,
    double **dC_array, icla_int_t ci, icla_int_t cj, icla_int_t lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_dsyrk_batched_core(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    double alpha,
    double const * const * dA_array, icla_int_t ai, icla_int_t aj, icla_int_t ldda,
    double const * const * dB_array, icla_int_t bi, icla_int_t bj, icla_int_t lddb,
    double beta,
    double **dC_array, icla_int_t ci, icla_int_t cj, icla_int_t lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_dsyrk_batched(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    double alpha,
    double const * const * dA_array, icla_int_t ldda,
    double beta,
    double **dC_array, icla_int_t lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
icla_dsyrk_batched(
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t n, icla_int_t k,
    double alpha,
    double const * const * dA_array, icla_int_t ldda,
    double beta,
    double **dC_array, icla_int_t lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_dsyrk_batched(
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t n, icla_int_t k,
    double alpha,
    double const * const * dA_array, icla_int_t ldda,
    double beta,
    double **dC_array, icla_int_t lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_dsyr2k_batched(
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t n, icla_int_t k,
    double alpha,
    double const * const * dA_array, icla_int_t ldda,
    double const * const * dB_array, icla_int_t lddb,
    double beta, double **dC_array, icla_int_t lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_dsyr2k_batched(
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t n, icla_int_t k,
    double alpha,
    double const * const * dA_array, icla_int_t ldda,
    double const * const * dB_array, icla_int_t lddb,
    double beta, double **dC_array, icla_int_t lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_dtrtri_diag_batched(
    icla_uplo_t uplo, icla_diag_t diag, icla_int_t n,
    double const * const *dA_array, icla_int_t ldda,
    double **dinvA_array,
    icla_int_t resetozero,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_dtrsm_small_batched(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t m, icla_int_t n,
        double alpha,
        double **dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
        double **dB_array, icla_int_t Bi, icla_int_t Bj, icla_int_t lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_dtrsm_recursive_batched(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t m, icla_int_t n,
        double alpha,
        double **dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
        double **dB_array, icla_int_t Bi, icla_int_t Bj, icla_int_t lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_dtrsm_batched(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t m, icla_int_t n,
        double alpha,
        double **dA_array, icla_int_t ldda,
        double **dB_array, icla_int_t lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_dtrsm_inv_batched(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    double alpha,
    double** dA_array,    icla_int_t ldda,
    double** dB_array,    icla_int_t lddb,
    icla_int_t batchCount, icla_queue_t queue);

void iclablas_dtrsm_inv_work_batched(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t flag, icla_int_t m, icla_int_t n,
    double alpha,
    double** dA_array,    icla_int_t ldda,
    double** dB_array,    icla_int_t lddb,
    double** dX_array,    icla_int_t lddx,
    double** dinvA_array, icla_int_t dinvA_length,
    double** dA_displ, double** dB_displ,
    double** dX_displ, double** dinvA_displ,
    icla_int_t resetozero,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_dtrsm_inv_outofplace_batched(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t flag, icla_int_t m, icla_int_t n,
    double alpha,
    double** dA_array,    icla_int_t ldda,
    double** dB_array,    icla_int_t lddb,
    double** dX_array,    icla_int_t lddx,
    double** dinvA_array, icla_int_t dinvA_length,
    double** dA_displ, double** dB_displ,
    double** dX_displ, double** dinvA_displ,
    icla_int_t resetozero,
    icla_int_t batchCount, icla_queue_t queue);

void iclablas_dtrsv_batched(
    icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t n,
    double** dA_array,    icla_int_t ldda,
    double** dB_array,    icla_int_t incb,
    icla_int_t batchCount, icla_queue_t queue);

void iclablas_dtrsv_work_batched(
    icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t n,
    double** dA_array,    icla_int_t ldda,
    double** dB_array,    icla_int_t incb,
    double** dX_array,
    icla_int_t batchCount, icla_queue_t queue);

void iclablas_dtrsv_outofplace_batched(
    icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t n,
    double ** A_array, icla_int_t lda,
    double **b_array, icla_int_t incb,
    double **x_array,
    icla_int_t batchCount, icla_queue_t queue, icla_int_t flag);

void
iclablas_dtrmm_batched_core(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t m, icla_int_t n,
        double alpha,
        double **dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
        double **dB_array, icla_int_t Bi, icla_int_t Bj, icla_int_t lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_dtrmm_batched(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t m, icla_int_t n,
        double alpha,
        double **dA_array, icla_int_t ldda,
        double **dB_array, icla_int_t lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_dsymm_batched_core(
        icla_side_t side, icla_uplo_t uplo,
        icla_int_t m, icla_int_t n,
        double alpha,
        double **dA_array, icla_int_t ldda,
        double **dB_array, icla_int_t lddb,
        double beta,
        double **dC_array, icla_int_t lddc,
        icla_int_t roffA, icla_int_t coffA, icla_int_t roffB, icla_int_t coffB, icla_int_t roffC, icla_int_t coffC,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_dsymm_batched(
        icla_side_t side, icla_uplo_t uplo,
        icla_int_t m, icla_int_t n,
        double alpha,
        double **dA_array, icla_int_t ldda,
        double **dB_array, icla_int_t lddb,
        double beta,
        double **dC_array, icla_int_t lddc,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_dsymv_batched_core(
        icla_uplo_t uplo, icla_int_t n,
        double alpha,
        double **dA_array, icla_int_t ldda,
        double **dX_array, icla_int_t incx,
        double beta,
        double **dY_array, icla_int_t incy,
        icla_int_t offA, icla_int_t offX, icla_int_t offY,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_dsymv_batched(
        icla_uplo_t uplo, icla_int_t n,
        double alpha,
        double **dA_array, icla_int_t ldda,
        double **dX_array, icla_int_t incx,
        double beta,
        double **dY_array, icla_int_t incy,
        icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_dpotrf_batched(
    icla_uplo_t uplo, icla_int_t n,
    double **dA_array, icla_int_t lda,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dpotf2_batched(
    icla_uplo_t uplo, icla_int_t n,
    double **dA_array, icla_int_t ai, icla_int_t aj, icla_int_t lda,
    icla_int_t *info_array, icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dpotrf_panel_batched(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nb,
    double** dA_array, icla_int_t ai, icla_int_t aj, icla_int_t ldda,
    icla_int_t *info_array, icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dpotrf_recpanel_batched(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n, icla_int_t min_recpnb,
    double** dA_array, icla_int_t ai, icla_int_t aj, icla_int_t ldda,
    icla_int_t *info_array, icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dpotrf_rectile_batched(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n, icla_int_t min_recpnb,
    double** dA_array, icla_int_t ai, icla_int_t aj, icla_int_t ldda,
    icla_int_t *info_array, icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dpotrs_batched(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nrhs,
    double **dA_array, icla_int_t ldda,
    double **dB_array, icla_int_t lddb,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dposv_batched(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nrhs,
    double **dA_array, icla_int_t ldda,
    double **dB_array, icla_int_t lddb,
    icla_int_t *dinfo_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dgetrs_batched(
    icla_trans_t trans, icla_int_t n, icla_int_t nrhs,
    double **dA_array, icla_int_t ldda,
    icla_int_t **dipiv_array,
    double **dB_array, icla_int_t lddb,
    icla_int_t batchCount, icla_queue_t queue);

void
icla_dlaswp_rowparallel_batched(
    icla_int_t n,
    double**  input_array, icla_int_t  input_i, icla_int_t  input_j, icla_int_t ldi,
    double** output_array, icla_int_t output_i, icla_int_t output_j, icla_int_t ldo,
    icla_int_t k1, icla_int_t k2,
    icla_int_t **pivinfo_array,
    icla_int_t batchCount, icla_queue_t queue);

void
icla_dlaswp_rowserial_batched(
    icla_int_t n, double** dA_array, icla_int_t lda,
    icla_int_t k1, icla_int_t k2,
    icla_int_t **ipiv_array,
    icla_int_t batchCount, icla_queue_t queue);

void
icla_dlaswp_columnserial_batched(
    icla_int_t n, double** dA_array, icla_int_t lda,
    icla_int_t k1, icla_int_t k2,
    icla_int_t **ipiv_array,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_dtranspose_batched(
    icla_int_t m, icla_int_t n,
    double **dA_array,  icla_int_t ldda,
    double **dAT_array, icla_int_t lddat,
    icla_int_t batchCount, icla_queue_t queue);

void iclablas_dlaset_internal_batched(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    double offdiag, double diag,
    iclaDouble_ptr dAarray[], icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_dlaset_batched(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    double offdiag, double diag,
    iclaDouble_ptr dAarray[], icla_int_t ldda,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dgbsv_batched(
    icla_int_t n, icla_int_t kl, icla_int_t ku, icla_int_t nrhs,
    double **dA_array, icla_int_t ldda, icla_int_t **dipiv_array,
    double** dB_array, icla_int_t lddb,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dgbsv_batched_fused_sm(
    icla_int_t n, icla_int_t kl, icla_int_t ku, icla_int_t nrhs,
    double** dA_array, icla_int_t ldda, icla_int_t** ipiv_array,
    double** dB_array, icla_int_t lddb, icla_int_t* info_array,
    icla_int_t nthreads, icla_int_t ntcol,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_dgbsv_batched_strided_work(
    icla_int_t n, icla_int_t kl, icla_int_t ku, icla_int_t nrhs,
    double* dA, icla_int_t ldda, icla_int_t strideA,
    icla_int_t* dipiv, icla_int_t stride_piv,
    double* dB, icla_int_t lddb, icla_int_t strideB,
    icla_int_t *info_array,
    void* device_work, icla_int_t *lwork,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dgbsv_batched_strided(
    icla_int_t n, icla_int_t kl, icla_int_t ku, icla_int_t nrhs,
    double* dA, icla_int_t ldda, icla_int_t strideA,
    icla_int_t* dipiv, icla_int_t stride_piv,
    double* dB, icla_int_t lddb, icla_int_t strideB,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dgbsv_batched_work(
    icla_int_t n, icla_int_t kl, icla_int_t ku, icla_int_t nrhs,
    double** dA_array, icla_int_t ldda, icla_int_t **dipiv_array,
    double** dB_array, icla_int_t lddb,
    icla_int_t *info_array,
    void* device_work, icla_int_t *lwork,
    icla_int_t batchCount, icla_queue_t queue);

void
icla_dgbtrf_set_fillin(
    icla_int_t n, icla_int_t kl, icla_int_t ku,
    double** dAB_array, icla_int_t lddab,
    icla_int_t** dipiv_array, int* ju_array, icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dgbtf2_dswap_batched(
    icla_int_t kl, icla_int_t ku,
    double **dAB_array, icla_int_t ai, icla_int_t aj, icla_int_t lddab,
    icla_int_t** dipiv_array, icla_int_t ipiv_offset,
    int* ju_array, icla_int_t gbstep, icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dgbtf2_scal_ger_batched(
    icla_int_t m, icla_int_t n, icla_int_t kl, icla_int_t ku,
    double **dAB_array, icla_int_t ai, icla_int_t aj, icla_int_t lddab,
    int* ju_array, icla_int_t gbstep, icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dgbtrf_batched_fused_sm(
    icla_int_t m,  icla_int_t n,
    icla_int_t kl, icla_int_t ku,
    double** dAB_array, icla_int_t lddab,
    icla_int_t** ipiv_array, icla_int_t* info_array,
    icla_int_t nthreads, icla_int_t ntcol,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_dgbtrf_batched_sliding_window_loopout(
    icla_int_t m,  icla_int_t n,
    icla_int_t kl, icla_int_t ku,
    double** dAB_array, icla_int_t lddab,
    icla_int_t** ipiv_array, icla_int_t* info_array,
    void* device_work, icla_int_t *lwork,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_dgbtrf_batched_sliding_window_loopin(
    icla_int_t m,  icla_int_t n,
    icla_int_t kl, icla_int_t ku,
    double** dAB_array, icla_int_t lddab,
    icla_int_t** ipiv_array, icla_int_t* info_array,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_dgbtrf_batched_strided(
        icla_int_t m, icla_int_t n,
        icla_int_t kl, icla_int_t ku,
        double* dAB, icla_int_t lddab, icla_int_t strideAB,
        icla_int_t* dipiv, icla_int_t stride_piv,
        icla_int_t *info_array,
        icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dgbtrf_batched_strided_work(
        icla_int_t m, icla_int_t n,
        icla_int_t kl, icla_int_t ku,
        double* dAB, icla_int_t lddab, icla_int_t strideAB,
        icla_int_t* dipiv, icla_int_t stride_piv,
        icla_int_t *info_array,
        void* device_work, icla_int_t *lwork,
        icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dgbtrf_batched_work(
        icla_int_t m, icla_int_t n,
        icla_int_t kl, icla_int_t ku,
        double **dAB_array, icla_int_t lddab,
        icla_int_t **dipiv_array, icla_int_t *info_array,
        void* device_work, icla_int_t *lwork,
        icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dgbtrf_batched(
    icla_int_t m, icla_int_t n,
    icla_int_t kl, icla_int_t ku,
    double **dAB_array, icla_int_t lddab,
    icla_int_t **dipiv_array, icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dgbtrs_batched(
    icla_trans_t transA,
    icla_int_t n, icla_int_t kl, icla_int_t ku, icla_int_t nrhs,
    double **dA_array, icla_int_t ldda, icla_int_t **dipiv_array,
    double** dB_array, icla_int_t lddb,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dgbtrs_lower_batched(
    icla_trans_t transA,
    icla_int_t n, icla_int_t kl, icla_int_t ku, icla_int_t nrhs,
    double** dA_array, icla_int_t ldda, icla_int_t **dipiv_array,
    double** dB_array, icla_int_t lddb,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
iclablas_dgbtrs_lower_blocked_batched(
        icla_int_t n, icla_int_t kl, icla_int_t ku, icla_int_t nrhs,
        double** dA_array, icla_int_t ldda, icla_int_t** dipiv_array,
        double** dB_array, icla_int_t lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_dgbtrs_swap_batched(
    icla_int_t n, double** dA_array, icla_int_t ldda,
    icla_int_t** dipiv_array, icla_int_t j,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dgbtrs_upper_batched(
    icla_trans_t transA,
    icla_int_t n, icla_int_t kl, icla_int_t ku, icla_int_t nrhs,
    double** dA_array, icla_int_t ldda, icla_int_t **dipiv_array,
    double** dB_array, icla_int_t lddb,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
iclablas_dgbtrs_upper_blocked_batched(
        icla_int_t n, icla_int_t kl, icla_int_t ku, icla_int_t nrhs,
        double** dA_array, icla_int_t ldda,
        double** dB_array, icla_int_t lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_dgbtrs_upper_columnwise_batched(
    icla_int_t n, icla_int_t kl, icla_int_t ku,
    icla_int_t nrhs, icla_int_t j,
    double** dA_array, icla_int_t ldda,
    double** dB_array, icla_int_t lddb,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_dger_batched_core(
    icla_int_t m, icla_int_t n,
    double alpha,
    double** dX_array, icla_int_t xi, icla_int_t xj, icla_int_t lddx, icla_int_t incx,
    double** dY_array, icla_int_t yi, icla_int_t yj, icla_int_t lddy, icla_int_t incy,
    double** dA_array, icla_int_t ai, icla_int_t aj, icla_int_t ldda,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_dgesv_batched_small(
    icla_int_t n, icla_int_t nrhs,
    double** dA_array, icla_int_t ldda,
    icla_int_t** dipiv_array,
    double **dB_array, icla_int_t lddb,
    icla_int_t* dinfo_array,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_dgetf2_batched(
    icla_int_t m, icla_int_t n,
    double **dA_array, icla_int_t ai, icla_int_t aj, icla_int_t lda,
    icla_int_t **ipiv_array,
    icla_int_t **dpivinfo_array,
    icla_int_t *info_array,
    icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dgetrf_recpanel_batched(
    icla_int_t m, icla_int_t n, icla_int_t min_recpnb,
    double** dA_array, icla_int_t ai, icla_int_t aj, icla_int_t ldda,
    icla_int_t** dipiv_array, icla_int_t** dpivinfo_array,
    icla_int_t *info_array, icla_int_t gbstep,
    icla_int_t batchCount,  icla_queue_t queue);

icla_int_t
icla_dgetrf_batched(
    icla_int_t m, icla_int_t n,
    double **dA_array,
    icla_int_t lda,
    icla_int_t **ipiv_array,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dgetf2_fused_batched(
    icla_int_t m, icla_int_t n,
    double **dA_array, icla_int_t ai, icla_int_t aj, icla_int_t ldda,
    icla_int_t **dipiv_array,
    icla_int_t *info_array, icla_int_t batchCount,
    icla_queue_t queue);

icla_int_t
icla_dgetrf_batched_smallsq_noshfl(
    icla_int_t n,
    double** dA_array, icla_int_t ldda,
    icla_int_t** ipiv_array, icla_int_t* info_array,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_dgetri_outofplace_batched(
    icla_int_t n,
    double **dA_array, icla_int_t ldda,
    icla_int_t **dipiv_array,
    double **dinvA_array, icla_int_t lddia,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

void
icla_ddisplace_intpointers(
    icla_int_t **output_array,
    icla_int_t **input_array, icla_int_t lda,
    icla_int_t row, icla_int_t column,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_idamax_atomic_batched(
    icla_int_t n,
    double** x_array, icla_int_t incx,
    icla_int_t **max_id_array,
    icla_int_t batchCount);

void
iclablas_idamax_tree_batched(
    icla_int_t n,
    double** x_array, icla_int_t incx,
    icla_int_t **max_id_array,
    icla_int_t batchCount);

void
iclablas_idamax_batched(
    icla_int_t n,
    double** x_array, icla_int_t incx,
    icla_int_t **max_id_array,
    icla_int_t batchCount);

void
iclablas_idamax(
    icla_int_t n,
    double* x, icla_int_t incx,
    icla_int_t *max_id);

icla_int_t
icla_idamax_batched(
        icla_int_t length,
        double **x_array, icla_int_t xi, icla_int_t xj, icla_int_t lda, icla_int_t incx,
        icla_int_t** ipiv_array, icla_int_t ipiv_i,
        icla_int_t step, icla_int_t gbstep, icla_int_t *info_array,
        icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dswap_batched(
    icla_int_t n, double **x_array, icla_int_t xi, icla_int_t xj, icla_int_t incx,
    icla_int_t step, icla_int_t** ipiv_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dscal_dger_batched(
    icla_int_t m, icla_int_t n,
    double **dA_array, icla_int_t ai, icla_int_t aj, icla_int_t lda,
    icla_int_t *info_array, icla_int_t step, icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dcomputecolumn_batched(
    icla_int_t m, icla_int_t paneloffset, icla_int_t step,
    double **dA_array,  icla_int_t lda,
    icla_int_t ai, icla_int_t aj,
    icla_int_t **ipiv_array,
    icla_int_t *info_array, icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue);

void
icla_dgetf2trsm_batched(
    icla_int_t ib, icla_int_t n,
    double **dA_array,  icla_int_t j, icla_int_t lda,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dgetf2_nopiv_internal_batched(
    icla_int_t m, icla_int_t n,
    double** dA_array, icla_int_t ai, icla_int_t aj, icla_int_t ldda,
    icla_int_t* info_array, icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_dgetf2_nopiv_batched(
    icla_int_t m, icla_int_t n,
    double **dA_array, icla_int_t ai, icla_int_t aj, icla_int_t ldda,
    icla_int_t *info_array, icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dgetrf_recpanel_nopiv_batched(
    icla_int_t m, icla_int_t n, icla_int_t min_recpnb,
    double** dA_array,    icla_int_t ldda,
    double** dX_array,    icla_int_t dX_length,
    double** dinvA_array, icla_int_t dinvA_length,
    double** dW1_displ, double** dW2_displ,
    double** dW3_displ, double** dW4_displ,
    double** dW5_displ,
    icla_int_t *info_array, icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dgetrf_nopiv_batched(
    icla_int_t m, icla_int_t n,
    double **dA_array,
    icla_int_t lda,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dgetrs_nopiv_batched(
    icla_trans_t trans, icla_int_t n, icla_int_t nrhs,
    double **dA_array, icla_int_t ldda,
    double **dB_array, icla_int_t lddb,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dgesv_nopiv_batched(
    icla_int_t n, icla_int_t nrhs,
    double **dA_array, icla_int_t ldda,
    double **dB_array, icla_int_t lddb,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dgesv_rbt_batched(
    icla_int_t n, icla_int_t nrhs,
    double **dA_array, icla_int_t ldda,
    double **dB_array, icla_int_t lddb,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dgesv_batched(
    icla_int_t n, icla_int_t nrhs,
    double **dA_array, icla_int_t ldda,
    icla_int_t **dipiv_array,
    double **dB_array, icla_int_t lddb,
    icla_int_t *dinfo_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dgerbt_batched(
    icla_bool_t gen, icla_int_t n, icla_int_t nrhs,
    double **dA_array, icla_int_t ldda,
    double **dB_array, icla_int_t lddb,
    double *U, double *V,
    icla_int_t *info,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_dprbt_batched(
    icla_int_t n,
    double **dA_array, icla_int_t ldda,
    double *du, double *dv,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_dprbt_mv_batched(
    icla_int_t n, icla_int_t nrhs,
    double *dv, double **db_array, icla_int_t lddb,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_dprbt_mtv_batched(
    icla_int_t n, icla_int_t nrhs,
    double *du, double **db_array, icla_int_t lddb,
    icla_int_t batchCount, icla_queue_t queue);

void
setup_pivinfo(
    icla_int_t *pivinfo, icla_int_t *ipiv,
    icla_int_t m, icla_int_t nb,
    icla_queue_t queue);

void
iclablas_dgeadd_batched(
    icla_int_t m, icla_int_t n,
    double alpha,
    iclaDouble_const_ptr  const dAarray[], icla_int_t ldda,
    iclaDouble_ptr              dBarray[], icla_int_t lddb,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_dlacpy_internal_batched(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    iclaDouble_const_ptr const dAarray[], icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    iclaDouble_ptr             dBarray[], icla_int_t Bi, icla_int_t Bj, icla_int_t lddb,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_dlacpy_batched(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    iclaDouble_const_ptr  const dAarray[], icla_int_t ldda,
    iclaDouble_ptr              dBarray[], icla_int_t lddb,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_dgemv_batched_core(
    icla_trans_t trans, icla_int_t m, icla_int_t n,
    const double alpha,
    double const * const * dA_array, const double* dA, icla_int_t ldda, icla_int_t strideA,
    double const * const * dx_array, const double* dx, icla_int_t incx, icla_int_t stridex,
    const double beta,
    double** dy_array, iclaDouble_ptr dy, icla_int_t incy, icla_int_t stridey,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_dgemv_batched(
    icla_trans_t trans, icla_int_t m, icla_int_t n,
    const double alpha,
    double const * const * dA_array, icla_int_t ldda,
    double const * const * dx_array, icla_int_t incx,
    const double beta,
    double** dy_array, icla_int_t incy,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_dgemv_batched_strided(
    icla_trans_t trans, icla_int_t m, icla_int_t n,
    const double alpha,
    const double* dA, icla_int_t ldda, icla_int_t strideA,
    const double* dx, icla_int_t incx, icla_int_t stridex,
    const double beta,
    double* dy, icla_int_t incy, icla_int_t stridey,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
iclablas_dgemv_batched_smallsq(
    icla_trans_t trans, icla_int_t n,
    const double alpha,
    double const * const * dA_array, icla_int_t ldda,
    double const * const * dx_array, icla_int_t incx,
    const double beta,
    double** dy_array, icla_int_t incy,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
iclablas_dgemv_batched_strided_smallsq(
    icla_trans_t transA, icla_int_t n,
    const double alpha,
    const double* dA, icla_int_t ldda, icla_int_t strideA,
    const double* dx, icla_int_t incx, icla_int_t stridex,
    const double beta,
    double* dy, icla_int_t incy, icla_int_t stridey,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dgeqrf_batched_smallsq(
    icla_int_t n,
    double** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    double **dtau_array, icla_int_t taui, icla_int_t* info_array,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_dgeqrf_batched(
    icla_int_t m, icla_int_t n,
    double **dA_array,
    icla_int_t lda,
    double **dtau_array,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dgeqrf_batched_work(
    icla_int_t m, icla_int_t n,
    double **dA_array, icla_int_t ldda,
    double **dtau_array,
    icla_int_t *info_array,
    void* device_work, icla_int_t* device_lwork,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dgeqrf_expert_batched(
    icla_int_t m, icla_int_t n, icla_int_t nb,
    double **dA_array, icla_int_t ldda,
    double **dR_array, icla_int_t lddr,
    double **dT_array, icla_int_t lddt,
    double **dtau_array, icla_int_t provide_RT,
    double **dW_array,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dgeqrf_batched_v4(
    icla_int_t m, icla_int_t n,
    double **dA_array,
    icla_int_t lda,
    double **tau_array,
    icla_int_t *info_array,
    icla_int_t batchCount);

icla_int_t
icla_dgeqrf_panel_fused_update_batched(
        icla_int_t m, icla_int_t n, icla_int_t nb,
        double** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
        double** tau_array, icla_int_t taui,
        double** dR_array, icla_int_t Ri, icla_int_t Rj, icla_int_t lddr,
        icla_int_t *info_array, icla_int_t separate_R_V,
        icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dgeqrf_panel_internal_batched(
        icla_int_t m, icla_int_t n, icla_int_t nb,
        double** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
        double** tau_array, icla_int_t taui,
        double** dT_array, icla_int_t Ti, icla_int_t Tj, icla_int_t lddt,
        double** dR_array, icla_int_t Ri, icla_int_t Rj, icla_int_t lddr,
        double** dwork_array,
        icla_int_t *info_array,
        icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dgeqrf_panel_batched(
    icla_int_t m, icla_int_t n, icla_int_t nb,
    double** dA_array,    icla_int_t ldda,
    double** tau_array,
    double** dT_array, icla_int_t ldt,
    double** dR_array, icla_int_t ldr,
    double** dwork_array,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dgels_batched(
    icla_trans_t trans, icla_int_t m, icla_int_t n, icla_int_t nrhs,
    double **dA_array, icla_int_t ldda,
    double **dB_array, icla_int_t lddb,
    double *hwork, icla_int_t lwork,
    icla_int_t *info,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dgeqr2_fused_reg_tall_batched(
    icla_int_t m, icla_int_t n,
    double** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    double **dtau_array, icla_int_t taui,
    icla_int_t* info_array, icla_int_t check_launch_only,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_dgeqr2_fused_reg_medium_batched(
    icla_int_t m, icla_int_t n,
    double** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    double **dtau_array, icla_int_t taui,
    icla_int_t* info_array, icla_int_t check_launch_only,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_dgeqr2_fused_reg_batched(
    icla_int_t m, icla_int_t n,
    double** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    double **dtau_array, icla_int_t taui,
    icla_int_t* info_array, icla_int_t check_launch_only,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_dgeqr2_fused_sm_batched(
    icla_int_t m, icla_int_t n,
    double** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    double **dtau_array, icla_int_t taui,
    icla_int_t* info_array, icla_int_t nthreads, icla_int_t check_launch_only,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_dgeqr2_batched(
    icla_int_t m, icla_int_t n,
    double **dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    double **dtau_array, icla_int_t taui,
    icla_int_t *info_array, icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dlarf_fused_reg_tall_batched(
    icla_int_t m, icla_int_t n, icla_int_t nb, icla_int_t ib,
    double** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    double** dV_array, icla_int_t Vi, icla_int_t Vj, icla_int_t lddv,
    double **dtau_array, icla_int_t taui,
    icla_int_t check_launch_only,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_dlarf_fused_reg_medium_batched(
    icla_int_t m, icla_int_t n, icla_int_t nb, icla_int_t ib,
    double** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    double** dV_array, icla_int_t Vi, icla_int_t Vj, icla_int_t lddv,
    double **dtau_array, icla_int_t taui,
    icla_int_t check_launch_only,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_dlarf_fused_reg_batched(
    icla_int_t m, icla_int_t n, icla_int_t nb, icla_int_t ib,
    double** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    double** dV_array, icla_int_t Vi, icla_int_t Vj, icla_int_t lddv,
    double **dtau_array, icla_int_t taui,
    icla_int_t check_launch_only,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_dlarf_fused_sm_batched(
    icla_int_t m, icla_int_t n, icla_int_t nb, icla_int_t ib,
    double** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    double** dV_array, icla_int_t Vi, icla_int_t Vj, icla_int_t lddv,
    double **dtau_array, icla_int_t taui,
    icla_int_t nthreads, icla_int_t check_launch_only,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_dlarfb_gemm_internal_batched(
    icla_side_t side, icla_trans_t trans, icla_direct_t direct, icla_storev_t storev,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDouble_const_ptr dV_array[],    icla_int_t vi, icla_int_t vj, icla_int_t lddv,
    iclaDouble_const_ptr dT_array[],    icla_int_t Ti, icla_int_t Tj, icla_int_t lddt,
    iclaDouble_ptr dC_array[],          icla_int_t Ci, icla_int_t Cj, icla_int_t lddc,
    iclaDouble_ptr dwork_array[],       icla_int_t ldwork,
    iclaDouble_ptr dworkvt_array[],     icla_int_t ldworkvt,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dlarfb_gemm_batched(
    icla_side_t side, icla_trans_t trans, icla_direct_t direct, icla_storev_t storev,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDouble_const_ptr dV_array[],    icla_int_t lddv,
    iclaDouble_const_ptr dT_array[],    icla_int_t lddt,
    iclaDouble_ptr dC_array[],          icla_int_t lddc,
    iclaDouble_ptr dwork_array[],       icla_int_t ldwork,
    iclaDouble_ptr dworkvt_array[],     icla_int_t ldworkvt,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dlarft_internal_batched(
        icla_int_t n, icla_int_t k, icla_int_t stair_T,
        double **v_array,   icla_int_t vi, icla_int_t vj, icla_int_t ldv,
        double **tau_array, icla_int_t taui,
        double **T_array,   icla_int_t Ti, icla_int_t Tj, icla_int_t ldt,
        double **work_array, icla_int_t lwork,
        icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dlarft_batched(
    icla_int_t n, icla_int_t k, icla_int_t stair_T,
    double **v_array, icla_int_t ldv,
    double **tau_array,
    double **T_array, icla_int_t ldt,
    double **work_array, icla_int_t lwork,
    icla_int_t batchCount, icla_queue_t queue);

void
icla_dlarft_sm32x32_batched(
    icla_int_t n, icla_int_t k,
    double **v_array, icla_int_t vi, icla_int_t vj, icla_int_t ldv,
    double **tau_array, icla_int_t taui,
    double **T_array, icla_int_t Ti, icla_int_t Tj, icla_int_t ldt,
    icla_int_t batchCount, icla_queue_t queue);

void iclablas_dlarft_recdtrmv_sm32x32(
    icla_int_t m, icla_int_t n,
    double *tau,
    double *Trec, icla_int_t ldtrec,
    double *Ttri, icla_int_t ldttri,
    icla_queue_t queue);

void iclablas_dlarft_recdtrmv_sm32x32_batched(
    icla_int_t m, icla_int_t n,
    double **tau_array,  icla_int_t taui,
    double **Trec_array, icla_int_t Treci, icla_int_t Trecj, icla_int_t ldtrec,
    double **Ttri_array, icla_int_t Ttrii, icla_int_t Ttrij, icla_int_t ldttri,
    icla_int_t batchCount, icla_queue_t queue);

void iclablas_dlarft_dtrmv_sm32x32(
    icla_int_t m, icla_int_t n,
    double *tau,
    double *Tin, icla_int_t ldtin,
    double *Tout, icla_int_t ldtout,
    icla_queue_t queue);

void iclablas_dlarft_dtrmv_sm32x32_batched(
    icla_int_t m, icla_int_t n,
    double **tau_array, icla_int_t taui,
    double **Tin_array, icla_int_t Tini, icla_int_t Tinj, icla_int_t ldtin,
    double **Tout_array, icla_int_t Touti, icla_int_t Toutj, icla_int_t ldtout,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_dnrm2_cols_batched(
    icla_int_t m, icla_int_t n,
    double **dA_array, icla_int_t lda,
    double **dxnorm_array,
    icla_int_t batchCount);

void
icla_dlarfgx_batched(
    icla_int_t n, double **dx0_array, double **dx_array,
    double **dtau_array, double **dxnorm_array,
    double **dR_array, icla_int_t it,
    icla_int_t batchCount);

void
icla_dlarfx_batched_v4(
    icla_int_t m, icla_int_t n,
    double **v_array,
    double **tau_array,
    double **C_array, icla_int_t ldc, double **xnorm_array,
    icla_int_t step,
    icla_int_t batchCount);

void
iclablas_dlarfg_batched(
    icla_int_t n,
    double** dalpha_array,
    double** dx_array, icla_int_t incx,
    double** dtau_array,
    icla_int_t batchCount );

icla_int_t
icla_dpotrf_lpout_batched(
    icla_uplo_t uplo, icla_int_t n,
    double **dA_array, icla_int_t ai, icla_int_t aj, icla_int_t lda, icla_int_t gbstep,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dpotrf_lpin_batched(
    icla_uplo_t uplo, icla_int_t n,
    double **dA_array, icla_int_t ai, icla_int_t aj, icla_int_t lda, icla_int_t gbstep,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_dpotrf_v33_batched(
    icla_uplo_t uplo, icla_int_t n,
    double **dA_array, icla_int_t lda,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

// host interface
void
blas_dlacpy_batched(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    double const * const * hA_array, icla_int_t lda,
    double               **hB_array, icla_int_t ldb,
    icla_int_t batchCount );

void
blas_dgemm_batched(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    double alpha,
    double const * const * hA_array, icla_int_t lda,
    double const * const * hB_array, icla_int_t ldb,
    double beta,
    double **hC_array, icla_int_t ldc,
    icla_int_t batchCount );

void
blas_dtrsm_batched(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t m, icla_int_t n,
        double alpha,
        double **hA_array, icla_int_t lda,
        double **hB_array, icla_int_t ldb,
        icla_int_t batchCount );

void
blas_dtrmm_batched(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t m, icla_int_t n,
        double alpha,
        double **hA_array, icla_int_t lda,
        double **hB_array, icla_int_t ldb,
        icla_int_t batchCount );

void
blas_dsymm_batched(
        icla_side_t side, icla_uplo_t uplo,
        icla_int_t m, icla_int_t n,
        double alpha,
        double **hA_array, icla_int_t lda,
        double **hB_array, icla_int_t ldb,
        double beta,
        double **hC_array, icla_int_t ldc,
        icla_int_t batchCount );

void
blas_dsyrk_batched(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    double alpha, double const * const * hA_array, icla_int_t lda,
    double beta,  double               **hC_array, icla_int_t ldc,
    icla_int_t batchCount );

void
blas_dsyr2k_batched(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    double alpha, double const * const * hA_array, icla_int_t lda,
                              double const * const * hB_array, icla_int_t ldb,
    double beta,              double               **hC_array, icla_int_t ldc,
    icla_int_t batchCount );

// for debugging purpose
void
dset_stepinit_ipiv(
    icla_int_t **ipiv_array,
    icla_int_t pm,
    icla_int_t batchCount);

#ifdef __cplusplus
}
#endif

#undef ICLA_REAL

#endif  /* ICLA_DBATCHED_H */
