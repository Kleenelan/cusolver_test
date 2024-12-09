
#ifndef ICLA_SBATCHED_H
#define ICLA_SBATCHED_H

#include "icla_types.h"

#define ICLA_REAL

#ifdef __cplusplus
extern "C" {
#endif

void
icla_sset_pointer(
    float **output_array,
    float *input,
    icla_int_t lda,
    icla_int_t row, icla_int_t column,
    icla_int_t batch_offset,
    icla_int_t batchCount, icla_queue_t queue);

void
icla_sdisplace_pointers(
    float **output_array,
    float **input_array, icla_int_t lda,
    icla_int_t row, icla_int_t column,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_srecommend_cublas_gemm_batched(
    icla_trans_t transa, icla_trans_t transb,
    icla_int_t m, icla_int_t n, icla_int_t k);

icla_int_t
icla_srecommend_cublas_gemm_stream(
    icla_trans_t transa, icla_trans_t transb,
    icla_int_t m, icla_int_t n, icla_int_t k);

void icla_get_spotrf_batched_nbparam(icla_int_t n, icla_int_t *nb, icla_int_t *recnb);

icla_int_t icla_get_spotrf_batched_crossover();

void icla_get_sgetrf_batched_nbparam(icla_int_t n, icla_int_t *nb, icla_int_t *recnb);
icla_int_t icla_get_sgetrf_batched_ntcol(icla_int_t m, icla_int_t n);
icla_int_t icla_get_sgemm_batched_ntcol(icla_int_t n);
icla_int_t icla_get_sgemm_batched_smallsq_limit(icla_int_t n);
icla_int_t icla_get_sgeqrf_batched_nb(icla_int_t m);
icla_int_t icla_use_sgeqrf_batched_fused_update(icla_int_t m, icla_int_t n, icla_int_t batchCount);
icla_int_t icla_get_sgeqr2_fused_sm_batched_nthreads(icla_int_t m, icla_int_t n);
icla_int_t icla_get_sgeqrf_batched_ntcol(icla_int_t m, icla_int_t n);
icla_int_t icla_get_sgetri_batched_ntcol(icla_int_t m, icla_int_t n);
icla_int_t icla_get_strsm_batched_stop_nb(icla_side_t side, icla_int_t m, icla_int_t n);
void icla_get_sgbtrf_batched_params(icla_int_t m, icla_int_t n, icla_int_t kl, icla_int_t ku, icla_int_t *nb, icla_int_t *threads);

void
iclablas_sswapdblk_batched(
    icla_int_t n, icla_int_t nb,
    float **dA, icla_int_t ldda, icla_int_t inca,
    float **dB, icla_int_t lddb, icla_int_t incb,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_sgemm_batched_core(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    float alpha,
    float const * const * dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    float const * const * dB_array, icla_int_t Bi, icla_int_t Bj, icla_int_t lddb,
    float beta,
    float **dC_array, icla_int_t Ci, icla_int_t Cj, icla_int_t lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
icla_sgemm_batched_core(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    float alpha,
    float const * const * dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    float const * const * dB_array, icla_int_t Bi, icla_int_t Bj, icla_int_t lddb,
    float beta,
    float **dC_array, icla_int_t Ci, icla_int_t Cj, icla_int_t lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
icla_sgemm_batched(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    float alpha,
    float const * const * dA_array, icla_int_t ldda,
    float const * const * dB_array, icla_int_t lddb,
    float beta,
    float **dC_array, icla_int_t lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_sgemm_batched(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    float alpha,
    float const * const * dA_array, icla_int_t ldda,
    float const * const * dB_array, icla_int_t lddb,
    float beta,
    float **dC_array, icla_int_t lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_sgemm_batched_strided(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    float alpha,
    float const * dA, icla_int_t ldda, icla_int_t strideA,
    float const * dB, icla_int_t lddb, icla_int_t strideB,
    float beta,
    float       * dC, icla_int_t lddc, icla_int_t strideC,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_sgemm_batched_smallsq(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    float alpha,
    float const * const * dA_array, icla_int_t ai, icla_int_t aj, icla_int_t ldda,
    float const * const * dB_array, icla_int_t bi, icla_int_t bj, icla_int_t lddb,
    float beta,
    float **dC_array, icla_int_t ci, icla_int_t cj, icla_int_t lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_ssyrk_batched_core(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    float alpha,
    float const * const * dA_array, icla_int_t ai, icla_int_t aj, icla_int_t ldda,
    float const * const * dB_array, icla_int_t bi, icla_int_t bj, icla_int_t lddb,
    float beta,
    float **dC_array, icla_int_t ci, icla_int_t cj, icla_int_t lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_ssyrk_batched_core(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    float alpha,
    float const * const * dA_array, icla_int_t ai, icla_int_t aj, icla_int_t ldda,
    float const * const * dB_array, icla_int_t bi, icla_int_t bj, icla_int_t lddb,
    float beta,
    float **dC_array, icla_int_t ci, icla_int_t cj, icla_int_t lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_ssyrk_batched(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    float alpha,
    float const * const * dA_array, icla_int_t ldda,
    float beta,
    float **dC_array, icla_int_t lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
icla_ssyrk_batched(
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t n, icla_int_t k,
    float alpha,
    float const * const * dA_array, icla_int_t ldda,
    float beta,
    float **dC_array, icla_int_t lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_ssyrk_batched(
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t n, icla_int_t k,
    float alpha,
    float const * const * dA_array, icla_int_t ldda,
    float beta,
    float **dC_array, icla_int_t lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_ssyr2k_batched(
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t n, icla_int_t k,
    float alpha,
    float const * const * dA_array, icla_int_t ldda,
    float const * const * dB_array, icla_int_t lddb,
    float beta, float **dC_array, icla_int_t lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_ssyr2k_batched(
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t n, icla_int_t k,
    float alpha,
    float const * const * dA_array, icla_int_t ldda,
    float const * const * dB_array, icla_int_t lddb,
    float beta, float **dC_array, icla_int_t lddc,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_strtri_diag_batched(
    icla_uplo_t uplo, icla_diag_t diag, icla_int_t n,
    float const * const *dA_array, icla_int_t ldda,
    float **dinvA_array,
    icla_int_t resetozero,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_strsm_small_batched(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t m, icla_int_t n,
        float alpha,
        float **dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
        float **dB_array, icla_int_t Bi, icla_int_t Bj, icla_int_t lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_strsm_recursive_batched(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t m, icla_int_t n,
        float alpha,
        float **dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
        float **dB_array, icla_int_t Bi, icla_int_t Bj, icla_int_t lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_strsm_batched(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t m, icla_int_t n,
        float alpha,
        float **dA_array, icla_int_t ldda,
        float **dB_array, icla_int_t lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_strsm_inv_batched(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    float alpha,
    float** dA_array,    icla_int_t ldda,
    float** dB_array,    icla_int_t lddb,
    icla_int_t batchCount, icla_queue_t queue);

void iclablas_strsm_inv_work_batched(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t flag, icla_int_t m, icla_int_t n,
    float alpha,
    float** dA_array,    icla_int_t ldda,
    float** dB_array,    icla_int_t lddb,
    float** dX_array,    icla_int_t lddx,
    float** dinvA_array, icla_int_t dinvA_length,
    float** dA_displ, float** dB_displ,
    float** dX_displ, float** dinvA_displ,
    icla_int_t resetozero,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_strsm_inv_outofplace_batched(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t flag, icla_int_t m, icla_int_t n,
    float alpha,
    float** dA_array,    icla_int_t ldda,
    float** dB_array,    icla_int_t lddb,
    float** dX_array,    icla_int_t lddx,
    float** dinvA_array, icla_int_t dinvA_length,
    float** dA_displ, float** dB_displ,
    float** dX_displ, float** dinvA_displ,
    icla_int_t resetozero,
    icla_int_t batchCount, icla_queue_t queue);

void iclablas_strsv_batched(
    icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t n,
    float** dA_array,    icla_int_t ldda,
    float** dB_array,    icla_int_t incb,
    icla_int_t batchCount, icla_queue_t queue);

void iclablas_strsv_work_batched(
    icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t n,
    float** dA_array,    icla_int_t ldda,
    float** dB_array,    icla_int_t incb,
    float** dX_array,
    icla_int_t batchCount, icla_queue_t queue);

void iclablas_strsv_outofplace_batched(
    icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t n,
    float ** A_array, icla_int_t lda,
    float **b_array, icla_int_t incb,
    float **x_array,
    icla_int_t batchCount, icla_queue_t queue, icla_int_t flag);

void
iclablas_strmm_batched_core(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t m, icla_int_t n,
        float alpha,
        float **dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
        float **dB_array, icla_int_t Bi, icla_int_t Bj, icla_int_t lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_strmm_batched(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t m, icla_int_t n,
        float alpha,
        float **dA_array, icla_int_t ldda,
        float **dB_array, icla_int_t lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_ssymm_batched_core(
        icla_side_t side, icla_uplo_t uplo,
        icla_int_t m, icla_int_t n,
        float alpha,
        float **dA_array, icla_int_t ldda,
        float **dB_array, icla_int_t lddb,
        float beta,
        float **dC_array, icla_int_t lddc,
        icla_int_t roffA, icla_int_t coffA, icla_int_t roffB, icla_int_t coffB, icla_int_t roffC, icla_int_t coffC,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_ssymm_batched(
        icla_side_t side, icla_uplo_t uplo,
        icla_int_t m, icla_int_t n,
        float alpha,
        float **dA_array, icla_int_t ldda,
        float **dB_array, icla_int_t lddb,
        float beta,
        float **dC_array, icla_int_t lddc,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_ssymv_batched_core(
        icla_uplo_t uplo, icla_int_t n,
        float alpha,
        float **dA_array, icla_int_t ldda,
        float **dX_array, icla_int_t incx,
        float beta,
        float **dY_array, icla_int_t incy,
        icla_int_t offA, icla_int_t offX, icla_int_t offY,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_ssymv_batched(
        icla_uplo_t uplo, icla_int_t n,
        float alpha,
        float **dA_array, icla_int_t ldda,
        float **dX_array, icla_int_t incx,
        float beta,
        float **dY_array, icla_int_t incy,
        icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_spotrf_batched(
    icla_uplo_t uplo, icla_int_t n,
    float **dA_array, icla_int_t lda,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_spotf2_batched(
    icla_uplo_t uplo, icla_int_t n,
    float **dA_array, icla_int_t ai, icla_int_t aj, icla_int_t lda,
    icla_int_t *info_array, icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_spotrf_panel_batched(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nb,
    float** dA_array, icla_int_t ai, icla_int_t aj, icla_int_t ldda,
    icla_int_t *info_array, icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_spotrf_recpanel_batched(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n, icla_int_t min_recpnb,
    float** dA_array, icla_int_t ai, icla_int_t aj, icla_int_t ldda,
    icla_int_t *info_array, icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_spotrf_rectile_batched(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n, icla_int_t min_recpnb,
    float** dA_array, icla_int_t ai, icla_int_t aj, icla_int_t ldda,
    icla_int_t *info_array, icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_spotrs_batched(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nrhs,
    float **dA_array, icla_int_t ldda,
    float **dB_array, icla_int_t lddb,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_sposv_batched(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nrhs,
    float **dA_array, icla_int_t ldda,
    float **dB_array, icla_int_t lddb,
    icla_int_t *dinfo_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_sgetrs_batched(
    icla_trans_t trans, icla_int_t n, icla_int_t nrhs,
    float **dA_array, icla_int_t ldda,
    icla_int_t **dipiv_array,
    float **dB_array, icla_int_t lddb,
    icla_int_t batchCount, icla_queue_t queue);

void
icla_slaswp_rowparallel_batched(
    icla_int_t n,
    float**  input_array, icla_int_t  input_i, icla_int_t  input_j, icla_int_t ldi,
    float** output_array, icla_int_t output_i, icla_int_t output_j, icla_int_t ldo,
    icla_int_t k1, icla_int_t k2,
    icla_int_t **pivinfo_array,
    icla_int_t batchCount, icla_queue_t queue);

void
icla_slaswp_rowserial_batched(
    icla_int_t n, float** dA_array, icla_int_t lda,
    icla_int_t k1, icla_int_t k2,
    icla_int_t **ipiv_array,
    icla_int_t batchCount, icla_queue_t queue);

void
icla_slaswp_columnserial_batched(
    icla_int_t n, float** dA_array, icla_int_t lda,
    icla_int_t k1, icla_int_t k2,
    icla_int_t **ipiv_array,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_stranspose_batched(
    icla_int_t m, icla_int_t n,
    float **dA_array,  icla_int_t ldda,
    float **dAT_array, icla_int_t lddat,
    icla_int_t batchCount, icla_queue_t queue);

void iclablas_slaset_internal_batched(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    float offdiag, float diag,
    iclaFloat_ptr dAarray[], icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_slaset_batched(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    float offdiag, float diag,
    iclaFloat_ptr dAarray[], icla_int_t ldda,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_sgbsv_batched(
    icla_int_t n, icla_int_t kl, icla_int_t ku, icla_int_t nrhs,
    float **dA_array, icla_int_t ldda, icla_int_t **dipiv_array,
    float** dB_array, icla_int_t lddb,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_sgbsv_batched_fused_sm(
    icla_int_t n, icla_int_t kl, icla_int_t ku, icla_int_t nrhs,
    float** dA_array, icla_int_t ldda, icla_int_t** ipiv_array,
    float** dB_array, icla_int_t lddb, icla_int_t* info_array,
    icla_int_t nthreads, icla_int_t ntcol,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_sgbsv_batched_strided_work(
    icla_int_t n, icla_int_t kl, icla_int_t ku, icla_int_t nrhs,
    float* dA, icla_int_t ldda, icla_int_t strideA,
    icla_int_t* dipiv, icla_int_t stride_piv,
    float* dB, icla_int_t lddb, icla_int_t strideB,
    icla_int_t *info_array,
    void* device_work, icla_int_t *lwork,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_sgbsv_batched_strided(
    icla_int_t n, icla_int_t kl, icla_int_t ku, icla_int_t nrhs,
    float* dA, icla_int_t ldda, icla_int_t strideA,
    icla_int_t* dipiv, icla_int_t stride_piv,
    float* dB, icla_int_t lddb, icla_int_t strideB,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_sgbsv_batched_work(
    icla_int_t n, icla_int_t kl, icla_int_t ku, icla_int_t nrhs,
    float** dA_array, icla_int_t ldda, icla_int_t **dipiv_array,
    float** dB_array, icla_int_t lddb,
    icla_int_t *info_array,
    void* device_work, icla_int_t *lwork,
    icla_int_t batchCount, icla_queue_t queue);

void
icla_sgbtrf_set_fillin(
    icla_int_t n, icla_int_t kl, icla_int_t ku,
    float** dAB_array, icla_int_t lddab,
    icla_int_t** dipiv_array, int* ju_array, icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_sgbtf2_sswap_batched(
    icla_int_t kl, icla_int_t ku,
    float **dAB_array, icla_int_t ai, icla_int_t aj, icla_int_t lddab,
    icla_int_t** dipiv_array, icla_int_t ipiv_offset,
    int* ju_array, icla_int_t gbstep, icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_sgbtf2_scal_ger_batched(
    icla_int_t m, icla_int_t n, icla_int_t kl, icla_int_t ku,
    float **dAB_array, icla_int_t ai, icla_int_t aj, icla_int_t lddab,
    int* ju_array, icla_int_t gbstep, icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_sgbtrf_batched_fused_sm(
    icla_int_t m,  icla_int_t n,
    icla_int_t kl, icla_int_t ku,
    float** dAB_array, icla_int_t lddab,
    icla_int_t** ipiv_array, icla_int_t* info_array,
    icla_int_t nthreads, icla_int_t ntcol,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_sgbtrf_batched_sliding_window_loopout(
    icla_int_t m,  icla_int_t n,
    icla_int_t kl, icla_int_t ku,
    float** dAB_array, icla_int_t lddab,
    icla_int_t** ipiv_array, icla_int_t* info_array,
    void* device_work, icla_int_t *lwork,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_sgbtrf_batched_sliding_window_loopin(
    icla_int_t m,  icla_int_t n,
    icla_int_t kl, icla_int_t ku,
    float** dAB_array, icla_int_t lddab,
    icla_int_t** ipiv_array, icla_int_t* info_array,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_sgbtrf_batched_strided(
        icla_int_t m, icla_int_t n,
        icla_int_t kl, icla_int_t ku,
        float* dAB, icla_int_t lddab, icla_int_t strideAB,
        icla_int_t* dipiv, icla_int_t stride_piv,
        icla_int_t *info_array,
        icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_sgbtrf_batched_strided_work(
        icla_int_t m, icla_int_t n,
        icla_int_t kl, icla_int_t ku,
        float* dAB, icla_int_t lddab, icla_int_t strideAB,
        icla_int_t* dipiv, icla_int_t stride_piv,
        icla_int_t *info_array,
        void* device_work, icla_int_t *lwork,
        icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_sgbtrf_batched_work(
        icla_int_t m, icla_int_t n,
        icla_int_t kl, icla_int_t ku,
        float **dAB_array, icla_int_t lddab,
        icla_int_t **dipiv_array, icla_int_t *info_array,
        void* device_work, icla_int_t *lwork,
        icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_sgbtrf_batched(
    icla_int_t m, icla_int_t n,
    icla_int_t kl, icla_int_t ku,
    float **dAB_array, icla_int_t lddab,
    icla_int_t **dipiv_array, icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_sgbtrs_batched(
    icla_trans_t transA,
    icla_int_t n, icla_int_t kl, icla_int_t ku, icla_int_t nrhs,
    float **dA_array, icla_int_t ldda, icla_int_t **dipiv_array,
    float** dB_array, icla_int_t lddb,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_sgbtrs_lower_batched(
    icla_trans_t transA,
    icla_int_t n, icla_int_t kl, icla_int_t ku, icla_int_t nrhs,
    float** dA_array, icla_int_t ldda, icla_int_t **dipiv_array,
    float** dB_array, icla_int_t lddb,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
iclablas_sgbtrs_lower_blocked_batched(
        icla_int_t n, icla_int_t kl, icla_int_t ku, icla_int_t nrhs,
        float** dA_array, icla_int_t ldda, icla_int_t** dipiv_array,
        float** dB_array, icla_int_t lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_sgbtrs_swap_batched(
    icla_int_t n, float** dA_array, icla_int_t ldda,
    icla_int_t** dipiv_array, icla_int_t j,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_sgbtrs_upper_batched(
    icla_trans_t transA,
    icla_int_t n, icla_int_t kl, icla_int_t ku, icla_int_t nrhs,
    float** dA_array, icla_int_t ldda, icla_int_t **dipiv_array,
    float** dB_array, icla_int_t lddb,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
iclablas_sgbtrs_upper_blocked_batched(
        icla_int_t n, icla_int_t kl, icla_int_t ku, icla_int_t nrhs,
        float** dA_array, icla_int_t ldda,
        float** dB_array, icla_int_t lddb,
        icla_int_t batchCount, icla_queue_t queue );

void
iclablas_sgbtrs_upper_columnwise_batched(
    icla_int_t n, icla_int_t kl, icla_int_t ku,
    icla_int_t nrhs, icla_int_t j,
    float** dA_array, icla_int_t ldda,
    float** dB_array, icla_int_t lddb,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_sger_batched_core(
    icla_int_t m, icla_int_t n,
    float alpha,
    float** dX_array, icla_int_t xi, icla_int_t xj, icla_int_t lddx, icla_int_t incx,
    float** dY_array, icla_int_t yi, icla_int_t yj, icla_int_t lddy, icla_int_t incy,
    float** dA_array, icla_int_t ai, icla_int_t aj, icla_int_t ldda,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_sgesv_batched_small(
    icla_int_t n, icla_int_t nrhs,
    float** dA_array, icla_int_t ldda,
    icla_int_t** dipiv_array,
    float **dB_array, icla_int_t lddb,
    icla_int_t* dinfo_array,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_sgetf2_batched(
    icla_int_t m, icla_int_t n,
    float **dA_array, icla_int_t ai, icla_int_t aj, icla_int_t lda,
    icla_int_t **ipiv_array,
    icla_int_t **dpivinfo_array,
    icla_int_t *info_array,
    icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_sgetrf_recpanel_batched(
    icla_int_t m, icla_int_t n, icla_int_t min_recpnb,
    float** dA_array, icla_int_t ai, icla_int_t aj, icla_int_t ldda,
    icla_int_t** dipiv_array, icla_int_t** dpivinfo_array,
    icla_int_t *info_array, icla_int_t gbstep,
    icla_int_t batchCount,  icla_queue_t queue);

icla_int_t
icla_sgetrf_batched(
    icla_int_t m, icla_int_t n,
    float **dA_array,
    icla_int_t lda,
    icla_int_t **ipiv_array,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_sgetf2_fused_batched(
    icla_int_t m, icla_int_t n,
    float **dA_array, icla_int_t ai, icla_int_t aj, icla_int_t ldda,
    icla_int_t **dipiv_array,
    icla_int_t *info_array, icla_int_t batchCount,
    icla_queue_t queue);

icla_int_t
icla_sgetrf_batched_smallsq_noshfl(
    icla_int_t n,
    float** dA_array, icla_int_t ldda,
    icla_int_t** ipiv_array, icla_int_t* info_array,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_sgetri_outofplace_batched(
    icla_int_t n,
    float **dA_array, icla_int_t ldda,
    icla_int_t **dipiv_array,
    float **dinvA_array, icla_int_t lddia,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

void
icla_sdisplace_intpointers(
    icla_int_t **output_array,
    icla_int_t **input_array, icla_int_t lda,
    icla_int_t row, icla_int_t column,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_isamax_atomic_batched(
    icla_int_t n,
    float** x_array, icla_int_t incx,
    icla_int_t **max_id_array,
    icla_int_t batchCount);

void
iclablas_isamax_tree_batched(
    icla_int_t n,
    float** x_array, icla_int_t incx,
    icla_int_t **max_id_array,
    icla_int_t batchCount);

void
iclablas_isamax_batched(
    icla_int_t n,
    float** x_array, icla_int_t incx,
    icla_int_t **max_id_array,
    icla_int_t batchCount);

void
iclablas_isamax(
    icla_int_t n,
    float* x, icla_int_t incx,
    icla_int_t *max_id);

icla_int_t
icla_isamax_batched(
        icla_int_t length,
        float **x_array, icla_int_t xi, icla_int_t xj, icla_int_t lda, icla_int_t incx,
        icla_int_t** ipiv_array, icla_int_t ipiv_i,
        icla_int_t step, icla_int_t gbstep, icla_int_t *info_array,
        icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_sswap_batched(
    icla_int_t n, float **x_array, icla_int_t xi, icla_int_t xj, icla_int_t incx,
    icla_int_t step, icla_int_t** ipiv_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_sscal_sger_batched(
    icla_int_t m, icla_int_t n,
    float **dA_array, icla_int_t ai, icla_int_t aj, icla_int_t lda,
    icla_int_t *info_array, icla_int_t step, icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_scomputecolumn_batched(
    icla_int_t m, icla_int_t paneloffset, icla_int_t step,
    float **dA_array,  icla_int_t lda,
    icla_int_t ai, icla_int_t aj,
    icla_int_t **ipiv_array,
    icla_int_t *info_array, icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue);

void
icla_sgetf2trsm_batched(
    icla_int_t ib, icla_int_t n,
    float **dA_array,  icla_int_t j, icla_int_t lda,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_sgetf2_nopiv_internal_batched(
    icla_int_t m, icla_int_t n,
    float** dA_array, icla_int_t ai, icla_int_t aj, icla_int_t ldda,
    icla_int_t* info_array, icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_sgetf2_nopiv_batched(
    icla_int_t m, icla_int_t n,
    float **dA_array, icla_int_t ai, icla_int_t aj, icla_int_t ldda,
    icla_int_t *info_array, icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_sgetrf_recpanel_nopiv_batched(
    icla_int_t m, icla_int_t n, icla_int_t min_recpnb,
    float** dA_array,    icla_int_t ldda,
    float** dX_array,    icla_int_t dX_length,
    float** dinvA_array, icla_int_t dinvA_length,
    float** dW1_displ, float** dW2_displ,
    float** dW3_displ, float** dW4_displ,
    float** dW5_displ,
    icla_int_t *info_array, icla_int_t gbstep,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_sgetrf_nopiv_batched(
    icla_int_t m, icla_int_t n,
    float **dA_array,
    icla_int_t lda,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_sgetrs_nopiv_batched(
    icla_trans_t trans, icla_int_t n, icla_int_t nrhs,
    float **dA_array, icla_int_t ldda,
    float **dB_array, icla_int_t lddb,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_sgesv_nopiv_batched(
    icla_int_t n, icla_int_t nrhs,
    float **dA_array, icla_int_t ldda,
    float **dB_array, icla_int_t lddb,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_sgesv_rbt_batched(
    icla_int_t n, icla_int_t nrhs,
    float **dA_array, icla_int_t ldda,
    float **dB_array, icla_int_t lddb,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_sgesv_batched(
    icla_int_t n, icla_int_t nrhs,
    float **dA_array, icla_int_t ldda,
    icla_int_t **dipiv_array,
    float **dB_array, icla_int_t lddb,
    icla_int_t *dinfo_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_sgerbt_batched(
    icla_bool_t gen, icla_int_t n, icla_int_t nrhs,
    float **dA_array, icla_int_t ldda,
    float **dB_array, icla_int_t lddb,
    float *U, float *V,
    icla_int_t *info,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_sprbt_batched(
    icla_int_t n,
    float **dA_array, icla_int_t ldda,
    float *du, float *dv,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_sprbt_mv_batched(
    icla_int_t n, icla_int_t nrhs,
    float *dv, float **db_array, icla_int_t lddb,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_sprbt_mtv_batched(
    icla_int_t n, icla_int_t nrhs,
    float *du, float **db_array, icla_int_t lddb,
    icla_int_t batchCount, icla_queue_t queue);

void
setup_pivinfo(
    icla_int_t *pivinfo, icla_int_t *ipiv,
    icla_int_t m, icla_int_t nb,
    icla_queue_t queue);

void
iclablas_sgeadd_batched(
    icla_int_t m, icla_int_t n,
    float alpha,
    iclaFloat_const_ptr  const dAarray[], icla_int_t ldda,
    iclaFloat_ptr              dBarray[], icla_int_t lddb,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_slacpy_internal_batched(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    iclaFloat_const_ptr const dAarray[], icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    iclaFloat_ptr             dBarray[], icla_int_t Bi, icla_int_t Bj, icla_int_t lddb,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_slacpy_batched(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    iclaFloat_const_ptr  const dAarray[], icla_int_t ldda,
    iclaFloat_ptr              dBarray[], icla_int_t lddb,
    icla_int_t batchCount, icla_queue_t queue );

void
iclablas_sgemv_batched_core(
    icla_trans_t trans, icla_int_t m, icla_int_t n,
    const float alpha,
    float const * const * dA_array, const float* dA, icla_int_t ldda, icla_int_t strideA,
    float const * const * dx_array, const float* dx, icla_int_t incx, icla_int_t stridex,
    const float beta,
    float** dy_array, iclaFloat_ptr dy, icla_int_t incy, icla_int_t stridey,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_sgemv_batched(
    icla_trans_t trans, icla_int_t m, icla_int_t n,
    const float alpha,
    float const * const * dA_array, icla_int_t ldda,
    float const * const * dx_array, icla_int_t incx,
    const float beta,
    float** dy_array, icla_int_t incy,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_sgemv_batched_strided(
    icla_trans_t trans, icla_int_t m, icla_int_t n,
    const float alpha,
    const float* dA, icla_int_t ldda, icla_int_t strideA,
    const float* dx, icla_int_t incx, icla_int_t stridex,
    const float beta,
    float* dy, icla_int_t incy, icla_int_t stridey,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
iclablas_sgemv_batched_smallsq(
    icla_trans_t trans, icla_int_t n,
    const float alpha,
    float const * const * dA_array, icla_int_t ldda,
    float const * const * dx_array, icla_int_t incx,
    const float beta,
    float** dy_array, icla_int_t incy,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
iclablas_sgemv_batched_strided_smallsq(
    icla_trans_t transA, icla_int_t n,
    const float alpha,
    const float* dA, icla_int_t ldda, icla_int_t strideA,
    const float* dx, icla_int_t incx, icla_int_t stridex,
    const float beta,
    float* dy, icla_int_t incy, icla_int_t stridey,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_sgeqrf_batched_smallsq(
    icla_int_t n,
    float** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    float **dtau_array, icla_int_t taui, icla_int_t* info_array,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_sgeqrf_batched(
    icla_int_t m, icla_int_t n,
    float **dA_array,
    icla_int_t lda,
    float **dtau_array,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_sgeqrf_batched_work(
    icla_int_t m, icla_int_t n,
    float **dA_array, icla_int_t ldda,
    float **dtau_array,
    icla_int_t *info_array,
    void* device_work, icla_int_t* device_lwork,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_sgeqrf_expert_batched(
    icla_int_t m, icla_int_t n, icla_int_t nb,
    float **dA_array, icla_int_t ldda,
    float **dR_array, icla_int_t lddr,
    float **dT_array, icla_int_t lddt,
    float **dtau_array, icla_int_t provide_RT,
    float **dW_array,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_sgeqrf_batched_v4(
    icla_int_t m, icla_int_t n,
    float **dA_array,
    icla_int_t lda,
    float **tau_array,
    icla_int_t *info_array,
    icla_int_t batchCount);

icla_int_t
icla_sgeqrf_panel_fused_update_batched(
        icla_int_t m, icla_int_t n, icla_int_t nb,
        float** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
        float** tau_array, icla_int_t taui,
        float** dR_array, icla_int_t Ri, icla_int_t Rj, icla_int_t lddr,
        icla_int_t *info_array, icla_int_t separate_R_V,
        icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_sgeqrf_panel_internal_batched(
        icla_int_t m, icla_int_t n, icla_int_t nb,
        float** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
        float** tau_array, icla_int_t taui,
        float** dT_array, icla_int_t Ti, icla_int_t Tj, icla_int_t lddt,
        float** dR_array, icla_int_t Ri, icla_int_t Rj, icla_int_t lddr,
        float** dwork_array,
        icla_int_t *info_array,
        icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_sgeqrf_panel_batched(
    icla_int_t m, icla_int_t n, icla_int_t nb,
    float** dA_array,    icla_int_t ldda,
    float** tau_array,
    float** dT_array, icla_int_t ldt,
    float** dR_array, icla_int_t ldr,
    float** dwork_array,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_sgels_batched(
    icla_trans_t trans, icla_int_t m, icla_int_t n, icla_int_t nrhs,
    float **dA_array, icla_int_t ldda,
    float **dB_array, icla_int_t lddb,
    float *hwork, icla_int_t lwork,
    icla_int_t *info,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_sgeqr2_fused_reg_tall_batched(
    icla_int_t m, icla_int_t n,
    float** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    float **dtau_array, icla_int_t taui,
    icla_int_t* info_array, icla_int_t check_launch_only,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_sgeqr2_fused_reg_medium_batched(
    icla_int_t m, icla_int_t n,
    float** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    float **dtau_array, icla_int_t taui,
    icla_int_t* info_array, icla_int_t check_launch_only,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_sgeqr2_fused_reg_batched(
    icla_int_t m, icla_int_t n,
    float** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    float **dtau_array, icla_int_t taui,
    icla_int_t* info_array, icla_int_t check_launch_only,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_sgeqr2_fused_sm_batched(
    icla_int_t m, icla_int_t n,
    float** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    float **dtau_array, icla_int_t taui,
    icla_int_t* info_array, icla_int_t nthreads, icla_int_t check_launch_only,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_sgeqr2_batched(
    icla_int_t m, icla_int_t n,
    float **dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    float **dtau_array, icla_int_t taui,
    icla_int_t *info_array, icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_slarf_fused_reg_tall_batched(
    icla_int_t m, icla_int_t n, icla_int_t nb, icla_int_t ib,
    float** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    float** dV_array, icla_int_t Vi, icla_int_t Vj, icla_int_t lddv,
    float **dtau_array, icla_int_t taui,
    icla_int_t check_launch_only,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_slarf_fused_reg_medium_batched(
    icla_int_t m, icla_int_t n, icla_int_t nb, icla_int_t ib,
    float** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    float** dV_array, icla_int_t Vi, icla_int_t Vj, icla_int_t lddv,
    float **dtau_array, icla_int_t taui,
    icla_int_t check_launch_only,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_slarf_fused_reg_batched(
    icla_int_t m, icla_int_t n, icla_int_t nb, icla_int_t ib,
    float** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    float** dV_array, icla_int_t Vi, icla_int_t Vj, icla_int_t lddv,
    float **dtau_array, icla_int_t taui,
    icla_int_t check_launch_only,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_slarf_fused_sm_batched(
    icla_int_t m, icla_int_t n, icla_int_t nb, icla_int_t ib,
    float** dA_array, icla_int_t Ai, icla_int_t Aj, icla_int_t ldda,
    float** dV_array, icla_int_t Vi, icla_int_t Vj, icla_int_t lddv,
    float **dtau_array, icla_int_t taui,
    icla_int_t nthreads, icla_int_t check_launch_only,
    icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_slarfb_gemm_internal_batched(
    icla_side_t side, icla_trans_t trans, icla_direct_t direct, icla_storev_t storev,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaFloat_const_ptr dV_array[],    icla_int_t vi, icla_int_t vj, icla_int_t lddv,
    iclaFloat_const_ptr dT_array[],    icla_int_t Ti, icla_int_t Tj, icla_int_t lddt,
    iclaFloat_ptr dC_array[],          icla_int_t Ci, icla_int_t Cj, icla_int_t lddc,
    iclaFloat_ptr dwork_array[],       icla_int_t ldwork,
    iclaFloat_ptr dworkvt_array[],     icla_int_t ldworkvt,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_slarfb_gemm_batched(
    icla_side_t side, icla_trans_t trans, icla_direct_t direct, icla_storev_t storev,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaFloat_const_ptr dV_array[],    icla_int_t lddv,
    iclaFloat_const_ptr dT_array[],    icla_int_t lddt,
    iclaFloat_ptr dC_array[],          icla_int_t lddc,
    iclaFloat_ptr dwork_array[],       icla_int_t ldwork,
    iclaFloat_ptr dworkvt_array[],     icla_int_t ldworkvt,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_slarft_internal_batched(
        icla_int_t n, icla_int_t k, icla_int_t stair_T,
        float **v_array,   icla_int_t vi, icla_int_t vj, icla_int_t ldv,
        float **tau_array, icla_int_t taui,
        float **T_array,   icla_int_t Ti, icla_int_t Tj, icla_int_t ldt,
        float **work_array, icla_int_t lwork,
        icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_slarft_batched(
    icla_int_t n, icla_int_t k, icla_int_t stair_T,
    float **v_array, icla_int_t ldv,
    float **tau_array,
    float **T_array, icla_int_t ldt,
    float **work_array, icla_int_t lwork,
    icla_int_t batchCount, icla_queue_t queue);

void
icla_slarft_sm32x32_batched(
    icla_int_t n, icla_int_t k,
    float **v_array, icla_int_t vi, icla_int_t vj, icla_int_t ldv,
    float **tau_array, icla_int_t taui,
    float **T_array, icla_int_t Ti, icla_int_t Tj, icla_int_t ldt,
    icla_int_t batchCount, icla_queue_t queue);

void iclablas_slarft_recstrmv_sm32x32(
    icla_int_t m, icla_int_t n,
    float *tau,
    float *Trec, icla_int_t ldtrec,
    float *Ttri, icla_int_t ldttri,
    icla_queue_t queue);

void iclablas_slarft_recstrmv_sm32x32_batched(
    icla_int_t m, icla_int_t n,
    float **tau_array,  icla_int_t taui,
    float **Trec_array, icla_int_t Treci, icla_int_t Trecj, icla_int_t ldtrec,
    float **Ttri_array, icla_int_t Ttrii, icla_int_t Ttrij, icla_int_t ldttri,
    icla_int_t batchCount, icla_queue_t queue);

void iclablas_slarft_strmv_sm32x32(
    icla_int_t m, icla_int_t n,
    float *tau,
    float *Tin, icla_int_t ldtin,
    float *Tout, icla_int_t ldtout,
    icla_queue_t queue);

void iclablas_slarft_strmv_sm32x32_batched(
    icla_int_t m, icla_int_t n,
    float **tau_array, icla_int_t taui,
    float **Tin_array, icla_int_t Tini, icla_int_t Tinj, icla_int_t ldtin,
    float **Tout_array, icla_int_t Touti, icla_int_t Toutj, icla_int_t ldtout,
    icla_int_t batchCount, icla_queue_t queue);

void
iclablas_snrm2_cols_batched(
    icla_int_t m, icla_int_t n,
    float **dA_array, icla_int_t lda,
    float **dxnorm_array,
    icla_int_t batchCount);

void
icla_slarfgx_batched(
    icla_int_t n, float **dx0_array, float **dx_array,
    float **dtau_array, float **dxnorm_array,
    float **dR_array, icla_int_t it,
    icla_int_t batchCount);

void
icla_slarfx_batched_v4(
    icla_int_t m, icla_int_t n,
    float **v_array,
    float **tau_array,
    float **C_array, icla_int_t ldc, float **xnorm_array,
    icla_int_t step,
    icla_int_t batchCount);

void
iclablas_slarfg_batched(
    icla_int_t n,
    float** dalpha_array,
    float** dx_array, icla_int_t incx,
    float** dtau_array,
    icla_int_t batchCount );

icla_int_t
icla_spotrf_lpout_batched(
    icla_uplo_t uplo, icla_int_t n,
    float **dA_array, icla_int_t ai, icla_int_t aj, icla_int_t lda, icla_int_t gbstep,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_spotrf_lpin_batched(
    icla_uplo_t uplo, icla_int_t n,
    float **dA_array, icla_int_t ai, icla_int_t aj, icla_int_t lda, icla_int_t gbstep,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_spotrf_v33_batched(
    icla_uplo_t uplo, icla_int_t n,
    float **dA_array, icla_int_t lda,
    icla_int_t *info_array,
    icla_int_t batchCount, icla_queue_t queue);


void
blas_slacpy_batched(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    float const * const * hA_array, icla_int_t lda,
    float               **hB_array, icla_int_t ldb,
    icla_int_t batchCount );

void
blas_sgemm_batched(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    float alpha,
    float const * const * hA_array, icla_int_t lda,
    float const * const * hB_array, icla_int_t ldb,
    float beta,
    float **hC_array, icla_int_t ldc,
    icla_int_t batchCount );

void
blas_strsm_batched(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t m, icla_int_t n,
        float alpha,
        float **hA_array, icla_int_t lda,
        float **hB_array, icla_int_t ldb,
        icla_int_t batchCount );

void
blas_strmm_batched(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t m, icla_int_t n,
        float alpha,
        float **hA_array, icla_int_t lda,
        float **hB_array, icla_int_t ldb,
        icla_int_t batchCount );

void
blas_ssymm_batched(
        icla_side_t side, icla_uplo_t uplo,
        icla_int_t m, icla_int_t n,
        float alpha,
        float **hA_array, icla_int_t lda,
        float **hB_array, icla_int_t ldb,
        float beta,
        float **hC_array, icla_int_t ldc,
        icla_int_t batchCount );

void
blas_ssyrk_batched(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    float alpha, float const * const * hA_array, icla_int_t lda,
    float beta,  float               **hC_array, icla_int_t ldc,
    icla_int_t batchCount );

void
blas_ssyr2k_batched(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    float alpha, float const * const * hA_array, icla_int_t lda,
                              float const * const * hB_array, icla_int_t ldb,
    float beta,              float               **hC_array, icla_int_t ldc,
    icla_int_t batchCount );


void
sset_stepinit_ipiv(
    icla_int_t **ipiv_array,
    icla_int_t pm,
    icla_int_t batchCount);

#ifdef __cplusplus
}
#endif

#undef ICLA_REAL

#endif

