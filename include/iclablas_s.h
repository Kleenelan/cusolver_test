/*
    -- ICLA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from include/iclablas_z.h, normal z -> s, Fri Nov 29 12:16:14 2024
*/

#ifndef ICLABLAS_S_H
#define ICLABLAS_S_H

#include "icla_types.h"
#include "icla_copy.h"

#define ICLA_REAL

#ifdef __cplusplus
extern "C" {
#endif

  /*
   * Transpose functions
   */
void
iclablas_stranspose_inplace(
    icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    icla_queue_t queue );

void
iclablas_stranspose_inplace(
    icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    icla_queue_t queue );

void
iclablas_stranspose(
    icla_int_t m, icla_int_t n,
    iclaFloat_const_ptr dA,  icla_int_t ldda,
    iclaFloat_ptr       dAT, icla_int_t lddat,
    icla_queue_t queue );

void
iclablas_stranspose(
    icla_int_t m, icla_int_t n,
    iclaFloat_const_ptr dA,  icla_int_t ldda,
    iclaFloat_ptr       dAT, icla_int_t lddat,
    icla_queue_t queue );

void
iclablas_sgetmatrix_transpose(
    icla_int_t m, icla_int_t n, icla_int_t nb,
    iclaFloat_const_ptr dAT,   icla_int_t ldda,
    float          *hA,    icla_int_t lda,
    iclaFloat_ptr       dwork, icla_int_t lddw,
    icla_queue_t queues[2] );

void
iclablas_ssetmatrix_transpose(
    icla_int_t m, icla_int_t n, icla_int_t nb,
    const float *hA,    icla_int_t lda,
    iclaFloat_ptr    dAT,   icla_int_t ldda,
    iclaFloat_ptr    dwork, icla_int_t lddw,
    icla_queue_t queues[2] );

  /*
   * RBT-related functions
   */
void
iclablas_sprbt(
    icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    iclaFloat_ptr du,
    iclaFloat_ptr dv,
    icla_queue_t queue );

void
iclablas_sprbt_mv(
    icla_int_t n, icla_int_t nrhs,
    iclaFloat_ptr dv,
    iclaFloat_ptr db, icla_int_t lddb,
    icla_queue_t queue );

void
iclablas_sprbt_mtv(
    icla_int_t n, icla_int_t nrhs,
    iclaFloat_ptr du,
    iclaFloat_ptr db, icla_int_t lddb,
    icla_queue_t queue );

  /*
   * Multi-GPU copy functions
   */
void
icla_sgetmatrix_1D_col_bcyclic(
    icla_int_t ngpu,
    icla_int_t m, icla_int_t n, icla_int_t nb,
    iclaFloat_const_ptr const dA[], icla_int_t ldda,
    float                *hA,   icla_int_t lda,
    icla_queue_t queue[] );

void
icla_ssetmatrix_1D_col_bcyclic(
    icla_int_t ngpu,
    icla_int_t m, icla_int_t n, icla_int_t nb,
    const float *hA,   icla_int_t lda,
    iclaFloat_ptr    dA[], icla_int_t ldda,
    icla_queue_t queue[] );

void
icla_sgetmatrix_1D_row_bcyclic(
    icla_int_t ngpu,
    icla_int_t m, icla_int_t n, icla_int_t nb,
    iclaFloat_const_ptr const dA[], icla_int_t ldda,
    float                *hA,   icla_int_t lda,
    icla_queue_t queue[] );

void
icla_ssetmatrix_1D_row_bcyclic(
    icla_int_t ngpu,
    icla_int_t m, icla_int_t n, icla_int_t nb,
    const float *hA,   icla_int_t lda,
    iclaFloat_ptr    dA[], icla_int_t ldda,
    icla_queue_t queue[] );

void
iclablas_sgetmatrix_transpose_mgpu(
    icla_int_t ngpu,
    icla_int_t m, icla_int_t n, icla_int_t nb,
    iclaFloat_const_ptr const dAT[],    icla_int_t ldda,
    float                *hA,       icla_int_t lda,
    iclaFloat_ptr             dwork[],  icla_int_t lddw,
    icla_queue_t queues[][2] );

void
iclablas_ssetmatrix_transpose_mgpu(
    icla_int_t ngpu,
    icla_int_t m, icla_int_t n, icla_int_t nb,
    const float *hA,      icla_int_t lda,
    iclaFloat_ptr    dAT[],   icla_int_t ldda,
    iclaFloat_ptr    dwork[], icla_int_t lddw,
    icla_queue_t queues[][2] );

// in src/ssytrd_mgpu.cpp
// TODO rename ssetmatrix_sy or similar
icla_int_t
icla_shtodhe(
    icla_int_t ngpu, icla_uplo_t uplo, icla_int_t n, icla_int_t nb,
    float     *A,   icla_int_t lda,
    iclaFloat_ptr dA[], icla_int_t ldda,
    icla_queue_t queues[][10],
    icla_int_t *info );

// in src/spotrf3_mgpu.cpp
// TODO same as icla_shtodhe?
icla_int_t
icla_shtodpo(
    icla_int_t ngpu, icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    icla_int_t off_i, icla_int_t off_j, icla_int_t nb,
    float     *A,   icla_int_t lda,
    iclaFloat_ptr dA[], icla_int_t ldda,
    icla_queue_t queues[][3],
    icla_int_t *info );

// in src/spotrf3_mgpu.cpp
// TODO rename sgetmatrix_sy or similar
icla_int_t
icla_sdtohpo(
    icla_int_t ngpu, icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    icla_int_t off_i, icla_int_t off_j, icla_int_t nb, icla_int_t NB,
    float     *A,   icla_int_t lda,
    iclaFloat_ptr dA[], icla_int_t ldda,
    icla_queue_t queues[][3],
    icla_int_t *info );


  /*
   * Multi-GPU BLAS functions (alphabetical order)
   */
void
iclablas_ssymm_mgpu(
    icla_side_t side, icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    float alpha,
    iclaFloat_ptr dA[],    icla_int_t ldda,  icla_int_t offset,
    iclaFloat_ptr dB[],    icla_int_t lddb,
    float beta,
    iclaFloat_ptr dC[],    icla_int_t lddc,
    iclaFloat_ptr dwork[], icla_int_t dworksiz,
    //float    *C,       icla_int_t ldc,
    //float    *work[],  icla_int_t worksiz,
    icla_int_t ngpu, icla_int_t nb,
    icla_queue_t queues[][20], icla_int_t nqueue,
    icla_event_t events[][iclaMaxGPUs*iclaMaxGPUs+10], icla_int_t nevents,
    icla_int_t gnode[iclaMaxGPUs][iclaMaxGPUs+2], icla_int_t ncmplx );

icla_int_t
iclablas_ssymv_mgpu(
    icla_uplo_t uplo,
    icla_int_t n,
    float alpha,
    iclaFloat_const_ptr const d_lA[], icla_int_t ldda, icla_int_t offset,
    iclaFloat_const_ptr dx,           icla_int_t incx,
    float beta,
    iclaFloat_ptr    dy,              icla_int_t incy,
    float       *hwork,           icla_int_t lhwork,
    iclaFloat_ptr    dwork[],         icla_int_t ldwork,
    icla_int_t ngpu,
    icla_int_t nb,
    icla_queue_t queues[] );

icla_int_t
iclablas_ssymv_mgpu_sync(
    icla_uplo_t uplo,
    icla_int_t n,
    float alpha,
    iclaFloat_const_ptr const d_lA[], icla_int_t ldda, icla_int_t offset,
    iclaFloat_const_ptr dx,           icla_int_t incx,
    float beta,
    iclaFloat_ptr    dy,              icla_int_t incy,
    float       *hwork,           icla_int_t lhwork,
    iclaFloat_ptr    dwork[],         icla_int_t ldwork,
    icla_int_t ngpu,
    icla_int_t nb,
    icla_queue_t queues[] );

icla_int_t
icla_ssytrs_gpu(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nrhs,
    float *dA, icla_int_t ldda,
    icla_int_t *ipiv,
    float *dB, icla_int_t lddb,
    icla_int_t *info,
    icla_queue_t queue );

// Ichi's version, in src/ssytrd_mgpu.cpp
void
icla_ssyr2k_mgpu(
    icla_int_t ngpu,
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t nb, icla_int_t n, icla_int_t k,
    float alpha,
    iclaFloat_ptr dB[], icla_int_t lddb, icla_int_t b_offset,
    float beta,
    iclaFloat_ptr dC[], icla_int_t lddc, icla_int_t c_offset,
    icla_int_t nqueue, icla_queue_t queues[][10] );

void
iclablas_ssyr2k_mgpu2(
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t n, icla_int_t k,
    float alpha,
    iclaFloat_ptr dA[], icla_int_t ldda, icla_int_t a_offset,
    iclaFloat_ptr dB[], icla_int_t lddb, icla_int_t b_offset,
    float beta,
    iclaFloat_ptr dC[], icla_int_t lddc, icla_int_t c_offset,
    icla_int_t ngpu, icla_int_t nb,
    icla_queue_t queues[][20], icla_int_t nqueue );

// in src/spotrf_mgpu_right.cpp
void
icla_ssyrk_mgpu(
    icla_int_t ngpu,
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t nb, icla_int_t n, icla_int_t k,
    float alpha,
    iclaFloat_ptr dB[], icla_int_t lddb, icla_int_t b_offset,
    float beta,
    iclaFloat_ptr dC[], icla_int_t lddc, icla_int_t c_offset,
    icla_int_t nqueue, icla_queue_t queues[][10] );

// in src/spotrf_mgpu_right.cpp
void
icla_ssyrk_mgpu2(
    icla_int_t ngpu,
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t nb, icla_int_t n, icla_int_t k,
    float alpha,
    iclaFloat_ptr dB[], icla_int_t lddb, icla_int_t b_offset,
    float beta,
    iclaFloat_ptr dC[], icla_int_t lddc, icla_int_t c_offset,
    icla_int_t nqueue, icla_queue_t queues[][10] );


  /*
   * LAPACK auxiliary functions (alphabetical order)
   */
icla_int_t
iclablas_sdiinertia(
    icla_int_t n,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    int *dneig,
    icla_queue_t queue );

void
iclablas_sgeadd(
    icla_int_t m, icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_ptr       dB, icla_int_t lddb,
    icla_queue_t queue );

void
iclablas_sgeadd2(
    icla_int_t m, icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    float beta,
    iclaFloat_ptr       dB, icla_int_t lddb,
    icla_queue_t queue );

void
iclablas_sgeam(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    float beta,
    iclaFloat_const_ptr dB, icla_int_t lddb,
    iclaFloat_ptr dC, icla_int_t lddc,
    icla_queue_t queue );

icla_int_t
iclablas_ssiinertia(
    icla_uplo_t uplo,
    icla_int_t n,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    icla_int_t *ipiv,
    int *dneig,
    icla_queue_t queue );

void
iclablas_slacpy(
    icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_ptr       dB, icla_int_t lddb,
    icla_queue_t queue );

void
iclablas_slacpy_conj(
    icla_int_t n,
    iclaFloat_ptr dA1, icla_int_t lda1,
    iclaFloat_ptr dA2, icla_int_t lda2,
    icla_queue_t queue );

void
iclablas_slacpy_sym_in(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    icla_int_t *rows, icla_int_t *perm,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_ptr       dB, icla_int_t lddb,
    icla_queue_t queue );

void
iclablas_slacpy_sym_out(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    icla_int_t *rows, icla_int_t *perm,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_ptr       dB, icla_int_t lddb,
    icla_queue_t queue );

float
iclablas_slange(
    icla_norm_t norm,
    icla_int_t m, icla_int_t n,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_ptr dwork, icla_int_t lwork,
    icla_queue_t queue );

float
iclablas_slansy(
    icla_norm_t norm, icla_uplo_t uplo,
    icla_int_t n,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_ptr dwork, icla_int_t lwork,
    icla_queue_t queue );

void
iclablas_slarfg(
    icla_int_t n,
    iclaFloat_ptr dalpha,
    iclaFloat_ptr dx, icla_int_t incx,
    iclaFloat_ptr dtau,
    icla_queue_t queue );

void
iclablas_slascl(
    icla_type_t type, icla_int_t kl, icla_int_t ku,
    float cfrom, float cto,
    icla_int_t m, icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    icla_queue_t queue,
    icla_int_t *info );

void
iclablas_slascl_2x2(
    icla_type_t type, icla_int_t m,
    iclaFloat_const_ptr dW, icla_int_t lddw,
    iclaFloat_ptr       dA, icla_int_t ldda,
    icla_queue_t queue,
    icla_int_t *info );

void
iclablas_slascl2(
    icla_type_t type,
    icla_int_t m, icla_int_t n,
    iclaFloat_const_ptr dD,
    iclaFloat_ptr dA, icla_int_t ldda,
    icla_queue_t queue,
    icla_int_t *info );

void
iclablas_slascl_diag(
    icla_type_t type, icla_int_t m, icla_int_t n,
    iclaFloat_const_ptr dD, icla_int_t lddd,
    iclaFloat_ptr       dA, icla_int_t ldda,
    icla_queue_t queue,
    icla_int_t *info );

void
iclablas_slaset(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    float offdiag, float diag,
    iclaFloat_ptr dA, icla_int_t ldda,
    icla_queue_t queue );

void
iclablas_slaset_band(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n, icla_int_t k,
    float offdiag, float diag,
    iclaFloat_ptr dA, icla_int_t ldda,
    icla_queue_t queue );

void
iclablas_slaswp(
    icla_int_t n,
    iclaFloat_ptr dAT, icla_int_t ldda,
    icla_int_t k1, icla_int_t k2,
    const icla_int_t *ipiv, icla_int_t inci,
    icla_queue_t queue );

void
iclablas_slaswp2(
    icla_int_t n,
    iclaFloat_ptr dAT, icla_int_t ldda,
    icla_int_t k1, icla_int_t k2,
    iclaInt_const_ptr d_ipiv, icla_int_t inci,
    icla_queue_t queue );

void
iclablas_slaswp_sym(
    icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    icla_int_t k1, icla_int_t k2,
    const icla_int_t *ipiv, icla_int_t inci,
    icla_queue_t queue );

void
iclablas_slaswpx(
    icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldx, icla_int_t ldy,
    icla_int_t k1, icla_int_t k2,
    const icla_int_t *ipiv, icla_int_t inci,
    icla_queue_t queue );

void
icla_slaswp_rowparallel_native(
    icla_int_t n,
    float* input, icla_int_t ldi,
    float* output, icla_int_t ldo,
    icla_int_t k1, icla_int_t k2,
    icla_int_t *pivinfo,
    icla_queue_t queue);

void
icla_slaswp_columnserial(
    icla_int_t n, iclaFloat_ptr dA, icla_int_t lda,
    icla_int_t k1, icla_int_t k2,
    icla_int_t *dipiv, icla_queue_t queue);

void
iclablas_ssymmetrize(
    icla_uplo_t uplo, icla_int_t m,
    iclaFloat_ptr dA, icla_int_t ldda,
    icla_queue_t queue );

void
iclablas_ssymmetrize_tiles(
    icla_uplo_t uplo, icla_int_t m,
    iclaFloat_ptr dA, icla_int_t ldda,
    icla_int_t ntile, icla_int_t mstride, icla_int_t nstride,
    icla_queue_t queue );

void
iclablas_strtri_diag(
    icla_uplo_t uplo, icla_diag_t diag, icla_int_t n,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_ptr d_dinvA,
    icla_queue_t queue );

  /*
   * to cleanup (alphabetical order)
   */
icla_int_t
icla_slarfb_gpu(
    icla_side_t side, icla_trans_t trans, icla_direct_t direct, icla_storev_t storev,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaFloat_const_ptr dV, icla_int_t lddv,
    iclaFloat_const_ptr dT, icla_int_t lddt,
    iclaFloat_ptr dC,       icla_int_t lddc,
    iclaFloat_ptr dwork,    icla_int_t ldwork,
    icla_queue_t queue );

icla_int_t
icla_slarfb_gpu_gemm(
    icla_side_t side, icla_trans_t trans, icla_direct_t direct, icla_storev_t storev,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaFloat_const_ptr dV, icla_int_t lddv,
    iclaFloat_const_ptr dT, icla_int_t lddt,
    iclaFloat_ptr dC,       icla_int_t lddc,
    iclaFloat_ptr dwork,    icla_int_t ldwork,
    iclaFloat_ptr dworkvt,  icla_int_t ldworkvt,
    icla_queue_t queue );

void
icla_slarfbx_gpu(
    icla_int_t m, icla_int_t k,
    iclaFloat_ptr V,  icla_int_t ldv,
    iclaFloat_ptr dT, icla_int_t ldt,
    iclaFloat_ptr c,
    iclaFloat_ptr dwork,
    icla_queue_t queue );

void
icla_slarfg_gpu(
    icla_int_t n,
    iclaFloat_ptr dx0,
    iclaFloat_ptr dx,
    iclaFloat_ptr dtau,
    iclaFloat_ptr        dxnorm,
    iclaFloat_ptr dAkk,
    icla_queue_t queue );

void
icla_slarfgtx_gpu(
    icla_int_t n,
    iclaFloat_ptr dx0,
    iclaFloat_ptr dx,
    iclaFloat_ptr dtau,
    iclaFloat_ptr        dxnorm,
    iclaFloat_ptr dA, icla_int_t iter,
    iclaFloat_ptr V,  icla_int_t ldv,
    iclaFloat_ptr T,  icla_int_t ldt,
    iclaFloat_ptr dwork,
    icla_queue_t queue );

void
icla_slarfgx_gpu(
    icla_int_t n,
    iclaFloat_ptr dx0,
    iclaFloat_ptr dx,
    iclaFloat_ptr dtau,
    iclaFloat_ptr        dxnorm,
    iclaFloat_ptr dA, icla_int_t iter,
    icla_queue_t queue );

void
icla_slarfx_gpu(
    icla_int_t m, icla_int_t n,
    iclaFloat_ptr v,
    iclaFloat_ptr tau,
    iclaFloat_ptr C, icla_int_t ldc,
    iclaFloat_ptr        xnorm,
    iclaFloat_ptr dT, icla_int_t iter,
    iclaFloat_ptr work,
    icla_queue_t queue );

  /*
   * Level 1 BLAS (alphabetical order)
   */
void
iclablas_saxpycp(
    icla_int_t m,
    iclaFloat_ptr dr,
    iclaFloat_ptr dx,
    iclaFloat_const_ptr db,
    icla_queue_t queue );

void
iclablas_sswap(
    icla_int_t n,
    iclaFloat_ptr dx, icla_int_t incx,
    iclaFloat_ptr dy, icla_int_t incy,
    icla_queue_t queue );

void
iclablas_sswapblk(
    icla_order_t order,
    icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    iclaFloat_ptr dB, icla_int_t lddb,
    icla_int_t i1, icla_int_t i2,
    const icla_int_t *ipiv, icla_int_t inci,
    icla_int_t offset,
    icla_queue_t queue );

void
iclablas_sswapdblk(
    icla_int_t n, icla_int_t nb,
    iclaFloat_ptr dA, icla_int_t ldda, icla_int_t inca,
    iclaFloat_ptr dB, icla_int_t lddb, icla_int_t incb,
    icla_queue_t queue );

void
iclablas_snrm2_adjust(
    icla_int_t k,
    iclaFloat_ptr dxnorm,
    iclaFloat_ptr dc,
    icla_queue_t queue );

#ifdef REAL
void
iclablas_snrm2_check(
    icla_int_t m, icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    iclaFloat_ptr dxnorm,
    iclaFloat_ptr dlsticc,
    icla_queue_t queue );
#endif

void
iclablas_snrm2_check(
    icla_int_t m, icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    iclaFloat_ptr dxnorm,
    iclaFloat_ptr dlsticc,
    icla_queue_t queue );

void
iclablas_snrm2_cols(
    icla_int_t m, icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    iclaFloat_ptr dxnorm,
    icla_queue_t queue );

void
iclablas_snrm2_row_check_adjust(
    icla_int_t k, float tol,
    iclaFloat_ptr dxnorm,
    iclaFloat_ptr dxnorm2,
    iclaFloat_ptr dC, icla_int_t lddc,
    iclaFloat_ptr dlsticc,
    icla_queue_t queue );

  /*
   * Level 2 BLAS (alphabetical order)
   */
// trsv were always queue versions
void
iclablas_strsv(
    icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t n,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_ptr       db, icla_int_t incb,
    icla_queue_t queue );

// todo: move flag before queue?
void
iclablas_strsv_outofplace(
    icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t n,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_ptr db,       icla_int_t incb,
    iclaFloat_ptr dx,
    icla_queue_t queue,
    icla_int_t flag );

void
iclablas_sgemv(
    icla_trans_t trans, icla_int_t m, icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_const_ptr dx, icla_int_t incx,
    float beta,
    iclaFloat_ptr dy, icla_int_t incy,
    icla_queue_t queue );

void
iclablas_sgemv_conj(
    icla_int_t m, icla_int_t n, float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_const_ptr dx, icla_int_t incx,
    float beta,
    iclaFloat_ptr dy, icla_int_t incy,
    icla_queue_t queue );

icla_int_t
iclablas_ssymv(
    icla_uplo_t uplo, icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_const_ptr dx, icla_int_t incx,
    float beta,
    iclaFloat_ptr       dy, icla_int_t incy,
    icla_queue_t queue );

icla_int_t
iclablas_ssymv(
    icla_uplo_t uplo, icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_const_ptr dx, icla_int_t incx,
    float beta,
    iclaFloat_ptr       dy, icla_int_t incy,
    icla_queue_t queue );

// hemv/symv_work were always queue versions
icla_int_t
iclablas_ssymv_work(
    icla_uplo_t uplo, icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_const_ptr dx, icla_int_t incx,
    float beta,
    iclaFloat_ptr       dy, icla_int_t incy,
    iclaFloat_ptr       dwork, icla_int_t lwork,
    icla_queue_t queue );

icla_int_t
iclablas_ssymv_work(
    icla_uplo_t uplo, icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_const_ptr dx, icla_int_t incx,
    float beta,
    iclaFloat_ptr       dy, icla_int_t incy,
    iclaFloat_ptr       dwork, icla_int_t lwork,
    icla_queue_t queue );

  /*
   * Level 3 BLAS (alphabetical order)
   */
void
iclablas_sgemm(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_const_ptr dB, icla_int_t lddb,
    float beta,
    iclaFloat_ptr       dC, icla_int_t lddc,
    icla_queue_t queue );

void
iclablas_sgemm_reduce(
    icla_int_t m, icla_int_t n, icla_int_t k,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_const_ptr dB, icla_int_t lddb,
    float beta,
    iclaFloat_ptr       dC, icla_int_t lddc,
    icla_queue_t queue );

void
iclablas_strsm(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_ptr       dB, icla_int_t lddb,
    icla_queue_t queue );

void
iclablas_strsm_outofplace(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_ptr       dB, icla_int_t lddb,
    iclaFloat_ptr       dX, icla_int_t lddx,
    icla_int_t flag,
    iclaFloat_ptr d_dinvA, icla_int_t dinvA_length,
    icla_queue_t queue );

void
iclablas_strsm_work(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_ptr       dB, icla_int_t lddb,
    iclaFloat_ptr       dX, icla_int_t lddx,
    icla_int_t flag,
    iclaFloat_ptr d_dinvA, icla_int_t dinvA_length,
    icla_queue_t queue );


  /*
   * Wrappers for platform independence.
   * These wrap CUBLAS or AMD OpenCL BLAS functions.
   */

// =============================================================================
// copying vectors
// set  copies host   to device
// get  copies device to host
// copy copies device to device
// (with CUDA unified addressing, copy can be between same or different devices)
// Add the function, file, and line for error-reporting purposes.

/// Type-safe version of icla_setvector() for float arrays.
/// @ingroup icla_setvector
#define icla_ssetvector(           n, hx_src, incx, dy_dst, incy, queue ) \
        icla_ssetvector_internal(  n, hx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of icla_getvector() for float arrays.
/// @ingroup icla_getvector
#define icla_sgetvector(           n, dx_src, incx, hy_dst, incy, queue ) \
        icla_sgetvector_internal(  n, dx_src, incx, hy_dst, incy, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of icla_copyvector() for float arrays.
/// @ingroup icla_copyvector
#define icla_scopyvector(          n, dx_src, incx, dy_dst, incy, queue ) \
        icla_scopyvector_internal( n, dx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of icla_setvector_async() for float arrays.
/// @ingroup icla_setvector
#define icla_ssetvector_async(           n, hx_src, incx, dy_dst, incy, queue ) \
        icla_ssetvector_async_internal(  n, hx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of icla_getvector_async() for float arrays.
/// @ingroup icla_getvector
#define icla_sgetvector_async(           n, dx_src, incx, hy_dst, incy, queue ) \
        icla_sgetvector_async_internal(  n, dx_src, incx, hy_dst, incy, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of icla_copyvector_async() for float arrays.
/// @ingroup icla_copyvector
#define icla_scopyvector_async(          n, dx_src, incx, dy_dst, incy, queue ) \
        icla_scopyvector_async_internal( n, dx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

static inline void
icla_ssetvector_internal(
    icla_int_t n,
    float const    *hx_src, icla_int_t incx,
    iclaFloat_ptr       dy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_setvector_internal( n, sizeof(float),
                              hx_src, incx,
                              dy_dst, incy, queue,
                              func, file, line );
}

static inline void
icla_sgetvector_internal(
    icla_int_t n,
    iclaFloat_const_ptr dx_src, icla_int_t incx,
    float          *hy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_getvector_internal( n, sizeof(float),
                              dx_src, incx,
                              hy_dst, incy, queue,
                              func, file, line );
}

static inline void
icla_scopyvector_internal(
    icla_int_t n,
    iclaFloat_const_ptr dx_src, icla_int_t incx,
    iclaFloat_ptr       dy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_copyvector_internal( n, sizeof(float),
                               dx_src, incx,
                               dy_dst, incy, queue,
                               func, file, line );
}

static inline void
icla_ssetvector_async_internal(
    icla_int_t n,
    float const    *hx_src, icla_int_t incx,
    iclaFloat_ptr       dy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_setvector_async_internal( n, sizeof(float),
                                    hx_src, incx,
                                    dy_dst, incy, queue,
                                    func, file, line );
}

static inline void
icla_sgetvector_async_internal(
    icla_int_t n,
    iclaFloat_const_ptr dx_src, icla_int_t incx,
    float          *hy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_getvector_async_internal( n, sizeof(float),
                                    dx_src, incx,
                                    hy_dst, incy, queue,
                                    func, file, line );
}

static inline void
icla_scopyvector_async_internal(
    icla_int_t n,
    iclaFloat_const_ptr dx_src, icla_int_t incx,
    iclaFloat_ptr       dy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_copyvector_async_internal( n, sizeof(float),
                                     dx_src, incx,
                                     dy_dst, incy, queue,
                                     func, file, line );
}


// =============================================================================
// copying sub-matrices (contiguous columns)

/// Type-safe version of icla_setmatrix() for float arrays.
/// @ingroup icla_setmatrix
#define icla_ssetmatrix(           m, n, hA_src, lda,  dB_dst, lddb, queue ) \
        icla_ssetmatrix_internal(  m, n, hA_src, lda,  dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of icla_getmatrix() for float arrays.
/// @ingroup icla_getmatrix
#define icla_sgetmatrix(           m, n, dA_src, ldda, hB_dst, ldb,  queue ) \
        icla_sgetmatrix_internal(  m, n, dA_src, ldda, hB_dst, ldb,  queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of icla_copymatrix() for float arrays.
/// @ingroup icla_copymatrix
#define icla_scopymatrix(          m, n, dA_src, ldda, dB_dst, lddb, queue ) \
        icla_scopymatrix_internal( m, n, dA_src, ldda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of icla_setmatrix_async() for float arrays.
/// @ingroup icla_setmatrix
#define icla_ssetmatrix_async(           m, n, hA_src, lda, dB_dst, lddb, queue ) \
        icla_ssetmatrix_async_internal(  m, n, hA_src, lda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of icla_getmatrix_async() for float arrays.
/// @ingroup icla_getmatrix
#define icla_sgetmatrix_async(           m, n, dA_src, ldda, hB_dst, ldb, queue ) \
        icla_sgetmatrix_async_internal(  m, n, dA_src, ldda, hB_dst, ldb, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of icla_copymatrix_async() for float arrays.
/// @ingroup icla_copymatrix
#define icla_scopymatrix_async(          m, n, dA_src, ldda, dB_dst, lddb, queue ) \
        icla_scopymatrix_async_internal( m, n, dA_src, ldda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

static inline void
icla_ssetmatrix_internal(
    icla_int_t m, icla_int_t n,
    float const    *hA_src, icla_int_t lda,
    iclaFloat_ptr       dB_dst, icla_int_t lddb,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_setmatrix_internal( m, n, sizeof(float),
                              hA_src, lda,
                              dB_dst, lddb, queue,
                              func, file, line );
}

static inline void
icla_sgetmatrix_internal(
    icla_int_t m, icla_int_t n,
    iclaFloat_const_ptr dA_src, icla_int_t ldda,
    float          *hB_dst, icla_int_t ldb,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_getmatrix_internal( m, n, sizeof(float),
                              dA_src, ldda,
                              hB_dst, ldb, queue,
                              func, file, line );
}

static inline void
icla_scopymatrix_internal(
    icla_int_t m, icla_int_t n,
    iclaFloat_const_ptr dA_src, icla_int_t ldda,
    iclaFloat_ptr       dB_dst, icla_int_t lddb,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_copymatrix_internal( m, n, sizeof(float),
                               dA_src, ldda,
                               dB_dst, lddb, queue,
                               func, file, line );
}

static inline void
icla_ssetmatrix_async_internal(
    icla_int_t m, icla_int_t n,
    float const    *hA_src, icla_int_t lda,
    iclaFloat_ptr       dB_dst, icla_int_t lddb,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_setmatrix_async_internal( m, n, sizeof(float),
                                    hA_src, lda,
                                    dB_dst, lddb, queue,
                                    func, file, line );
}

static inline void
icla_sgetmatrix_async_internal(
    icla_int_t m, icla_int_t n,
    iclaFloat_const_ptr dA_src, icla_int_t ldda,
    float          *hB_dst, icla_int_t ldb,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_getmatrix_async_internal( m, n, sizeof(float),
                                    dA_src, ldda,
                                    hB_dst, ldb, queue,
                                    func, file, line );
}

static inline void
icla_scopymatrix_async_internal(
    icla_int_t m, icla_int_t n,
    iclaFloat_const_ptr dA_src, icla_int_t ldda,
    iclaFloat_ptr       dB_dst, icla_int_t lddb,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_copymatrix_async_internal( m, n, sizeof(float),
                                     dA_src, ldda,
                                     dB_dst, lddb, queue,
                                     func, file, line );
}


// =============================================================================
// Level 1 BLAS (alphabetical order)

icla_int_t
icla_isamax(
    icla_int_t n,
    iclaFloat_const_ptr dx, icla_int_t incx,
    icla_queue_t queue );

icla_int_t
icla_isamax_native(
    icla_int_t length,
    iclaFloat_ptr x, icla_int_t incx,
    icla_int_t* ipiv, icla_int_t *info,
    icla_int_t step, icla_int_t gbstep, icla_queue_t queue);

icla_int_t
icla_isamin(
    icla_int_t n,
    iclaFloat_const_ptr dx, icla_int_t incx,
    icla_queue_t queue );

float
icla_sasum(
    icla_int_t n,
    iclaFloat_const_ptr dx, icla_int_t incx,
    icla_queue_t queue );

void
icla_saxpy(
    icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dx, icla_int_t incx,
    iclaFloat_ptr       dy, icla_int_t incy,
    icla_queue_t queue );

void
icla_scopy(
    icla_int_t n,
    iclaFloat_const_ptr dx, icla_int_t incx,
    iclaFloat_ptr       dy, icla_int_t incy,
    icla_queue_t queue );

float
icla_sdot(
    icla_int_t n,
    iclaFloat_const_ptr dx, icla_int_t incx,
    iclaFloat_const_ptr dy, icla_int_t incy,
    icla_queue_t queue );

float
icla_sdot(
    icla_int_t n,
    iclaFloat_const_ptr dx, icla_int_t incx,
    iclaFloat_const_ptr dy, icla_int_t incy,
    icla_queue_t queue );

float
icla_snrm2(
    icla_int_t n,
    iclaFloat_const_ptr dx, icla_int_t incx,
    icla_queue_t queue );

void
icla_srot(
    icla_int_t n,
    iclaFloat_ptr dx, icla_int_t incx,
    iclaFloat_ptr dy, icla_int_t incy,
    float dc, float ds,
    icla_queue_t queue );

void
icla_srot(
    icla_int_t n,
    iclaFloat_ptr dx, icla_int_t incx,
    iclaFloat_ptr dy, icla_int_t incy,
    float dc, float ds,
    icla_queue_t queue );

void
icla_srotg(
    iclaFloat_ptr a,
    iclaFloat_ptr b,
    iclaFloat_ptr        c,
    iclaFloat_ptr s,
    icla_queue_t queue );

#ifdef ICLA_REAL
void
icla_srotm(
    icla_int_t n,
    iclaFloat_ptr dx, icla_int_t incx,
    iclaFloat_ptr dy, icla_int_t incy,
    iclaFloat_const_ptr param,
    icla_queue_t queue );

void
icla_srotmg(
    iclaFloat_ptr       d1,
    iclaFloat_ptr       d2,
    iclaFloat_ptr       x1,
    iclaFloat_const_ptr y1,
    iclaFloat_ptr param,
    icla_queue_t queue );
#endif  // ICLA_REAL

void
icla_sscal(
    icla_int_t n,
    float alpha,
    iclaFloat_ptr dx, icla_int_t incx,
    icla_queue_t queue );

void
icla_sscal(
    icla_int_t n,
    float alpha,
    iclaFloat_ptr dx, icla_int_t incx,
    icla_queue_t queue );

icla_int_t
icla_sscal_sger_native(
    icla_int_t m, icla_int_t n,
    iclaFloat_ptr dA, icla_int_t lda,
    icla_int_t *info, icla_int_t step, icla_int_t gbstep,
    icla_queue_t queue);

void
icla_sswap(
    icla_int_t n,
    iclaFloat_ptr dx, icla_int_t incx,
    iclaFloat_ptr dy, icla_int_t incy,
    icla_queue_t queue );

void
icla_sswap_native(
    icla_int_t n, iclaFloat_ptr x, icla_int_t incx,
    icla_int_t step, icla_int_t* ipiv,
    icla_queue_t queue);

// =============================================================================
// Level 2 BLAS (alphabetical order)

void
icla_sgemv(
    icla_trans_t transA,
    icla_int_t m, icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_const_ptr dx, icla_int_t incx,
    float beta,
    iclaFloat_ptr       dy, icla_int_t incy,
    icla_queue_t queue );

void
icla_sger(
    icla_int_t m, icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dx, icla_int_t incx,
    iclaFloat_const_ptr dy, icla_int_t incy,
    iclaFloat_ptr       dA, icla_int_t ldda,
    icla_queue_t queue );

#ifdef ICLA_COMPLEX
void
icla_sger(
    icla_int_t m, icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dx, icla_int_t incx,
    iclaFloat_const_ptr dy, icla_int_t incy,
    iclaFloat_ptr       dA, icla_int_t ldda,
    icla_queue_t queue );

void
icla_ssymv(
    icla_uplo_t uplo,
    icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_const_ptr dx, icla_int_t incx,
    float beta,
    iclaFloat_ptr       dy, icla_int_t incy,
    icla_queue_t queue );

void
icla_ssyr(
    icla_uplo_t uplo,
    icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dx, icla_int_t incx,
    iclaFloat_ptr       dA, icla_int_t ldda,
    icla_queue_t queue );

void
icla_ssyr2(
    icla_uplo_t uplo,
    icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dx, icla_int_t incx,
    iclaFloat_const_ptr dy, icla_int_t incy,
    iclaFloat_ptr       dA, icla_int_t ldda,
    icla_queue_t queue );
#endif // ICLA_COMPLEX

void
icla_ssymv(
    icla_uplo_t uplo,
    icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_const_ptr dx, icla_int_t incx,
    float beta,
    iclaFloat_ptr       dy, icla_int_t incy,
    icla_queue_t queue );

void
icla_ssyr(
    icla_uplo_t uplo,
    icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dx, icla_int_t incx,
    iclaFloat_ptr       dA, icla_int_t ldda,
    icla_queue_t queue );

void
icla_ssyr2(
    icla_uplo_t uplo,
    icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dx, icla_int_t incx,
    iclaFloat_const_ptr dy, icla_int_t incy,
    iclaFloat_ptr       dA, icla_int_t ldda,
    icla_queue_t queue );

void
icla_strmv(
    icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t n,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_ptr       dx, icla_int_t incx,
    icla_queue_t queue );

void
iclablas_strmv(
    icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t n,
    float *dA, icla_int_t ldda,
    float *dx, icla_int_t incx,
    icla_queue_t queue );

void
icla_strsv(
    icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t n,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_ptr       dx, icla_int_t incx,
    icla_queue_t queue );

// =============================================================================
// Level 3 BLAS (alphabetical order)

void
icla_sgemm(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_const_ptr dB, icla_int_t lddb,
    float beta,
    iclaFloat_ptr       dC, icla_int_t lddc,
    icla_queue_t queue );

void
icla_ssymm(
    icla_side_t side, icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_const_ptr dB, icla_int_t lddb,
    float beta,
    iclaFloat_ptr       dC, icla_int_t lddc,
    icla_queue_t queue );

void
iclablas_ssymm(
    icla_side_t side, icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_const_ptr dB, icla_int_t lddb,
    float beta,
    iclaFloat_ptr       dC, icla_int_t lddc,
    icla_queue_t queue );

void
icla_ssyr2k(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_const_ptr dB, icla_int_t lddb,
    float beta,
    iclaFloat_ptr       dC, icla_int_t lddc,
    icla_queue_t queue );

void iclablas_ssyr2k(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    float alpha,
    iclaFloat_ptr dA, icla_int_t ldda,
    iclaFloat_ptr dB, icla_int_t lddb,
    float beta,
    iclaFloat_ptr       dC, icla_int_t lddc,
    icla_queue_t queue );

void
icla_ssyrk(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    float beta,
    iclaFloat_ptr       dC, icla_int_t lddc,
    icla_queue_t queue );

void iclablas_ssyrk(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    float alpha,
    iclaFloat_ptr dA, icla_int_t ldda,
    float beta,
    iclaFloat_ptr dC, icla_int_t lddc,
    icla_queue_t queue);

void iclablas_ssyrk_internal(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k, icla_int_t nb,
    float alpha,
    iclaFloat_ptr dA, icla_int_t ldda,
    iclaFloat_ptr dB, icla_int_t lddb,
    float beta,
    iclaFloat_ptr dC, icla_int_t lddc,
    icla_int_t conjugate, icla_queue_t queue);

void
iclablas_ssyrk_small_reduce(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    float alpha, float* dA, icla_int_t ldda,
    float beta,  float* dC, icla_int_t lddc,
    icla_int_t nthread_blocks, icla_queue_t queue );

void
icla_ssymm(
    icla_side_t side, icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_const_ptr dB, icla_int_t lddb,
    float beta,
    iclaFloat_ptr       dC, icla_int_t lddc,
    icla_queue_t queue );

void
iclablas_ssymm(
    icla_side_t side, icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_const_ptr dB, icla_int_t lddb,
    float beta,
    iclaFloat_ptr       dC, icla_int_t lddc,
    icla_queue_t queue );

void
icla_ssyr2k(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_const_ptr dB, icla_int_t lddb,
    float beta,
    iclaFloat_ptr       dC, icla_int_t lddc,
    icla_queue_t queue );

void iclablas_ssyr2k(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    float alpha,
    iclaFloat_ptr dA, icla_int_t ldda,
    iclaFloat_ptr dB, icla_int_t lddb,
    float beta,
    iclaFloat_ptr dC, icla_int_t lddc,
    icla_queue_t queue );

void
icla_ssyrk(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    float beta,
    iclaFloat_ptr       dC, icla_int_t lddc,
    icla_queue_t queue );

void iclablas_ssyrk(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    float alpha,
    iclaFloat_ptr dA, icla_int_t ldda,
    float beta,
    iclaFloat_ptr dC, icla_int_t lddc,
    icla_queue_t queue);

void
icla_strmm(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_ptr       dB, icla_int_t lddb,
    icla_queue_t queue );

void
iclablas_strmm(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t m, icla_int_t n,
        float alpha,
        float *dA, icla_int_t ldda,
        float *dB, icla_int_t lddb,
        icla_queue_t queue );

void
icla_strsm(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_ptr       dB, icla_int_t lddb,
    icla_queue_t queue );

void
icla_sgetf2trsm_2d_native(
    icla_int_t m, icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    iclaFloat_ptr dB, icla_int_t lddb,
    icla_queue_t queue);

icla_int_t
icla_spotf2_lpout(
        icla_uplo_t uplo, icla_int_t n,
        float *dA, icla_int_t lda, icla_int_t gbstep,
        icla_int_t *dinfo, icla_queue_t queue);

icla_int_t
icla_spotf2_lpin(
        icla_uplo_t uplo, icla_int_t n,
        float *dA, icla_int_t lda, icla_int_t gbstep,
        icla_int_t *dinfo, icla_queue_t queue);

#ifdef __cplusplus
}
#endif

#undef ICLA_REAL

#endif // ICLABLAS_S_H
