/*
    -- ICLA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from include/iclablas_z.h, normal z -> c, Fri Nov 29 12:16:14 2024
*/

#ifndef ICLABLAS_C_H
#define ICLABLAS_C_H

#include "icla_types.h"
#include "icla_copy.h"

#define ICLA_COMPLEX

#ifdef __cplusplus
extern "C" {
#endif

  /*
   * Transpose functions
   */
void
iclablas_ctranspose_inplace(
    icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    icla_queue_t queue );

void
iclablas_ctranspose_conj_inplace(
    icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    icla_queue_t queue );

void
iclablas_ctranspose(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex_const_ptr dA,  icla_int_t ldda,
    iclaFloatComplex_ptr       dAT, icla_int_t lddat,
    icla_queue_t queue );

void
iclablas_ctranspose_conj(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex_const_ptr dA,  icla_int_t ldda,
    iclaFloatComplex_ptr       dAT, icla_int_t lddat,
    icla_queue_t queue );

void
iclablas_cgetmatrix_transpose(
    icla_int_t m, icla_int_t n, icla_int_t nb,
    iclaFloatComplex_const_ptr dAT,   icla_int_t ldda,
    iclaFloatComplex          *hA,    icla_int_t lda,
    iclaFloatComplex_ptr       dwork, icla_int_t lddw,
    icla_queue_t queues[2] );

void
iclablas_csetmatrix_transpose(
    icla_int_t m, icla_int_t n, icla_int_t nb,
    const iclaFloatComplex *hA,    icla_int_t lda,
    iclaFloatComplex_ptr    dAT,   icla_int_t ldda,
    iclaFloatComplex_ptr    dwork, icla_int_t lddw,
    icla_queue_t queues[2] );

  /*
   * RBT-related functions
   */
void
iclablas_cprbt(
    icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr du,
    iclaFloatComplex_ptr dv,
    icla_queue_t queue );

void
iclablas_cprbt_mv(
    icla_int_t n, icla_int_t nrhs,
    iclaFloatComplex_ptr dv,
    iclaFloatComplex_ptr db, icla_int_t lddb,
    icla_queue_t queue );

void
iclablas_cprbt_mtv(
    icla_int_t n, icla_int_t nrhs,
    iclaFloatComplex_ptr du,
    iclaFloatComplex_ptr db, icla_int_t lddb,
    icla_queue_t queue );

  /*
   * Multi-GPU copy functions
   */
void
icla_cgetmatrix_1D_col_bcyclic(
    icla_int_t ngpu,
    icla_int_t m, icla_int_t n, icla_int_t nb,
    iclaFloatComplex_const_ptr const dA[], icla_int_t ldda,
    iclaFloatComplex                *hA,   icla_int_t lda,
    icla_queue_t queue[] );

void
icla_csetmatrix_1D_col_bcyclic(
    icla_int_t ngpu,
    icla_int_t m, icla_int_t n, icla_int_t nb,
    const iclaFloatComplex *hA,   icla_int_t lda,
    iclaFloatComplex_ptr    dA[], icla_int_t ldda,
    icla_queue_t queue[] );

void
icla_cgetmatrix_1D_row_bcyclic(
    icla_int_t ngpu,
    icla_int_t m, icla_int_t n, icla_int_t nb,
    iclaFloatComplex_const_ptr const dA[], icla_int_t ldda,
    iclaFloatComplex                *hA,   icla_int_t lda,
    icla_queue_t queue[] );

void
icla_csetmatrix_1D_row_bcyclic(
    icla_int_t ngpu,
    icla_int_t m, icla_int_t n, icla_int_t nb,
    const iclaFloatComplex *hA,   icla_int_t lda,
    iclaFloatComplex_ptr    dA[], icla_int_t ldda,
    icla_queue_t queue[] );

void
iclablas_cgetmatrix_transpose_mgpu(
    icla_int_t ngpu,
    icla_int_t m, icla_int_t n, icla_int_t nb,
    iclaFloatComplex_const_ptr const dAT[],    icla_int_t ldda,
    iclaFloatComplex                *hA,       icla_int_t lda,
    iclaFloatComplex_ptr             dwork[],  icla_int_t lddw,
    icla_queue_t queues[][2] );

void
iclablas_csetmatrix_transpose_mgpu(
    icla_int_t ngpu,
    icla_int_t m, icla_int_t n, icla_int_t nb,
    const iclaFloatComplex *hA,      icla_int_t lda,
    iclaFloatComplex_ptr    dAT[],   icla_int_t ldda,
    iclaFloatComplex_ptr    dwork[], icla_int_t lddw,
    icla_queue_t queues[][2] );

// in src/chetrd_mgpu.cpp
// TODO rename csetmatrix_sy or similar
icla_int_t
icla_chtodhe(
    icla_int_t ngpu, icla_uplo_t uplo, icla_int_t n, icla_int_t nb,
    iclaFloatComplex     *A,   icla_int_t lda,
    iclaFloatComplex_ptr dA[], icla_int_t ldda,
    icla_queue_t queues[][10],
    icla_int_t *info );

// in src/cpotrf3_mgpu.cpp
// TODO same as icla_chtodhe?
icla_int_t
icla_chtodpo(
    icla_int_t ngpu, icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    icla_int_t off_i, icla_int_t off_j, icla_int_t nb,
    iclaFloatComplex     *A,   icla_int_t lda,
    iclaFloatComplex_ptr dA[], icla_int_t ldda,
    icla_queue_t queues[][3],
    icla_int_t *info );

// in src/cpotrf3_mgpu.cpp
// TODO rename cgetmatrix_sy or similar
icla_int_t
icla_cdtohpo(
    icla_int_t ngpu, icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    icla_int_t off_i, icla_int_t off_j, icla_int_t nb, icla_int_t NB,
    iclaFloatComplex     *A,   icla_int_t lda,
    iclaFloatComplex_ptr dA[], icla_int_t ldda,
    icla_queue_t queues[][3],
    icla_int_t *info );


  /*
   * Multi-GPU BLAS functions (alphabetical order)
   */
void
iclablas_chemm_mgpu(
    icla_side_t side, icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_ptr dA[],    icla_int_t ldda,  icla_int_t offset,
    iclaFloatComplex_ptr dB[],    icla_int_t lddb,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr dC[],    icla_int_t lddc,
    iclaFloatComplex_ptr dwork[], icla_int_t dworksiz,
    //iclaFloatComplex    *C,       icla_int_t ldc,
    //iclaFloatComplex    *work[],  icla_int_t worksiz,
    icla_int_t ngpu, icla_int_t nb,
    icla_queue_t queues[][20], icla_int_t nqueue,
    icla_event_t events[][iclaMaxGPUs*iclaMaxGPUs+10], icla_int_t nevents,
    icla_int_t gnode[iclaMaxGPUs][iclaMaxGPUs+2], icla_int_t ncmplx );

icla_int_t
iclablas_chemv_mgpu(
    icla_uplo_t uplo,
    icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr const d_lA[], icla_int_t ldda, icla_int_t offset,
    iclaFloatComplex_const_ptr dx,           icla_int_t incx,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr    dy,              icla_int_t incy,
    iclaFloatComplex       *hwork,           icla_int_t lhwork,
    iclaFloatComplex_ptr    dwork[],         icla_int_t ldwork,
    icla_int_t ngpu,
    icla_int_t nb,
    icla_queue_t queues[] );

icla_int_t
iclablas_chemv_mgpu_sync(
    icla_uplo_t uplo,
    icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr const d_lA[], icla_int_t ldda, icla_int_t offset,
    iclaFloatComplex_const_ptr dx,           icla_int_t incx,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr    dy,              icla_int_t incy,
    iclaFloatComplex       *hwork,           icla_int_t lhwork,
    iclaFloatComplex_ptr    dwork[],         icla_int_t ldwork,
    icla_int_t ngpu,
    icla_int_t nb,
    icla_queue_t queues[] );

icla_int_t
icla_chetrs_gpu(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nrhs,
    iclaFloatComplex *dA, icla_int_t ldda,
    icla_int_t *ipiv,
    iclaFloatComplex *dB, icla_int_t lddb,
    icla_int_t *info,
    icla_queue_t queue );

// Ichi's version, in src/chetrd_mgpu.cpp
void
icla_cher2k_mgpu(
    icla_int_t ngpu,
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t nb, icla_int_t n, icla_int_t k,
    iclaFloatComplex alpha,
    iclaFloatComplex_ptr dB[], icla_int_t lddb, icla_int_t b_offset,
    float beta,
    iclaFloatComplex_ptr dC[], icla_int_t lddc, icla_int_t c_offset,
    icla_int_t nqueue, icla_queue_t queues[][10] );

void
iclablas_cher2k_mgpu2(
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t n, icla_int_t k,
    iclaFloatComplex alpha,
    iclaFloatComplex_ptr dA[], icla_int_t ldda, icla_int_t a_offset,
    iclaFloatComplex_ptr dB[], icla_int_t lddb, icla_int_t b_offset,
    float beta,
    iclaFloatComplex_ptr dC[], icla_int_t lddc, icla_int_t c_offset,
    icla_int_t ngpu, icla_int_t nb,
    icla_queue_t queues[][20], icla_int_t nqueue );

// in src/cpotrf_mgpu_right.cpp
void
icla_cherk_mgpu(
    icla_int_t ngpu,
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t nb, icla_int_t n, icla_int_t k,
    float alpha,
    iclaFloatComplex_ptr dB[], icla_int_t lddb, icla_int_t b_offset,
    float beta,
    iclaFloatComplex_ptr dC[], icla_int_t lddc, icla_int_t c_offset,
    icla_int_t nqueue, icla_queue_t queues[][10] );

// in src/cpotrf_mgpu_right.cpp
void
icla_cherk_mgpu2(
    icla_int_t ngpu,
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t nb, icla_int_t n, icla_int_t k,
    float alpha,
    iclaFloatComplex_ptr dB[], icla_int_t lddb, icla_int_t b_offset,
    float beta,
    iclaFloatComplex_ptr dC[], icla_int_t lddc, icla_int_t c_offset,
    icla_int_t nqueue, icla_queue_t queues[][10] );


  /*
   * LAPACK auxiliary functions (alphabetical order)
   */
icla_int_t
iclablas_cdiinertia(
    icla_int_t n,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    int *dneig,
    icla_queue_t queue );

void
iclablas_cgeadd(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr       dB, icla_int_t lddb,
    icla_queue_t queue );

void
iclablas_cgeadd2(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr       dB, icla_int_t lddb,
    icla_queue_t queue );

void
iclablas_cgeam(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex beta,
    iclaFloatComplex_const_ptr dB, icla_int_t lddb,
    iclaFloatComplex_ptr dC, icla_int_t lddc,
    icla_queue_t queue );

icla_int_t
iclablas_cheinertia(
    icla_uplo_t uplo,
    icla_int_t n,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    icla_int_t *ipiv,
    int *dneig,
    icla_queue_t queue );

void
iclablas_clacpy(
    icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr       dB, icla_int_t lddb,
    icla_queue_t queue );

void
iclablas_clacpy_conj(
    icla_int_t n,
    iclaFloatComplex_ptr dA1, icla_int_t lda1,
    iclaFloatComplex_ptr dA2, icla_int_t lda2,
    icla_queue_t queue );

void
iclablas_clacpy_sym_in(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    icla_int_t *rows, icla_int_t *perm,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr       dB, icla_int_t lddb,
    icla_queue_t queue );

void
iclablas_clacpy_sym_out(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    icla_int_t *rows, icla_int_t *perm,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr       dB, icla_int_t lddb,
    icla_queue_t queue );

float
iclablas_clange(
    icla_norm_t norm,
    icla_int_t m, icla_int_t n,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloat_ptr dwork, icla_int_t lwork,
    icla_queue_t queue );

float
iclablas_clanhe(
    icla_norm_t norm, icla_uplo_t uplo,
    icla_int_t n,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloat_ptr dwork, icla_int_t lwork,
    icla_queue_t queue );

void
iclablas_clarfg(
    icla_int_t n,
    iclaFloatComplex_ptr dalpha,
    iclaFloatComplex_ptr dx, icla_int_t incx,
    iclaFloatComplex_ptr dtau,
    icla_queue_t queue );

void
iclablas_clascl(
    icla_type_t type, icla_int_t kl, icla_int_t ku,
    float cfrom, float cto,
    icla_int_t m, icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    icla_queue_t queue,
    icla_int_t *info );

void
iclablas_clascl_2x2(
    icla_type_t type, icla_int_t m,
    iclaFloatComplex_const_ptr dW, icla_int_t lddw,
    iclaFloatComplex_ptr       dA, icla_int_t ldda,
    icla_queue_t queue,
    icla_int_t *info );

void
iclablas_clascl2(
    icla_type_t type,
    icla_int_t m, icla_int_t n,
    iclaFloat_const_ptr dD,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    icla_queue_t queue,
    icla_int_t *info );

void
iclablas_clascl_diag(
    icla_type_t type, icla_int_t m, icla_int_t n,
    iclaFloatComplex_const_ptr dD, icla_int_t lddd,
    iclaFloatComplex_ptr       dA, icla_int_t ldda,
    icla_queue_t queue,
    icla_int_t *info );

void
iclablas_claset(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    iclaFloatComplex offdiag, iclaFloatComplex diag,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    icla_queue_t queue );

void
iclablas_claset_band(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n, icla_int_t k,
    iclaFloatComplex offdiag, iclaFloatComplex diag,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    icla_queue_t queue );

void
iclablas_claswp(
    icla_int_t n,
    iclaFloatComplex_ptr dAT, icla_int_t ldda,
    icla_int_t k1, icla_int_t k2,
    const icla_int_t *ipiv, icla_int_t inci,
    icla_queue_t queue );

void
iclablas_claswp2(
    icla_int_t n,
    iclaFloatComplex_ptr dAT, icla_int_t ldda,
    icla_int_t k1, icla_int_t k2,
    iclaInt_const_ptr d_ipiv, icla_int_t inci,
    icla_queue_t queue );

void
iclablas_claswp_sym(
    icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    icla_int_t k1, icla_int_t k2,
    const icla_int_t *ipiv, icla_int_t inci,
    icla_queue_t queue );

void
iclablas_claswpx(
    icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldx, icla_int_t ldy,
    icla_int_t k1, icla_int_t k2,
    const icla_int_t *ipiv, icla_int_t inci,
    icla_queue_t queue );

void
icla_claswp_rowparallel_native(
    icla_int_t n,
    iclaFloatComplex* input, icla_int_t ldi,
    iclaFloatComplex* output, icla_int_t ldo,
    icla_int_t k1, icla_int_t k2,
    icla_int_t *pivinfo,
    icla_queue_t queue);

void
icla_claswp_columnserial(
    icla_int_t n, iclaFloatComplex_ptr dA, icla_int_t lda,
    icla_int_t k1, icla_int_t k2,
    icla_int_t *dipiv, icla_queue_t queue);

void
iclablas_csymmetrize(
    icla_uplo_t uplo, icla_int_t m,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    icla_queue_t queue );

void
iclablas_csymmetrize_tiles(
    icla_uplo_t uplo, icla_int_t m,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    icla_int_t ntile, icla_int_t mstride, icla_int_t nstride,
    icla_queue_t queue );

void
iclablas_ctrtri_diag(
    icla_uplo_t uplo, icla_diag_t diag, icla_int_t n,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr d_dinvA,
    icla_queue_t queue );

  /*
   * to cleanup (alphabetical order)
   */
icla_int_t
icla_clarfb_gpu(
    icla_side_t side, icla_trans_t trans, icla_direct_t direct, icla_storev_t storev,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaFloatComplex_const_ptr dV, icla_int_t lddv,
    iclaFloatComplex_const_ptr dT, icla_int_t lddt,
    iclaFloatComplex_ptr dC,       icla_int_t lddc,
    iclaFloatComplex_ptr dwork,    icla_int_t ldwork,
    icla_queue_t queue );

icla_int_t
icla_clarfb_gpu_gemm(
    icla_side_t side, icla_trans_t trans, icla_direct_t direct, icla_storev_t storev,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaFloatComplex_const_ptr dV, icla_int_t lddv,
    iclaFloatComplex_const_ptr dT, icla_int_t lddt,
    iclaFloatComplex_ptr dC,       icla_int_t lddc,
    iclaFloatComplex_ptr dwork,    icla_int_t ldwork,
    iclaFloatComplex_ptr dworkvt,  icla_int_t ldworkvt,
    icla_queue_t queue );

void
icla_clarfbx_gpu(
    icla_int_t m, icla_int_t k,
    iclaFloatComplex_ptr V,  icla_int_t ldv,
    iclaFloatComplex_ptr dT, icla_int_t ldt,
    iclaFloatComplex_ptr c,
    iclaFloatComplex_ptr dwork,
    icla_queue_t queue );

void
icla_clarfg_gpu(
    icla_int_t n,
    iclaFloatComplex_ptr dx0,
    iclaFloatComplex_ptr dx,
    iclaFloatComplex_ptr dtau,
    iclaFloat_ptr        dxnorm,
    iclaFloatComplex_ptr dAkk,
    icla_queue_t queue );

void
icla_clarfgtx_gpu(
    icla_int_t n,
    iclaFloatComplex_ptr dx0,
    iclaFloatComplex_ptr dx,
    iclaFloatComplex_ptr dtau,
    iclaFloat_ptr        dxnorm,
    iclaFloatComplex_ptr dA, icla_int_t iter,
    iclaFloatComplex_ptr V,  icla_int_t ldv,
    iclaFloatComplex_ptr T,  icla_int_t ldt,
    iclaFloatComplex_ptr dwork,
    icla_queue_t queue );

void
icla_clarfgx_gpu(
    icla_int_t n,
    iclaFloatComplex_ptr dx0,
    iclaFloatComplex_ptr dx,
    iclaFloatComplex_ptr dtau,
    iclaFloat_ptr        dxnorm,
    iclaFloatComplex_ptr dA, icla_int_t iter,
    icla_queue_t queue );

void
icla_clarfx_gpu(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex_ptr v,
    iclaFloatComplex_ptr tau,
    iclaFloatComplex_ptr C, icla_int_t ldc,
    iclaFloat_ptr        xnorm,
    iclaFloatComplex_ptr dT, icla_int_t iter,
    iclaFloatComplex_ptr work,
    icla_queue_t queue );

  /*
   * Level 1 BLAS (alphabetical order)
   */
void
iclablas_caxpycp(
    icla_int_t m,
    iclaFloatComplex_ptr dr,
    iclaFloatComplex_ptr dx,
    iclaFloatComplex_const_ptr db,
    icla_queue_t queue );

void
iclablas_cswap(
    icla_int_t n,
    iclaFloatComplex_ptr dx, icla_int_t incx,
    iclaFloatComplex_ptr dy, icla_int_t incy,
    icla_queue_t queue );

void
iclablas_cswapblk(
    icla_order_t order,
    icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr dB, icla_int_t lddb,
    icla_int_t i1, icla_int_t i2,
    const icla_int_t *ipiv, icla_int_t inci,
    icla_int_t offset,
    icla_queue_t queue );

void
iclablas_cswapdblk(
    icla_int_t n, icla_int_t nb,
    iclaFloatComplex_ptr dA, icla_int_t ldda, icla_int_t inca,
    iclaFloatComplex_ptr dB, icla_int_t lddb, icla_int_t incb,
    icla_queue_t queue );

void
iclablas_scnrm2_adjust(
    icla_int_t k,
    iclaFloat_ptr dxnorm,
    iclaFloatComplex_ptr dc,
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
iclablas_scnrm2_check(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    iclaFloat_ptr dxnorm,
    iclaFloat_ptr dlsticc,
    icla_queue_t queue );

void
iclablas_scnrm2_cols(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    iclaFloat_ptr dxnorm,
    icla_queue_t queue );

void
iclablas_scnrm2_row_check_adjust(
    icla_int_t k, float tol,
    iclaFloat_ptr dxnorm,
    iclaFloat_ptr dxnorm2,
    iclaFloatComplex_ptr dC, icla_int_t lddc,
    iclaFloat_ptr dlsticc,
    icla_queue_t queue );

  /*
   * Level 2 BLAS (alphabetical order)
   */
// trsv were always queue versions
void
iclablas_ctrsv(
    icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t n,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr       db, icla_int_t incb,
    icla_queue_t queue );

// todo: move flag before queue?
void
iclablas_ctrsv_outofplace(
    icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t n,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr db,       icla_int_t incb,
    iclaFloatComplex_ptr dx,
    icla_queue_t queue,
    icla_int_t flag );

void
iclablas_cgemv(
    icla_trans_t trans, icla_int_t m, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr dy, icla_int_t incy,
    icla_queue_t queue );

void
iclablas_cgemv_conj(
    icla_int_t m, icla_int_t n, iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr dy, icla_int_t incy,
    icla_queue_t queue );

icla_int_t
iclablas_chemv(
    icla_uplo_t uplo, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr       dy, icla_int_t incy,
    icla_queue_t queue );

icla_int_t
iclablas_csymv(
    icla_uplo_t uplo, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr       dy, icla_int_t incy,
    icla_queue_t queue );

// hemv/symv_work were always queue versions
icla_int_t
iclablas_chemv_work(
    icla_uplo_t uplo, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr       dy, icla_int_t incy,
    iclaFloatComplex_ptr       dwork, icla_int_t lwork,
    icla_queue_t queue );

icla_int_t
iclablas_csymv_work(
    icla_uplo_t uplo, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr       dy, icla_int_t incy,
    iclaFloatComplex_ptr       dwork, icla_int_t lwork,
    icla_queue_t queue );

  /*
   * Level 3 BLAS (alphabetical order)
   */
void
iclablas_cgemm(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_const_ptr dB, icla_int_t lddb,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr       dC, icla_int_t lddc,
    icla_queue_t queue );

void
iclablas_cgemm_reduce(
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_const_ptr dB, icla_int_t lddb,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr       dC, icla_int_t lddc,
    icla_queue_t queue );

void
iclablas_ctrsm(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr       dB, icla_int_t lddb,
    icla_queue_t queue );

void
iclablas_ctrsm_outofplace(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr       dB, icla_int_t lddb,
    iclaFloatComplex_ptr       dX, icla_int_t lddx,
    icla_int_t flag,
    iclaFloatComplex_ptr d_dinvA, icla_int_t dinvA_length,
    icla_queue_t queue );

void
iclablas_ctrsm_work(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr       dB, icla_int_t lddb,
    iclaFloatComplex_ptr       dX, icla_int_t lddx,
    icla_int_t flag,
    iclaFloatComplex_ptr d_dinvA, icla_int_t dinvA_length,
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

/// Type-safe version of icla_setvector() for iclaFloatComplex arrays.
/// @ingroup icla_setvector
#define icla_csetvector(           n, hx_src, incx, dy_dst, incy, queue ) \
        icla_csetvector_internal(  n, hx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of icla_getvector() for iclaFloatComplex arrays.
/// @ingroup icla_getvector
#define icla_cgetvector(           n, dx_src, incx, hy_dst, incy, queue ) \
        icla_cgetvector_internal(  n, dx_src, incx, hy_dst, incy, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of icla_copyvector() for iclaFloatComplex arrays.
/// @ingroup icla_copyvector
#define icla_ccopyvector(          n, dx_src, incx, dy_dst, incy, queue ) \
        icla_ccopyvector_internal( n, dx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of icla_setvector_async() for iclaFloatComplex arrays.
/// @ingroup icla_setvector
#define icla_csetvector_async(           n, hx_src, incx, dy_dst, incy, queue ) \
        icla_csetvector_async_internal(  n, hx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of icla_getvector_async() for iclaFloatComplex arrays.
/// @ingroup icla_getvector
#define icla_cgetvector_async(           n, dx_src, incx, hy_dst, incy, queue ) \
        icla_cgetvector_async_internal(  n, dx_src, incx, hy_dst, incy, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of icla_copyvector_async() for iclaFloatComplex arrays.
/// @ingroup icla_copyvector
#define icla_ccopyvector_async(          n, dx_src, incx, dy_dst, incy, queue ) \
        icla_ccopyvector_async_internal( n, dx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

static inline void
icla_csetvector_internal(
    icla_int_t n,
    iclaFloatComplex const    *hx_src, icla_int_t incx,
    iclaFloatComplex_ptr       dy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_setvector_internal( n, sizeof(iclaFloatComplex),
                              hx_src, incx,
                              dy_dst, incy, queue,
                              func, file, line );
}

static inline void
icla_cgetvector_internal(
    icla_int_t n,
    iclaFloatComplex_const_ptr dx_src, icla_int_t incx,
    iclaFloatComplex          *hy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_getvector_internal( n, sizeof(iclaFloatComplex),
                              dx_src, incx,
                              hy_dst, incy, queue,
                              func, file, line );
}

static inline void
icla_ccopyvector_internal(
    icla_int_t n,
    iclaFloatComplex_const_ptr dx_src, icla_int_t incx,
    iclaFloatComplex_ptr       dy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_copyvector_internal( n, sizeof(iclaFloatComplex),
                               dx_src, incx,
                               dy_dst, incy, queue,
                               func, file, line );
}

static inline void
icla_csetvector_async_internal(
    icla_int_t n,
    iclaFloatComplex const    *hx_src, icla_int_t incx,
    iclaFloatComplex_ptr       dy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_setvector_async_internal( n, sizeof(iclaFloatComplex),
                                    hx_src, incx,
                                    dy_dst, incy, queue,
                                    func, file, line );
}

static inline void
icla_cgetvector_async_internal(
    icla_int_t n,
    iclaFloatComplex_const_ptr dx_src, icla_int_t incx,
    iclaFloatComplex          *hy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_getvector_async_internal( n, sizeof(iclaFloatComplex),
                                    dx_src, incx,
                                    hy_dst, incy, queue,
                                    func, file, line );
}

static inline void
icla_ccopyvector_async_internal(
    icla_int_t n,
    iclaFloatComplex_const_ptr dx_src, icla_int_t incx,
    iclaFloatComplex_ptr       dy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_copyvector_async_internal( n, sizeof(iclaFloatComplex),
                                     dx_src, incx,
                                     dy_dst, incy, queue,
                                     func, file, line );
}


// =============================================================================
// copying sub-matrices (contiguous columns)

/// Type-safe version of icla_setmatrix() for iclaFloatComplex arrays.
/// @ingroup icla_setmatrix
#define icla_csetmatrix(           m, n, hA_src, lda,  dB_dst, lddb, queue ) \
        icla_csetmatrix_internal(  m, n, hA_src, lda,  dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of icla_getmatrix() for iclaFloatComplex arrays.
/// @ingroup icla_getmatrix
#define icla_cgetmatrix(           m, n, dA_src, ldda, hB_dst, ldb,  queue ) \
        icla_cgetmatrix_internal(  m, n, dA_src, ldda, hB_dst, ldb,  queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of icla_copymatrix() for iclaFloatComplex arrays.
/// @ingroup icla_copymatrix
#define icla_ccopymatrix(          m, n, dA_src, ldda, dB_dst, lddb, queue ) \
        icla_ccopymatrix_internal( m, n, dA_src, ldda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of icla_setmatrix_async() for iclaFloatComplex arrays.
/// @ingroup icla_setmatrix
#define icla_csetmatrix_async(           m, n, hA_src, lda, dB_dst, lddb, queue ) \
        icla_csetmatrix_async_internal(  m, n, hA_src, lda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of icla_getmatrix_async() for iclaFloatComplex arrays.
/// @ingroup icla_getmatrix
#define icla_cgetmatrix_async(           m, n, dA_src, ldda, hB_dst, ldb, queue ) \
        icla_cgetmatrix_async_internal(  m, n, dA_src, ldda, hB_dst, ldb, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of icla_copymatrix_async() for iclaFloatComplex arrays.
/// @ingroup icla_copymatrix
#define icla_ccopymatrix_async(          m, n, dA_src, ldda, dB_dst, lddb, queue ) \
        icla_ccopymatrix_async_internal( m, n, dA_src, ldda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

static inline void
icla_csetmatrix_internal(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex const    *hA_src, icla_int_t lda,
    iclaFloatComplex_ptr       dB_dst, icla_int_t lddb,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_setmatrix_internal( m, n, sizeof(iclaFloatComplex),
                              hA_src, lda,
                              dB_dst, lddb, queue,
                              func, file, line );
}

static inline void
icla_cgetmatrix_internal(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex_const_ptr dA_src, icla_int_t ldda,
    iclaFloatComplex          *hB_dst, icla_int_t ldb,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_getmatrix_internal( m, n, sizeof(iclaFloatComplex),
                              dA_src, ldda,
                              hB_dst, ldb, queue,
                              func, file, line );
}

static inline void
icla_ccopymatrix_internal(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex_const_ptr dA_src, icla_int_t ldda,
    iclaFloatComplex_ptr       dB_dst, icla_int_t lddb,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_copymatrix_internal( m, n, sizeof(iclaFloatComplex),
                               dA_src, ldda,
                               dB_dst, lddb, queue,
                               func, file, line );
}

static inline void
icla_csetmatrix_async_internal(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex const    *hA_src, icla_int_t lda,
    iclaFloatComplex_ptr       dB_dst, icla_int_t lddb,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_setmatrix_async_internal( m, n, sizeof(iclaFloatComplex),
                                    hA_src, lda,
                                    dB_dst, lddb, queue,
                                    func, file, line );
}

static inline void
icla_cgetmatrix_async_internal(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex_const_ptr dA_src, icla_int_t ldda,
    iclaFloatComplex          *hB_dst, icla_int_t ldb,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_getmatrix_async_internal( m, n, sizeof(iclaFloatComplex),
                                    dA_src, ldda,
                                    hB_dst, ldb, queue,
                                    func, file, line );
}

static inline void
icla_ccopymatrix_async_internal(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex_const_ptr dA_src, icla_int_t ldda,
    iclaFloatComplex_ptr       dB_dst, icla_int_t lddb,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_copymatrix_async_internal( m, n, sizeof(iclaFloatComplex),
                                     dA_src, ldda,
                                     dB_dst, lddb, queue,
                                     func, file, line );
}


// =============================================================================
// Level 1 BLAS (alphabetical order)

icla_int_t
icla_icamax(
    icla_int_t n,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    icla_queue_t queue );

icla_int_t
icla_icamax_native(
    icla_int_t length,
    iclaFloatComplex_ptr x, icla_int_t incx,
    icla_int_t* ipiv, icla_int_t *info,
    icla_int_t step, icla_int_t gbstep, icla_queue_t queue);

icla_int_t
icla_icamin(
    icla_int_t n,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    icla_queue_t queue );

float
icla_scasum(
    icla_int_t n,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    icla_queue_t queue );

void
icla_caxpy(
    icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex_ptr       dy, icla_int_t incy,
    icla_queue_t queue );

void
icla_ccopy(
    icla_int_t n,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex_ptr       dy, icla_int_t incy,
    icla_queue_t queue );

iclaFloatComplex
icla_cdotc(
    icla_int_t n,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex_const_ptr dy, icla_int_t incy,
    icla_queue_t queue );

iclaFloatComplex
icla_cdotu(
    icla_int_t n,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex_const_ptr dy, icla_int_t incy,
    icla_queue_t queue );

float
icla_scnrm2(
    icla_int_t n,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    icla_queue_t queue );

void
icla_crot(
    icla_int_t n,
    iclaFloatComplex_ptr dx, icla_int_t incx,
    iclaFloatComplex_ptr dy, icla_int_t incy,
    float dc, iclaFloatComplex ds,
    icla_queue_t queue );

void
icla_csrot(
    icla_int_t n,
    iclaFloatComplex_ptr dx, icla_int_t incx,
    iclaFloatComplex_ptr dy, icla_int_t incy,
    float dc, float ds,
    icla_queue_t queue );

void
icla_crotg(
    iclaFloatComplex_ptr a,
    iclaFloatComplex_ptr b,
    iclaFloat_ptr        c,
    iclaFloatComplex_ptr s,
    icla_queue_t queue );

#ifdef ICLA_REAL
void
icla_crotm(
    icla_int_t n,
    iclaFloat_ptr dx, icla_int_t incx,
    iclaFloat_ptr dy, icla_int_t incy,
    iclaFloat_const_ptr param,
    icla_queue_t queue );

void
icla_crotmg(
    iclaFloat_ptr       d1,
    iclaFloat_ptr       d2,
    iclaFloat_ptr       x1,
    iclaFloat_const_ptr y1,
    iclaFloat_ptr param,
    icla_queue_t queue );
#endif  // ICLA_REAL

void
icla_cscal(
    icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_ptr dx, icla_int_t incx,
    icla_queue_t queue );

void
icla_csscal(
    icla_int_t n,
    float alpha,
    iclaFloatComplex_ptr dx, icla_int_t incx,
    icla_queue_t queue );

icla_int_t
icla_cscal_cgeru_native(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t lda,
    icla_int_t *info, icla_int_t step, icla_int_t gbstep,
    icla_queue_t queue);

void
icla_cswap(
    icla_int_t n,
    iclaFloatComplex_ptr dx, icla_int_t incx,
    iclaFloatComplex_ptr dy, icla_int_t incy,
    icla_queue_t queue );

void
icla_cswap_native(
    icla_int_t n, iclaFloatComplex_ptr x, icla_int_t incx,
    icla_int_t step, icla_int_t* ipiv,
    icla_queue_t queue);

// =============================================================================
// Level 2 BLAS (alphabetical order)

void
icla_cgemv(
    icla_trans_t transA,
    icla_int_t m, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr       dy, icla_int_t incy,
    icla_queue_t queue );

void
icla_cgerc(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex_const_ptr dy, icla_int_t incy,
    iclaFloatComplex_ptr       dA, icla_int_t ldda,
    icla_queue_t queue );

#ifdef ICLA_COMPLEX
void
icla_cgeru(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex_const_ptr dy, icla_int_t incy,
    iclaFloatComplex_ptr       dA, icla_int_t ldda,
    icla_queue_t queue );

void
icla_chemv(
    icla_uplo_t uplo,
    icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr       dy, icla_int_t incy,
    icla_queue_t queue );

void
icla_cher(
    icla_uplo_t uplo,
    icla_int_t n,
    float alpha,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex_ptr       dA, icla_int_t ldda,
    icla_queue_t queue );

void
icla_cher2(
    icla_uplo_t uplo,
    icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex_const_ptr dy, icla_int_t incy,
    iclaFloatComplex_ptr       dA, icla_int_t ldda,
    icla_queue_t queue );
#endif // ICLA_COMPLEX

void
icla_csymv(
    icla_uplo_t uplo,
    icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr       dy, icla_int_t incy,
    icla_queue_t queue );

void
icla_csyr(
    icla_uplo_t uplo,
    icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex_ptr       dA, icla_int_t ldda,
    icla_queue_t queue );

void
icla_csyr2(
    icla_uplo_t uplo,
    icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex_const_ptr dy, icla_int_t incy,
    iclaFloatComplex_ptr       dA, icla_int_t ldda,
    icla_queue_t queue );

void
icla_ctrmv(
    icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t n,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr       dx, icla_int_t incx,
    icla_queue_t queue );

void
iclablas_ctrmv(
    icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t n,
    iclaFloatComplex *dA, icla_int_t ldda,
    iclaFloatComplex *dx, icla_int_t incx,
    icla_queue_t queue );

void
icla_ctrsv(
    icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t n,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr       dx, icla_int_t incx,
    icla_queue_t queue );

// =============================================================================
// Level 3 BLAS (alphabetical order)

void
icla_cgemm(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_const_ptr dB, icla_int_t lddb,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr       dC, icla_int_t lddc,
    icla_queue_t queue );

void
icla_chemm(
    icla_side_t side, icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_const_ptr dB, icla_int_t lddb,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr       dC, icla_int_t lddc,
    icla_queue_t queue );

void
iclablas_chemm(
    icla_side_t side, icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_const_ptr dB, icla_int_t lddb,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr       dC, icla_int_t lddc,
    icla_queue_t queue );

void
icla_cher2k(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_const_ptr dB, icla_int_t lddb,
    float beta,
    iclaFloatComplex_ptr       dC, icla_int_t lddc,
    icla_queue_t queue );

void iclablas_cher2k(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    iclaFloatComplex alpha,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr dB, icla_int_t lddb,
    float beta,
    iclaFloatComplex_ptr       dC, icla_int_t lddc,
    icla_queue_t queue );

void
icla_cherk(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    float alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    float beta,
    iclaFloatComplex_ptr       dC, icla_int_t lddc,
    icla_queue_t queue );

void iclablas_cherk(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    float alpha,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    float beta,
    iclaFloatComplex_ptr dC, icla_int_t lddc,
    icla_queue_t queue);

void iclablas_cherk_internal(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k, icla_int_t nb,
    iclaFloatComplex alpha,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr dB, icla_int_t lddb,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr dC, icla_int_t lddc,
    icla_int_t conjugate, icla_queue_t queue);

void
iclablas_cherk_small_reduce(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    float alpha, iclaFloatComplex* dA, icla_int_t ldda,
    float beta,  iclaFloatComplex* dC, icla_int_t lddc,
    icla_int_t nthread_blocks, icla_queue_t queue );

void
icla_csymm(
    icla_side_t side, icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_const_ptr dB, icla_int_t lddb,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr       dC, icla_int_t lddc,
    icla_queue_t queue );

void
iclablas_csymm(
    icla_side_t side, icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_const_ptr dB, icla_int_t lddb,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr       dC, icla_int_t lddc,
    icla_queue_t queue );

void
icla_csyr2k(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_const_ptr dB, icla_int_t lddb,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr       dC, icla_int_t lddc,
    icla_queue_t queue );

void iclablas_csyr2k(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    iclaFloatComplex alpha,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr dB, icla_int_t lddb,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr dC, icla_int_t lddc,
    icla_queue_t queue );

void
icla_csyrk(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr       dC, icla_int_t lddc,
    icla_queue_t queue );

void iclablas_csyrk(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    iclaFloatComplex alpha,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr dC, icla_int_t lddc,
    icla_queue_t queue);

void
icla_ctrmm(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr       dB, icla_int_t lddb,
    icla_queue_t queue );

void
iclablas_ctrmm(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t m, icla_int_t n,
        iclaFloatComplex alpha,
        iclaFloatComplex *dA, icla_int_t ldda,
        iclaFloatComplex *dB, icla_int_t lddb,
        icla_queue_t queue );

void
icla_ctrsm(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr       dB, icla_int_t lddb,
    icla_queue_t queue );

void
icla_cgetf2trsm_2d_native(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr dB, icla_int_t lddb,
    icla_queue_t queue);

icla_int_t
icla_cpotf2_lpout(
        icla_uplo_t uplo, icla_int_t n,
        iclaFloatComplex *dA, icla_int_t lda, icla_int_t gbstep,
        icla_int_t *dinfo, icla_queue_t queue);

icla_int_t
icla_cpotf2_lpin(
        icla_uplo_t uplo, icla_int_t n,
        iclaFloatComplex *dA, icla_int_t lda, icla_int_t gbstep,
        icla_int_t *dinfo, icla_queue_t queue);

#ifdef __cplusplus
}
#endif

#undef ICLA_COMPLEX

#endif // ICLABLAS_C_H
