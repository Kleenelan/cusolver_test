/*
    -- ICLA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from include/iclablas_z.h, normal z -> d, Fri Nov 29 12:16:14 2024
*/

#ifndef ICLABLAS_D_H
#define ICLABLAS_D_H

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
iclablas_dtranspose_inplace(
    icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    icla_queue_t queue );

void
iclablas_dtranspose_inplace(
    icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    icla_queue_t queue );

void
iclablas_dtranspose(
    icla_int_t m, icla_int_t n,
    iclaDouble_const_ptr dA,  icla_int_t ldda,
    iclaDouble_ptr       dAT, icla_int_t lddat,
    icla_queue_t queue );

void
iclablas_dtranspose(
    icla_int_t m, icla_int_t n,
    iclaDouble_const_ptr dA,  icla_int_t ldda,
    iclaDouble_ptr       dAT, icla_int_t lddat,
    icla_queue_t queue );

void
iclablas_dgetmatrix_transpose(
    icla_int_t m, icla_int_t n, icla_int_t nb,
    iclaDouble_const_ptr dAT,   icla_int_t ldda,
    double          *hA,    icla_int_t lda,
    iclaDouble_ptr       dwork, icla_int_t lddw,
    icla_queue_t queues[2] );

void
iclablas_dsetmatrix_transpose(
    icla_int_t m, icla_int_t n, icla_int_t nb,
    const double *hA,    icla_int_t lda,
    iclaDouble_ptr    dAT,   icla_int_t ldda,
    iclaDouble_ptr    dwork, icla_int_t lddw,
    icla_queue_t queues[2] );

  /*
   * RBT-related functions
   */
void
iclablas_dprbt(
    icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    iclaDouble_ptr du,
    iclaDouble_ptr dv,
    icla_queue_t queue );

void
iclablas_dprbt_mv(
    icla_int_t n, icla_int_t nrhs,
    iclaDouble_ptr dv,
    iclaDouble_ptr db, icla_int_t lddb,
    icla_queue_t queue );

void
iclablas_dprbt_mtv(
    icla_int_t n, icla_int_t nrhs,
    iclaDouble_ptr du,
    iclaDouble_ptr db, icla_int_t lddb,
    icla_queue_t queue );

  /*
   * Multi-GPU copy functions
   */
void
icla_dgetmatrix_1D_col_bcyclic(
    icla_int_t ngpu,
    icla_int_t m, icla_int_t n, icla_int_t nb,
    iclaDouble_const_ptr const dA[], icla_int_t ldda,
    double                *hA,   icla_int_t lda,
    icla_queue_t queue[] );

void
icla_dsetmatrix_1D_col_bcyclic(
    icla_int_t ngpu,
    icla_int_t m, icla_int_t n, icla_int_t nb,
    const double *hA,   icla_int_t lda,
    iclaDouble_ptr    dA[], icla_int_t ldda,
    icla_queue_t queue[] );

void
icla_dgetmatrix_1D_row_bcyclic(
    icla_int_t ngpu,
    icla_int_t m, icla_int_t n, icla_int_t nb,
    iclaDouble_const_ptr const dA[], icla_int_t ldda,
    double                *hA,   icla_int_t lda,
    icla_queue_t queue[] );

void
icla_dsetmatrix_1D_row_bcyclic(
    icla_int_t ngpu,
    icla_int_t m, icla_int_t n, icla_int_t nb,
    const double *hA,   icla_int_t lda,
    iclaDouble_ptr    dA[], icla_int_t ldda,
    icla_queue_t queue[] );

void
iclablas_dgetmatrix_transpose_mgpu(
    icla_int_t ngpu,
    icla_int_t m, icla_int_t n, icla_int_t nb,
    iclaDouble_const_ptr const dAT[],    icla_int_t ldda,
    double                *hA,       icla_int_t lda,
    iclaDouble_ptr             dwork[],  icla_int_t lddw,
    icla_queue_t queues[][2] );

void
iclablas_dsetmatrix_transpose_mgpu(
    icla_int_t ngpu,
    icla_int_t m, icla_int_t n, icla_int_t nb,
    const double *hA,      icla_int_t lda,
    iclaDouble_ptr    dAT[],   icla_int_t ldda,
    iclaDouble_ptr    dwork[], icla_int_t lddw,
    icla_queue_t queues[][2] );

// in src/dsytrd_mgpu.cpp
// TODO rename dsetmatrix_sy or similar
icla_int_t
icla_dhtodhe(
    icla_int_t ngpu, icla_uplo_t uplo, icla_int_t n, icla_int_t nb,
    double     *A,   icla_int_t lda,
    iclaDouble_ptr dA[], icla_int_t ldda,
    icla_queue_t queues[][10],
    icla_int_t *info );

// in src/dpotrf3_mgpu.cpp
// TODO same as icla_dhtodhe?
icla_int_t
icla_dhtodpo(
    icla_int_t ngpu, icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    icla_int_t off_i, icla_int_t off_j, icla_int_t nb,
    double     *A,   icla_int_t lda,
    iclaDouble_ptr dA[], icla_int_t ldda,
    icla_queue_t queues[][3],
    icla_int_t *info );

// in src/dpotrf3_mgpu.cpp
// TODO rename dgetmatrix_sy or similar
icla_int_t
icla_ddtohpo(
    icla_int_t ngpu, icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    icla_int_t off_i, icla_int_t off_j, icla_int_t nb, icla_int_t NB,
    double     *A,   icla_int_t lda,
    iclaDouble_ptr dA[], icla_int_t ldda,
    icla_queue_t queues[][3],
    icla_int_t *info );


  /*
   * Multi-GPU BLAS functions (alphabetical order)
   */
void
iclablas_dsymm_mgpu(
    icla_side_t side, icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    double alpha,
    iclaDouble_ptr dA[],    icla_int_t ldda,  icla_int_t offset,
    iclaDouble_ptr dB[],    icla_int_t lddb,
    double beta,
    iclaDouble_ptr dC[],    icla_int_t lddc,
    iclaDouble_ptr dwork[], icla_int_t dworksiz,
    //double    *C,       icla_int_t ldc,
    //double    *work[],  icla_int_t worksiz,
    icla_int_t ngpu, icla_int_t nb,
    icla_queue_t queues[][20], icla_int_t nqueue,
    icla_event_t events[][iclaMaxGPUs*iclaMaxGPUs+10], icla_int_t nevents,
    icla_int_t gnode[iclaMaxGPUs][iclaMaxGPUs+2], icla_int_t ncmplx );

icla_int_t
iclablas_dsymv_mgpu(
    icla_uplo_t uplo,
    icla_int_t n,
    double alpha,
    iclaDouble_const_ptr const d_lA[], icla_int_t ldda, icla_int_t offset,
    iclaDouble_const_ptr dx,           icla_int_t incx,
    double beta,
    iclaDouble_ptr    dy,              icla_int_t incy,
    double       *hwork,           icla_int_t lhwork,
    iclaDouble_ptr    dwork[],         icla_int_t ldwork,
    icla_int_t ngpu,
    icla_int_t nb,
    icla_queue_t queues[] );

icla_int_t
iclablas_dsymv_mgpu_sync(
    icla_uplo_t uplo,
    icla_int_t n,
    double alpha,
    iclaDouble_const_ptr const d_lA[], icla_int_t ldda, icla_int_t offset,
    iclaDouble_const_ptr dx,           icla_int_t incx,
    double beta,
    iclaDouble_ptr    dy,              icla_int_t incy,
    double       *hwork,           icla_int_t lhwork,
    iclaDouble_ptr    dwork[],         icla_int_t ldwork,
    icla_int_t ngpu,
    icla_int_t nb,
    icla_queue_t queues[] );

icla_int_t
icla_dsytrs_gpu(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nrhs,
    double *dA, icla_int_t ldda,
    icla_int_t *ipiv,
    double *dB, icla_int_t lddb,
    icla_int_t *info,
    icla_queue_t queue );

// Ichi's version, in src/dsytrd_mgpu.cpp
void
icla_dsyr2k_mgpu(
    icla_int_t ngpu,
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t nb, icla_int_t n, icla_int_t k,
    double alpha,
    iclaDouble_ptr dB[], icla_int_t lddb, icla_int_t b_offset,
    double beta,
    iclaDouble_ptr dC[], icla_int_t lddc, icla_int_t c_offset,
    icla_int_t nqueue, icla_queue_t queues[][10] );

void
iclablas_dsyr2k_mgpu2(
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t n, icla_int_t k,
    double alpha,
    iclaDouble_ptr dA[], icla_int_t ldda, icla_int_t a_offset,
    iclaDouble_ptr dB[], icla_int_t lddb, icla_int_t b_offset,
    double beta,
    iclaDouble_ptr dC[], icla_int_t lddc, icla_int_t c_offset,
    icla_int_t ngpu, icla_int_t nb,
    icla_queue_t queues[][20], icla_int_t nqueue );

// in src/dpotrf_mgpu_right.cpp
void
icla_dsyrk_mgpu(
    icla_int_t ngpu,
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t nb, icla_int_t n, icla_int_t k,
    double alpha,
    iclaDouble_ptr dB[], icla_int_t lddb, icla_int_t b_offset,
    double beta,
    iclaDouble_ptr dC[], icla_int_t lddc, icla_int_t c_offset,
    icla_int_t nqueue, icla_queue_t queues[][10] );

// in src/dpotrf_mgpu_right.cpp
void
icla_dsyrk_mgpu2(
    icla_int_t ngpu,
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t nb, icla_int_t n, icla_int_t k,
    double alpha,
    iclaDouble_ptr dB[], icla_int_t lddb, icla_int_t b_offset,
    double beta,
    iclaDouble_ptr dC[], icla_int_t lddc, icla_int_t c_offset,
    icla_int_t nqueue, icla_queue_t queues[][10] );


  /*
   * LAPACK auxiliary functions (alphabetical order)
   */
icla_int_t
iclablas_ddiinertia(
    icla_int_t n,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    int *dneig,
    icla_queue_t queue );

void
iclablas_dgeadd(
    icla_int_t m, icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_ptr       dB, icla_int_t lddb,
    icla_queue_t queue );

void
iclablas_dgeadd2(
    icla_int_t m, icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    double beta,
    iclaDouble_ptr       dB, icla_int_t lddb,
    icla_queue_t queue );

void
iclablas_dgeam(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    double beta,
    iclaDouble_const_ptr dB, icla_int_t lddb,
    iclaDouble_ptr dC, icla_int_t lddc,
    icla_queue_t queue );

icla_int_t
iclablas_dsiinertia(
    icla_uplo_t uplo,
    icla_int_t n,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    icla_int_t *ipiv,
    int *dneig,
    icla_queue_t queue );

void
iclablas_dlacpy(
    icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_ptr       dB, icla_int_t lddb,
    icla_queue_t queue );

void
iclablas_dlacpy_conj(
    icla_int_t n,
    iclaDouble_ptr dA1, icla_int_t lda1,
    iclaDouble_ptr dA2, icla_int_t lda2,
    icla_queue_t queue );

void
iclablas_dlacpy_sym_in(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    icla_int_t *rows, icla_int_t *perm,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_ptr       dB, icla_int_t lddb,
    icla_queue_t queue );

void
iclablas_dlacpy_sym_out(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    icla_int_t *rows, icla_int_t *perm,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_ptr       dB, icla_int_t lddb,
    icla_queue_t queue );

double
iclablas_dlange(
    icla_norm_t norm,
    icla_int_t m, icla_int_t n,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_ptr dwork, icla_int_t lwork,
    icla_queue_t queue );

double
iclablas_dlansy(
    icla_norm_t norm, icla_uplo_t uplo,
    icla_int_t n,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_ptr dwork, icla_int_t lwork,
    icla_queue_t queue );

void
iclablas_dlarfg(
    icla_int_t n,
    iclaDouble_ptr dalpha,
    iclaDouble_ptr dx, icla_int_t incx,
    iclaDouble_ptr dtau,
    icla_queue_t queue );

void
iclablas_dlascl(
    icla_type_t type, icla_int_t kl, icla_int_t ku,
    double cfrom, double cto,
    icla_int_t m, icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    icla_queue_t queue,
    icla_int_t *info );

void
iclablas_dlascl_2x2(
    icla_type_t type, icla_int_t m,
    iclaDouble_const_ptr dW, icla_int_t lddw,
    iclaDouble_ptr       dA, icla_int_t ldda,
    icla_queue_t queue,
    icla_int_t *info );

void
iclablas_dlascl2(
    icla_type_t type,
    icla_int_t m, icla_int_t n,
    iclaDouble_const_ptr dD,
    iclaDouble_ptr dA, icla_int_t ldda,
    icla_queue_t queue,
    icla_int_t *info );

void
iclablas_dlascl_diag(
    icla_type_t type, icla_int_t m, icla_int_t n,
    iclaDouble_const_ptr dD, icla_int_t lddd,
    iclaDouble_ptr       dA, icla_int_t ldda,
    icla_queue_t queue,
    icla_int_t *info );

void
iclablas_dlaset(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    double offdiag, double diag,
    iclaDouble_ptr dA, icla_int_t ldda,
    icla_queue_t queue );

void
iclablas_dlaset_band(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n, icla_int_t k,
    double offdiag, double diag,
    iclaDouble_ptr dA, icla_int_t ldda,
    icla_queue_t queue );

void
iclablas_dlaswp(
    icla_int_t n,
    iclaDouble_ptr dAT, icla_int_t ldda,
    icla_int_t k1, icla_int_t k2,
    const icla_int_t *ipiv, icla_int_t inci,
    icla_queue_t queue );

void
iclablas_dlaswp2(
    icla_int_t n,
    iclaDouble_ptr dAT, icla_int_t ldda,
    icla_int_t k1, icla_int_t k2,
    iclaInt_const_ptr d_ipiv, icla_int_t inci,
    icla_queue_t queue );

void
iclablas_dlaswp_sym(
    icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    icla_int_t k1, icla_int_t k2,
    const icla_int_t *ipiv, icla_int_t inci,
    icla_queue_t queue );

void
iclablas_dlaswpx(
    icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldx, icla_int_t ldy,
    icla_int_t k1, icla_int_t k2,
    const icla_int_t *ipiv, icla_int_t inci,
    icla_queue_t queue );

void
icla_dlaswp_rowparallel_native(
    icla_int_t n,
    double* input, icla_int_t ldi,
    double* output, icla_int_t ldo,
    icla_int_t k1, icla_int_t k2,
    icla_int_t *pivinfo,
    icla_queue_t queue);

void
icla_dlaswp_columnserial(
    icla_int_t n, iclaDouble_ptr dA, icla_int_t lda,
    icla_int_t k1, icla_int_t k2,
    icla_int_t *dipiv, icla_queue_t queue);

void
iclablas_dsymmetrize(
    icla_uplo_t uplo, icla_int_t m,
    iclaDouble_ptr dA, icla_int_t ldda,
    icla_queue_t queue );

void
iclablas_dsymmetrize_tiles(
    icla_uplo_t uplo, icla_int_t m,
    iclaDouble_ptr dA, icla_int_t ldda,
    icla_int_t ntile, icla_int_t mstride, icla_int_t nstride,
    icla_queue_t queue );

void
iclablas_dtrtri_diag(
    icla_uplo_t uplo, icla_diag_t diag, icla_int_t n,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_ptr d_dinvA,
    icla_queue_t queue );

  /*
   * to cleanup (alphabetical order)
   */
icla_int_t
icla_dlarfb_gpu(
    icla_side_t side, icla_trans_t trans, icla_direct_t direct, icla_storev_t storev,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDouble_const_ptr dV, icla_int_t lddv,
    iclaDouble_const_ptr dT, icla_int_t lddt,
    iclaDouble_ptr dC,       icla_int_t lddc,
    iclaDouble_ptr dwork,    icla_int_t ldwork,
    icla_queue_t queue );

icla_int_t
icla_dlarfb_gpu_gemm(
    icla_side_t side, icla_trans_t trans, icla_direct_t direct, icla_storev_t storev,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDouble_const_ptr dV, icla_int_t lddv,
    iclaDouble_const_ptr dT, icla_int_t lddt,
    iclaDouble_ptr dC,       icla_int_t lddc,
    iclaDouble_ptr dwork,    icla_int_t ldwork,
    iclaDouble_ptr dworkvt,  icla_int_t ldworkvt,
    icla_queue_t queue );

void
icla_dlarfbx_gpu(
    icla_int_t m, icla_int_t k,
    iclaDouble_ptr V,  icla_int_t ldv,
    iclaDouble_ptr dT, icla_int_t ldt,
    iclaDouble_ptr c,
    iclaDouble_ptr dwork,
    icla_queue_t queue );

void
icla_dlarfg_gpu(
    icla_int_t n,
    iclaDouble_ptr dx0,
    iclaDouble_ptr dx,
    iclaDouble_ptr dtau,
    iclaDouble_ptr        dxnorm,
    iclaDouble_ptr dAkk,
    icla_queue_t queue );

void
icla_dlarfgtx_gpu(
    icla_int_t n,
    iclaDouble_ptr dx0,
    iclaDouble_ptr dx,
    iclaDouble_ptr dtau,
    iclaDouble_ptr        dxnorm,
    iclaDouble_ptr dA, icla_int_t iter,
    iclaDouble_ptr V,  icla_int_t ldv,
    iclaDouble_ptr T,  icla_int_t ldt,
    iclaDouble_ptr dwork,
    icla_queue_t queue );

void
icla_dlarfgx_gpu(
    icla_int_t n,
    iclaDouble_ptr dx0,
    iclaDouble_ptr dx,
    iclaDouble_ptr dtau,
    iclaDouble_ptr        dxnorm,
    iclaDouble_ptr dA, icla_int_t iter,
    icla_queue_t queue );

void
icla_dlarfx_gpu(
    icla_int_t m, icla_int_t n,
    iclaDouble_ptr v,
    iclaDouble_ptr tau,
    iclaDouble_ptr C, icla_int_t ldc,
    iclaDouble_ptr        xnorm,
    iclaDouble_ptr dT, icla_int_t iter,
    iclaDouble_ptr work,
    icla_queue_t queue );

  /*
   * Level 1 BLAS (alphabetical order)
   */
void
iclablas_daxpycp(
    icla_int_t m,
    iclaDouble_ptr dr,
    iclaDouble_ptr dx,
    iclaDouble_const_ptr db,
    icla_queue_t queue );

void
iclablas_dswap(
    icla_int_t n,
    iclaDouble_ptr dx, icla_int_t incx,
    iclaDouble_ptr dy, icla_int_t incy,
    icla_queue_t queue );

void
iclablas_dswapblk(
    icla_order_t order,
    icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    iclaDouble_ptr dB, icla_int_t lddb,
    icla_int_t i1, icla_int_t i2,
    const icla_int_t *ipiv, icla_int_t inci,
    icla_int_t offset,
    icla_queue_t queue );

void
iclablas_dswapdblk(
    icla_int_t n, icla_int_t nb,
    iclaDouble_ptr dA, icla_int_t ldda, icla_int_t inca,
    iclaDouble_ptr dB, icla_int_t lddb, icla_int_t incb,
    icla_queue_t queue );

void
iclablas_dnrm2_adjust(
    icla_int_t k,
    iclaDouble_ptr dxnorm,
    iclaDouble_ptr dc,
    icla_queue_t queue );

#ifdef REAL
void
iclablas_dnrm2_check(
    icla_int_t m, icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    iclaDouble_ptr dxnorm,
    iclaDouble_ptr dlsticc,
    icla_queue_t queue );
#endif

void
iclablas_dnrm2_check(
    icla_int_t m, icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    iclaDouble_ptr dxnorm,
    iclaDouble_ptr dlsticc,
    icla_queue_t queue );

void
iclablas_dnrm2_cols(
    icla_int_t m, icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    iclaDouble_ptr dxnorm,
    icla_queue_t queue );

void
iclablas_dnrm2_row_check_adjust(
    icla_int_t k, double tol,
    iclaDouble_ptr dxnorm,
    iclaDouble_ptr dxnorm2,
    iclaDouble_ptr dC, icla_int_t lddc,
    iclaDouble_ptr dlsticc,
    icla_queue_t queue );

  /*
   * Level 2 BLAS (alphabetical order)
   */
// trsv were always queue versions
void
iclablas_dtrsv(
    icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t n,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_ptr       db, icla_int_t incb,
    icla_queue_t queue );

// todo: move flag before queue?
void
iclablas_dtrsv_outofplace(
    icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t n,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_ptr db,       icla_int_t incb,
    iclaDouble_ptr dx,
    icla_queue_t queue,
    icla_int_t flag );

void
iclablas_dgemv(
    icla_trans_t trans, icla_int_t m, icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_const_ptr dx, icla_int_t incx,
    double beta,
    iclaDouble_ptr dy, icla_int_t incy,
    icla_queue_t queue );

void
iclablas_dgemv_conj(
    icla_int_t m, icla_int_t n, double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_const_ptr dx, icla_int_t incx,
    double beta,
    iclaDouble_ptr dy, icla_int_t incy,
    icla_queue_t queue );

icla_int_t
iclablas_dsymv(
    icla_uplo_t uplo, icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_const_ptr dx, icla_int_t incx,
    double beta,
    iclaDouble_ptr       dy, icla_int_t incy,
    icla_queue_t queue );

icla_int_t
iclablas_dsymv(
    icla_uplo_t uplo, icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_const_ptr dx, icla_int_t incx,
    double beta,
    iclaDouble_ptr       dy, icla_int_t incy,
    icla_queue_t queue );

// hemv/symv_work were always queue versions
icla_int_t
iclablas_dsymv_work(
    icla_uplo_t uplo, icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_const_ptr dx, icla_int_t incx,
    double beta,
    iclaDouble_ptr       dy, icla_int_t incy,
    iclaDouble_ptr       dwork, icla_int_t lwork,
    icla_queue_t queue );

icla_int_t
iclablas_dsymv_work(
    icla_uplo_t uplo, icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_const_ptr dx, icla_int_t incx,
    double beta,
    iclaDouble_ptr       dy, icla_int_t incy,
    iclaDouble_ptr       dwork, icla_int_t lwork,
    icla_queue_t queue );

  /*
   * Level 3 BLAS (alphabetical order)
   */
void
iclablas_dgemm(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_const_ptr dB, icla_int_t lddb,
    double beta,
    iclaDouble_ptr       dC, icla_int_t lddc,
    icla_queue_t queue );

void
iclablas_dgemm_reduce(
    icla_int_t m, icla_int_t n, icla_int_t k,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_const_ptr dB, icla_int_t lddb,
    double beta,
    iclaDouble_ptr       dC, icla_int_t lddc,
    icla_queue_t queue );

void
iclablas_dtrsm(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_ptr       dB, icla_int_t lddb,
    icla_queue_t queue );

void
iclablas_dtrsm_outofplace(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_ptr       dB, icla_int_t lddb,
    iclaDouble_ptr       dX, icla_int_t lddx,
    icla_int_t flag,
    iclaDouble_ptr d_dinvA, icla_int_t dinvA_length,
    icla_queue_t queue );

void
iclablas_dtrsm_work(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_ptr       dB, icla_int_t lddb,
    iclaDouble_ptr       dX, icla_int_t lddx,
    icla_int_t flag,
    iclaDouble_ptr d_dinvA, icla_int_t dinvA_length,
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

/// Type-safe version of icla_setvector() for double arrays.
/// @ingroup icla_setvector
#define icla_dsetvector(           n, hx_src, incx, dy_dst, incy, queue ) \
        icla_dsetvector_internal(  n, hx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of icla_getvector() for double arrays.
/// @ingroup icla_getvector
#define icla_dgetvector(           n, dx_src, incx, hy_dst, incy, queue ) \
        icla_dgetvector_internal(  n, dx_src, incx, hy_dst, incy, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of icla_copyvector() for double arrays.
/// @ingroup icla_copyvector
#define icla_dcopyvector(          n, dx_src, incx, dy_dst, incy, queue ) \
        icla_dcopyvector_internal( n, dx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of icla_setvector_async() for double arrays.
/// @ingroup icla_setvector
#define icla_dsetvector_async(           n, hx_src, incx, dy_dst, incy, queue ) \
        icla_dsetvector_async_internal(  n, hx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of icla_getvector_async() for double arrays.
/// @ingroup icla_getvector
#define icla_dgetvector_async(           n, dx_src, incx, hy_dst, incy, queue ) \
        icla_dgetvector_async_internal(  n, dx_src, incx, hy_dst, incy, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of icla_copyvector_async() for double arrays.
/// @ingroup icla_copyvector
#define icla_dcopyvector_async(          n, dx_src, incx, dy_dst, incy, queue ) \
        icla_dcopyvector_async_internal( n, dx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

static inline void
icla_dsetvector_internal(
    icla_int_t n,
    double const    *hx_src, icla_int_t incx,
    iclaDouble_ptr       dy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_setvector_internal( n, sizeof(double),
                              hx_src, incx,
                              dy_dst, incy, queue,
                              func, file, line );
}

static inline void
icla_dgetvector_internal(
    icla_int_t n,
    iclaDouble_const_ptr dx_src, icla_int_t incx,
    double          *hy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_getvector_internal( n, sizeof(double),
                              dx_src, incx,
                              hy_dst, incy, queue,
                              func, file, line );
}

static inline void
icla_dcopyvector_internal(
    icla_int_t n,
    iclaDouble_const_ptr dx_src, icla_int_t incx,
    iclaDouble_ptr       dy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_copyvector_internal( n, sizeof(double),
                               dx_src, incx,
                               dy_dst, incy, queue,
                               func, file, line );
}

static inline void
icla_dsetvector_async_internal(
    icla_int_t n,
    double const    *hx_src, icla_int_t incx,
    iclaDouble_ptr       dy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_setvector_async_internal( n, sizeof(double),
                                    hx_src, incx,
                                    dy_dst, incy, queue,
                                    func, file, line );
}

static inline void
icla_dgetvector_async_internal(
    icla_int_t n,
    iclaDouble_const_ptr dx_src, icla_int_t incx,
    double          *hy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_getvector_async_internal( n, sizeof(double),
                                    dx_src, incx,
                                    hy_dst, incy, queue,
                                    func, file, line );
}

static inline void
icla_dcopyvector_async_internal(
    icla_int_t n,
    iclaDouble_const_ptr dx_src, icla_int_t incx,
    iclaDouble_ptr       dy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_copyvector_async_internal( n, sizeof(double),
                                     dx_src, incx,
                                     dy_dst, incy, queue,
                                     func, file, line );
}


// =============================================================================
// copying sub-matrices (contiguous columns)

/// Type-safe version of icla_setmatrix() for double arrays.
/// @ingroup icla_setmatrix
#define icla_dsetmatrix(           m, n, hA_src, lda,  dB_dst, lddb, queue ) \
        icla_dsetmatrix_internal(  m, n, hA_src, lda,  dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of icla_getmatrix() for double arrays.
/// @ingroup icla_getmatrix
#define icla_dgetmatrix(           m, n, dA_src, ldda, hB_dst, ldb,  queue ) \
        icla_dgetmatrix_internal(  m, n, dA_src, ldda, hB_dst, ldb,  queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of icla_copymatrix() for double arrays.
/// @ingroup icla_copymatrix
#define icla_dcopymatrix(          m, n, dA_src, ldda, dB_dst, lddb, queue ) \
        icla_dcopymatrix_internal( m, n, dA_src, ldda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of icla_setmatrix_async() for double arrays.
/// @ingroup icla_setmatrix
#define icla_dsetmatrix_async(           m, n, hA_src, lda, dB_dst, lddb, queue ) \
        icla_dsetmatrix_async_internal(  m, n, hA_src, lda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of icla_getmatrix_async() for double arrays.
/// @ingroup icla_getmatrix
#define icla_dgetmatrix_async(           m, n, dA_src, ldda, hB_dst, ldb, queue ) \
        icla_dgetmatrix_async_internal(  m, n, dA_src, ldda, hB_dst, ldb, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of icla_copymatrix_async() for double arrays.
/// @ingroup icla_copymatrix
#define icla_dcopymatrix_async(          m, n, dA_src, ldda, dB_dst, lddb, queue ) \
        icla_dcopymatrix_async_internal( m, n, dA_src, ldda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

static inline void
icla_dsetmatrix_internal(
    icla_int_t m, icla_int_t n,
    double const    *hA_src, icla_int_t lda,
    iclaDouble_ptr       dB_dst, icla_int_t lddb,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_setmatrix_internal( m, n, sizeof(double),
                              hA_src, lda,
                              dB_dst, lddb, queue,
                              func, file, line );
}

static inline void
icla_dgetmatrix_internal(
    icla_int_t m, icla_int_t n,
    iclaDouble_const_ptr dA_src, icla_int_t ldda,
    double          *hB_dst, icla_int_t ldb,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_getmatrix_internal( m, n, sizeof(double),
                              dA_src, ldda,
                              hB_dst, ldb, queue,
                              func, file, line );
}

static inline void
icla_dcopymatrix_internal(
    icla_int_t m, icla_int_t n,
    iclaDouble_const_ptr dA_src, icla_int_t ldda,
    iclaDouble_ptr       dB_dst, icla_int_t lddb,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_copymatrix_internal( m, n, sizeof(double),
                               dA_src, ldda,
                               dB_dst, lddb, queue,
                               func, file, line );
}

static inline void
icla_dsetmatrix_async_internal(
    icla_int_t m, icla_int_t n,
    double const    *hA_src, icla_int_t lda,
    iclaDouble_ptr       dB_dst, icla_int_t lddb,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_setmatrix_async_internal( m, n, sizeof(double),
                                    hA_src, lda,
                                    dB_dst, lddb, queue,
                                    func, file, line );
}

static inline void
icla_dgetmatrix_async_internal(
    icla_int_t m, icla_int_t n,
    iclaDouble_const_ptr dA_src, icla_int_t ldda,
    double          *hB_dst, icla_int_t ldb,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_getmatrix_async_internal( m, n, sizeof(double),
                                    dA_src, ldda,
                                    hB_dst, ldb, queue,
                                    func, file, line );
}

static inline void
icla_dcopymatrix_async_internal(
    icla_int_t m, icla_int_t n,
    iclaDouble_const_ptr dA_src, icla_int_t ldda,
    iclaDouble_ptr       dB_dst, icla_int_t lddb,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_copymatrix_async_internal( m, n, sizeof(double),
                                     dA_src, ldda,
                                     dB_dst, lddb, queue,
                                     func, file, line );
}


// =============================================================================
// Level 1 BLAS (alphabetical order)

icla_int_t
icla_idamax(
    icla_int_t n,
    iclaDouble_const_ptr dx, icla_int_t incx,
    icla_queue_t queue );

icla_int_t
icla_idamax_native(
    icla_int_t length,
    iclaDouble_ptr x, icla_int_t incx,
    icla_int_t* ipiv, icla_int_t *info,
    icla_int_t step, icla_int_t gbstep, icla_queue_t queue);

icla_int_t
icla_idamin(
    icla_int_t n,
    iclaDouble_const_ptr dx, icla_int_t incx,
    icla_queue_t queue );

double
icla_dasum(
    icla_int_t n,
    iclaDouble_const_ptr dx, icla_int_t incx,
    icla_queue_t queue );

void
icla_daxpy(
    icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dx, icla_int_t incx,
    iclaDouble_ptr       dy, icla_int_t incy,
    icla_queue_t queue );

void
icla_dcopy(
    icla_int_t n,
    iclaDouble_const_ptr dx, icla_int_t incx,
    iclaDouble_ptr       dy, icla_int_t incy,
    icla_queue_t queue );

double
icla_ddot(
    icla_int_t n,
    iclaDouble_const_ptr dx, icla_int_t incx,
    iclaDouble_const_ptr dy, icla_int_t incy,
    icla_queue_t queue );

double
icla_ddot(
    icla_int_t n,
    iclaDouble_const_ptr dx, icla_int_t incx,
    iclaDouble_const_ptr dy, icla_int_t incy,
    icla_queue_t queue );

double
icla_dnrm2(
    icla_int_t n,
    iclaDouble_const_ptr dx, icla_int_t incx,
    icla_queue_t queue );

void
icla_drot(
    icla_int_t n,
    iclaDouble_ptr dx, icla_int_t incx,
    iclaDouble_ptr dy, icla_int_t incy,
    double dc, double ds,
    icla_queue_t queue );

void
icla_drot(
    icla_int_t n,
    iclaDouble_ptr dx, icla_int_t incx,
    iclaDouble_ptr dy, icla_int_t incy,
    double dc, double ds,
    icla_queue_t queue );

void
icla_drotg(
    iclaDouble_ptr a,
    iclaDouble_ptr b,
    iclaDouble_ptr        c,
    iclaDouble_ptr s,
    icla_queue_t queue );

#ifdef ICLA_REAL
void
icla_drotm(
    icla_int_t n,
    iclaDouble_ptr dx, icla_int_t incx,
    iclaDouble_ptr dy, icla_int_t incy,
    iclaDouble_const_ptr param,
    icla_queue_t queue );

void
icla_drotmg(
    iclaDouble_ptr       d1,
    iclaDouble_ptr       d2,
    iclaDouble_ptr       x1,
    iclaDouble_const_ptr y1,
    iclaDouble_ptr param,
    icla_queue_t queue );
#endif  // ICLA_REAL

void
icla_dscal(
    icla_int_t n,
    double alpha,
    iclaDouble_ptr dx, icla_int_t incx,
    icla_queue_t queue );

void
icla_dscal(
    icla_int_t n,
    double alpha,
    iclaDouble_ptr dx, icla_int_t incx,
    icla_queue_t queue );

icla_int_t
icla_dscal_dger_native(
    icla_int_t m, icla_int_t n,
    iclaDouble_ptr dA, icla_int_t lda,
    icla_int_t *info, icla_int_t step, icla_int_t gbstep,
    icla_queue_t queue);

void
icla_dswap(
    icla_int_t n,
    iclaDouble_ptr dx, icla_int_t incx,
    iclaDouble_ptr dy, icla_int_t incy,
    icla_queue_t queue );

void
icla_dswap_native(
    icla_int_t n, iclaDouble_ptr x, icla_int_t incx,
    icla_int_t step, icla_int_t* ipiv,
    icla_queue_t queue);

// =============================================================================
// Level 2 BLAS (alphabetical order)

void
icla_dgemv(
    icla_trans_t transA,
    icla_int_t m, icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_const_ptr dx, icla_int_t incx,
    double beta,
    iclaDouble_ptr       dy, icla_int_t incy,
    icla_queue_t queue );

void
icla_dger(
    icla_int_t m, icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dx, icla_int_t incx,
    iclaDouble_const_ptr dy, icla_int_t incy,
    iclaDouble_ptr       dA, icla_int_t ldda,
    icla_queue_t queue );

#ifdef ICLA_COMPLEX
void
icla_dger(
    icla_int_t m, icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dx, icla_int_t incx,
    iclaDouble_const_ptr dy, icla_int_t incy,
    iclaDouble_ptr       dA, icla_int_t ldda,
    icla_queue_t queue );

void
icla_dsymv(
    icla_uplo_t uplo,
    icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_const_ptr dx, icla_int_t incx,
    double beta,
    iclaDouble_ptr       dy, icla_int_t incy,
    icla_queue_t queue );

void
icla_dsyr(
    icla_uplo_t uplo,
    icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dx, icla_int_t incx,
    iclaDouble_ptr       dA, icla_int_t ldda,
    icla_queue_t queue );

void
icla_dsyr2(
    icla_uplo_t uplo,
    icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dx, icla_int_t incx,
    iclaDouble_const_ptr dy, icla_int_t incy,
    iclaDouble_ptr       dA, icla_int_t ldda,
    icla_queue_t queue );
#endif // ICLA_COMPLEX

void
icla_dsymv(
    icla_uplo_t uplo,
    icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_const_ptr dx, icla_int_t incx,
    double beta,
    iclaDouble_ptr       dy, icla_int_t incy,
    icla_queue_t queue );

void
icla_dsyr(
    icla_uplo_t uplo,
    icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dx, icla_int_t incx,
    iclaDouble_ptr       dA, icla_int_t ldda,
    icla_queue_t queue );

void
icla_dsyr2(
    icla_uplo_t uplo,
    icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dx, icla_int_t incx,
    iclaDouble_const_ptr dy, icla_int_t incy,
    iclaDouble_ptr       dA, icla_int_t ldda,
    icla_queue_t queue );

void
icla_dtrmv(
    icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t n,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_ptr       dx, icla_int_t incx,
    icla_queue_t queue );

void
iclablas_dtrmv(
    icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t n,
    double *dA, icla_int_t ldda,
    double *dx, icla_int_t incx,
    icla_queue_t queue );

void
icla_dtrsv(
    icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t n,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_ptr       dx, icla_int_t incx,
    icla_queue_t queue );

// =============================================================================
// Level 3 BLAS (alphabetical order)

void
icla_dgemm(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_const_ptr dB, icla_int_t lddb,
    double beta,
    iclaDouble_ptr       dC, icla_int_t lddc,
    icla_queue_t queue );

void
icla_dsymm(
    icla_side_t side, icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_const_ptr dB, icla_int_t lddb,
    double beta,
    iclaDouble_ptr       dC, icla_int_t lddc,
    icla_queue_t queue );

void
iclablas_dsymm(
    icla_side_t side, icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_const_ptr dB, icla_int_t lddb,
    double beta,
    iclaDouble_ptr       dC, icla_int_t lddc,
    icla_queue_t queue );

void
icla_dsyr2k(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_const_ptr dB, icla_int_t lddb,
    double beta,
    iclaDouble_ptr       dC, icla_int_t lddc,
    icla_queue_t queue );

void iclablas_dsyr2k(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    double alpha,
    iclaDouble_ptr dA, icla_int_t ldda,
    iclaDouble_ptr dB, icla_int_t lddb,
    double beta,
    iclaDouble_ptr       dC, icla_int_t lddc,
    icla_queue_t queue );

void
icla_dsyrk(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    double beta,
    iclaDouble_ptr       dC, icla_int_t lddc,
    icla_queue_t queue );

void iclablas_dsyrk(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    double alpha,
    iclaDouble_ptr dA, icla_int_t ldda,
    double beta,
    iclaDouble_ptr dC, icla_int_t lddc,
    icla_queue_t queue);

void iclablas_dsyrk_internal(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k, icla_int_t nb,
    double alpha,
    iclaDouble_ptr dA, icla_int_t ldda,
    iclaDouble_ptr dB, icla_int_t lddb,
    double beta,
    iclaDouble_ptr dC, icla_int_t lddc,
    icla_int_t conjugate, icla_queue_t queue);

void
iclablas_dsyrk_small_reduce(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    double alpha, double* dA, icla_int_t ldda,
    double beta,  double* dC, icla_int_t lddc,
    icla_int_t nthread_blocks, icla_queue_t queue );

void
icla_dsymm(
    icla_side_t side, icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_const_ptr dB, icla_int_t lddb,
    double beta,
    iclaDouble_ptr       dC, icla_int_t lddc,
    icla_queue_t queue );

void
iclablas_dsymm(
    icla_side_t side, icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_const_ptr dB, icla_int_t lddb,
    double beta,
    iclaDouble_ptr       dC, icla_int_t lddc,
    icla_queue_t queue );

void
icla_dsyr2k(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_const_ptr dB, icla_int_t lddb,
    double beta,
    iclaDouble_ptr       dC, icla_int_t lddc,
    icla_queue_t queue );

void iclablas_dsyr2k(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    double alpha,
    iclaDouble_ptr dA, icla_int_t ldda,
    iclaDouble_ptr dB, icla_int_t lddb,
    double beta,
    iclaDouble_ptr dC, icla_int_t lddc,
    icla_queue_t queue );

void
icla_dsyrk(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    double beta,
    iclaDouble_ptr       dC, icla_int_t lddc,
    icla_queue_t queue );

void iclablas_dsyrk(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    double alpha,
    iclaDouble_ptr dA, icla_int_t ldda,
    double beta,
    iclaDouble_ptr dC, icla_int_t lddc,
    icla_queue_t queue);

void
icla_dtrmm(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_ptr       dB, icla_int_t lddb,
    icla_queue_t queue );

void
iclablas_dtrmm(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t m, icla_int_t n,
        double alpha,
        double *dA, icla_int_t ldda,
        double *dB, icla_int_t lddb,
        icla_queue_t queue );

void
icla_dtrsm(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_ptr       dB, icla_int_t lddb,
    icla_queue_t queue );

void
icla_dgetf2trsm_2d_native(
    icla_int_t m, icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    iclaDouble_ptr dB, icla_int_t lddb,
    icla_queue_t queue);

icla_int_t
icla_dpotf2_lpout(
        icla_uplo_t uplo, icla_int_t n,
        double *dA, icla_int_t lda, icla_int_t gbstep,
        icla_int_t *dinfo, icla_queue_t queue);

icla_int_t
icla_dpotf2_lpin(
        icla_uplo_t uplo, icla_int_t n,
        double *dA, icla_int_t lda, icla_int_t gbstep,
        icla_int_t *dinfo, icla_queue_t queue);

#ifdef __cplusplus
}
#endif

#undef ICLA_REAL

#endif // ICLABLAS_D_H
