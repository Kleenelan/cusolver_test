/*
    -- ICLA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from include/iclablas_z_v1.h, normal z -> d, Fri Nov 29 12:16:14 2024
*/

#ifndef ICLABLAS_D_V1_H
#define ICLABLAS_D_V1_H

#ifdef ICLA_NO_V1
#error "Since ICLA_NO_V1 is defined, icla.h is invalid; use icla_v2.h"
#endif

#include "icla_types.h"
#include "icla_copy_v1.h"

#define ICLA_REAL

#ifdef __cplusplus
extern "C" {
#endif

  /*
   * Transpose functions
   */
void
iclablas_dtranspose_inplace_v1(
    icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda );

void
iclablas_dtranspose_inplace_v1(
    icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda );

void
iclablas_dtranspose_v1(
    icla_int_t m, icla_int_t n,
    iclaDouble_const_ptr dA,  icla_int_t ldda,
    iclaDouble_ptr       dAT, icla_int_t lddat );

void
iclablas_dtranspose_v1(
    icla_int_t m, icla_int_t n,
    iclaDouble_const_ptr dA,  icla_int_t ldda,
    iclaDouble_ptr       dAT, icla_int_t lddat );

void
iclablas_dgetmatrix_transpose_v1(
    icla_int_t m, icla_int_t n,
    iclaDouble_const_ptr dAT,   icla_int_t ldda,
    double          *hA,    icla_int_t lda,
    iclaDouble_ptr       dwork, icla_int_t lddwork, icla_int_t nb );

void
iclablas_dsetmatrix_transpose_v1(
    icla_int_t m, icla_int_t n,
    const double *hA,    icla_int_t lda,
    iclaDouble_ptr    dAT,   icla_int_t ldda,
    iclaDouble_ptr    dwork, icla_int_t lddwork, icla_int_t nb );

  /*
   * RBT-related functions
   */
void
iclablas_dprbt_v1(
    icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    iclaDouble_ptr du,
    iclaDouble_ptr dv );

void
iclablas_dprbt_mv_v1(
    icla_int_t n, icla_int_t nrhs,
    iclaDouble_ptr dv,
    iclaDouble_ptr db, icla_int_t lddb);

void
iclablas_dprbt_mtv_v1(
    icla_int_t n, icla_int_t nrhs,
    iclaDouble_ptr du,
    iclaDouble_ptr db, icla_int_t lddb);

  /*
   * Multi-GPU copy functions
   */
void
icla_dgetmatrix_1D_col_bcyclic_v1(
    icla_int_t m, icla_int_t n,
    iclaDouble_const_ptr const dA[], icla_int_t ldda,
    double                *hA,   icla_int_t lda,
    icla_int_t ngpu, icla_int_t nb );

void
icla_dsetmatrix_1D_col_bcyclic_v1(
    icla_int_t m, icla_int_t n,
    const double *hA,   icla_int_t lda,
    iclaDouble_ptr    dA[], icla_int_t ldda,
    icla_int_t ngpu, icla_int_t nb );

void
icla_dgetmatrix_1D_row_bcyclic_v1(
    icla_int_t m, icla_int_t n,
    iclaDouble_const_ptr const dA[], icla_int_t ldda,
    double                *hA,   icla_int_t lda,
    icla_int_t ngpu, icla_int_t nb );

void
icla_dsetmatrix_1D_row_bcyclic_v1(
    icla_int_t m, icla_int_t n,
    const double *hA,   icla_int_t lda,
    iclaDouble_ptr    dA[], icla_int_t ldda,
    icla_int_t ngpu, icla_int_t nb );


  /*
   * LAPACK auxiliary functions (alphabetical order)
   */
void
iclablas_dgeadd_v1(
    icla_int_t m, icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_ptr       dB, icla_int_t lddb );

void
iclablas_dgeadd2_v1(
    icla_int_t m, icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    double beta,
    iclaDouble_ptr       dB, icla_int_t lddb );

void
iclablas_dlacpy_v1(
    icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_ptr       dB, icla_int_t lddb );

void
iclablas_dlacpy_conj_v1(
    icla_int_t n,
    iclaDouble_ptr dA1, icla_int_t lda1,
    iclaDouble_ptr dA2, icla_int_t lda2 );

void
iclablas_dlacpy_sym_in_v1(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    icla_int_t *rows, icla_int_t *perm,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_ptr       dB, icla_int_t lddb );

void
iclablas_dlacpy_sym_out_v1(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    icla_int_t *rows, icla_int_t *perm,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_ptr       dB, icla_int_t lddb );

double
iclablas_dlange_v1(
    icla_norm_t norm,
    icla_int_t m, icla_int_t n,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_ptr dwork, icla_int_t lwork );

double
iclablas_dlansy_v1(
    icla_norm_t norm, icla_uplo_t uplo,
    icla_int_t n,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_ptr dwork, icla_int_t lwork );

double
iclablas_dlansy_v1(
    icla_norm_t norm, icla_uplo_t uplo,
    icla_int_t n,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_ptr dwork, icla_int_t lwork );

void
iclablas_dlarfg_v1(
    icla_int_t n,
    iclaDouble_ptr dalpha,
    iclaDouble_ptr dx, icla_int_t incx,
    iclaDouble_ptr dtau );

void
iclablas_dlascl_v1(
    icla_type_t type, icla_int_t kl, icla_int_t ku,
    double cfrom, double cto,
    icla_int_t m, icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    icla_int_t *info );

void
iclablas_dlascl_2x2_v1(
    icla_type_t type, icla_int_t m,
    iclaDouble_const_ptr dW, icla_int_t lddw,
    iclaDouble_ptr       dA, icla_int_t ldda,
    icla_int_t *info );

void
iclablas_dlascl2_v1(
    icla_type_t type,
    icla_int_t m, icla_int_t n,
    iclaDouble_const_ptr dD,
    iclaDouble_ptr dA, icla_int_t ldda,
    icla_int_t *info );

void
iclablas_dlascl_diag_v1(
    icla_type_t type, icla_int_t m, icla_int_t n,
    iclaDouble_const_ptr dD, icla_int_t lddd,
    iclaDouble_ptr       dA, icla_int_t ldda,
    icla_int_t *info );

void
iclablas_dlaset_v1(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    double offdiag, double diag,
    iclaDouble_ptr dA, icla_int_t ldda );

void
iclablas_dlaset_band_v1(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n, icla_int_t k,
    double offdiag, double diag,
    iclaDouble_ptr dA, icla_int_t ldda );

void
iclablas_dlaswp_v1(
    icla_int_t n,
    iclaDouble_ptr dAT, icla_int_t ldda,
    icla_int_t k1, icla_int_t k2,
    const icla_int_t *ipiv, icla_int_t inci );

void
iclablas_dlaswp2_v1(
    icla_int_t n,
    iclaDouble_ptr dAT, icla_int_t ldda,
    icla_int_t k1, icla_int_t k2,
    iclaInt_const_ptr d_ipiv, icla_int_t inci );

void
iclablas_dlaswp_sym_v1(
    icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    icla_int_t k1, icla_int_t k2,
    const icla_int_t *ipiv, icla_int_t inci );

void
iclablas_dlaswpx_v1(
    icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldx, icla_int_t ldy,
    icla_int_t k1, icla_int_t k2,
    const icla_int_t *ipiv, icla_int_t inci );

void
iclablas_dsymmetrize_v1(
    icla_uplo_t uplo, icla_int_t m,
    iclaDouble_ptr dA, icla_int_t ldda );

void
iclablas_dsymmetrize_tiles_v1(
    icla_uplo_t uplo, icla_int_t m,
    iclaDouble_ptr dA, icla_int_t ldda,
    icla_int_t ntile, icla_int_t mstride, icla_int_t nstride );

void
iclablas_dtrtri_diag_v1(
    icla_uplo_t uplo, icla_diag_t diag, icla_int_t n,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_ptr d_dinvA );

  /*
   * to cleanup (alphabetical order)
   */
void
iclablas_dnrm2_adjust_v1(
    icla_int_t k,
    iclaDouble_ptr dxnorm,
    iclaDouble_ptr dc );

void
iclablas_dnrm2_check_v1(
    icla_int_t m, icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    iclaDouble_ptr dxnorm,
    iclaDouble_ptr dlsticc );

void
iclablas_dnrm2_cols_v1(
    icla_int_t m, icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    iclaDouble_ptr dxnorm );

void
iclablas_dnrm2_row_check_adjust_v1(
    icla_int_t k, double tol,
    iclaDouble_ptr dxnorm,
    iclaDouble_ptr dxnorm2,
    iclaDouble_ptr dC, icla_int_t lddc,
    iclaDouble_ptr dlsticc );

icla_int_t
icla_dlarfb_gpu_v1(
    icla_side_t side, icla_trans_t trans, icla_direct_t direct, icla_storev_t storev,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDouble_const_ptr dV, icla_int_t lddv,
    iclaDouble_const_ptr dT, icla_int_t lddt,
    iclaDouble_ptr dC,       icla_int_t lddc,
    iclaDouble_ptr dwork,    icla_int_t ldwork );

icla_int_t
icla_dlarfb_gpu_gemm_v1(
    icla_side_t side, icla_trans_t trans, icla_direct_t direct, icla_storev_t storev,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDouble_const_ptr dV, icla_int_t lddv,
    iclaDouble_const_ptr dT, icla_int_t lddt,
    iclaDouble_ptr dC,       icla_int_t lddc,
    iclaDouble_ptr dwork,    icla_int_t ldwork,
    iclaDouble_ptr dworkvt,  icla_int_t ldworkvt );

void
icla_dlarfbx_gpu_v1(
    icla_int_t m, icla_int_t k,
    iclaDouble_ptr V,  icla_int_t ldv,
    iclaDouble_ptr dT, icla_int_t ldt,
    iclaDouble_ptr c,
    iclaDouble_ptr dwork );

void
icla_dlarfg_gpu_v1(
    icla_int_t n,
    iclaDouble_ptr dx0,
    iclaDouble_ptr dx,
    iclaDouble_ptr dtau,
    iclaDouble_ptr        dxnorm,
    iclaDouble_ptr dAkk );

void
icla_dlarfgtx_gpu_v1(
    icla_int_t n,
    iclaDouble_ptr dx0,
    iclaDouble_ptr dx,
    iclaDouble_ptr dtau,
    iclaDouble_ptr        dxnorm,
    iclaDouble_ptr dA, icla_int_t iter,
    iclaDouble_ptr V,  icla_int_t ldv,
    iclaDouble_ptr T,  icla_int_t ldt,
    iclaDouble_ptr dwork );

void
icla_dlarfgx_gpu_v1(
    icla_int_t n,
    iclaDouble_ptr dx0,
    iclaDouble_ptr dx,
    iclaDouble_ptr dtau,
    iclaDouble_ptr        dxnorm,
    iclaDouble_ptr dA, icla_int_t iter );

void
icla_dlarfx_gpu_v1(
    icla_int_t m, icla_int_t n,
    iclaDouble_ptr v,
    iclaDouble_ptr tau,
    iclaDouble_ptr C,  icla_int_t ldc,
    iclaDouble_ptr        xnorm,
    iclaDouble_ptr dT, icla_int_t iter,
    iclaDouble_ptr work );


  /*
   * Level 1 BLAS (alphabetical order)
   */
void
iclablas_daxpycp_v1(
    icla_int_t m,
    iclaDouble_ptr dr,
    iclaDouble_ptr dx,
    iclaDouble_const_ptr db );

void
iclablas_dswap_v1(
    icla_int_t n,
    iclaDouble_ptr dx, icla_int_t incx,
    iclaDouble_ptr dy, icla_int_t incy );

void
iclablas_dswapblk_v1(
    icla_order_t order,
    icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    iclaDouble_ptr dB, icla_int_t lddb,
    icla_int_t i1, icla_int_t i2,
    const icla_int_t *ipiv, icla_int_t inci,
    icla_int_t offset );

void
iclablas_dswapdblk_v1(
    icla_int_t n, icla_int_t nb,
    iclaDouble_ptr dA, icla_int_t ldda, icla_int_t inca,
    iclaDouble_ptr dB, icla_int_t lddb, icla_int_t incb );

  /*
   * Level 2 BLAS (alphabetical order)
   */
void
iclablas_dgemv_v1(
    icla_trans_t trans, icla_int_t m, icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_const_ptr dx, icla_int_t incx,
    double beta,
    iclaDouble_ptr       dy, icla_int_t incy );

void
iclablas_dgemv_conj_v1(
    icla_int_t m, icla_int_t n, double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_const_ptr dx, icla_int_t incx,
    double beta,
    iclaDouble_ptr dy, icla_int_t incy );

icla_int_t
iclablas_dsymv_v1(
    icla_uplo_t uplo, icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_const_ptr dx, icla_int_t incx,
    double beta,
    iclaDouble_ptr       dy, icla_int_t incy );

icla_int_t
iclablas_dsymv_v1(
    icla_uplo_t uplo, icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_const_ptr dx, icla_int_t incx,
    double beta,
    iclaDouble_ptr       dy, icla_int_t incy );

  /*
   * Level 3 BLAS (alphabetical order)
   */
void
iclablas_dgemm_v1(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_const_ptr dB, icla_int_t lddb,
    double beta,
    iclaDouble_ptr       dC, icla_int_t lddc );

void
iclablas_dgemm_reduce_v1(
    icla_int_t m, icla_int_t n, icla_int_t k,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_const_ptr dB, icla_int_t lddb,
    double beta,
    iclaDouble_ptr       dC, icla_int_t lddc );

void
iclablas_dsymm_v1(
    icla_side_t side, icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_const_ptr dB, icla_int_t lddb,
    double beta,
    iclaDouble_ptr       dC, icla_int_t lddc );

void
iclablas_dsymm_v1(
    icla_side_t side, icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_const_ptr dB, icla_int_t lddb,
    double beta,
    iclaDouble_ptr       dC, icla_int_t lddc );

void
iclablas_dsyr2k_v1(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_const_ptr dB, icla_int_t lddb,
    double beta,
    iclaDouble_ptr       dC, icla_int_t lddc );

void
iclablas_dsyr2k_v1(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_const_ptr dB, icla_int_t lddb,
    double  beta,
    iclaDouble_ptr       dC, icla_int_t lddc );

void
iclablas_dsyrk_v1(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    double beta,
    iclaDouble_ptr       dC, icla_int_t lddc );

void
iclablas_dsyrk_v1(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    double  alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    double  beta,
    iclaDouble_ptr       dC, icla_int_t lddc );

void
iclablas_dtrsm_v1(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_ptr       dB, icla_int_t lddb );

void
iclablas_dtrsm_outofplace_v1(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_ptr       dB, icla_int_t lddb,
    iclaDouble_ptr       dX, icla_int_t lddx,
    icla_int_t flag,
    iclaDouble_ptr d_dinvA, icla_int_t dinvA_length );

void
iclablas_dtrsm_work_v1(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_ptr       dB, icla_int_t lddb,
    iclaDouble_ptr       dX, icla_int_t lddx,
    icla_int_t flag,
    iclaDouble_ptr d_dinvA, icla_int_t dinvA_length );


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

#define icla_dsetvector_v1(           n, hx_src, incx, dy_dst, incy ) \
        icla_dsetvector_v1_internal(  n, hx_src, incx, dy_dst, incy, __func__, __FILE__, __LINE__ )

#define icla_dgetvector_v1(           n, dx_src, incx, hy_dst, incy ) \
        icla_dgetvector_v1_internal(  n, dx_src, incx, hy_dst, incy, __func__, __FILE__, __LINE__ )

#define icla_dcopyvector_v1(          n, dx_src, incx, dy_dst, incy ) \
        icla_dcopyvector_v1_internal( n, dx_src, incx, dy_dst, incy, __func__, __FILE__, __LINE__ )

static inline void
icla_dsetvector_v1_internal(
    icla_int_t n,
    double const    *hx_src, icla_int_t incx,
    iclaDouble_ptr       dy_dst, icla_int_t incy,
    const char* func, const char* file, int line )
{
    icla_setvector_v1_internal( n, sizeof(double),
                                 hx_src, incx,
                                 dy_dst, incy,
                                 func, file, line );
}

static inline void
icla_dgetvector_v1_internal(
    icla_int_t n,
    iclaDouble_const_ptr dx_src, icla_int_t incx,
    double          *hy_dst, icla_int_t incy,
    const char* func, const char* file, int line )
{
    icla_getvector_v1_internal( n, sizeof(double),
                                 dx_src, incx,
                                 hy_dst, incy,
                                 func, file, line );
}

static inline void
icla_dcopyvector_v1_internal(
    icla_int_t n,
    iclaDouble_const_ptr dx_src, icla_int_t incx,
    iclaDouble_ptr       dy_dst, icla_int_t incy,
    const char* func, const char* file, int line )
{
    icla_copyvector_v1_internal( n, sizeof(double),
                                  dx_src, incx,
                                  dy_dst, incy,
                                  func, file, line );
}


// =============================================================================
// copying sub-matrices (contiguous columns)

#define icla_dsetmatrix_v1(           m, n, hA_src, lda,  dB_dst, lddb ) \
        icla_dsetmatrix_v1_internal(  m, n, hA_src, lda,  dB_dst, lddb, __func__, __FILE__, __LINE__ )

#define icla_dgetmatrix_v1(           m, n, dA_src, ldda, hB_dst, ldb  ) \
        icla_dgetmatrix_v1_internal(  m, n, dA_src, ldda, hB_dst, ldb,  __func__, __FILE__, __LINE__ )

#define icla_dcopymatrix_v1(          m, n, dA_src, ldda, dB_dst, lddb ) \
        icla_dcopymatrix_v1_internal( m, n, dA_src, ldda, dB_dst, lddb, __func__, __FILE__, __LINE__ )

static inline void
icla_dsetmatrix_v1_internal(
    icla_int_t m, icla_int_t n,
    double const    *hA_src, icla_int_t lda,
    iclaDouble_ptr       dB_dst, icla_int_t lddb,
    const char* func, const char* file, int line )
{
    icla_setmatrix_v1_internal( m, n, sizeof(double),
                                 hA_src, lda,
                                 dB_dst, lddb,
                                 func, file, line );
}

static inline void
icla_dgetmatrix_v1_internal(
    icla_int_t m, icla_int_t n,
    iclaDouble_const_ptr dA_src, icla_int_t ldda,
    double          *hB_dst, icla_int_t ldb,
    const char* func, const char* file, int line )
{
    icla_getmatrix_v1_internal( m, n, sizeof(double),
                                 dA_src, ldda,
                                 hB_dst, ldb,
                                 func, file, line );
}

static inline void
icla_dcopymatrix_v1_internal(
    icla_int_t m, icla_int_t n,
    iclaDouble_const_ptr dA_src, icla_int_t ldda,
    iclaDouble_ptr       dB_dst, icla_int_t lddb,
    const char* func, const char* file, int line )
{
    icla_copymatrix_v1_internal( m, n, sizeof(double),
                                  dA_src, ldda,
                                  dB_dst, lddb,
                                  func, file, line );
}


// =============================================================================
// Level 1 BLAS (alphabetical order)

// in cublas_v2, result returned through output argument
icla_int_t
icla_idamax_v1(
    icla_int_t n,
    iclaDouble_const_ptr dx, icla_int_t incx );

// in cublas_v2, result returned through output argument
icla_int_t
icla_idamin_v1(
    icla_int_t n,
    iclaDouble_const_ptr dx, icla_int_t incx );

// in cublas_v2, result returned through output argument
double
icla_dasum_v1(
    icla_int_t n,
    iclaDouble_const_ptr dx, icla_int_t incx );

void
icla_daxpy_v1(
    icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dx, icla_int_t incx,
    iclaDouble_ptr       dy, icla_int_t incy );

void
icla_dcopy_v1(
    icla_int_t n,
    iclaDouble_const_ptr dx, icla_int_t incx,
    iclaDouble_ptr       dy, icla_int_t incy );

// in cublas_v2, result returned through output argument
double
icla_ddot_v1(
    icla_int_t n,
    iclaDouble_const_ptr dx, icla_int_t incx,
    iclaDouble_const_ptr dy, icla_int_t incy );

// in cublas_v2, result returned through output argument
double
icla_ddot_v1(
    icla_int_t n,
    iclaDouble_const_ptr dx, icla_int_t incx,
    iclaDouble_const_ptr dy, icla_int_t incy );

// in cublas_v2, result returned through output argument
double
icla_dnrm2_v1(
    icla_int_t n,
    iclaDouble_const_ptr dx, icla_int_t incx );

void
icla_drot_v1(
    icla_int_t n,
    iclaDouble_ptr dx, icla_int_t incx,
    iclaDouble_ptr dy, icla_int_t incy,
    double dc, double ds );

void
icla_drot_v1(
    icla_int_t n,
    iclaDouble_ptr dx, icla_int_t incx,
    iclaDouble_ptr dy, icla_int_t incy,
    double dc, double ds );

#ifdef ICLA_REAL
void
icla_drotm_v1(
    icla_int_t n,
    iclaDouble_ptr dx, icla_int_t incx,
    iclaDouble_ptr dy, icla_int_t incy,
    iclaDouble_const_ptr param );

void
icla_drotmg_v1(
    iclaDouble_ptr d1, iclaDouble_ptr       d2,
    iclaDouble_ptr x1, iclaDouble_const_ptr y1,
    iclaDouble_ptr param );
#endif  // ICLA_REAL

void
icla_dscal_v1(
    icla_int_t n,
    double alpha,
    iclaDouble_ptr dx, icla_int_t incx );

void
icla_dscal_v1(
    icla_int_t n,
    double alpha,
    iclaDouble_ptr dx, icla_int_t incx );

void
icla_dswap_v1(
    icla_int_t n,
    iclaDouble_ptr dx, icla_int_t incx,
    iclaDouble_ptr dy, icla_int_t incy );

// =============================================================================
// Level 2 BLAS (alphabetical order)

void
icla_dgemv_v1(
    icla_trans_t transA,
    icla_int_t m, icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_const_ptr dx, icla_int_t incx,
    double beta,
    iclaDouble_ptr       dy, icla_int_t incy );

void
icla_dger_v1(
    icla_int_t m, icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dx, icla_int_t incx,
    iclaDouble_const_ptr dy, icla_int_t incy,
    iclaDouble_ptr       dA, icla_int_t ldda );

void
icla_dger_v1(
    icla_int_t m, icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dx, icla_int_t incx,
    iclaDouble_const_ptr dy, icla_int_t incy,
    iclaDouble_ptr       dA, icla_int_t ldda );

void
icla_dsymv_v1(
    icla_uplo_t uplo,
    icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_const_ptr dx, icla_int_t incx,
    double beta,
    iclaDouble_ptr       dy, icla_int_t incy );

void
icla_dsyr_v1(
    icla_uplo_t uplo,
    icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dx, icla_int_t incx,
    iclaDouble_ptr       dA, icla_int_t ldda );

void
icla_dsyr2_v1(
    icla_uplo_t uplo,
    icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dx, icla_int_t incx,
    iclaDouble_const_ptr dy, icla_int_t incy,
    iclaDouble_ptr       dA, icla_int_t ldda );

void
icla_dtrmv_v1(
    icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t n,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_ptr       dx, icla_int_t incx );

void
icla_dtrsv_v1(
    icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t n,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_ptr       dx, icla_int_t incx );

// =============================================================================
// Level 3 BLAS (alphabetical order)

void
icla_dgemm_v1(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_const_ptr dB, icla_int_t lddb,
    double beta,
    iclaDouble_ptr       dC, icla_int_t lddc );

void
icla_dsymm_v1(
    icla_side_t side, icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_const_ptr dB, icla_int_t lddb,
    double beta,
    iclaDouble_ptr       dC, icla_int_t lddc );

void
icla_dsymm_v1(
    icla_side_t side, icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_const_ptr dB, icla_int_t lddb,
    double beta,
    iclaDouble_ptr       dC, icla_int_t lddc );

void
icla_dsyr2k_v1(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_const_ptr dB, icla_int_t lddb,
    double beta,
    iclaDouble_ptr       dC, icla_int_t lddc );

void
icla_dsyr2k_v1(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_const_ptr dB, icla_int_t lddb,
    double beta,
    iclaDouble_ptr       dC, icla_int_t lddc );

void
icla_dsyrk_v1(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    double beta,
    iclaDouble_ptr       dC, icla_int_t lddc );

void
icla_dsyrk_v1(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    double beta,
    iclaDouble_ptr       dC, icla_int_t lddc );

void
icla_dtrmm_v1(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_ptr       dB, icla_int_t lddb );

void
icla_dtrsm_v1(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    double alpha,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    iclaDouble_ptr       dB, icla_int_t lddb );


#ifdef __cplusplus
}
#endif

#undef ICLA_REAL

#endif // ICLABLAS_D_H
