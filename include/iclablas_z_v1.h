/*
    -- ICLA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
*/

#ifndef ICLABLAS_Z_V1_H
#define ICLABLAS_Z_V1_H

#ifdef ICLA_NO_V1
#error "Since ICLA_NO_V1 is defined, icla.h is invalid; use icla_v2.h"
#endif

#include "icla_types.h"
#include "icla_copy_v1.h"

#define ICLA_COMPLEX

#ifdef __cplusplus
extern "C" {
#endif

  /*
   * Transpose functions
   */
void
iclablas_ztranspose_inplace_v1(
    icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda );

void
iclablas_ztranspose_conj_inplace_v1(
    icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda );

void
iclablas_ztranspose_v1(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex_const_ptr dA,  icla_int_t ldda,
    iclaDoubleComplex_ptr       dAT, icla_int_t lddat );

void
iclablas_ztranspose_conj_v1(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex_const_ptr dA,  icla_int_t ldda,
    iclaDoubleComplex_ptr       dAT, icla_int_t lddat );

void
iclablas_zgetmatrix_transpose_v1(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex_const_ptr dAT,   icla_int_t ldda,
    iclaDoubleComplex          *hA,    icla_int_t lda,
    iclaDoubleComplex_ptr       dwork, icla_int_t lddwork, icla_int_t nb );

void
iclablas_zsetmatrix_transpose_v1(
    icla_int_t m, icla_int_t n,
    const iclaDoubleComplex *hA,    icla_int_t lda,
    iclaDoubleComplex_ptr    dAT,   icla_int_t ldda,
    iclaDoubleComplex_ptr    dwork, icla_int_t lddwork, icla_int_t nb );

  /*
   * RBT-related functions
   */
void
iclablas_zprbt_v1(
    icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr du,
    iclaDoubleComplex_ptr dv );

void
iclablas_zprbt_mv_v1(
    icla_int_t n, icla_int_t nrhs,
    iclaDoubleComplex_ptr dv,
    iclaDoubleComplex_ptr db, icla_int_t lddb);

void
iclablas_zprbt_mtv_v1(
    icla_int_t n, icla_int_t nrhs,
    iclaDoubleComplex_ptr du,
    iclaDoubleComplex_ptr db, icla_int_t lddb);

  /*
   * Multi-GPU copy functions
   */
void
icla_zgetmatrix_1D_col_bcyclic_v1(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex_const_ptr const dA[], icla_int_t ldda,
    iclaDoubleComplex                *hA,   icla_int_t lda,
    icla_int_t ngpu, icla_int_t nb );

void
icla_zsetmatrix_1D_col_bcyclic_v1(
    icla_int_t m, icla_int_t n,
    const iclaDoubleComplex *hA,   icla_int_t lda,
    iclaDoubleComplex_ptr    dA[], icla_int_t ldda,
    icla_int_t ngpu, icla_int_t nb );

void
icla_zgetmatrix_1D_row_bcyclic_v1(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex_const_ptr const dA[], icla_int_t ldda,
    iclaDoubleComplex                *hA,   icla_int_t lda,
    icla_int_t ngpu, icla_int_t nb );

void
icla_zsetmatrix_1D_row_bcyclic_v1(
    icla_int_t m, icla_int_t n,
    const iclaDoubleComplex *hA,   icla_int_t lda,
    iclaDoubleComplex_ptr    dA[], icla_int_t ldda,
    icla_int_t ngpu, icla_int_t nb );


  /*
   * LAPACK auxiliary functions (alphabetical order)
   */
void
iclablas_zgeadd_v1(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr       dB, icla_int_t lddb );

void
iclablas_zgeadd2_v1(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dB, icla_int_t lddb );

void
iclablas_zlacpy_v1(
    icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr       dB, icla_int_t lddb );

void
iclablas_zlacpy_conj_v1(
    icla_int_t n,
    iclaDoubleComplex_ptr dA1, icla_int_t lda1,
    iclaDoubleComplex_ptr dA2, icla_int_t lda2 );

void
iclablas_zlacpy_sym_in_v1(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    icla_int_t *rows, icla_int_t *perm,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr       dB, icla_int_t lddb );

void
iclablas_zlacpy_sym_out_v1(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    icla_int_t *rows, icla_int_t *perm,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr       dB, icla_int_t lddb );

double
iclablas_zlange_v1(
    icla_norm_t norm,
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDouble_ptr dwork, icla_int_t lwork );

double
iclablas_zlanhe_v1(
    icla_norm_t norm, icla_uplo_t uplo,
    icla_int_t n,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDouble_ptr dwork, icla_int_t lwork );

double
iclablas_zlansy_v1(
    icla_norm_t norm, icla_uplo_t uplo,
    icla_int_t n,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDouble_ptr dwork, icla_int_t lwork );

void
iclablas_zlarfg_v1(
    icla_int_t n,
    iclaDoubleComplex_ptr dalpha,
    iclaDoubleComplex_ptr dx, icla_int_t incx,
    iclaDoubleComplex_ptr dtau );

void
iclablas_zlascl_v1(
    icla_type_t type, icla_int_t kl, icla_int_t ku,
    double cfrom, double cto,
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    icla_int_t *info );

void
iclablas_zlascl_2x2_v1(
    icla_type_t type, icla_int_t m,
    iclaDoubleComplex_const_ptr dW, icla_int_t lddw,
    iclaDoubleComplex_ptr       dA, icla_int_t ldda,
    icla_int_t *info );

void
iclablas_zlascl2_v1(
    icla_type_t type,
    icla_int_t m, icla_int_t n,
    iclaDouble_const_ptr dD,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    icla_int_t *info );

void
iclablas_zlascl_diag_v1(
    icla_type_t type, icla_int_t m, icla_int_t n,
    iclaDoubleComplex_const_ptr dD, icla_int_t lddd,
    iclaDoubleComplex_ptr       dA, icla_int_t ldda,
    icla_int_t *info );

void
iclablas_zlaset_v1(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    iclaDoubleComplex offdiag, iclaDoubleComplex diag,
    iclaDoubleComplex_ptr dA, icla_int_t ldda );

void
iclablas_zlaset_band_v1(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDoubleComplex offdiag, iclaDoubleComplex diag,
    iclaDoubleComplex_ptr dA, icla_int_t ldda );

void
iclablas_zlaswp_v1(
    icla_int_t n,
    iclaDoubleComplex_ptr dAT, icla_int_t ldda,
    icla_int_t k1, icla_int_t k2,
    const icla_int_t *ipiv, icla_int_t inci );

void
iclablas_zlaswp2_v1(
    icla_int_t n,
    iclaDoubleComplex_ptr dAT, icla_int_t ldda,
    icla_int_t k1, icla_int_t k2,
    iclaInt_const_ptr d_ipiv, icla_int_t inci );

void
iclablas_zlaswp_sym_v1(
    icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    icla_int_t k1, icla_int_t k2,
    const icla_int_t *ipiv, icla_int_t inci );

void
iclablas_zlaswpx_v1(
    icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldx, icla_int_t ldy,
    icla_int_t k1, icla_int_t k2,
    const icla_int_t *ipiv, icla_int_t inci );

void
iclablas_zsymmetrize_v1(
    icla_uplo_t uplo, icla_int_t m,
    iclaDoubleComplex_ptr dA, icla_int_t ldda );

void
iclablas_zsymmetrize_tiles_v1(
    icla_uplo_t uplo, icla_int_t m,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    icla_int_t ntile, icla_int_t mstride, icla_int_t nstride );

void
iclablas_ztrtri_diag_v1(
    icla_uplo_t uplo, icla_diag_t diag, icla_int_t n,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr d_dinvA );

  /*
   * to cleanup (alphabetical order)
   */
void
iclablas_dznrm2_adjust_v1(
    icla_int_t k,
    iclaDouble_ptr dxnorm,
    iclaDoubleComplex_ptr dc );

void
iclablas_dznrm2_check_v1(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    iclaDouble_ptr dxnorm,
    iclaDouble_ptr dlsticc );

void
iclablas_dznrm2_cols_v1(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    iclaDouble_ptr dxnorm );

void
iclablas_dznrm2_row_check_adjust_v1(
    icla_int_t k, double tol,
    iclaDouble_ptr dxnorm,
    iclaDouble_ptr dxnorm2,
    iclaDoubleComplex_ptr dC, icla_int_t lddc,
    iclaDouble_ptr dlsticc );

icla_int_t
icla_zlarfb_gpu_v1(
    icla_side_t side, icla_trans_t trans, icla_direct_t direct, icla_storev_t storev,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDoubleComplex_const_ptr dV, icla_int_t lddv,
    iclaDoubleComplex_const_ptr dT, icla_int_t lddt,
    iclaDoubleComplex_ptr dC,       icla_int_t lddc,
    iclaDoubleComplex_ptr dwork,    icla_int_t ldwork );

icla_int_t
icla_zlarfb_gpu_gemm_v1(
    icla_side_t side, icla_trans_t trans, icla_direct_t direct, icla_storev_t storev,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDoubleComplex_const_ptr dV, icla_int_t lddv,
    iclaDoubleComplex_const_ptr dT, icla_int_t lddt,
    iclaDoubleComplex_ptr dC,       icla_int_t lddc,
    iclaDoubleComplex_ptr dwork,    icla_int_t ldwork,
    iclaDoubleComplex_ptr dworkvt,  icla_int_t ldworkvt );

void
icla_zlarfbx_gpu_v1(
    icla_int_t m, icla_int_t k,
    iclaDoubleComplex_ptr V,  icla_int_t ldv,
    iclaDoubleComplex_ptr dT, icla_int_t ldt,
    iclaDoubleComplex_ptr c,
    iclaDoubleComplex_ptr dwork );

void
icla_zlarfg_gpu_v1(
    icla_int_t n,
    iclaDoubleComplex_ptr dx0,
    iclaDoubleComplex_ptr dx,
    iclaDoubleComplex_ptr dtau,
    iclaDouble_ptr        dxnorm,
    iclaDoubleComplex_ptr dAkk );

void
icla_zlarfgtx_gpu_v1(
    icla_int_t n,
    iclaDoubleComplex_ptr dx0,
    iclaDoubleComplex_ptr dx,
    iclaDoubleComplex_ptr dtau,
    iclaDouble_ptr        dxnorm,
    iclaDoubleComplex_ptr dA, icla_int_t iter,
    iclaDoubleComplex_ptr V,  icla_int_t ldv,
    iclaDoubleComplex_ptr T,  icla_int_t ldt,
    iclaDoubleComplex_ptr dwork );

void
icla_zlarfgx_gpu_v1(
    icla_int_t n,
    iclaDoubleComplex_ptr dx0,
    iclaDoubleComplex_ptr dx,
    iclaDoubleComplex_ptr dtau,
    iclaDouble_ptr        dxnorm,
    iclaDoubleComplex_ptr dA, icla_int_t iter );

void
icla_zlarfx_gpu_v1(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex_ptr v,
    iclaDoubleComplex_ptr tau,
    iclaDoubleComplex_ptr C,  icla_int_t ldc,
    iclaDouble_ptr        xnorm,
    iclaDoubleComplex_ptr dT, icla_int_t iter,
    iclaDoubleComplex_ptr work );


  /*
   * Level 1 BLAS (alphabetical order)
   */
void
iclablas_zaxpycp_v1(
    icla_int_t m,
    iclaDoubleComplex_ptr dr,
    iclaDoubleComplex_ptr dx,
    iclaDoubleComplex_const_ptr db );

void
iclablas_zswap_v1(
    icla_int_t n,
    iclaDoubleComplex_ptr dx, icla_int_t incx,
    iclaDoubleComplex_ptr dy, icla_int_t incy );

void
iclablas_zswapblk_v1(
    icla_order_t order,
    icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr dB, icla_int_t lddb,
    icla_int_t i1, icla_int_t i2,
    const icla_int_t *ipiv, icla_int_t inci,
    icla_int_t offset );

void
iclablas_zswapdblk_v1(
    icla_int_t n, icla_int_t nb,
    iclaDoubleComplex_ptr dA, icla_int_t ldda, icla_int_t inca,
    iclaDoubleComplex_ptr dB, icla_int_t lddb, icla_int_t incb );

  /*
   * Level 2 BLAS (alphabetical order)
   */
void
iclablas_zgemv_v1(
    icla_trans_t trans, icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dy, icla_int_t incy );

void
iclablas_zgemv_conj_v1(
    icla_int_t m, icla_int_t n, iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr dy, icla_int_t incy );

icla_int_t
iclablas_zhemv_v1(
    icla_uplo_t uplo, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dy, icla_int_t incy );

icla_int_t
iclablas_zsymv_v1(
    icla_uplo_t uplo, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dy, icla_int_t incy );

  /*
   * Level 3 BLAS (alphabetical order)
   */
void
iclablas_zgemm_v1(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dB, icla_int_t lddb,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dC, icla_int_t lddc );

void
iclablas_zgemm_reduce_v1(
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dB, icla_int_t lddb,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dC, icla_int_t lddc );

void
iclablas_zhemm_v1(
    icla_side_t side, icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dB, icla_int_t lddb,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dC, icla_int_t lddc );

void
iclablas_zsymm_v1(
    icla_side_t side, icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dB, icla_int_t lddb,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dC, icla_int_t lddc );

void
iclablas_zsyr2k_v1(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dB, icla_int_t lddb,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dC, icla_int_t lddc );

void
iclablas_zher2k_v1(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dB, icla_int_t lddb,
    double  beta,
    iclaDoubleComplex_ptr       dC, icla_int_t lddc );

void
iclablas_zsyrk_v1(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dC, icla_int_t lddc );

void
iclablas_zherk_v1(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    double  alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    double  beta,
    iclaDoubleComplex_ptr       dC, icla_int_t lddc );

void
iclablas_ztrsm_v1(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr       dB, icla_int_t lddb );

void
iclablas_ztrsm_outofplace_v1(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr       dB, icla_int_t lddb,
    iclaDoubleComplex_ptr       dX, icla_int_t lddx,
    icla_int_t flag,
    iclaDoubleComplex_ptr d_dinvA, icla_int_t dinvA_length );

void
iclablas_ztrsm_work_v1(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr       dB, icla_int_t lddb,
    iclaDoubleComplex_ptr       dX, icla_int_t lddx,
    icla_int_t flag,
    iclaDoubleComplex_ptr d_dinvA, icla_int_t dinvA_length );


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

#define icla_zsetvector_v1(           n, hx_src, incx, dy_dst, incy ) \
        icla_zsetvector_v1_internal(  n, hx_src, incx, dy_dst, incy, __func__, __FILE__, __LINE__ )

#define icla_zgetvector_v1(           n, dx_src, incx, hy_dst, incy ) \
        icla_zgetvector_v1_internal(  n, dx_src, incx, hy_dst, incy, __func__, __FILE__, __LINE__ )

#define icla_zcopyvector_v1(          n, dx_src, incx, dy_dst, incy ) \
        icla_zcopyvector_v1_internal( n, dx_src, incx, dy_dst, incy, __func__, __FILE__, __LINE__ )

static inline void
icla_zsetvector_v1_internal(
    icla_int_t n,
    iclaDoubleComplex const    *hx_src, icla_int_t incx,
    iclaDoubleComplex_ptr       dy_dst, icla_int_t incy,
    const char* func, const char* file, int line )
{
    icla_setvector_v1_internal( n, sizeof(iclaDoubleComplex),
                                 hx_src, incx,
                                 dy_dst, incy,
                                 func, file, line );
}

static inline void
icla_zgetvector_v1_internal(
    icla_int_t n,
    iclaDoubleComplex_const_ptr dx_src, icla_int_t incx,
    iclaDoubleComplex          *hy_dst, icla_int_t incy,
    const char* func, const char* file, int line )
{
    icla_getvector_v1_internal( n, sizeof(iclaDoubleComplex),
                                 dx_src, incx,
                                 hy_dst, incy,
                                 func, file, line );
}

static inline void
icla_zcopyvector_v1_internal(
    icla_int_t n,
    iclaDoubleComplex_const_ptr dx_src, icla_int_t incx,
    iclaDoubleComplex_ptr       dy_dst, icla_int_t incy,
    const char* func, const char* file, int line )
{
    icla_copyvector_v1_internal( n, sizeof(iclaDoubleComplex),
                                  dx_src, incx,
                                  dy_dst, incy,
                                  func, file, line );
}


// =============================================================================
// copying sub-matrices (contiguous columns)

#define icla_zsetmatrix_v1(           m, n, hA_src, lda,  dB_dst, lddb ) \
        icla_zsetmatrix_v1_internal(  m, n, hA_src, lda,  dB_dst, lddb, __func__, __FILE__, __LINE__ )

#define icla_zgetmatrix_v1(           m, n, dA_src, ldda, hB_dst, ldb  ) \
        icla_zgetmatrix_v1_internal(  m, n, dA_src, ldda, hB_dst, ldb,  __func__, __FILE__, __LINE__ )

#define icla_zcopymatrix_v1(          m, n, dA_src, ldda, dB_dst, lddb ) \
        icla_zcopymatrix_v1_internal( m, n, dA_src, ldda, dB_dst, lddb, __func__, __FILE__, __LINE__ )

static inline void
icla_zsetmatrix_v1_internal(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex const    *hA_src, icla_int_t lda,
    iclaDoubleComplex_ptr       dB_dst, icla_int_t lddb,
    const char* func, const char* file, int line )
{
    icla_setmatrix_v1_internal( m, n, sizeof(iclaDoubleComplex),
                                 hA_src, lda,
                                 dB_dst, lddb,
                                 func, file, line );
}

static inline void
icla_zgetmatrix_v1_internal(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex_const_ptr dA_src, icla_int_t ldda,
    iclaDoubleComplex          *hB_dst, icla_int_t ldb,
    const char* func, const char* file, int line )
{
    icla_getmatrix_v1_internal( m, n, sizeof(iclaDoubleComplex),
                                 dA_src, ldda,
                                 hB_dst, ldb,
                                 func, file, line );
}

static inline void
icla_zcopymatrix_v1_internal(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex_const_ptr dA_src, icla_int_t ldda,
    iclaDoubleComplex_ptr       dB_dst, icla_int_t lddb,
    const char* func, const char* file, int line )
{
    icla_copymatrix_v1_internal( m, n, sizeof(iclaDoubleComplex),
                                  dA_src, ldda,
                                  dB_dst, lddb,
                                  func, file, line );
}


// =============================================================================
// Level 1 BLAS (alphabetical order)

// in cublas_v2, result returned through output argument
icla_int_t
icla_izamax_v1(
    icla_int_t n,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx );

// in cublas_v2, result returned through output argument
icla_int_t
icla_izamin_v1(
    icla_int_t n,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx );

// in cublas_v2, result returned through output argument
double
icla_dzasum_v1(
    icla_int_t n,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx );

void
icla_zaxpy_v1(
    icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex_ptr       dy, icla_int_t incy );

void
icla_zcopy_v1(
    icla_int_t n,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex_ptr       dy, icla_int_t incy );

// in cublas_v2, result returned through output argument
iclaDoubleComplex
icla_zdotc_v1(
    icla_int_t n,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex_const_ptr dy, icla_int_t incy );

// in cublas_v2, result returned through output argument
iclaDoubleComplex
icla_zdotu_v1(
    icla_int_t n,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex_const_ptr dy, icla_int_t incy );

// in cublas_v2, result returned through output argument
double
icla_dznrm2_v1(
    icla_int_t n,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx );

void
icla_zrot_v1(
    icla_int_t n,
    iclaDoubleComplex_ptr dx, icla_int_t incx,
    iclaDoubleComplex_ptr dy, icla_int_t incy,
    double dc, iclaDoubleComplex ds );

void
icla_zdrot_v1(
    icla_int_t n,
    iclaDoubleComplex_ptr dx, icla_int_t incx,
    iclaDoubleComplex_ptr dy, icla_int_t incy,
    double dc, double ds );

#ifdef ICLA_REAL
void
icla_zrotm_v1(
    icla_int_t n,
    iclaDouble_ptr dx, icla_int_t incx,
    iclaDouble_ptr dy, icla_int_t incy,
    iclaDouble_const_ptr param );

void
icla_zrotmg_v1(
    iclaDouble_ptr d1, iclaDouble_ptr       d2,
    iclaDouble_ptr x1, iclaDouble_const_ptr y1,
    iclaDouble_ptr param );
#endif  // ICLA_REAL

void
icla_zscal_v1(
    icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_ptr dx, icla_int_t incx );

void
icla_zdscal_v1(
    icla_int_t n,
    double alpha,
    iclaDoubleComplex_ptr dx, icla_int_t incx );

void
icla_zswap_v1(
    icla_int_t n,
    iclaDoubleComplex_ptr dx, icla_int_t incx,
    iclaDoubleComplex_ptr dy, icla_int_t incy );

// =============================================================================
// Level 2 BLAS (alphabetical order)

void
icla_zgemv_v1(
    icla_trans_t transA,
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dy, icla_int_t incy );

void
icla_zgerc_v1(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex_const_ptr dy, icla_int_t incy,
    iclaDoubleComplex_ptr       dA, icla_int_t ldda );

void
icla_zgeru_v1(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex_const_ptr dy, icla_int_t incy,
    iclaDoubleComplex_ptr       dA, icla_int_t ldda );

void
icla_zhemv_v1(
    icla_uplo_t uplo,
    icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dy, icla_int_t incy );

void
icla_zher_v1(
    icla_uplo_t uplo,
    icla_int_t n,
    double alpha,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex_ptr       dA, icla_int_t ldda );

void
icla_zher2_v1(
    icla_uplo_t uplo,
    icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex_const_ptr dy, icla_int_t incy,
    iclaDoubleComplex_ptr       dA, icla_int_t ldda );

void
icla_ztrmv_v1(
    icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t n,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr       dx, icla_int_t incx );

void
icla_ztrsv_v1(
    icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t n,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr       dx, icla_int_t incx );

// =============================================================================
// Level 3 BLAS (alphabetical order)

void
icla_zgemm_v1(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dB, icla_int_t lddb,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dC, icla_int_t lddc );

void
icla_zsymm_v1(
    icla_side_t side, icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dB, icla_int_t lddb,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dC, icla_int_t lddc );

void
icla_zhemm_v1(
    icla_side_t side, icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dB, icla_int_t lddb,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dC, icla_int_t lddc );

void
icla_zsyr2k_v1(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dB, icla_int_t lddb,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dC, icla_int_t lddc );

void
icla_zher2k_v1(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dB, icla_int_t lddb,
    double beta,
    iclaDoubleComplex_ptr       dC, icla_int_t lddc );

void
icla_zsyrk_v1(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dC, icla_int_t lddc );

void
icla_zherk_v1(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    double alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    double beta,
    iclaDoubleComplex_ptr       dC, icla_int_t lddc );

void
icla_ztrmm_v1(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr       dB, icla_int_t lddb );

void
icla_ztrsm_v1(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr       dB, icla_int_t lddb );


#ifdef __cplusplus
}
#endif

#undef ICLA_COMPLEX

#endif // ICLABLAS_Z_H
