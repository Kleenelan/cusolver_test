
#ifndef ICLABLAS_C_V1_H
#define ICLABLAS_C_V1_H

#ifdef ICLA_NO_V1
#error "Since ICLA_NO_V1 is defined, icla.h is invalid; use icla_v2.h"
#endif

#include "icla_types.h"
#include "icla_copy_v1.h"

#define ICLA_COMPLEX

#ifdef __cplusplus
extern "C" {
#endif

void
iclablas_ctranspose_inplace_v1(
    icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda );

void
iclablas_ctranspose_conj_inplace_v1(
    icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda );

void
iclablas_ctranspose_v1(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex_const_ptr dA,  icla_int_t ldda,
    iclaFloatComplex_ptr       dAT, icla_int_t lddat );

void
iclablas_ctranspose_conj_v1(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex_const_ptr dA,  icla_int_t ldda,
    iclaFloatComplex_ptr       dAT, icla_int_t lddat );

void
iclablas_cgetmatrix_transpose_v1(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex_const_ptr dAT,   icla_int_t ldda,
    iclaFloatComplex          *hA,    icla_int_t lda,
    iclaFloatComplex_ptr       dwork, icla_int_t lddwork, icla_int_t nb );

void
iclablas_csetmatrix_transpose_v1(
    icla_int_t m, icla_int_t n,
    const iclaFloatComplex *hA,    icla_int_t lda,
    iclaFloatComplex_ptr    dAT,   icla_int_t ldda,
    iclaFloatComplex_ptr    dwork, icla_int_t lddwork, icla_int_t nb );

void
iclablas_cprbt_v1(
    icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr du,
    iclaFloatComplex_ptr dv );

void
iclablas_cprbt_mv_v1(
    icla_int_t n, icla_int_t nrhs,
    iclaFloatComplex_ptr dv,
    iclaFloatComplex_ptr db, icla_int_t lddb);

void
iclablas_cprbt_mtv_v1(
    icla_int_t n, icla_int_t nrhs,
    iclaFloatComplex_ptr du,
    iclaFloatComplex_ptr db, icla_int_t lddb);

void
icla_cgetmatrix_1D_col_bcyclic_v1(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex_const_ptr const dA[], icla_int_t ldda,
    iclaFloatComplex                *hA,   icla_int_t lda,
    icla_int_t ngpu, icla_int_t nb );

void
icla_csetmatrix_1D_col_bcyclic_v1(
    icla_int_t m, icla_int_t n,
    const iclaFloatComplex *hA,   icla_int_t lda,
    iclaFloatComplex_ptr    dA[], icla_int_t ldda,
    icla_int_t ngpu, icla_int_t nb );

void
icla_cgetmatrix_1D_row_bcyclic_v1(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex_const_ptr const dA[], icla_int_t ldda,
    iclaFloatComplex                *hA,   icla_int_t lda,
    icla_int_t ngpu, icla_int_t nb );

void
icla_csetmatrix_1D_row_bcyclic_v1(
    icla_int_t m, icla_int_t n,
    const iclaFloatComplex *hA,   icla_int_t lda,
    iclaFloatComplex_ptr    dA[], icla_int_t ldda,
    icla_int_t ngpu, icla_int_t nb );

void
iclablas_cgeadd_v1(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr       dB, icla_int_t lddb );

void
iclablas_cgeadd2_v1(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr       dB, icla_int_t lddb );

void
iclablas_clacpy_v1(
    icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr       dB, icla_int_t lddb );

void
iclablas_clacpy_conj_v1(
    icla_int_t n,
    iclaFloatComplex_ptr dA1, icla_int_t lda1,
    iclaFloatComplex_ptr dA2, icla_int_t lda2 );

void
iclablas_clacpy_sym_in_v1(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    icla_int_t *rows, icla_int_t *perm,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr       dB, icla_int_t lddb );

void
iclablas_clacpy_sym_out_v1(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    icla_int_t *rows, icla_int_t *perm,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr       dB, icla_int_t lddb );

float
iclablas_clange_v1(
    icla_norm_t norm,
    icla_int_t m, icla_int_t n,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloat_ptr dwork, icla_int_t lwork );

float
iclablas_clanhe_v1(
    icla_norm_t norm, icla_uplo_t uplo,
    icla_int_t n,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloat_ptr dwork, icla_int_t lwork );

float
iclablas_clansy_v1(
    icla_norm_t norm, icla_uplo_t uplo,
    icla_int_t n,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloat_ptr dwork, icla_int_t lwork );

void
iclablas_clarfg_v1(
    icla_int_t n,
    iclaFloatComplex_ptr dalpha,
    iclaFloatComplex_ptr dx, icla_int_t incx,
    iclaFloatComplex_ptr dtau );

void
iclablas_clascl_v1(
    icla_type_t type, icla_int_t kl, icla_int_t ku,
    float cfrom, float cto,
    icla_int_t m, icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    icla_int_t *info );

void
iclablas_clascl_2x2_v1(
    icla_type_t type, icla_int_t m,
    iclaFloatComplex_const_ptr dW, icla_int_t lddw,
    iclaFloatComplex_ptr       dA, icla_int_t ldda,
    icla_int_t *info );

void
iclablas_clascl2_v1(
    icla_type_t type,
    icla_int_t m, icla_int_t n,
    iclaFloat_const_ptr dD,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    icla_int_t *info );

void
iclablas_clascl_diag_v1(
    icla_type_t type, icla_int_t m, icla_int_t n,
    iclaFloatComplex_const_ptr dD, icla_int_t lddd,
    iclaFloatComplex_ptr       dA, icla_int_t ldda,
    icla_int_t *info );

void
iclablas_claset_v1(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    iclaFloatComplex offdiag, iclaFloatComplex diag,
    iclaFloatComplex_ptr dA, icla_int_t ldda );

void
iclablas_claset_band_v1(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n, icla_int_t k,
    iclaFloatComplex offdiag, iclaFloatComplex diag,
    iclaFloatComplex_ptr dA, icla_int_t ldda );

void
iclablas_claswp_v1(
    icla_int_t n,
    iclaFloatComplex_ptr dAT, icla_int_t ldda,
    icla_int_t k1, icla_int_t k2,
    const icla_int_t *ipiv, icla_int_t inci );

void
iclablas_claswp2_v1(
    icla_int_t n,
    iclaFloatComplex_ptr dAT, icla_int_t ldda,
    icla_int_t k1, icla_int_t k2,
    iclaInt_const_ptr d_ipiv, icla_int_t inci );

void
iclablas_claswp_sym_v1(
    icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    icla_int_t k1, icla_int_t k2,
    const icla_int_t *ipiv, icla_int_t inci );

void
iclablas_claswpx_v1(
    icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldx, icla_int_t ldy,
    icla_int_t k1, icla_int_t k2,
    const icla_int_t *ipiv, icla_int_t inci );

void
iclablas_csymmetrize_v1(
    icla_uplo_t uplo, icla_int_t m,
    iclaFloatComplex_ptr dA, icla_int_t ldda );

void
iclablas_csymmetrize_tiles_v1(
    icla_uplo_t uplo, icla_int_t m,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    icla_int_t ntile, icla_int_t mstride, icla_int_t nstride );

void
iclablas_ctrtri_diag_v1(
    icla_uplo_t uplo, icla_diag_t diag, icla_int_t n,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr d_dinvA );

void
iclablas_scnrm2_adjust_v1(
    icla_int_t k,
    iclaFloat_ptr dxnorm,
    iclaFloatComplex_ptr dc );

void
iclablas_scnrm2_check_v1(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    iclaFloat_ptr dxnorm,
    iclaFloat_ptr dlsticc );

void
iclablas_scnrm2_cols_v1(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    iclaFloat_ptr dxnorm );

void
iclablas_scnrm2_row_check_adjust_v1(
    icla_int_t k, float tol,
    iclaFloat_ptr dxnorm,
    iclaFloat_ptr dxnorm2,
    iclaFloatComplex_ptr dC, icla_int_t lddc,
    iclaFloat_ptr dlsticc );

icla_int_t
icla_clarfb_gpu_v1(
    icla_side_t side, icla_trans_t trans, icla_direct_t direct, icla_storev_t storev,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaFloatComplex_const_ptr dV, icla_int_t lddv,
    iclaFloatComplex_const_ptr dT, icla_int_t lddt,
    iclaFloatComplex_ptr dC,       icla_int_t lddc,
    iclaFloatComplex_ptr dwork,    icla_int_t ldwork );

icla_int_t
icla_clarfb_gpu_gemm_v1(
    icla_side_t side, icla_trans_t trans, icla_direct_t direct, icla_storev_t storev,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaFloatComplex_const_ptr dV, icla_int_t lddv,
    iclaFloatComplex_const_ptr dT, icla_int_t lddt,
    iclaFloatComplex_ptr dC,       icla_int_t lddc,
    iclaFloatComplex_ptr dwork,    icla_int_t ldwork,
    iclaFloatComplex_ptr dworkvt,  icla_int_t ldworkvt );

void
icla_clarfbx_gpu_v1(
    icla_int_t m, icla_int_t k,
    iclaFloatComplex_ptr V,  icla_int_t ldv,
    iclaFloatComplex_ptr dT, icla_int_t ldt,
    iclaFloatComplex_ptr c,
    iclaFloatComplex_ptr dwork );

void
icla_clarfg_gpu_v1(
    icla_int_t n,
    iclaFloatComplex_ptr dx0,
    iclaFloatComplex_ptr dx,
    iclaFloatComplex_ptr dtau,
    iclaFloat_ptr        dxnorm,
    iclaFloatComplex_ptr dAkk );

void
icla_clarfgtx_gpu_v1(
    icla_int_t n,
    iclaFloatComplex_ptr dx0,
    iclaFloatComplex_ptr dx,
    iclaFloatComplex_ptr dtau,
    iclaFloat_ptr        dxnorm,
    iclaFloatComplex_ptr dA, icla_int_t iter,
    iclaFloatComplex_ptr V,  icla_int_t ldv,
    iclaFloatComplex_ptr T,  icla_int_t ldt,
    iclaFloatComplex_ptr dwork );

void
icla_clarfgx_gpu_v1(
    icla_int_t n,
    iclaFloatComplex_ptr dx0,
    iclaFloatComplex_ptr dx,
    iclaFloatComplex_ptr dtau,
    iclaFloat_ptr        dxnorm,
    iclaFloatComplex_ptr dA, icla_int_t iter );

void
icla_clarfx_gpu_v1(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex_ptr v,
    iclaFloatComplex_ptr tau,
    iclaFloatComplex_ptr C,  icla_int_t ldc,
    iclaFloat_ptr        xnorm,
    iclaFloatComplex_ptr dT, icla_int_t iter,
    iclaFloatComplex_ptr work );

void
iclablas_caxpycp_v1(
    icla_int_t m,
    iclaFloatComplex_ptr dr,
    iclaFloatComplex_ptr dx,
    iclaFloatComplex_const_ptr db );

void
iclablas_cswap_v1(
    icla_int_t n,
    iclaFloatComplex_ptr dx, icla_int_t incx,
    iclaFloatComplex_ptr dy, icla_int_t incy );

void
iclablas_cswapblk_v1(
    icla_order_t order,
    icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr dB, icla_int_t lddb,
    icla_int_t i1, icla_int_t i2,
    const icla_int_t *ipiv, icla_int_t inci,
    icla_int_t offset );

void
iclablas_cswapdblk_v1(
    icla_int_t n, icla_int_t nb,
    iclaFloatComplex_ptr dA, icla_int_t ldda, icla_int_t inca,
    iclaFloatComplex_ptr dB, icla_int_t lddb, icla_int_t incb );

void
iclablas_cgemv_v1(
    icla_trans_t trans, icla_int_t m, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr       dy, icla_int_t incy );

void
iclablas_cgemv_conj_v1(
    icla_int_t m, icla_int_t n, iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr dy, icla_int_t incy );

icla_int_t
iclablas_chemv_v1(
    icla_uplo_t uplo, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr       dy, icla_int_t incy );

icla_int_t
iclablas_csymv_v1(
    icla_uplo_t uplo, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr       dy, icla_int_t incy );

void
iclablas_cgemm_v1(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_const_ptr dB, icla_int_t lddb,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr       dC, icla_int_t lddc );

void
iclablas_cgemm_reduce_v1(
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_const_ptr dB, icla_int_t lddb,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr       dC, icla_int_t lddc );

void
iclablas_chemm_v1(
    icla_side_t side, icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_const_ptr dB, icla_int_t lddb,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr       dC, icla_int_t lddc );

void
iclablas_csymm_v1(
    icla_side_t side, icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_const_ptr dB, icla_int_t lddb,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr       dC, icla_int_t lddc );

void
iclablas_csyr2k_v1(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_const_ptr dB, icla_int_t lddb,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr       dC, icla_int_t lddc );

void
iclablas_cher2k_v1(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_const_ptr dB, icla_int_t lddb,
    float  beta,
    iclaFloatComplex_ptr       dC, icla_int_t lddc );

void
iclablas_csyrk_v1(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr       dC, icla_int_t lddc );

void
iclablas_cherk_v1(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    float  alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    float  beta,
    iclaFloatComplex_ptr       dC, icla_int_t lddc );

void
iclablas_ctrsm_v1(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr       dB, icla_int_t lddb );

void
iclablas_ctrsm_outofplace_v1(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr       dB, icla_int_t lddb,
    iclaFloatComplex_ptr       dX, icla_int_t lddx,
    icla_int_t flag,
    iclaFloatComplex_ptr d_dinvA, icla_int_t dinvA_length );

void
iclablas_ctrsm_work_v1(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr       dB, icla_int_t lddb,
    iclaFloatComplex_ptr       dX, icla_int_t lddx,
    icla_int_t flag,
    iclaFloatComplex_ptr d_dinvA, icla_int_t dinvA_length );

#define icla_csetvector_v1(           n, hx_src, incx, dy_dst, incy ) \
        icla_csetvector_v1_internal(  n, hx_src, incx, dy_dst, incy, __func__, __FILE__, __LINE__ )

#define icla_cgetvector_v1(           n, dx_src, incx, hy_dst, incy ) \
        icla_cgetvector_v1_internal(  n, dx_src, incx, hy_dst, incy, __func__, __FILE__, __LINE__ )

#define icla_ccopyvector_v1(          n, dx_src, incx, dy_dst, incy ) \
        icla_ccopyvector_v1_internal( n, dx_src, incx, dy_dst, incy, __func__, __FILE__, __LINE__ )

static inline void
icla_csetvector_v1_internal(
    icla_int_t n,
    iclaFloatComplex const    *hx_src, icla_int_t incx,
    iclaFloatComplex_ptr       dy_dst, icla_int_t incy,
    const char* func, const char* file, int line )
{
    icla_setvector_v1_internal( n, sizeof(iclaFloatComplex),
                                 hx_src, incx,
                                 dy_dst, incy,
                                 func, file, line );
}

static inline void
icla_cgetvector_v1_internal(
    icla_int_t n,
    iclaFloatComplex_const_ptr dx_src, icla_int_t incx,
    iclaFloatComplex          *hy_dst, icla_int_t incy,
    const char* func, const char* file, int line )
{
    icla_getvector_v1_internal( n, sizeof(iclaFloatComplex),
                                 dx_src, incx,
                                 hy_dst, incy,
                                 func, file, line );
}

static inline void
icla_ccopyvector_v1_internal(
    icla_int_t n,
    iclaFloatComplex_const_ptr dx_src, icla_int_t incx,
    iclaFloatComplex_ptr       dy_dst, icla_int_t incy,
    const char* func, const char* file, int line )
{
    icla_copyvector_v1_internal( n, sizeof(iclaFloatComplex),
                                  dx_src, incx,
                                  dy_dst, incy,
                                  func, file, line );
}

#define icla_csetmatrix_v1(           m, n, hA_src, lda,  dB_dst, lddb ) \
        icla_csetmatrix_v1_internal(  m, n, hA_src, lda,  dB_dst, lddb, __func__, __FILE__, __LINE__ )

#define icla_cgetmatrix_v1(           m, n, dA_src, ldda, hB_dst, ldb  ) \
        icla_cgetmatrix_v1_internal(  m, n, dA_src, ldda, hB_dst, ldb,  __func__, __FILE__, __LINE__ )

#define icla_ccopymatrix_v1(          m, n, dA_src, ldda, dB_dst, lddb ) \
        icla_ccopymatrix_v1_internal( m, n, dA_src, ldda, dB_dst, lddb, __func__, __FILE__, __LINE__ )

static inline void
icla_csetmatrix_v1_internal(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex const    *hA_src, icla_int_t lda,
    iclaFloatComplex_ptr       dB_dst, icla_int_t lddb,
    const char* func, const char* file, int line )
{
    icla_setmatrix_v1_internal( m, n, sizeof(iclaFloatComplex),
                                 hA_src, lda,
                                 dB_dst, lddb,
                                 func, file, line );
}

static inline void
icla_cgetmatrix_v1_internal(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex_const_ptr dA_src, icla_int_t ldda,
    iclaFloatComplex          *hB_dst, icla_int_t ldb,
    const char* func, const char* file, int line )
{
    icla_getmatrix_v1_internal( m, n, sizeof(iclaFloatComplex),
                                 dA_src, ldda,
                                 hB_dst, ldb,
                                 func, file, line );
}

static inline void
icla_ccopymatrix_v1_internal(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex_const_ptr dA_src, icla_int_t ldda,
    iclaFloatComplex_ptr       dB_dst, icla_int_t lddb,
    const char* func, const char* file, int line )
{
    icla_copymatrix_v1_internal( m, n, sizeof(iclaFloatComplex),
                                  dA_src, ldda,
                                  dB_dst, lddb,
                                  func, file, line );
}

icla_int_t
icla_icamax_v1(
    icla_int_t n,
    iclaFloatComplex_const_ptr dx, icla_int_t incx );

icla_int_t
icla_icamin_v1(
    icla_int_t n,
    iclaFloatComplex_const_ptr dx, icla_int_t incx );

float
icla_scasum_v1(
    icla_int_t n,
    iclaFloatComplex_const_ptr dx, icla_int_t incx );

void
icla_caxpy_v1(
    icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex_ptr       dy, icla_int_t incy );

void
icla_ccopy_v1(
    icla_int_t n,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex_ptr       dy, icla_int_t incy );

iclaFloatComplex
icla_cdotc_v1(
    icla_int_t n,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex_const_ptr dy, icla_int_t incy );

iclaFloatComplex
icla_cdotu_v1(
    icla_int_t n,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex_const_ptr dy, icla_int_t incy );

float
icla_scnrm2_v1(
    icla_int_t n,
    iclaFloatComplex_const_ptr dx, icla_int_t incx );

void
icla_crot_v1(
    icla_int_t n,
    iclaFloatComplex_ptr dx, icla_int_t incx,
    iclaFloatComplex_ptr dy, icla_int_t incy,
    float dc, iclaFloatComplex ds );

void
icla_csrot_v1(
    icla_int_t n,
    iclaFloatComplex_ptr dx, icla_int_t incx,
    iclaFloatComplex_ptr dy, icla_int_t incy,
    float dc, float ds );

#ifdef ICLA_REAL
void
icla_crotm_v1(
    icla_int_t n,
    iclaFloat_ptr dx, icla_int_t incx,
    iclaFloat_ptr dy, icla_int_t incy,
    iclaFloat_const_ptr param );

void
icla_crotmg_v1(
    iclaFloat_ptr d1, iclaFloat_ptr       d2,
    iclaFloat_ptr x1, iclaFloat_const_ptr y1,
    iclaFloat_ptr param );
#endif

void
icla_cscal_v1(
    icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_ptr dx, icla_int_t incx );

void
icla_csscal_v1(
    icla_int_t n,
    float alpha,
    iclaFloatComplex_ptr dx, icla_int_t incx );

void
icla_cswap_v1(
    icla_int_t n,
    iclaFloatComplex_ptr dx, icla_int_t incx,
    iclaFloatComplex_ptr dy, icla_int_t incy );

void
icla_cgemv_v1(
    icla_trans_t transA,
    icla_int_t m, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr       dy, icla_int_t incy );

void
icla_cgerc_v1(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex_const_ptr dy, icla_int_t incy,
    iclaFloatComplex_ptr       dA, icla_int_t ldda );

void
icla_cgeru_v1(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex_const_ptr dy, icla_int_t incy,
    iclaFloatComplex_ptr       dA, icla_int_t ldda );

void
icla_chemv_v1(
    icla_uplo_t uplo,
    icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr       dy, icla_int_t incy );

void
icla_cher_v1(
    icla_uplo_t uplo,
    icla_int_t n,
    float alpha,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex_ptr       dA, icla_int_t ldda );

void
icla_cher2_v1(
    icla_uplo_t uplo,
    icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dx, icla_int_t incx,
    iclaFloatComplex_const_ptr dy, icla_int_t incy,
    iclaFloatComplex_ptr       dA, icla_int_t ldda );

void
icla_ctrmv_v1(
    icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t n,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr       dx, icla_int_t incx );

void
icla_ctrsv_v1(
    icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t n,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr       dx, icla_int_t incx );

void
icla_cgemm_v1(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_const_ptr dB, icla_int_t lddb,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr       dC, icla_int_t lddc );

void
icla_csymm_v1(
    icla_side_t side, icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_const_ptr dB, icla_int_t lddb,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr       dC, icla_int_t lddc );

void
icla_chemm_v1(
    icla_side_t side, icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_const_ptr dB, icla_int_t lddb,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr       dC, icla_int_t lddc );

void
icla_csyr2k_v1(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_const_ptr dB, icla_int_t lddb,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr       dC, icla_int_t lddc );

void
icla_cher2k_v1(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_const_ptr dB, icla_int_t lddb,
    float beta,
    iclaFloatComplex_ptr       dC, icla_int_t lddc );

void
icla_csyrk_v1(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex beta,
    iclaFloatComplex_ptr       dC, icla_int_t lddc );

void
icla_cherk_v1(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    float alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    float beta,
    iclaFloatComplex_ptr       dC, icla_int_t lddc );

void
icla_ctrmm_v1(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr       dB, icla_int_t lddb );

void
icla_ctrsm_v1(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    iclaFloatComplex alpha,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr       dB, icla_int_t lddb );

#ifdef __cplusplus
}
#endif

#undef ICLA_COMPLEX

#endif

