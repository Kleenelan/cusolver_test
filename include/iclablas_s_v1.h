
#ifndef ICLABLAS_S_V1_H
#define ICLABLAS_S_V1_H

#ifdef ICLA_NO_V1
#error "Since ICLA_NO_V1 is defined, icla.h is invalid; use icla_v2.h"
#endif

#include "icla_types.h"
#include "icla_copy_v1.h"

#define ICLA_REAL

#ifdef __cplusplus
extern "C" {
#endif

void
iclablas_stranspose_inplace_v1(
    icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda );

void
iclablas_stranspose_inplace_v1(
    icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda );

void
iclablas_stranspose_v1(
    icla_int_t m, icla_int_t n,
    iclaFloat_const_ptr dA,  icla_int_t ldda,
    iclaFloat_ptr       dAT, icla_int_t lddat );

void
iclablas_stranspose_v1(
    icla_int_t m, icla_int_t n,
    iclaFloat_const_ptr dA,  icla_int_t ldda,
    iclaFloat_ptr       dAT, icla_int_t lddat );

void
iclablas_sgetmatrix_transpose_v1(
    icla_int_t m, icla_int_t n,
    iclaFloat_const_ptr dAT,   icla_int_t ldda,
    float          *hA,    icla_int_t lda,
    iclaFloat_ptr       dwork, icla_int_t lddwork, icla_int_t nb );

void
iclablas_ssetmatrix_transpose_v1(
    icla_int_t m, icla_int_t n,
    const float *hA,    icla_int_t lda,
    iclaFloat_ptr    dAT,   icla_int_t ldda,
    iclaFloat_ptr    dwork, icla_int_t lddwork, icla_int_t nb );

void
iclablas_sprbt_v1(
    icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    iclaFloat_ptr du,
    iclaFloat_ptr dv );

void
iclablas_sprbt_mv_v1(
    icla_int_t n, icla_int_t nrhs,
    iclaFloat_ptr dv,
    iclaFloat_ptr db, icla_int_t lddb);

void
iclablas_sprbt_mtv_v1(
    icla_int_t n, icla_int_t nrhs,
    iclaFloat_ptr du,
    iclaFloat_ptr db, icla_int_t lddb);

void
icla_sgetmatrix_1D_col_bcyclic_v1(
    icla_int_t m, icla_int_t n,
    iclaFloat_const_ptr const dA[], icla_int_t ldda,
    float                *hA,   icla_int_t lda,
    icla_int_t ngpu, icla_int_t nb );

void
icla_ssetmatrix_1D_col_bcyclic_v1(
    icla_int_t m, icla_int_t n,
    const float *hA,   icla_int_t lda,
    iclaFloat_ptr    dA[], icla_int_t ldda,
    icla_int_t ngpu, icla_int_t nb );

void
icla_sgetmatrix_1D_row_bcyclic_v1(
    icla_int_t m, icla_int_t n,
    iclaFloat_const_ptr const dA[], icla_int_t ldda,
    float                *hA,   icla_int_t lda,
    icla_int_t ngpu, icla_int_t nb );

void
icla_ssetmatrix_1D_row_bcyclic_v1(
    icla_int_t m, icla_int_t n,
    const float *hA,   icla_int_t lda,
    iclaFloat_ptr    dA[], icla_int_t ldda,
    icla_int_t ngpu, icla_int_t nb );

void
iclablas_sgeadd_v1(
    icla_int_t m, icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_ptr       dB, icla_int_t lddb );

void
iclablas_sgeadd2_v1(
    icla_int_t m, icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    float beta,
    iclaFloat_ptr       dB, icla_int_t lddb );

void
iclablas_slacpy_v1(
    icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_ptr       dB, icla_int_t lddb );

void
iclablas_slacpy_conj_v1(
    icla_int_t n,
    iclaFloat_ptr dA1, icla_int_t lda1,
    iclaFloat_ptr dA2, icla_int_t lda2 );

void
iclablas_slacpy_sym_in_v1(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    icla_int_t *rows, icla_int_t *perm,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_ptr       dB, icla_int_t lddb );

void
iclablas_slacpy_sym_out_v1(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    icla_int_t *rows, icla_int_t *perm,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_ptr       dB, icla_int_t lddb );

float
iclablas_slange_v1(
    icla_norm_t norm,
    icla_int_t m, icla_int_t n,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_ptr dwork, icla_int_t lwork );

float
iclablas_slansy_v1(
    icla_norm_t norm, icla_uplo_t uplo,
    icla_int_t n,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_ptr dwork, icla_int_t lwork );

float
iclablas_slansy_v1(
    icla_norm_t norm, icla_uplo_t uplo,
    icla_int_t n,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_ptr dwork, icla_int_t lwork );

void
iclablas_slarfg_v1(
    icla_int_t n,
    iclaFloat_ptr dalpha,
    iclaFloat_ptr dx, icla_int_t incx,
    iclaFloat_ptr dtau );

void
iclablas_slascl_v1(
    icla_type_t type, icla_int_t kl, icla_int_t ku,
    float cfrom, float cto,
    icla_int_t m, icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    icla_int_t *info );

void
iclablas_slascl_2x2_v1(
    icla_type_t type, icla_int_t m,
    iclaFloat_const_ptr dW, icla_int_t lddw,
    iclaFloat_ptr       dA, icla_int_t ldda,
    icla_int_t *info );

void
iclablas_slascl2_v1(
    icla_type_t type,
    icla_int_t m, icla_int_t n,
    iclaFloat_const_ptr dD,
    iclaFloat_ptr dA, icla_int_t ldda,
    icla_int_t *info );

void
iclablas_slascl_diag_v1(
    icla_type_t type, icla_int_t m, icla_int_t n,
    iclaFloat_const_ptr dD, icla_int_t lddd,
    iclaFloat_ptr       dA, icla_int_t ldda,
    icla_int_t *info );

void
iclablas_slaset_v1(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    float offdiag, float diag,
    iclaFloat_ptr dA, icla_int_t ldda );

void
iclablas_slaset_band_v1(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n, icla_int_t k,
    float offdiag, float diag,
    iclaFloat_ptr dA, icla_int_t ldda );

void
iclablas_slaswp_v1(
    icla_int_t n,
    iclaFloat_ptr dAT, icla_int_t ldda,
    icla_int_t k1, icla_int_t k2,
    const icla_int_t *ipiv, icla_int_t inci );

void
iclablas_slaswp2_v1(
    icla_int_t n,
    iclaFloat_ptr dAT, icla_int_t ldda,
    icla_int_t k1, icla_int_t k2,
    iclaInt_const_ptr d_ipiv, icla_int_t inci );

void
iclablas_slaswp_sym_v1(
    icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    icla_int_t k1, icla_int_t k2,
    const icla_int_t *ipiv, icla_int_t inci );

void
iclablas_slaswpx_v1(
    icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldx, icla_int_t ldy,
    icla_int_t k1, icla_int_t k2,
    const icla_int_t *ipiv, icla_int_t inci );

void
iclablas_ssymmetrize_v1(
    icla_uplo_t uplo, icla_int_t m,
    iclaFloat_ptr dA, icla_int_t ldda );

void
iclablas_ssymmetrize_tiles_v1(
    icla_uplo_t uplo, icla_int_t m,
    iclaFloat_ptr dA, icla_int_t ldda,
    icla_int_t ntile, icla_int_t mstride, icla_int_t nstride );

void
iclablas_strtri_diag_v1(
    icla_uplo_t uplo, icla_diag_t diag, icla_int_t n,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_ptr d_dinvA );

void
iclablas_snrm2_adjust_v1(
    icla_int_t k,
    iclaFloat_ptr dxnorm,
    iclaFloat_ptr dc );

void
iclablas_snrm2_check_v1(
    icla_int_t m, icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    iclaFloat_ptr dxnorm,
    iclaFloat_ptr dlsticc );

void
iclablas_snrm2_cols_v1(
    icla_int_t m, icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    iclaFloat_ptr dxnorm );

void
iclablas_snrm2_row_check_adjust_v1(
    icla_int_t k, float tol,
    iclaFloat_ptr dxnorm,
    iclaFloat_ptr dxnorm2,
    iclaFloat_ptr dC, icla_int_t lddc,
    iclaFloat_ptr dlsticc );

icla_int_t
icla_slarfb_gpu_v1(
    icla_side_t side, icla_trans_t trans, icla_direct_t direct, icla_storev_t storev,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaFloat_const_ptr dV, icla_int_t lddv,
    iclaFloat_const_ptr dT, icla_int_t lddt,
    iclaFloat_ptr dC,       icla_int_t lddc,
    iclaFloat_ptr dwork,    icla_int_t ldwork );

icla_int_t
icla_slarfb_gpu_gemm_v1(
    icla_side_t side, icla_trans_t trans, icla_direct_t direct, icla_storev_t storev,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaFloat_const_ptr dV, icla_int_t lddv,
    iclaFloat_const_ptr dT, icla_int_t lddt,
    iclaFloat_ptr dC,       icla_int_t lddc,
    iclaFloat_ptr dwork,    icla_int_t ldwork,
    iclaFloat_ptr dworkvt,  icla_int_t ldworkvt );

void
icla_slarfbx_gpu_v1(
    icla_int_t m, icla_int_t k,
    iclaFloat_ptr V,  icla_int_t ldv,
    iclaFloat_ptr dT, icla_int_t ldt,
    iclaFloat_ptr c,
    iclaFloat_ptr dwork );

void
icla_slarfg_gpu_v1(
    icla_int_t n,
    iclaFloat_ptr dx0,
    iclaFloat_ptr dx,
    iclaFloat_ptr dtau,
    iclaFloat_ptr        dxnorm,
    iclaFloat_ptr dAkk );

void
icla_slarfgtx_gpu_v1(
    icla_int_t n,
    iclaFloat_ptr dx0,
    iclaFloat_ptr dx,
    iclaFloat_ptr dtau,
    iclaFloat_ptr        dxnorm,
    iclaFloat_ptr dA, icla_int_t iter,
    iclaFloat_ptr V,  icla_int_t ldv,
    iclaFloat_ptr T,  icla_int_t ldt,
    iclaFloat_ptr dwork );

void
icla_slarfgx_gpu_v1(
    icla_int_t n,
    iclaFloat_ptr dx0,
    iclaFloat_ptr dx,
    iclaFloat_ptr dtau,
    iclaFloat_ptr        dxnorm,
    iclaFloat_ptr dA, icla_int_t iter );

void
icla_slarfx_gpu_v1(
    icla_int_t m, icla_int_t n,
    iclaFloat_ptr v,
    iclaFloat_ptr tau,
    iclaFloat_ptr C,  icla_int_t ldc,
    iclaFloat_ptr        xnorm,
    iclaFloat_ptr dT, icla_int_t iter,
    iclaFloat_ptr work );

void
iclablas_saxpycp_v1(
    icla_int_t m,
    iclaFloat_ptr dr,
    iclaFloat_ptr dx,
    iclaFloat_const_ptr db );

void
iclablas_sswap_v1(
    icla_int_t n,
    iclaFloat_ptr dx, icla_int_t incx,
    iclaFloat_ptr dy, icla_int_t incy );

void
iclablas_sswapblk_v1(
    icla_order_t order,
    icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    iclaFloat_ptr dB, icla_int_t lddb,
    icla_int_t i1, icla_int_t i2,
    const icla_int_t *ipiv, icla_int_t inci,
    icla_int_t offset );

void
iclablas_sswapdblk_v1(
    icla_int_t n, icla_int_t nb,
    iclaFloat_ptr dA, icla_int_t ldda, icla_int_t inca,
    iclaFloat_ptr dB, icla_int_t lddb, icla_int_t incb );

void
iclablas_sgemv_v1(
    icla_trans_t trans, icla_int_t m, icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_const_ptr dx, icla_int_t incx,
    float beta,
    iclaFloat_ptr       dy, icla_int_t incy );

void
iclablas_sgemv_conj_v1(
    icla_int_t m, icla_int_t n, float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_const_ptr dx, icla_int_t incx,
    float beta,
    iclaFloat_ptr dy, icla_int_t incy );

icla_int_t
iclablas_ssymv_v1(
    icla_uplo_t uplo, icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_const_ptr dx, icla_int_t incx,
    float beta,
    iclaFloat_ptr       dy, icla_int_t incy );

icla_int_t
iclablas_ssymv_v1(
    icla_uplo_t uplo, icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_const_ptr dx, icla_int_t incx,
    float beta,
    iclaFloat_ptr       dy, icla_int_t incy );

void
iclablas_sgemm_v1(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_const_ptr dB, icla_int_t lddb,
    float beta,
    iclaFloat_ptr       dC, icla_int_t lddc );

void
iclablas_sgemm_reduce_v1(
    icla_int_t m, icla_int_t n, icla_int_t k,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_const_ptr dB, icla_int_t lddb,
    float beta,
    iclaFloat_ptr       dC, icla_int_t lddc );

void
iclablas_ssymm_v1(
    icla_side_t side, icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_const_ptr dB, icla_int_t lddb,
    float beta,
    iclaFloat_ptr       dC, icla_int_t lddc );

void
iclablas_ssymm_v1(
    icla_side_t side, icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_const_ptr dB, icla_int_t lddb,
    float beta,
    iclaFloat_ptr       dC, icla_int_t lddc );

void
iclablas_ssyr2k_v1(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_const_ptr dB, icla_int_t lddb,
    float beta,
    iclaFloat_ptr       dC, icla_int_t lddc );

void
iclablas_ssyr2k_v1(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_const_ptr dB, icla_int_t lddb,
    float  beta,
    iclaFloat_ptr       dC, icla_int_t lddc );

void
iclablas_ssyrk_v1(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    float beta,
    iclaFloat_ptr       dC, icla_int_t lddc );

void
iclablas_ssyrk_v1(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    float  alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    float  beta,
    iclaFloat_ptr       dC, icla_int_t lddc );

void
iclablas_strsm_v1(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_ptr       dB, icla_int_t lddb );

void
iclablas_strsm_outofplace_v1(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_ptr       dB, icla_int_t lddb,
    iclaFloat_ptr       dX, icla_int_t lddx,
    icla_int_t flag,
    iclaFloat_ptr d_dinvA, icla_int_t dinvA_length );

void
iclablas_strsm_work_v1(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_ptr       dB, icla_int_t lddb,
    iclaFloat_ptr       dX, icla_int_t lddx,
    icla_int_t flag,
    iclaFloat_ptr d_dinvA, icla_int_t dinvA_length );

#define icla_ssetvector_v1(           n, hx_src, incx, dy_dst, incy ) \
        icla_ssetvector_v1_internal(  n, hx_src, incx, dy_dst, incy, __func__, __FILE__, __LINE__ )

#define icla_sgetvector_v1(           n, dx_src, incx, hy_dst, incy ) \
        icla_sgetvector_v1_internal(  n, dx_src, incx, hy_dst, incy, __func__, __FILE__, __LINE__ )

#define icla_scopyvector_v1(          n, dx_src, incx, dy_dst, incy ) \
        icla_scopyvector_v1_internal( n, dx_src, incx, dy_dst, incy, __func__, __FILE__, __LINE__ )

static inline void
icla_ssetvector_v1_internal(
    icla_int_t n,
    float const    *hx_src, icla_int_t incx,
    iclaFloat_ptr       dy_dst, icla_int_t incy,
    const char* func, const char* file, int line )
{
    icla_setvector_v1_internal( n, sizeof(float),
                                 hx_src, incx,
                                 dy_dst, incy,
                                 func, file, line );
}

static inline void
icla_sgetvector_v1_internal(
    icla_int_t n,
    iclaFloat_const_ptr dx_src, icla_int_t incx,
    float          *hy_dst, icla_int_t incy,
    const char* func, const char* file, int line )
{
    icla_getvector_v1_internal( n, sizeof(float),
                                 dx_src, incx,
                                 hy_dst, incy,
                                 func, file, line );
}

static inline void
icla_scopyvector_v1_internal(
    icla_int_t n,
    iclaFloat_const_ptr dx_src, icla_int_t incx,
    iclaFloat_ptr       dy_dst, icla_int_t incy,
    const char* func, const char* file, int line )
{
    icla_copyvector_v1_internal( n, sizeof(float),
                                  dx_src, incx,
                                  dy_dst, incy,
                                  func, file, line );
}

#define icla_ssetmatrix_v1(           m, n, hA_src, lda,  dB_dst, lddb ) \
        icla_ssetmatrix_v1_internal(  m, n, hA_src, lda,  dB_dst, lddb, __func__, __FILE__, __LINE__ )

#define icla_sgetmatrix_v1(           m, n, dA_src, ldda, hB_dst, ldb  ) \
        icla_sgetmatrix_v1_internal(  m, n, dA_src, ldda, hB_dst, ldb,  __func__, __FILE__, __LINE__ )

#define icla_scopymatrix_v1(          m, n, dA_src, ldda, dB_dst, lddb ) \
        icla_scopymatrix_v1_internal( m, n, dA_src, ldda, dB_dst, lddb, __func__, __FILE__, __LINE__ )

static inline void
icla_ssetmatrix_v1_internal(
    icla_int_t m, icla_int_t n,
    float const    *hA_src, icla_int_t lda,
    iclaFloat_ptr       dB_dst, icla_int_t lddb,
    const char* func, const char* file, int line )
{
    icla_setmatrix_v1_internal( m, n, sizeof(float),
                                 hA_src, lda,
                                 dB_dst, lddb,
                                 func, file, line );
}

static inline void
icla_sgetmatrix_v1_internal(
    icla_int_t m, icla_int_t n,
    iclaFloat_const_ptr dA_src, icla_int_t ldda,
    float          *hB_dst, icla_int_t ldb,
    const char* func, const char* file, int line )
{
    icla_getmatrix_v1_internal( m, n, sizeof(float),
                                 dA_src, ldda,
                                 hB_dst, ldb,
                                 func, file, line );
}

static inline void
icla_scopymatrix_v1_internal(
    icla_int_t m, icla_int_t n,
    iclaFloat_const_ptr dA_src, icla_int_t ldda,
    iclaFloat_ptr       dB_dst, icla_int_t lddb,
    const char* func, const char* file, int line )
{
    icla_copymatrix_v1_internal( m, n, sizeof(float),
                                  dA_src, ldda,
                                  dB_dst, lddb,
                                  func, file, line );
}

icla_int_t
icla_isamax_v1(
    icla_int_t n,
    iclaFloat_const_ptr dx, icla_int_t incx );

icla_int_t
icla_isamin_v1(
    icla_int_t n,
    iclaFloat_const_ptr dx, icla_int_t incx );

float
icla_sasum_v1(
    icla_int_t n,
    iclaFloat_const_ptr dx, icla_int_t incx );

void
icla_saxpy_v1(
    icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dx, icla_int_t incx,
    iclaFloat_ptr       dy, icla_int_t incy );

void
icla_scopy_v1(
    icla_int_t n,
    iclaFloat_const_ptr dx, icla_int_t incx,
    iclaFloat_ptr       dy, icla_int_t incy );

float
icla_sdot_v1(
    icla_int_t n,
    iclaFloat_const_ptr dx, icla_int_t incx,
    iclaFloat_const_ptr dy, icla_int_t incy );

float
icla_sdot_v1(
    icla_int_t n,
    iclaFloat_const_ptr dx, icla_int_t incx,
    iclaFloat_const_ptr dy, icla_int_t incy );

float
icla_snrm2_v1(
    icla_int_t n,
    iclaFloat_const_ptr dx, icla_int_t incx );

void
icla_srot_v1(
    icla_int_t n,
    iclaFloat_ptr dx, icla_int_t incx,
    iclaFloat_ptr dy, icla_int_t incy,
    float dc, float ds );

void
icla_srot_v1(
    icla_int_t n,
    iclaFloat_ptr dx, icla_int_t incx,
    iclaFloat_ptr dy, icla_int_t incy,
    float dc, float ds );

#ifdef ICLA_REAL
void
icla_srotm_v1(
    icla_int_t n,
    iclaFloat_ptr dx, icla_int_t incx,
    iclaFloat_ptr dy, icla_int_t incy,
    iclaFloat_const_ptr param );

void
icla_srotmg_v1(
    iclaFloat_ptr d1, iclaFloat_ptr       d2,
    iclaFloat_ptr x1, iclaFloat_const_ptr y1,
    iclaFloat_ptr param );
#endif

void
icla_sscal_v1(
    icla_int_t n,
    float alpha,
    iclaFloat_ptr dx, icla_int_t incx );

void
icla_sscal_v1(
    icla_int_t n,
    float alpha,
    iclaFloat_ptr dx, icla_int_t incx );

void
icla_sswap_v1(
    icla_int_t n,
    iclaFloat_ptr dx, icla_int_t incx,
    iclaFloat_ptr dy, icla_int_t incy );

void
icla_sgemv_v1(
    icla_trans_t transA,
    icla_int_t m, icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_const_ptr dx, icla_int_t incx,
    float beta,
    iclaFloat_ptr       dy, icla_int_t incy );

void
icla_sger_v1(
    icla_int_t m, icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dx, icla_int_t incx,
    iclaFloat_const_ptr dy, icla_int_t incy,
    iclaFloat_ptr       dA, icla_int_t ldda );

void
icla_sger_v1(
    icla_int_t m, icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dx, icla_int_t incx,
    iclaFloat_const_ptr dy, icla_int_t incy,
    iclaFloat_ptr       dA, icla_int_t ldda );

void
icla_ssymv_v1(
    icla_uplo_t uplo,
    icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_const_ptr dx, icla_int_t incx,
    float beta,
    iclaFloat_ptr       dy, icla_int_t incy );

void
icla_ssyr_v1(
    icla_uplo_t uplo,
    icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dx, icla_int_t incx,
    iclaFloat_ptr       dA, icla_int_t ldda );

void
icla_ssyr2_v1(
    icla_uplo_t uplo,
    icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dx, icla_int_t incx,
    iclaFloat_const_ptr dy, icla_int_t incy,
    iclaFloat_ptr       dA, icla_int_t ldda );

void
icla_strmv_v1(
    icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t n,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_ptr       dx, icla_int_t incx );

void
icla_strsv_v1(
    icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t n,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_ptr       dx, icla_int_t incx );

void
icla_sgemm_v1(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_const_ptr dB, icla_int_t lddb,
    float beta,
    iclaFloat_ptr       dC, icla_int_t lddc );

void
icla_ssymm_v1(
    icla_side_t side, icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_const_ptr dB, icla_int_t lddb,
    float beta,
    iclaFloat_ptr       dC, icla_int_t lddc );

void
icla_ssymm_v1(
    icla_side_t side, icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_const_ptr dB, icla_int_t lddb,
    float beta,
    iclaFloat_ptr       dC, icla_int_t lddc );

void
icla_ssyr2k_v1(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_const_ptr dB, icla_int_t lddb,
    float beta,
    iclaFloat_ptr       dC, icla_int_t lddc );

void
icla_ssyr2k_v1(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_const_ptr dB, icla_int_t lddb,
    float beta,
    iclaFloat_ptr       dC, icla_int_t lddc );

void
icla_ssyrk_v1(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    float beta,
    iclaFloat_ptr       dC, icla_int_t lddc );

void
icla_ssyrk_v1(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    float beta,
    iclaFloat_ptr       dC, icla_int_t lddc );

void
icla_strmm_v1(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_ptr       dB, icla_int_t lddb );

void
icla_strsm_v1(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    float alpha,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    iclaFloat_ptr       dB, icla_int_t lddb );

#ifdef __cplusplus
}
#endif

#undef ICLA_REAL

#endif

