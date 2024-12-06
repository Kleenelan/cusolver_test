
#ifndef ICLABLAS_Z_H
#define ICLABLAS_Z_H

#include "icla_types.h"
#include "icla_copy.h"

#define ICLA_COMPLEX

#ifdef __cplusplus
extern "C" {
#endif

void
iclablas_ztranspose_inplace(
    icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    icla_queue_t queue );

void
iclablas_ztranspose_conj_inplace(
    icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    icla_queue_t queue );

void
iclablas_ztranspose(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex_const_ptr dA,  icla_int_t ldda,
    iclaDoubleComplex_ptr       dAT, icla_int_t lddat,
    icla_queue_t queue );

void
iclablas_ztranspose_conj(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex_const_ptr dA,  icla_int_t ldda,
    iclaDoubleComplex_ptr       dAT, icla_int_t lddat,
    icla_queue_t queue );

void
iclablas_zgetmatrix_transpose(
    icla_int_t m, icla_int_t n, icla_int_t nb,
    iclaDoubleComplex_const_ptr dAT,   icla_int_t ldda,
    iclaDoubleComplex          *hA,    icla_int_t lda,
    iclaDoubleComplex_ptr       dwork, icla_int_t lddw,
    icla_queue_t queues[2] );

void
iclablas_zsetmatrix_transpose(
    icla_int_t m, icla_int_t n, icla_int_t nb,
    const iclaDoubleComplex *hA,    icla_int_t lda,
    iclaDoubleComplex_ptr    dAT,   icla_int_t ldda,
    iclaDoubleComplex_ptr    dwork, icla_int_t lddw,
    icla_queue_t queues[2] );

void
iclablas_zprbt(
    icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr du,
    iclaDoubleComplex_ptr dv,
    icla_queue_t queue );

void
iclablas_zprbt_mv(
    icla_int_t n, icla_int_t nrhs,
    iclaDoubleComplex_ptr dv,
    iclaDoubleComplex_ptr db, icla_int_t lddb,
    icla_queue_t queue );

void
iclablas_zprbt_mtv(
    icla_int_t n, icla_int_t nrhs,
    iclaDoubleComplex_ptr du,
    iclaDoubleComplex_ptr db, icla_int_t lddb,
    icla_queue_t queue );

void
icla_zgetmatrix_1D_col_bcyclic(
    icla_int_t ngpu,
    icla_int_t m, icla_int_t n, icla_int_t nb,
    iclaDoubleComplex_const_ptr const dA[], icla_int_t ldda,
    iclaDoubleComplex                *hA,   icla_int_t lda,
    icla_queue_t queue[] );

void
icla_zsetmatrix_1D_col_bcyclic(
    icla_int_t ngpu,
    icla_int_t m, icla_int_t n, icla_int_t nb,
    const iclaDoubleComplex *hA,   icla_int_t lda,
    iclaDoubleComplex_ptr    dA[], icla_int_t ldda,
    icla_queue_t queue[] );

void
icla_zgetmatrix_1D_row_bcyclic(
    icla_int_t ngpu,
    icla_int_t m, icla_int_t n, icla_int_t nb,
    iclaDoubleComplex_const_ptr const dA[], icla_int_t ldda,
    iclaDoubleComplex                *hA,   icla_int_t lda,
    icla_queue_t queue[] );

void
icla_zsetmatrix_1D_row_bcyclic(
    icla_int_t ngpu,
    icla_int_t m, icla_int_t n, icla_int_t nb,
    const iclaDoubleComplex *hA,   icla_int_t lda,
    iclaDoubleComplex_ptr    dA[], icla_int_t ldda,
    icla_queue_t queue[] );

void
iclablas_zgetmatrix_transpose_mgpu(
    icla_int_t ngpu,
    icla_int_t m, icla_int_t n, icla_int_t nb,
    iclaDoubleComplex_const_ptr const dAT[],    icla_int_t ldda,
    iclaDoubleComplex                *hA,       icla_int_t lda,
    iclaDoubleComplex_ptr             dwork[],  icla_int_t lddw,
    icla_queue_t queues[][2] );

void
iclablas_zsetmatrix_transpose_mgpu(
    icla_int_t ngpu,
    icla_int_t m, icla_int_t n, icla_int_t nb,
    const iclaDoubleComplex *hA,      icla_int_t lda,
    iclaDoubleComplex_ptr    dAT[],   icla_int_t ldda,
    iclaDoubleComplex_ptr    dwork[], icla_int_t lddw,
    icla_queue_t queues[][2] );



icla_int_t
icla_zhtodhe(
    icla_int_t ngpu, icla_uplo_t uplo, icla_int_t n, icla_int_t nb,
    iclaDoubleComplex     *A,   icla_int_t lda,
    iclaDoubleComplex_ptr dA[], icla_int_t ldda,
    icla_queue_t queues[][10],
    icla_int_t *info );



icla_int_t
icla_zhtodpo(
    icla_int_t ngpu, icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    icla_int_t off_i, icla_int_t off_j, icla_int_t nb,
    iclaDoubleComplex     *A,   icla_int_t lda,
    iclaDoubleComplex_ptr dA[], icla_int_t ldda,
    icla_queue_t queues[][3],
    icla_int_t *info );



icla_int_t
icla_zdtohpo(
    icla_int_t ngpu, icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    icla_int_t off_i, icla_int_t off_j, icla_int_t nb, icla_int_t NB,
    iclaDoubleComplex     *A,   icla_int_t lda,
    iclaDoubleComplex_ptr dA[], icla_int_t ldda,
    icla_queue_t queues[][3],
    icla_int_t *info );

void
iclablas_zhemm_mgpu(
    icla_side_t side, icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_ptr dA[],    icla_int_t ldda,  icla_int_t offset,
    iclaDoubleComplex_ptr dB[],    icla_int_t lddb,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr dC[],    icla_int_t lddc,
    iclaDoubleComplex_ptr dwork[], icla_int_t dworksiz,


    icla_int_t ngpu, icla_int_t nb,
    icla_queue_t queues[][20], icla_int_t nqueue,
    icla_event_t events[][IclaMaxGPUs*IclaMaxGPUs+10], icla_int_t nevents,
    icla_int_t gnode[IclaMaxGPUs][IclaMaxGPUs+2], icla_int_t ncmplx );

icla_int_t
iclablas_zhemv_mgpu(
    icla_uplo_t uplo,
    icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr const d_lA[], icla_int_t ldda, icla_int_t offset,
    iclaDoubleComplex_const_ptr dx,           icla_int_t incx,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr    dy,              icla_int_t incy,
    iclaDoubleComplex       *hwork,           icla_int_t lhwork,
    iclaDoubleComplex_ptr    dwork[],         icla_int_t ldwork,
    icla_int_t ngpu,
    icla_int_t nb,
    icla_queue_t queues[] );

icla_int_t
iclablas_zhemv_mgpu_sync(
    icla_uplo_t uplo,
    icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr const d_lA[], icla_int_t ldda, icla_int_t offset,
    iclaDoubleComplex_const_ptr dx,           icla_int_t incx,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr    dy,              icla_int_t incy,
    iclaDoubleComplex       *hwork,           icla_int_t lhwork,
    iclaDoubleComplex_ptr    dwork[],         icla_int_t ldwork,
    icla_int_t ngpu,
    icla_int_t nb,
    icla_queue_t queues[] );

icla_int_t
icla_zhetrs_gpu(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nrhs,
    iclaDoubleComplex *dA, icla_int_t ldda,
    icla_int_t *ipiv,
    iclaDoubleComplex *dB, icla_int_t lddb,
    icla_int_t *info,
    icla_queue_t queue );


void
icla_zher2k_mgpu(
    icla_int_t ngpu,
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t nb, icla_int_t n, icla_int_t k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_ptr dB[], icla_int_t lddb, icla_int_t b_offset,
    double beta,
    iclaDoubleComplex_ptr dC[], icla_int_t lddc, icla_int_t c_offset,
    icla_int_t nqueue, icla_queue_t queues[][10] );

void
iclablas_zher2k_mgpu2(
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t n, icla_int_t k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_ptr dA[], icla_int_t ldda, icla_int_t a_offset,
    iclaDoubleComplex_ptr dB[], icla_int_t lddb, icla_int_t b_offset,
    double beta,
    iclaDoubleComplex_ptr dC[], icla_int_t lddc, icla_int_t c_offset,
    icla_int_t ngpu, icla_int_t nb,
    icla_queue_t queues[][20], icla_int_t nqueue );


void
icla_zherk_mgpu(
    icla_int_t ngpu,
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t nb, icla_int_t n, icla_int_t k,
    double alpha,
    iclaDoubleComplex_ptr dB[], icla_int_t lddb, icla_int_t b_offset,
    double beta,
    iclaDoubleComplex_ptr dC[], icla_int_t lddc, icla_int_t c_offset,
    icla_int_t nqueue, icla_queue_t queues[][10] );


void
icla_zherk_mgpu2(
    icla_int_t ngpu,
    icla_uplo_t uplo, icla_trans_t trans, icla_int_t nb, icla_int_t n, icla_int_t k,
    double alpha,
    iclaDoubleComplex_ptr dB[], icla_int_t lddb, icla_int_t b_offset,
    double beta,
    iclaDoubleComplex_ptr dC[], icla_int_t lddc, icla_int_t c_offset,
    icla_int_t nqueue, icla_queue_t queues[][10] );


icla_int_t
iclablas_zdiinertia(
    icla_int_t n,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    int *dneig,
    icla_queue_t queue );

void
iclablas_zgeadd(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr       dB, icla_int_t lddb,
    icla_queue_t queue );

void
iclablas_zgeadd2(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dB, icla_int_t lddb,
    icla_queue_t queue );

void
iclablas_zgeam(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex beta,
    iclaDoubleComplex_const_ptr dB, icla_int_t lddb,
    iclaDoubleComplex_ptr dC, icla_int_t lddc,
    icla_queue_t queue );

icla_int_t
iclablas_zheinertia(
    icla_uplo_t uplo,
    icla_int_t n,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    icla_int_t *ipiv,
    int *dneig,
    icla_queue_t queue );

void
iclablas_zlacpy(
    icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr       dB, icla_int_t lddb,
    icla_queue_t queue );

void
iclablas_zlacpy_conj(
    icla_int_t n,
    iclaDoubleComplex_ptr dA1, icla_int_t lda1,
    iclaDoubleComplex_ptr dA2, icla_int_t lda2,
    icla_queue_t queue );

void
iclablas_zlacpy_sym_in(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    icla_int_t *rows, icla_int_t *perm,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr       dB, icla_int_t lddb,
    icla_queue_t queue );

void
iclablas_zlacpy_sym_out(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    icla_int_t *rows, icla_int_t *perm,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr       dB, icla_int_t lddb,
    icla_queue_t queue );

double
iclablas_zlange(
    icla_norm_t norm,
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDouble_ptr dwork, icla_int_t lwork,
    icla_queue_t queue );

double
iclablas_zlanhe(
    icla_norm_t norm, icla_uplo_t uplo,
    icla_int_t n,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDouble_ptr dwork, icla_int_t lwork,
    icla_queue_t queue );

void
iclablas_zlarfg(
    icla_int_t n,
    iclaDoubleComplex_ptr dalpha,
    iclaDoubleComplex_ptr dx, icla_int_t incx,
    iclaDoubleComplex_ptr dtau,
    icla_queue_t queue );

void
iclablas_zlascl(
    icla_type_t type, icla_int_t kl, icla_int_t ku,
    double cfrom, double cto,
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    icla_queue_t queue,
    icla_int_t *info );

void
iclablas_zlascl_2x2(
    icla_type_t type, icla_int_t m,
    iclaDoubleComplex_const_ptr dW, icla_int_t lddw,
    iclaDoubleComplex_ptr       dA, icla_int_t ldda,
    icla_queue_t queue,
    icla_int_t *info );

void
iclablas_zlascl2(
    icla_type_t type,
    icla_int_t m, icla_int_t n,
    iclaDouble_const_ptr dD,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    icla_queue_t queue,
    icla_int_t *info );

void
iclablas_zlascl_diag(
    icla_type_t type, icla_int_t m, icla_int_t n,
    iclaDoubleComplex_const_ptr dD, icla_int_t lddd,
    iclaDoubleComplex_ptr       dA, icla_int_t ldda,
    icla_queue_t queue,
    icla_int_t *info );

void
iclablas_zlaset(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    iclaDoubleComplex offdiag, iclaDoubleComplex diag,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    icla_queue_t queue );

void
iclablas_zlaset_band(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDoubleComplex offdiag, iclaDoubleComplex diag,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    icla_queue_t queue );

void
iclablas_zlaswp(
    icla_int_t n,
    iclaDoubleComplex_ptr dAT, icla_int_t ldda,
    icla_int_t k1, icla_int_t k2,
    const icla_int_t *ipiv, icla_int_t inci,
    icla_queue_t queue );

void
iclablas_zlaswp2(
    icla_int_t n,
    iclaDoubleComplex_ptr dAT, icla_int_t ldda,
    icla_int_t k1, icla_int_t k2,
    iclaInt_const_ptr d_ipiv, icla_int_t inci,
    icla_queue_t queue );

void
iclablas_zlaswp_sym(
    icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    icla_int_t k1, icla_int_t k2,
    const icla_int_t *ipiv, icla_int_t inci,
    icla_queue_t queue );

void
iclablas_zlaswpx(
    icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldx, icla_int_t ldy,
    icla_int_t k1, icla_int_t k2,
    const icla_int_t *ipiv, icla_int_t inci,
    icla_queue_t queue );

void
icla_zlaswp_rowparallel_native(
    icla_int_t n,
    iclaDoubleComplex* input, icla_int_t ldi,
    iclaDoubleComplex* output, icla_int_t ldo,
    icla_int_t k1, icla_int_t k2,
    icla_int_t *pivinfo,
    icla_queue_t queue);

void
icla_zlaswp_columnserial(
    icla_int_t n, iclaDoubleComplex_ptr dA, icla_int_t lda,
    icla_int_t k1, icla_int_t k2,
    icla_int_t *dipiv, icla_queue_t queue);

void
iclablas_zsymmetrize(
    icla_uplo_t uplo, icla_int_t m,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    icla_queue_t queue );

void
iclablas_zsymmetrize_tiles(
    icla_uplo_t uplo, icla_int_t m,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    icla_int_t ntile, icla_int_t mstride, icla_int_t nstride,
    icla_queue_t queue );

void
iclablas_ztrtri_diag(
    icla_uplo_t uplo, icla_diag_t diag, icla_int_t n,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr d_dinvA,
    icla_queue_t queue );

icla_int_t
icla_zlarfb_gpu(
    icla_side_t side, icla_trans_t trans, icla_direct_t direct, icla_storev_t storev,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDoubleComplex_const_ptr dV, icla_int_t lddv,
    iclaDoubleComplex_const_ptr dT, icla_int_t lddt,
    iclaDoubleComplex_ptr dC,       icla_int_t lddc,
    iclaDoubleComplex_ptr dwork,    icla_int_t ldwork,
    icla_queue_t queue );

icla_int_t
icla_zlarfb_gpu_gemm(
    icla_side_t side, icla_trans_t trans, icla_direct_t direct, icla_storev_t storev,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDoubleComplex_const_ptr dV, icla_int_t lddv,
    iclaDoubleComplex_const_ptr dT, icla_int_t lddt,
    iclaDoubleComplex_ptr dC,       icla_int_t lddc,
    iclaDoubleComplex_ptr dwork,    icla_int_t ldwork,
    iclaDoubleComplex_ptr dworkvt,  icla_int_t ldworkvt,
    icla_queue_t queue );

void
icla_zlarfbx_gpu(
    icla_int_t m, icla_int_t k,
    iclaDoubleComplex_ptr V,  icla_int_t ldv,
    iclaDoubleComplex_ptr dT, icla_int_t ldt,
    iclaDoubleComplex_ptr c,
    iclaDoubleComplex_ptr dwork,
    icla_queue_t queue );

void
icla_zlarfg_gpu(
    icla_int_t n,
    iclaDoubleComplex_ptr dx0,
    iclaDoubleComplex_ptr dx,
    iclaDoubleComplex_ptr dtau,
    iclaDouble_ptr        dxnorm,
    iclaDoubleComplex_ptr dAkk,
    icla_queue_t queue );

void
icla_zlarfgtx_gpu(
    icla_int_t n,
    iclaDoubleComplex_ptr dx0,
    iclaDoubleComplex_ptr dx,
    iclaDoubleComplex_ptr dtau,
    iclaDouble_ptr        dxnorm,
    iclaDoubleComplex_ptr dA, icla_int_t iter,
    iclaDoubleComplex_ptr V,  icla_int_t ldv,
    iclaDoubleComplex_ptr T,  icla_int_t ldt,
    iclaDoubleComplex_ptr dwork,
    icla_queue_t queue );

void
icla_zlarfgx_gpu(
    icla_int_t n,
    iclaDoubleComplex_ptr dx0,
    iclaDoubleComplex_ptr dx,
    iclaDoubleComplex_ptr dtau,
    iclaDouble_ptr        dxnorm,
    iclaDoubleComplex_ptr dA, icla_int_t iter,
    icla_queue_t queue );

void
icla_zlarfx_gpu(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex_ptr v,
    iclaDoubleComplex_ptr tau,
    iclaDoubleComplex_ptr C, icla_int_t ldc,
    iclaDouble_ptr        xnorm,
    iclaDoubleComplex_ptr dT, icla_int_t iter,
    iclaDoubleComplex_ptr work,
    icla_queue_t queue );

void
iclablas_zaxpycp(
    icla_int_t m,
    iclaDoubleComplex_ptr dr,
    iclaDoubleComplex_ptr dx,
    iclaDoubleComplex_const_ptr db,
    icla_queue_t queue );

void
iclablas_zswap(
    icla_int_t n,
    iclaDoubleComplex_ptr dx, icla_int_t incx,
    iclaDoubleComplex_ptr dy, icla_int_t incy,
    icla_queue_t queue );

void
iclablas_zswapblk(
    icla_order_t order,
    icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr dB, icla_int_t lddb,
    icla_int_t i1, icla_int_t i2,
    const icla_int_t *ipiv, icla_int_t inci,
    icla_int_t offset,
    icla_queue_t queue );

void
iclablas_zswapdblk(
    icla_int_t n, icla_int_t nb,
    iclaDoubleComplex_ptr dA, icla_int_t ldda, icla_int_t inca,
    iclaDoubleComplex_ptr dB, icla_int_t lddb, icla_int_t incb,
    icla_queue_t queue );

void
iclablas_dznrm2_adjust(
    icla_int_t k,
    iclaDouble_ptr dxnorm,
    iclaDoubleComplex_ptr dc,
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
iclablas_dznrm2_check(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    iclaDouble_ptr dxnorm,
    iclaDouble_ptr dlsticc,
    icla_queue_t queue );

void
iclablas_dznrm2_cols(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    iclaDouble_ptr dxnorm,
    icla_queue_t queue );

void
iclablas_dznrm2_row_check_adjust(
    icla_int_t k, double tol,
    iclaDouble_ptr dxnorm,
    iclaDouble_ptr dxnorm2,
    iclaDoubleComplex_ptr dC, icla_int_t lddc,
    iclaDouble_ptr dlsticc,
    icla_queue_t queue );

void
iclablas_ztrsv(
    icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t n,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr       db, icla_int_t incb,
    icla_queue_t queue );


void
iclablas_ztrsv_outofplace(
    icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t n,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr db,       icla_int_t incb,
    iclaDoubleComplex_ptr dx,
    icla_queue_t queue,
    icla_int_t flag );

void
iclablas_zgemv(
    icla_trans_t trans, icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr dy, icla_int_t incy,
    icla_queue_t queue );

void
iclablas_zgemv_conj(
    icla_int_t m, icla_int_t n, iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr dy, icla_int_t incy,
    icla_queue_t queue );

icla_int_t
iclablas_zhemv(
    icla_uplo_t uplo, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dy, icla_int_t incy,
    icla_queue_t queue );

icla_int_t
iclablas_zsymv(
    icla_uplo_t uplo, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dy, icla_int_t incy,
    icla_queue_t queue );


icla_int_t
iclablas_zhemv_work(
    icla_uplo_t uplo, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dy, icla_int_t incy,
    iclaDoubleComplex_ptr       dwork, icla_int_t lwork,
    icla_queue_t queue );

icla_int_t
iclablas_zsymv_work(
    icla_uplo_t uplo, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dy, icla_int_t incy,
    iclaDoubleComplex_ptr       dwork, icla_int_t lwork,
    icla_queue_t queue );

void
iclablas_zgemm(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dB, icla_int_t lddb,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dC, icla_int_t lddc,
    icla_queue_t queue );

void
iclablas_zgemm_reduce(
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dB, icla_int_t lddb,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dC, icla_int_t lddc,
    icla_queue_t queue );

void
iclablas_ztrsm(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr       dB, icla_int_t lddb,
    icla_queue_t queue );

void
iclablas_ztrsm_outofplace(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr       dB, icla_int_t lddb,
    iclaDoubleComplex_ptr       dX, icla_int_t lddx,
    icla_int_t flag,
    iclaDoubleComplex_ptr d_dinvA, icla_int_t dinvA_length,
    icla_queue_t queue );

void
iclablas_ztrsm_work(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr       dB, icla_int_t lddb,
    iclaDoubleComplex_ptr       dX, icla_int_t lddx,
    icla_int_t flag,
    iclaDoubleComplex_ptr d_dinvA, icla_int_t dinvA_length,
    icla_queue_t queue );


#define icla_zsetvector(           n, hx_src, incx, dy_dst, incy, queue ) \
        icla_zsetvector_internal(  n, hx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )



#define icla_zgetvector(           n, dx_src, incx, hy_dst, incy, queue ) \
        icla_zgetvector_internal(  n, dx_src, incx, hy_dst, incy, queue, __func__, __FILE__, __LINE__ )



#define icla_zcopyvector(          n, dx_src, incx, dy_dst, incy, queue ) \
        icla_zcopyvector_internal( n, dx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )



#define icla_zsetvector_async(           n, hx_src, incx, dy_dst, incy, queue ) \
        icla_zsetvector_async_internal(  n, hx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )



#define icla_zgetvector_async(           n, dx_src, incx, hy_dst, incy, queue ) \
        icla_zgetvector_async_internal(  n, dx_src, incx, hy_dst, incy, queue, __func__, __FILE__, __LINE__ )



#define icla_zcopyvector_async(          n, dx_src, incx, dy_dst, incy, queue ) \
        icla_zcopyvector_async_internal( n, dx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

static inline void
icla_zsetvector_internal(
    icla_int_t n,
    iclaDoubleComplex const    *hx_src, icla_int_t incx,
    iclaDoubleComplex_ptr       dy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_setvector_internal( n, sizeof(iclaDoubleComplex),
                              hx_src, incx,
                              dy_dst, incy, queue,
                              func, file, line );
}

static inline void
icla_zgetvector_internal(
    icla_int_t n,
    iclaDoubleComplex_const_ptr dx_src, icla_int_t incx,
    iclaDoubleComplex          *hy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_getvector_internal( n, sizeof(iclaDoubleComplex),
                              dx_src, incx,
                              hy_dst, incy, queue,
                              func, file, line );
}

static inline void
icla_zcopyvector_internal(
    icla_int_t n,
    iclaDoubleComplex_const_ptr dx_src, icla_int_t incx,
    iclaDoubleComplex_ptr       dy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_copyvector_internal( n, sizeof(iclaDoubleComplex),
                               dx_src, incx,
                               dy_dst, incy, queue,
                               func, file, line );
}

static inline void
icla_zsetvector_async_internal(
    icla_int_t n,
    iclaDoubleComplex const    *hx_src, icla_int_t incx,
    iclaDoubleComplex_ptr       dy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_setvector_async_internal( n, sizeof(iclaDoubleComplex),
                                    hx_src, incx,
                                    dy_dst, incy, queue,
                                    func, file, line );
}

static inline void
icla_zgetvector_async_internal(
    icla_int_t n,
    iclaDoubleComplex_const_ptr dx_src, icla_int_t incx,
    iclaDoubleComplex          *hy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_getvector_async_internal( n, sizeof(iclaDoubleComplex),
                                    dx_src, incx,
                                    hy_dst, incy, queue,
                                    func, file, line );
}

static inline void
icla_zcopyvector_async_internal(
    icla_int_t n,
    iclaDoubleComplex_const_ptr dx_src, icla_int_t incx,
    iclaDoubleComplex_ptr       dy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_copyvector_async_internal( n, sizeof(iclaDoubleComplex),
                                     dx_src, incx,
                                     dy_dst, incy, queue,
                                     func, file, line );
}







#define icla_zsetmatrix(           m, n, hA_src, lda,  dB_dst, lddb, queue ) \
        icla_zsetmatrix_internal(  m, n, hA_src, lda,  dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )



#define icla_zgetmatrix(           m, n, dA_src, ldda, hB_dst, ldb,  queue ) \
        icla_zgetmatrix_internal(  m, n, dA_src, ldda, hB_dst, ldb,  queue, __func__, __FILE__, __LINE__ )



#define icla_zcopymatrix(          m, n, dA_src, ldda, dB_dst, lddb, queue ) \
        icla_zcopymatrix_internal( m, n, dA_src, ldda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )



#define icla_zsetmatrix_async(           m, n, hA_src, lda, dB_dst, lddb, queue ) \
        icla_zsetmatrix_async_internal(  m, n, hA_src, lda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )



#define icla_zgetmatrix_async(           m, n, dA_src, ldda, hB_dst, ldb, queue ) \
        icla_zgetmatrix_async_internal(  m, n, dA_src, ldda, hB_dst, ldb, queue, __func__, __FILE__, __LINE__ )



#define icla_zcopymatrix_async(          m, n, dA_src, ldda, dB_dst, lddb, queue ) \
        icla_zcopymatrix_async_internal( m, n, dA_src, ldda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

static inline void
icla_zsetmatrix_internal(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex const    *hA_src, icla_int_t lda,
    iclaDoubleComplex_ptr       dB_dst, icla_int_t lddb,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_setmatrix_internal( m, n, sizeof(iclaDoubleComplex),
                              hA_src, lda,
                              dB_dst, lddb, queue,
                              func, file, line );
}

static inline void
icla_zgetmatrix_internal(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex_const_ptr dA_src, icla_int_t ldda,
    iclaDoubleComplex          *hB_dst, icla_int_t ldb,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_getmatrix_internal( m, n, sizeof(iclaDoubleComplex),
                              dA_src, ldda,
                              hB_dst, ldb, queue,
                              func, file, line );
}

static inline void
icla_zcopymatrix_internal(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex_const_ptr dA_src, icla_int_t ldda,
    iclaDoubleComplex_ptr       dB_dst, icla_int_t lddb,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_copymatrix_internal( m, n, sizeof(iclaDoubleComplex),
                               dA_src, ldda,
                               dB_dst, lddb, queue,
                               func, file, line );
}

static inline void
icla_zsetmatrix_async_internal(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex const    *hA_src, icla_int_t lda,
    iclaDoubleComplex_ptr       dB_dst, icla_int_t lddb,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_setmatrix_async_internal( m, n, sizeof(iclaDoubleComplex),
                                    hA_src, lda,
                                    dB_dst, lddb, queue,
                                    func, file, line );
}

static inline void
icla_zgetmatrix_async_internal(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex_const_ptr dA_src, icla_int_t ldda,
    iclaDoubleComplex          *hB_dst, icla_int_t ldb,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_getmatrix_async_internal( m, n, sizeof(iclaDoubleComplex),
                                    dA_src, ldda,
                                    hB_dst, ldb, queue,
                                    func, file, line );
}

static inline void
icla_zcopymatrix_async_internal(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex_const_ptr dA_src, icla_int_t ldda,
    iclaDoubleComplex_ptr       dB_dst, icla_int_t lddb,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_copymatrix_async_internal( m, n, sizeof(iclaDoubleComplex),
                                     dA_src, ldda,
                                     dB_dst, lddb, queue,
                                     func, file, line );
}





icla_int_t
icla_izamax(
    icla_int_t n,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    icla_queue_t queue );

icla_int_t
icla_izamax_native(
    icla_int_t length,
    iclaDoubleComplex_ptr x, icla_int_t incx,
    icla_int_t* ipiv, icla_int_t *info,
    icla_int_t step, icla_int_t gbstep, icla_queue_t queue);

icla_int_t
icla_izamin(
    icla_int_t n,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    icla_queue_t queue );

double
icla_dzasum(
    icla_int_t n,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    icla_queue_t queue );

void
icla_zaxpy(
    icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex_ptr       dy, icla_int_t incy,
    icla_queue_t queue );

void
icla_zcopy(
    icla_int_t n,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex_ptr       dy, icla_int_t incy,
    icla_queue_t queue );

iclaDoubleComplex
icla_zdotc(
    icla_int_t n,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex_const_ptr dy, icla_int_t incy,
    icla_queue_t queue );

iclaDoubleComplex
icla_zdotu(
    icla_int_t n,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex_const_ptr dy, icla_int_t incy,
    icla_queue_t queue );

double
icla_dznrm2(
    icla_int_t n,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    icla_queue_t queue );

void
icla_zrot(
    icla_int_t n,
    iclaDoubleComplex_ptr dx, icla_int_t incx,
    iclaDoubleComplex_ptr dy, icla_int_t incy,
    double dc, iclaDoubleComplex ds,
    icla_queue_t queue );

void
icla_zdrot(
    icla_int_t n,
    iclaDoubleComplex_ptr dx, icla_int_t incx,
    iclaDoubleComplex_ptr dy, icla_int_t incy,
    double dc, double ds,
    icla_queue_t queue );

void
icla_zrotg(
    iclaDoubleComplex_ptr a,
    iclaDoubleComplex_ptr b,
    iclaDouble_ptr        c,
    iclaDoubleComplex_ptr s,
    icla_queue_t queue );

#ifdef ICLA_REAL
void
icla_zrotm(
    icla_int_t n,
    iclaDouble_ptr dx, icla_int_t incx,
    iclaDouble_ptr dy, icla_int_t incy,
    iclaDouble_const_ptr param,
    icla_queue_t queue );

void
icla_zrotmg(
    iclaDouble_ptr       d1,
    iclaDouble_ptr       d2,
    iclaDouble_ptr       x1,
    iclaDouble_const_ptr y1,
    iclaDouble_ptr param,
    icla_queue_t queue );
#endif

void
icla_zscal(
    icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_ptr dx, icla_int_t incx,
    icla_queue_t queue );

void
icla_zdscal(
    icla_int_t n,
    double alpha,
    iclaDoubleComplex_ptr dx, icla_int_t incx,
    icla_queue_t queue );

icla_int_t
icla_zscal_zgeru_native(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t lda,
    icla_int_t *info, icla_int_t step, icla_int_t gbstep,
    icla_queue_t queue);

void
icla_zswap(
    icla_int_t n,
    iclaDoubleComplex_ptr dx, icla_int_t incx,
    iclaDoubleComplex_ptr dy, icla_int_t incy,
    icla_queue_t queue );

void
icla_zswap_native(
    icla_int_t n, iclaDoubleComplex_ptr x, icla_int_t incx,
    icla_int_t step, icla_int_t* ipiv,
    icla_queue_t queue);




void
icla_zgemv(
    icla_trans_t transA,
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dy, icla_int_t incy,
    icla_queue_t queue );

void
icla_zgerc(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex_const_ptr dy, icla_int_t incy,
    iclaDoubleComplex_ptr       dA, icla_int_t ldda,
    icla_queue_t queue );

#ifdef ICLA_COMPLEX
void
icla_zgeru(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex_const_ptr dy, icla_int_t incy,
    iclaDoubleComplex_ptr       dA, icla_int_t ldda,
    icla_queue_t queue );

void
icla_zhemv(
    icla_uplo_t uplo,
    icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dy, icla_int_t incy,
    icla_queue_t queue );

void
icla_zher(
    icla_uplo_t uplo,
    icla_int_t n,
    double alpha,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex_ptr       dA, icla_int_t ldda,
    icla_queue_t queue );

void
icla_zher2(
    icla_uplo_t uplo,
    icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex_const_ptr dy, icla_int_t incy,
    iclaDoubleComplex_ptr       dA, icla_int_t ldda,
    icla_queue_t queue );
#endif

void
icla_zsymv(
    icla_uplo_t uplo,
    icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dy, icla_int_t incy,
    icla_queue_t queue );

void
icla_zsyr(
    icla_uplo_t uplo,
    icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex_ptr       dA, icla_int_t ldda,
    icla_queue_t queue );

void
icla_zsyr2(
    icla_uplo_t uplo,
    icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex_const_ptr dy, icla_int_t incy,
    iclaDoubleComplex_ptr       dA, icla_int_t ldda,
    icla_queue_t queue );

void
icla_ztrmv(
    icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t n,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr       dx, icla_int_t incx,
    icla_queue_t queue );

void
iclablas_ztrmv(
    icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t n,
    iclaDoubleComplex *dA, icla_int_t ldda,
    iclaDoubleComplex *dx, icla_int_t incx,
    icla_queue_t queue );

void
icla_ztrsv(
    icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t n,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr       dx, icla_int_t incx,
    icla_queue_t queue );




void
icla_zgemm(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dB, icla_int_t lddb,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dC, icla_int_t lddc,
    icla_queue_t queue );

void
icla_zhemm(
    icla_side_t side, icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dB, icla_int_t lddb,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dC, icla_int_t lddc,
    icla_queue_t queue );

void
iclablas_zhemm(
    icla_side_t side, icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dB, icla_int_t lddb,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dC, icla_int_t lddc,
    icla_queue_t queue );

void
icla_zher2k(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dB, icla_int_t lddb,
    double beta,
    iclaDoubleComplex_ptr       dC, icla_int_t lddc,
    icla_queue_t queue );

void iclablas_zher2k(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr dB, icla_int_t lddb,
    double beta,
    iclaDoubleComplex_ptr       dC, icla_int_t lddc,
    icla_queue_t queue );

void
icla_zherk(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    double alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    double beta,
    iclaDoubleComplex_ptr       dC, icla_int_t lddc,
    icla_queue_t queue );

void iclablas_zherk(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    double alpha,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    double beta,
    iclaDoubleComplex_ptr dC, icla_int_t lddc,
    icla_queue_t queue);

void iclablas_zherk_internal(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k, icla_int_t nb,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr dB, icla_int_t lddb,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr dC, icla_int_t lddc,
    icla_int_t conjugate, icla_queue_t queue);

void
iclablas_zherk_small_reduce(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    double alpha, iclaDoubleComplex* dA, icla_int_t ldda,
    double beta,  iclaDoubleComplex* dC, icla_int_t lddc,
    icla_int_t nthread_blocks, icla_queue_t queue );

void
icla_zsymm(
    icla_side_t side, icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dB, icla_int_t lddb,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dC, icla_int_t lddc,
    icla_queue_t queue );

void
iclablas_zsymm(
    icla_side_t side, icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dB, icla_int_t lddb,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dC, icla_int_t lddc,
    icla_queue_t queue );

void
icla_zsyr2k(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dB, icla_int_t lddb,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dC, icla_int_t lddc,
    icla_queue_t queue );

void iclablas_zsyr2k(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr dB, icla_int_t lddb,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr dC, icla_int_t lddc,
    icla_queue_t queue );

void
icla_zsyrk(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dC, icla_int_t lddc,
    icla_queue_t queue );

void iclablas_zsyrk(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr dC, icla_int_t lddc,
    icla_queue_t queue);

void
icla_ztrmm(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr       dB, icla_int_t lddb,
    icla_queue_t queue );

void
iclablas_ztrmm(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t m, icla_int_t n,
        iclaDoubleComplex alpha,
        iclaDoubleComplex *dA, icla_int_t ldda,
        iclaDoubleComplex *dB, icla_int_t lddb,
        icla_queue_t queue );

void
icla_ztrsm(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr       dB, icla_int_t lddb,
    icla_queue_t queue );

void
icla_zgetf2trsm_2d_native(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr dB, icla_int_t lddb,
    icla_queue_t queue);

icla_int_t
icla_zpotf2_lpout(
        icla_uplo_t uplo, icla_int_t n,
        iclaDoubleComplex *dA, icla_int_t lda, icla_int_t gbstep,
        icla_int_t *dinfo, icla_queue_t queue);

icla_int_t
icla_zpotf2_lpin(
        icla_uplo_t uplo, icla_int_t n,
        iclaDoubleComplex *dA, icla_int_t lda, icla_int_t gbstep,
        icla_int_t *dinfo, icla_queue_t queue);

#ifdef __cplusplus
}
#endif

#undef ICLA_COMPLEX

#endif
