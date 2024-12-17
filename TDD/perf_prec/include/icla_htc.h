

#ifndef ICLA_HTC_H
#define ICLA_HTC_H

#include "icla_types.h"

#ifdef __cplusplus
extern "C" {
#endif


icla_int_t
icla_dhgesv_iteref_gpu(
    icla_trans_t trans, icla_int_t n, icla_int_t nrhs,
    iclaDouble_ptr dA, icla_int_t ldda,
    icla_int_t *ipiv,
    iclaInt_ptr dipiv,
    iclaDouble_ptr dB, icla_int_t lddb,
    iclaDouble_ptr dX, icla_int_t lddx,
    iclaDouble_ptr dworkd, iclaFloat_ptr dworks,
    icla_int_t *iter,
    icla_int_t *info);

icla_int_t
icla_dsgesv_iteref_gpu(
    icla_trans_t trans, icla_int_t n, icla_int_t nrhs,
    iclaDouble_ptr dA, icla_int_t ldda,
    icla_int_t *ipiv,
    iclaInt_ptr dipiv,
    iclaDouble_ptr dB, icla_int_t lddb,
    iclaDouble_ptr dX, icla_int_t lddx,
    iclaDouble_ptr dworkd, iclaFloat_ptr dworks,
    icla_int_t *iter,
    icla_int_t *info);

icla_int_t
icla_dxgesv_gmres_gpu(
    icla_trans_t trans, icla_int_t n, icla_int_t nrhs,
    iclaDouble_ptr dA, icla_int_t ldda,
    icla_int_t *ipiv, iclaInt_ptr dipiv,
    iclaDouble_ptr dB, icla_int_t lddb,
    iclaDouble_ptr dX, icla_int_t lddx,
    iclaDouble_ptr dworkd, iclaFloat_ptr dworks,
    icla_refinement_t facto_type,
    icla_refinement_t solver_type,
    icla_int_t *iter,
    icla_int_t *info,
    real_Double_t *facto_time);

icla_int_t
icla_dfgmres_plu_gpu(
    icla_trans_t trans, icla_int_t n, icla_int_t nrhs,
    iclaDouble_ptr dA, icla_int_t ldda,
    iclaFloat_ptr dLU_sprec, icla_int_t lddlusp,
    iclaDouble_ptr dLU_dprec, icla_int_t lddludp,
    iclaInt_ptr ipiv, iclaInt_ptr dipiv,
    iclaDouble_ptr dB, icla_int_t lddb,
    iclaDouble_ptr dX, icla_int_t lddx,
    iclaFloat_ptr dSX,
    icla_int_t maxiter, icla_int_t restrt,
    icla_int_t maxiter_inner, icla_int_t restrt_inner,
    icla_int_t userinitguess,
    double tol, double innertol,
	double *rnorm0, icla_int_t *niters,
    icla_refinement_t solver_type,
    char *algoname, icla_int_t is_inner,
    icla_queue_t queue);

icla_int_t
icla_dsgelatrs_cpu(
    icla_trans_t trans, icla_int_t n, icla_int_t nrhs,
    iclaFloat_ptr  dA, icla_int_t ldda,
    iclaInt_ptr        dipiv,
    iclaDouble_ptr dB, icla_int_t lddb,
    iclaDouble_ptr dX, icla_int_t lddx,
    iclaFloat_ptr dSX,
    icla_int_t *info);




icla_int_t
icla_hgetrf_gpu(
    icla_int_t m, icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    icla_int_t *ipiv,
    icla_int_t *info );

icla_int_t
icla_htgetrf_gpu(
    icla_int_t m, icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    icla_int_t *ipiv,
    icla_int_t *info );

icla_int_t
icla_xhsgetrf_gpu(
    icla_int_t m, icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    icla_int_t *ipiv,
    icla_int_t *info,
    icla_mp_type_t enable_tc,
    icla_mp_type_t mp_algo_type);

icla_int_t
icla_xshgetrf_gpu(
    icla_int_t m, icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    icla_int_t *ipiv,
    icla_int_t *info,
    icla_mp_type_t enable_tc,
    icla_mp_type_t mp_algo_type);

icla_int_t
    icla_get_hgetrf_nb( icla_int_t m, icla_int_t n );

icla_int_t
icla_get_xgetrf_nb(
        icla_int_t m, icla_int_t n, icla_int_t prev_nb,
        icla_mp_type_t enable_tc, icla_mp_type_t mp_algo_type);



icla_int_t
icla_dshposv_gpu_expert(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nrhs,
    iclaDouble_ptr dA, icla_int_t ldda,
    iclaDouble_ptr dB, icla_int_t lddb,
    iclaDouble_ptr dX, icla_int_t lddx,
    iclaDouble_ptr dworkd, iclaFloat_ptr dworks,
    icla_int_t *iter, icla_mode_t mode, icla_int_t use_gmres, icla_int_t preprocess,
    float cn, float theta, icla_int_t *info);

icla_int_t
icla_dshposv_gpu(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nrhs,
    iclaDouble_ptr dA, icla_int_t ldda,
    iclaDouble_ptr dB, icla_int_t lddb,
    iclaDouble_ptr dX, icla_int_t lddx,
    icla_int_t *iter, icla_int_t *info);

icla_int_t
icla_dshposv_native(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nrhs,
    iclaDouble_ptr dA, icla_int_t ldda,
    iclaDouble_ptr dB, icla_int_t lddb,
    iclaDouble_ptr dX, icla_int_t lddx,
    icla_int_t *iter, icla_int_t *info);



icla_int_t
icla_shpotrf_gpu(
    icla_uplo_t uplo, icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    icla_int_t *info );

icla_int_t
icla_shpotrf_native(
    icla_uplo_t uplo, icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    icla_int_t *info );

icla_int_t
icla_dfgmres_spd_gpu(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nrhs,
    double  *dA, icla_int_t ldda,
    float   *dL, icla_int_t lddl, float* dD,
    double  *dB, icla_int_t lddb,
    double  *dX, icla_int_t lddx,
    float   *dSX,
    icla_int_t maxiter, icla_int_t restrt,
    icla_int_t maxiter_inner, icla_int_t restrt_inner,
    double tol, double innertol,
	double *rnorm0, icla_int_t *niters, icla_int_t is_inner,
	icla_int_t is_preprocessed, float miu,
    icla_queue_t queue);



void
iclablas_convert_dp2hp(
    icla_int_t m, icla_int_t n,
    const double  *dA, icla_int_t ldda,
    iclaHalf  *dB, icla_int_t lddb,
    icla_queue_t queue );

void
iclablas_convert_hp2dp(
    icla_int_t m, icla_int_t n,
    const iclaHalf  *dA, icla_int_t ldda,
    double  *dB, icla_int_t lddb,
    icla_queue_t queue );

void
iclablas_convert_hp2sp(
    icla_int_t m, icla_int_t n,
    const iclaHalf  *dA, icla_int_t ldda,
    float  *dB, icla_int_t lddb,
    icla_queue_t queue );

void
iclablas_convert_sp2hp(
    icla_int_t m, icla_int_t n,
    const float  *dA, icla_int_t ldda,
    iclaHalf  *dB, icla_int_t lddb,
    icla_queue_t queue );

void
iclablas_hlaswp(
    icla_int_t n,
    iclaHalf *dAT, icla_int_t ldda,
    icla_int_t k1, icla_int_t k2,
    const icla_int_t *ipiv, icla_int_t inci,
    icla_queue_t queue );

#ifdef __cplusplus
}
#endif

#endif

