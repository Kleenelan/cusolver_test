
#ifndef ICLA_D_H
#define ICLA_D_H

#include "icla_types.h"
#include "icla_dgehrd_m.h"

#define ICLA_REAL

#ifdef __cplusplus
extern "C" {
#endif

#ifdef ICLA_REAL
icla_int_t icla_get_dlaex3_m_nb();

#endif

icla_int_t icla_get_dpotrf_nb( icla_int_t n );
icla_int_t icla_get_dgetrf_nb( icla_int_t m, icla_int_t n );
icla_int_t icla_get_dgetrf_native_nb( icla_int_t m, icla_int_t n );
icla_int_t icla_get_dgetri_nb( icla_int_t n );
icla_int_t icla_get_dsytrf_nb( icla_int_t n );
icla_int_t icla_get_dsytrf_nopiv_nb( icla_int_t n );
icla_int_t icla_get_dsytrf_aasen_nb( icla_int_t n );

icla_int_t icla_get_dgeqp3_nb( icla_int_t m, icla_int_t n );
icla_int_t icla_get_dgeqrf_nb( icla_int_t m, icla_int_t n );
icla_int_t icla_get_dgeqlf_nb( icla_int_t m, icla_int_t n );
icla_int_t icla_get_dgelqf_nb( icla_int_t m, icla_int_t n );

icla_int_t icla_get_dgehrd_nb( icla_int_t n );
icla_int_t icla_get_dsytrd_nb( icla_int_t n );
icla_int_t icla_get_dsygst_nb( icla_int_t n );
icla_int_t icla_get_dsygst_m_nb( icla_int_t n );

icla_int_t icla_get_dgebrd_nb( icla_int_t m, icla_int_t n );
icla_int_t icla_get_dgesvd_nb( icla_int_t m, icla_int_t n );

icla_int_t icla_get_dbulge_nb( icla_int_t n, icla_int_t nbthreads );
icla_int_t icla_get_dbulge_nb_mgpu( icla_int_t n );
icla_int_t icla_get_dbulge_vblksiz( icla_int_t n, icla_int_t nb, icla_int_t nbthreads );
icla_int_t icla_get_dbulge_gcperf();

bool icla_dgetrf_gpu_recommend_cpu(icla_int_t m, icla_int_t n, icla_int_t nb);
bool icla_dgetrf_native_recommend_notrans(icla_int_t m, icla_int_t n, icla_int_t nb);

#ifdef ICLA_REAL

icla_int_t
icla_dsidi(
    icla_uplo_t uplo,
    double *A, icla_int_t lda, icla_int_t n, icla_int_t *ipiv,
    double *det, icla_int_t *inert,
    double *work, icla_int_t job,
    icla_int_t *info);

void
icla_dmove_eig(
    icla_range_t range, icla_int_t n, double *w,
    icla_int_t *il, icla_int_t *iu, double vl, double vu, icla_int_t *mout);

void
icla_dvrange(
    icla_int_t k, double *d, icla_int_t *il, icla_int_t *iu, double vl, double vu);

void
icla_dirange(
    icla_int_t k, icla_int_t *indxq, icla_int_t *iil, icla_int_t *iiu, icla_int_t il, icla_int_t iu);
#endif

icla_int_t
icla_dgbsv_native(
        icla_int_t n, icla_int_t kl, icla_int_t ku, icla_int_t nrhs,
        double* dA, icla_int_t ldda, icla_int_t* dipiv,
        double* dB, icla_int_t lddb,
        icla_int_t *info);

icla_int_t
icla_dgbtf2_native_v2(
    icla_int_t m, icla_int_t n, icla_int_t kl, icla_int_t ku,
    double* dA, icla_int_t ldda, icla_int_t* ipiv,
    icla_int_t* info, icla_queue_t queue);

icla_int_t
icla_dgbtf2_native_v2_work(
    icla_int_t m, icla_int_t n, icla_int_t kl, icla_int_t ku,
    double* dA, icla_int_t ldda, icla_int_t* ipiv,
    icla_int_t* info,
    void* device_work, icla_int_t* lwork,
    icla_queue_t queue);

void
icla_dgbsv_native_work(
        icla_int_t n, icla_int_t kl, icla_int_t ku, icla_int_t nrhs,
        double* dA, icla_int_t ldda, icla_int_t* dipiv,
        double* dB, icla_int_t lddb,
        icla_int_t *info, void* device_work, icla_int_t* lwork,
        icla_queue_t queue);

icla_int_t
icla_dgbtf2_native(
    icla_int_t m, icla_int_t n, icla_int_t kl, icla_int_t ku,
    double* dA, icla_int_t ldda, icla_int_t* ipiv,
    icla_int_t* info, icla_queue_t queue);

icla_int_t
icla_dgbtf2_native_work(
    icla_int_t m, icla_int_t n, icla_int_t kl, icla_int_t ku,
    double* dA, icla_int_t ldda, icla_int_t* ipiv,
    icla_int_t* info,
    void* device_work, icla_int_t* lwork,
    icla_queue_t queue);

icla_int_t
icla_dgbtrf_native(
    icla_int_t m, icla_int_t n,
    icla_int_t kl, icla_int_t ku,
    double* dAB, icla_int_t lddab, icla_int_t* dipiv,
    icla_int_t *info);

void
icla_dgbtrf_native_work(
    icla_int_t m, icla_int_t n,
    icla_int_t kl, icla_int_t ku,
    double* dAB, icla_int_t lddab,
    icla_int_t* dipiv, icla_int_t *info,
    void* device_work, icla_int_t* lwork,
    icla_queue_t queue);

icla_int_t
icla_dgebrd(
    icla_int_t m, icla_int_t n,
    double *A, icla_int_t lda,
    double *d, double *e,
    double *tauq, double *taup,
    double *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_dgeev(
    icla_vec_t jobvl, icla_vec_t jobvr, icla_int_t n,
    double *A, icla_int_t lda,
    #ifdef ICLA_COMPLEX
    double *w,
    #else
    double *wr, double *wi,
    #endif
    double *VL, icla_int_t ldvl,
    double *VR, icla_int_t ldvr,
    double *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    double *rwork,
    #endif
    icla_int_t *info);

icla_int_t
icla_dgeev_m(
    icla_vec_t jobvl, icla_vec_t jobvr, icla_int_t n,
    double *A, icla_int_t lda,
    #ifdef ICLA_COMPLEX
    double *w,
    #else
    double *wr, double *wi,
    #endif
    double *VL, icla_int_t ldvl,
    double *VR, icla_int_t ldvr,
    double *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    double *rwork,
    #endif
    icla_int_t *info);

icla_int_t
icla_dgegqr_gpu(
    icla_int_t ikind, icla_int_t m, icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    iclaDouble_ptr dwork, double *work,
    icla_int_t *info);

icla_int_t
icla_dgegqr_expert_gpu_work(
    icla_int_t ikind, icla_int_t m, icla_int_t n,
    iclaDouble_ptr dA,   icla_int_t ldda,
    void *host_work,   icla_int_t *lwork_host,
    void *device_work, icla_int_t *lwork_device,
    icla_int_t *info, icla_queue_t queue );

icla_int_t
icla_dgehrd(
    icla_int_t n, icla_int_t ilo, icla_int_t ihi,
    double *A, icla_int_t lda,
    double *tau,
    double *work, icla_int_t lwork,
    iclaDouble_ptr dT,
    icla_int_t *info);

icla_int_t
icla_dgehrd_m(
    icla_int_t n, icla_int_t ilo, icla_int_t ihi,
    double *A, icla_int_t lda,
    double *tau,
    double *work, icla_int_t lwork,
    double *T,
    icla_int_t *info);

icla_int_t
icla_dgehrd2(
    icla_int_t n, icla_int_t ilo, icla_int_t ihi,
    double *A, icla_int_t lda,
    double *tau,
    double *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_dgelqf(
    icla_int_t m, icla_int_t n,
    double *A,    icla_int_t lda,
    double *tau,
    double *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_dgelqf_gpu(
    icla_int_t m, icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    double *tau,
    double *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_dgels(
    icla_trans_t trans, icla_int_t m, icla_int_t n, icla_int_t nrhs,
    iclaDouble_ptr A, icla_int_t lda,
    iclaDouble_ptr B, icla_int_t ldb,
    double *hwork, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_dggrqf(
    icla_int_t m, icla_int_t p, icla_int_t n,
    double *A, icla_int_t lda,
    double *taua,
    double *B, icla_int_t ldb,
    double *taub,
    double *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_dgglse(
    icla_int_t m, icla_int_t n, icla_int_t p,
    double *A, icla_int_t lda,
    double *B, icla_int_t ldb,
    double *c, double *d, double *x,
    double *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_dgels_gpu(
    icla_trans_t trans, icla_int_t m, icla_int_t n, icla_int_t nrhs,
    iclaDouble_ptr dA, icla_int_t ldda,
    iclaDouble_ptr dB, icla_int_t lddb,
    double *hwork, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_dgels3_gpu(
    icla_trans_t trans, icla_int_t m, icla_int_t n, icla_int_t nrhs,
    iclaDouble_ptr dA, icla_int_t ldda,
    iclaDouble_ptr dB, icla_int_t lddb,
    double *hwork, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_dgeqlf(
    icla_int_t m, icla_int_t n,
    double *A,    icla_int_t lda,
    double *tau,
    double *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_dgeqp3(
    icla_int_t m, icla_int_t n,
    double *A, icla_int_t lda,
    icla_int_t *jpvt, double *tau,
    double *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    double *rwork,
    #endif
    icla_int_t *info);

icla_int_t
icla_dgeqp3_gpu(
    icla_int_t m, icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    icla_int_t *jpvt, double *tau,
    iclaDouble_ptr dwork, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    double *rwork,
    #endif
    icla_int_t *info);

icla_int_t
icla_dgeqp3_expert_gpu_work(
    icla_int_t m, icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    icla_int_t *jpvt, double *tau,
    void* host_work,   icla_int_t *lwork_host,
    void* device_work, icla_int_t *lwork_device,
    icla_int_t *info, icla_queue_t queue );

icla_int_t
icla_dgeqr2_gpu(
    icla_int_t m, icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    iclaDouble_ptr dtau,
    iclaDouble_ptr        dwork,
    icla_queue_t queue,
    icla_int_t *info);

icla_int_t
icla_dgeqr2x_gpu(
    icla_int_t m, icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    iclaDouble_ptr dtau,
    iclaDouble_ptr dT, iclaDouble_ptr ddA,
    iclaDouble_ptr        dwork,
    icla_int_t *info);

icla_int_t
icla_dgeqr2x2_gpu(
    icla_int_t m, icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    iclaDouble_ptr dtau,
    iclaDouble_ptr dT, iclaDouble_ptr ddA,
    iclaDouble_ptr        dwork,
    icla_int_t *info);

icla_int_t
icla_dgeqr2x3_gpu(
    icla_int_t m, icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    iclaDouble_ptr dtau,
    iclaDouble_ptr dT,
    iclaDouble_ptr ddA,
    iclaDouble_ptr        dwork,
    icla_int_t *info);

icla_int_t
icla_dgeqr2x4_gpu(
    icla_int_t m, icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    iclaDouble_ptr dtau,
    iclaDouble_ptr dT,
    iclaDouble_ptr ddA,
    iclaDouble_ptr        dwork,
    icla_queue_t queue,
    icla_int_t *info);

icla_int_t
icla_dgeqrf(
    icla_int_t m, icla_int_t n,
    double *A, icla_int_t lda,
    double *tau,
    double *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_dgeqrf_gpu(
    icla_int_t m, icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    double *tau,
    iclaDouble_ptr dT,
    icla_int_t *info);

icla_int_t
icla_dgeqrf_expert_gpu_work(
    icla_int_t m, icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    double *tau, iclaDouble_ptr dT,
    icla_int_t *info,
    icla_mode_t mode, icla_int_t nb,
    void* host_work,   icla_int_t *lwork_host,
    void* device_work, icla_int_t *lwork_device,
    icla_queue_t queues[2] );

icla_int_t
icla_dgeqrf_m(
    icla_int_t ngpu,
    icla_int_t m, icla_int_t n,
    double *A,    icla_int_t lda,
    double *tau,
    double *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_dgeqrf_ooc(
    icla_int_t m, icla_int_t n,
    double *A, icla_int_t lda,
    double *tau,
    double *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_dgeqrf2_gpu(
    icla_int_t m, icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    double *tau,
    icla_int_t *info);

icla_int_t
icla_dgeqrf2_mgpu(
    icla_int_t ngpu,
    icla_int_t m, icla_int_t n,
    iclaDouble_ptr d_lA[], icla_int_t ldda,
    double *tau,
    icla_int_t *info);

icla_int_t
icla_dgeqrf3_gpu(
    icla_int_t m, icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    double *tau,
    iclaDouble_ptr dT,
    icla_int_t *info);

icla_int_t
icla_dgeqrs_gpu(
    icla_int_t m, icla_int_t n, icla_int_t nrhs,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    double const *tau,
    iclaDouble_ptr dT,
    iclaDouble_ptr dB, icla_int_t lddb,
    double *hwork, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_dgeqrs3_gpu(
    icla_int_t m, icla_int_t n, icla_int_t nrhs,
    iclaDouble_ptr dA, icla_int_t ldda,
    double const *tau,
    iclaDouble_ptr dT,
    iclaDouble_ptr dB, icla_int_t lddb,
    double *hwork, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_dgerbt_gpu(
    icla_bool_t gen, icla_int_t n, icla_int_t nrhs,
    iclaDouble_ptr dA, icla_int_t ldda,
    iclaDouble_ptr dB, icla_int_t lddb,
    double *U, double *V,
    icla_int_t *info);

icla_int_t
icla_dgerfs_nopiv_gpu(
    icla_trans_t trans, icla_int_t n, icla_int_t nrhs,
    iclaDouble_ptr dA, icla_int_t ldda,
    iclaDouble_ptr dB, icla_int_t lddb,
    iclaDouble_ptr dX, icla_int_t lddx,
    iclaDouble_ptr dworkd, iclaDouble_ptr dAF,
    icla_int_t *iter,
    icla_int_t *info);

icla_int_t
icla_dgesdd(
    icla_vec_t jobz, icla_int_t m, icla_int_t n,
    double *A, icla_int_t lda,
    double *s,
    double *U, icla_int_t ldu,
    double *VT, icla_int_t ldvt,
    double *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    double *rwork,
    #endif
    icla_int_t *iwork,
    icla_int_t *info);

icla_int_t
icla_dgesv(
    icla_int_t n, icla_int_t nrhs,
    double *A, icla_int_t lda,
    icla_int_t *ipiv,
    double *B, icla_int_t ldb,
    icla_int_t *info);

icla_int_t
icla_dgesv_gpu(
    icla_int_t n, icla_int_t nrhs,
    iclaDouble_ptr dA, icla_int_t ldda,
    icla_int_t *ipiv,
    iclaDouble_ptr dB, icla_int_t lddb,
    icla_int_t *info);

icla_int_t
icla_dgesv_nopiv_gpu(
    icla_int_t n, icla_int_t nrhs,
    iclaDouble_ptr dA, icla_int_t ldda,
    iclaDouble_ptr dB, icla_int_t lddb,
    icla_int_t *info);

icla_int_t
icla_dgesv_rbt(
    icla_bool_t ref, icla_int_t n, icla_int_t nrhs,
    double *A, icla_int_t lda,
    double *B, icla_int_t ldb,
    icla_int_t *info);

icla_int_t
icla_dgesvd(
    icla_vec_t jobu, icla_vec_t jobvt, icla_int_t m, icla_int_t n,
    double *A,    icla_int_t lda, double *s,
    double *U,    icla_int_t ldu,
    double *VT,   icla_int_t ldvt,
    double *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    double *rwork,
    #endif
    icla_int_t *info);

icla_int_t
icla_dgetf2_gpu(
    icla_int_t m, icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    icla_int_t *ipiv,
    icla_queue_t queue,
    icla_int_t *info);

icla_int_t
icla_dgetf2_native_fused(
    icla_int_t m, icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    icla_int_t *ipiv, icla_int_t gbstep,
    icla_int_t *flags,
    icla_int_t *info, icla_queue_t queue );

icla_int_t
icla_dgetf2_native(
    icla_int_t m, icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    icla_int_t *dipiv, icla_int_t* dipivinfo,
    icla_int_t *dinfo, icla_int_t gbstep,
    icla_event_t events[2],
    icla_queue_t queue, icla_queue_t update_queue);

icla_int_t
icla_dgetf2_nopiv(
    icla_int_t m, icla_int_t n,
    double *A, icla_int_t lda,
    icla_int_t *info);

icla_int_t
icla_dgetrf_recpanel_native(
    icla_int_t m, icla_int_t n, icla_int_t recnb,
    iclaDouble_ptr dA, icla_int_t ldda,
    icla_int_t* dipiv, icla_int_t* dipivinfo,
    icla_int_t *dinfo, icla_int_t gbstep,
    icla_event_t events[2], icla_queue_t queue, icla_queue_t update_queue);

icla_int_t
icla_dgetrf(
    icla_int_t m, icla_int_t n,
    double *A, icla_int_t lda,
    icla_int_t *ipiv,
    icla_int_t *info);

icla_int_t
icla_dgetrf_gpu(
    icla_int_t m, icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    icla_int_t *ipiv,
    icla_int_t *info);

icla_int_t
icla_dgetrf_expert_gpu_work(
    icla_int_t m, icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    icla_int_t *ipiv,
    icla_int_t *info, icla_mode_t mode,
    icla_int_t nb, icla_int_t recnb,
    void* host_work,   icla_int_t *lwork_host,
    void* device_work, icla_int_t *lwork_device,
    icla_event_t events[2], icla_queue_t queues[2] );

icla_int_t
icla_dgetrf_gpu_expert(
    icla_int_t m, icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    icla_int_t *ipiv,
    icla_int_t *info,
    icla_int_t nb, icla_mode_t mode);

icla_int_t
icla_dgetrf_native(
    icla_int_t m, icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    icla_int_t *ipiv,
    icla_int_t *info );

icla_int_t
icla_dgetrf_m(
    icla_int_t ngpu,
    icla_int_t m, icla_int_t n,
    double *A, icla_int_t lda,
    icla_int_t *ipiv,
    icla_int_t *info);

icla_int_t
icla_dgetrf_mgpu(
    icla_int_t ngpu,
    icla_int_t m, icla_int_t n,
    iclaDouble_ptr d_lA[], icla_int_t ldda,
    icla_int_t *ipiv,
    icla_int_t *info);

icla_int_t
icla_dgetrf2(
    icla_int_t m, icla_int_t n,
    double *A, icla_int_t lda,
    icla_int_t *ipiv,
    icla_int_t *info);

icla_int_t
icla_dgetrf2_mgpu(
    icla_int_t ngpu,
    icla_int_t m, icla_int_t n, icla_int_t nb, icla_int_t offset,
    iclaDouble_ptr d_lAT[], icla_int_t lddat,
    icla_int_t *ipiv,
    iclaDouble_ptr d_lAP[],
    double *W, icla_int_t ldw,
    icla_queue_t queues[][2],
    icla_int_t *info);

icla_int_t
icla_dgetrf_nopiv(
    icla_int_t m, icla_int_t n,
    double *A, icla_int_t lda,
    icla_int_t *info);

icla_int_t
icla_dgetrf_nopiv_gpu(
    icla_int_t m, icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    icla_int_t *info);

icla_int_t
icla_dgetri_gpu(
    icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    icla_int_t *ipiv,
    iclaDouble_ptr dwork, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_dgetri_expert_gpu_work(
    icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda, icla_int_t *ipiv,
    icla_int_t *info,
    icla_mode_t mode,
    void* host_work,   icla_int_t *lwork_host,
    void* device_work, icla_int_t *lwork_device,
    icla_queue_t queues[2] );

icla_int_t
icla_dgetrs_gpu(
    icla_trans_t trans, icla_int_t n, icla_int_t nrhs,
    iclaDouble_ptr dA, icla_int_t ldda,
    icla_int_t *ipiv,
    iclaDouble_ptr dB, icla_int_t lddb,
    icla_int_t *info);

icla_int_t
icla_dgetrs_expert_gpu_work(
    icla_trans_t trans, icla_int_t n, icla_int_t nrhs,
    iclaDouble_ptr dA, icla_int_t ldda, icla_int_t *ipiv,
    iclaDouble_ptr dB, icla_int_t lddb,
    icla_int_t *info,
    icla_mode_t mode,
    void* host_work,   icla_int_t *lwork_host,
    void* device_work, icla_int_t *lwork_device,
    icla_queue_t queue );

icla_int_t
icla_dgetrs_nopiv_gpu(
    icla_trans_t trans, icla_int_t n, icla_int_t nrhs,
    iclaDouble_ptr dA, icla_int_t ldda,
    iclaDouble_ptr dB, icla_int_t lddb,
    icla_int_t *info);

icla_int_t
icla_dsyevd(
    icla_vec_t jobz, icla_uplo_t uplo, icla_int_t n,
    double *A, icla_int_t lda,
    double *w,
    double *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    double *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

icla_int_t
icla_dsyevd_gpu(
    icla_vec_t jobz, icla_uplo_t uplo,
    icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    double *w,
    double *wA,  icla_int_t ldwa,
    double *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    double *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

icla_int_t
icla_dsyevd_m(
    icla_int_t ngpu,
    icla_vec_t jobz, icla_uplo_t uplo,
    icla_int_t n,
    double *A, icla_int_t lda,
    double *w,
    double *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    double *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

icla_int_t
icla_dsyevdx(
    icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo, icla_int_t n,
    double *A, icla_int_t lda,
    double vl, double vu, icla_int_t il, icla_int_t iu,
    icla_int_t *mout, double *w,
    double *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    double *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

icla_int_t
icla_dsyevdx_gpu(
    icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo,
    icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    double vl, double vu,
    icla_int_t il, icla_int_t iu,
    icla_int_t *mout, double *w,
    double *wA,  icla_int_t ldwa,
    double *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    double *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

icla_int_t
icla_dsyevdx_m(
    icla_int_t ngpu,
    icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo,
    icla_int_t n,
    double *A, icla_int_t lda,
    double vl, double vu, icla_int_t il, icla_int_t iu,
    icla_int_t *mout, double *w,
    double *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    double *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

icla_int_t
icla_dsyevdx_2stage(
    icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo,
    icla_int_t n,
    double *A, icla_int_t lda,
    double vl, double vu, icla_int_t il, icla_int_t iu,
    icla_int_t *mout, double *w,
    double *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    double *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

icla_int_t
icla_dsyevdx_2stage_m(
    icla_int_t ngpu,
    icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo,
    icla_int_t n,
    double *A, icla_int_t lda,
    double vl, double vu, icla_int_t il, icla_int_t iu,
    icla_int_t *mout, double *w,
    double *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    double *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

#ifdef ICLA_COMPLEX

icla_int_t
icla_dsyevr(
    icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo, icla_int_t n,
    double *A, icla_int_t lda,
    double vl, double vu,
    icla_int_t il, icla_int_t iu, double abstol, icla_int_t *mout,
    double *w,
    double *Z, icla_int_t ldz,
    icla_int_t *isuppz,
    double *work, icla_int_t lwork,
    double *rwork, icla_int_t lrwork,
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

icla_int_t
icla_dsyevr_gpu(
    icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo, icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    double vl, double vu,
    icla_int_t il, icla_int_t iu, double abstol, icla_int_t *mout,
    double *w,
    iclaDouble_ptr dZ, icla_int_t lddz,
    icla_int_t *isuppz,
    double *wA, icla_int_t ldwa,
    double *wZ, icla_int_t ldwz,
    double *work, icla_int_t lwork,
    double *rwork, icla_int_t lrwork,
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

icla_int_t
icla_dsyevx(
    icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo, icla_int_t n,
    double *A, icla_int_t lda,
    double vl, double vu,
    icla_int_t il, icla_int_t iu, double abstol, icla_int_t *mout,
    double *w,
    double *Z, icla_int_t ldz,
    double *work, icla_int_t lwork,
    double *rwork, icla_int_t *iwork,
    icla_int_t *ifail,
    icla_int_t *info);

icla_int_t
icla_dsyevx_gpu(
    icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo, icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    double vl, double vu, icla_int_t il, icla_int_t iu,
    double abstol, icla_int_t *mout,
    double *w,
    iclaDouble_ptr dZ, icla_int_t lddz,
    double *wA, icla_int_t ldwa,
    double *wZ, icla_int_t ldwz,
    double *work, icla_int_t lwork,
    double *rwork, icla_int_t *iwork,
    icla_int_t *ifail,
    icla_int_t *info);
#endif

icla_int_t
icla_dsygst(
    icla_int_t itype, icla_uplo_t uplo, icla_int_t n,
    double *A, icla_int_t lda,
    double *B, icla_int_t ldb,
    icla_int_t *info);

icla_int_t
icla_dsygst_gpu(
    icla_int_t itype, icla_uplo_t uplo, icla_int_t n,
    iclaDouble_ptr       dA, icla_int_t ldda,
    iclaDouble_const_ptr dB, icla_int_t lddb,
    icla_int_t *info);

icla_int_t
icla_dsygst_m(
    icla_int_t ngpu,
    icla_int_t itype, icla_uplo_t uplo, icla_int_t n,
    double *A, icla_int_t lda,
    double *B, icla_int_t ldb,
    icla_int_t *info);

icla_int_t
icla_dsygvd(
    icla_int_t itype, icla_vec_t jobz, icla_uplo_t uplo, icla_int_t n,
    double *A, icla_int_t lda,
    double *B, icla_int_t ldb,
    double *w, double *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    double *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

icla_int_t
icla_dsygvd_m(
    icla_int_t ngpu,
    icla_int_t itype, icla_vec_t jobz, icla_uplo_t uplo,
    icla_int_t n,
    double *A, icla_int_t lda,
    double *B, icla_int_t ldb,
    double *w,
    double *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    double *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

icla_int_t
icla_dsygvdx(
    icla_int_t itype, icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo,
    icla_int_t n, double *A, icla_int_t lda,
    double *B, icla_int_t ldb,
    double vl, double vu, icla_int_t il, icla_int_t iu,
    icla_int_t *mout, double *w,
    double *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    double *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

icla_int_t
icla_dsygvdx_m(
    icla_int_t ngpu,
    icla_int_t itype, icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo,
    icla_int_t n,
    double *A, icla_int_t lda,
    double *B, icla_int_t ldb,
    double vl, double vu, icla_int_t il, icla_int_t iu,
    icla_int_t *mout, double *w,
    double *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    double *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

icla_int_t
icla_dsygvdx_2stage(
    icla_int_t itype, icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo, icla_int_t n,
    double *A, icla_int_t lda,
    double *B, icla_int_t ldb,
    double vl, double vu, icla_int_t il, icla_int_t iu,
    icla_int_t *mout, double *w,
    double *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    double *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

icla_int_t
icla_dsygvdx_2stage_m(
    icla_int_t ngpu,
    icla_int_t itype, icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo,
    icla_int_t n,
    double *A, icla_int_t lda,
    double *B, icla_int_t ldb,
    double vl, double vu, icla_int_t il, icla_int_t iu,
    icla_int_t *mout, double *w,
    double *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    double *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

#ifdef ICLA_COMPLEX

icla_int_t
icla_dsygvr(
    icla_int_t itype, icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo, icla_int_t n,
    double *A, icla_int_t lda,
    double *B, icla_int_t ldb,
    double vl, double vu, icla_int_t il, icla_int_t iu,
    double abstol, icla_int_t *mout, double *w,
    double *Z, icla_int_t ldz,
    icla_int_t *isuppz, double *work, icla_int_t lwork,
    double *rwork, icla_int_t lrwork,
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

icla_int_t
icla_dsygvx(
    icla_int_t itype, icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo,
    icla_int_t n, double *A, icla_int_t lda,
    double *B, icla_int_t ldb,
    double vl, double vu, icla_int_t il, icla_int_t iu,
    double abstol, icla_int_t *mout, double *w,
    double *Z, icla_int_t ldz,
    double *work, icla_int_t lwork, double *rwork,
    icla_int_t *iwork, icla_int_t *ifail,
    icla_int_t *info);
#endif

icla_int_t
icla_dsysv(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nrhs,
    double *A, icla_int_t lda,
    icla_int_t *ipiv,
    double *B, icla_int_t ldb,
    icla_int_t *info);

icla_int_t
icla_dsysv_nopiv_gpu(
    icla_uplo_t uplo,  icla_int_t n, icla_int_t nrhs,
    iclaDouble_ptr dA, icla_int_t ldda,
    iclaDouble_ptr dB, icla_int_t lddb,
    icla_int_t *info);

icla_int_t
icla_dsytrd(
    icla_uplo_t uplo, icla_int_t n,
    double *A, icla_int_t lda,
    double *d, double *e, double *tau,
    double *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_dsytrd_gpu(
    icla_uplo_t uplo, icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    double *d, double *e, double *tau,
    double *wA,  icla_int_t ldwa,
    double *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_dsytrd2_gpu(
    icla_uplo_t uplo, icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    double *d, double *e, double *tau,
    double *wA,  icla_int_t ldwa,
    double *work, icla_int_t lwork,
    iclaDouble_ptr dwork, icla_int_t ldwork,
    icla_int_t *info);

icla_int_t
icla_dsytrd_mgpu(
    icla_int_t ngpu, icla_int_t nqueue,
    icla_uplo_t uplo, icla_int_t n,
    double *A, icla_int_t lda,
    double *d, double *e, double *tau,
    double *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_dsytrd_sb2st(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nb, icla_int_t Vblksiz,
    double *A, icla_int_t lda,
    double *d, double *e,
    double *V, icla_int_t ldv,
    double *TAU, icla_int_t compT,
    double *T, icla_int_t ldt);

icla_int_t
icla_dsytrd_sy2sb(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nb,
    double *A, icla_int_t lda,
    double *tau,
    double *work, icla_int_t lwork,
    iclaDouble_ptr dT,
    icla_int_t *info);

icla_int_t
icla_dsytrd_sy2sb_mgpu(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nb,
    double *A, icla_int_t lda,
    double *tau,
    double *work, icla_int_t lwork,
    iclaDouble_ptr dAmgpu[], icla_int_t ldda,
    iclaDouble_ptr dTmgpu[], icla_int_t lddt,
    icla_int_t ngpu, icla_int_t distblk,
    icla_queue_t queues[][20], icla_int_t nqueue,
    icla_int_t *info);

icla_int_t
icla_dsytrf(
    icla_uplo_t uplo, icla_int_t n,
    double *A, icla_int_t lda,
    icla_int_t *ipiv,
    icla_int_t *info);

icla_int_t
icla_dsytrf_gpu(
   icla_uplo_t uplo, icla_int_t n,
   double *dA, icla_int_t ldda,
   icla_int_t *ipiv,
   icla_int_t *info);

icla_int_t
icla_dsytrf_aasen(
    icla_uplo_t uplo, icla_int_t cpu_panel, icla_int_t n,
    double *A, icla_int_t lda,
    icla_int_t *ipiv, icla_int_t *info);

icla_int_t
icla_dsytrf_nopiv(
    icla_uplo_t uplo, icla_int_t n,
    double *A, icla_int_t lda,
    icla_int_t *info);

icla_int_t
icla_dsytrf_nopiv_cpu(
    icla_uplo_t uplo, icla_int_t n, icla_int_t ib,
    double *A, icla_int_t lda,
    icla_int_t *info);

icla_int_t
icla_dsytrf_nopiv_gpu(
    icla_uplo_t uplo, icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    icla_int_t *info);

icla_int_t
icla_dsytrs_nopiv_gpu(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nrhs,
    iclaDouble_ptr dA, icla_int_t ldda,
    iclaDouble_ptr dB, icla_int_t lddb,
    icla_int_t *info);

#ifdef ICLA_REAL

icla_int_t
icla_dlaex0(
    icla_int_t n, double *d, double *e,
    double *Q, icla_int_t ldq,
    double *work, icla_int_t *iwork,
    iclaDouble_ptr dwork,
    icla_range_t range, double vl, double vu, icla_int_t il, icla_int_t iu,
    icla_int_t *info);

icla_int_t
icla_dlaex0_m(
    icla_int_t ngpu,
    icla_int_t n, double *d, double *e,
    double *Q, icla_int_t ldq,
    double *work, icla_int_t *iwork,
    icla_range_t range, double vl, double vu,
    icla_int_t il, icla_int_t iu,
    icla_int_t *info);

icla_int_t
icla_dlaex1(
    icla_int_t n, double *d,
    double *Q, icla_int_t ldq,
    icla_int_t *indxq, double rho, icla_int_t cutpnt,
    double *work, icla_int_t *iwork,
    iclaDouble_ptr dwork,
    icla_queue_t queue,
    icla_range_t range, double vl, double vu, icla_int_t il, icla_int_t iu,
    icla_int_t *info);

icla_int_t
icla_dlaex1_m(
    icla_int_t ngpu,
    icla_int_t n, double *d,
    double *Q, icla_int_t ldq,
    icla_int_t *indxq, double rho, icla_int_t cutpnt,
    double *work, icla_int_t *iwork,
    iclaDouble_ptr dwork[],
    icla_queue_t queues[iclaMaxGPUs][2],
    icla_range_t range, double vl, double vu,
    icla_int_t il, icla_int_t iu, icla_int_t *info);

icla_int_t
icla_dlaex3(
    icla_int_t k, icla_int_t n, icla_int_t n1, double *d,
    double *Q, icla_int_t ldq,
    double rho,
    double *dlamda, double *Q2, icla_int_t *indx,
    icla_int_t *ctot, double *w, double *s, icla_int_t *indxq,
    iclaDouble_ptr dwork,
    icla_queue_t queue,
    icla_range_t range, double vl, double vu, icla_int_t il, icla_int_t iu,
    icla_int_t *info);

icla_int_t
icla_dlaex3_m(
    icla_int_t ngpu,
    icla_int_t k, icla_int_t n, icla_int_t n1, double *d,
    double *Q, icla_int_t ldq, double rho,
    double *dlamda, double *Q2, icla_int_t *indx,
    icla_int_t *ctot, double *w, double *s, icla_int_t *indxq,
    iclaDouble_ptr dwork[],
    icla_queue_t queues[iclaMaxGPUs][2],
    icla_range_t range, double vl, double vu, icla_int_t il, icla_int_t iu,
    icla_int_t *info);
#endif

icla_int_t
icla_dlabrd_gpu(
    icla_int_t m, icla_int_t n, icla_int_t nb,
    double     *A, icla_int_t lda,
    iclaDouble_ptr dA, icla_int_t ldda,
    double *d, double *e, double *tauq, double *taup,
    double     *X, icla_int_t ldx,
    iclaDouble_ptr dX, icla_int_t lddx,
    double     *Y, icla_int_t ldy,
    iclaDouble_ptr dY, icla_int_t lddy,
    double  *work, icla_int_t lwork,
    icla_queue_t queue);

icla_int_t
icla_dlasyf_gpu(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nb, icla_int_t *kb,
    iclaDouble_ptr dA, icla_int_t ldda,
    icla_int_t *ipiv,
    iclaDouble_ptr dW, icla_int_t lddw,
    icla_queue_t queues[],
    icla_int_t *info);

icla_int_t
icla_dlahr2(
    icla_int_t n, icla_int_t k, icla_int_t nb,
    iclaDouble_ptr dA, icla_int_t ldda,
    iclaDouble_ptr dV, icla_int_t lddv,
    double *A,  icla_int_t lda,
    double *tau,
    double *T,  icla_int_t ldt,
    double *Y,  icla_int_t ldy,
    icla_queue_t queue);

icla_int_t
icla_dlahr2_m(
    icla_int_t n, icla_int_t k, icla_int_t nb,
    double *A, icla_int_t lda,
    double *tau,
    double *T, icla_int_t ldt,
    double *Y, icla_int_t ldy,
    struct dgehrd_data *data);

icla_int_t
icla_dlahru(
    icla_int_t n, icla_int_t ihi, icla_int_t k, icla_int_t nb,
    double     *A, icla_int_t lda,
    iclaDouble_ptr dA, icla_int_t ldda,
    iclaDouble_ptr dY, icla_int_t lddy,
    iclaDouble_ptr dV, icla_int_t lddv,
    iclaDouble_ptr dT,
    iclaDouble_ptr dwork,
    icla_queue_t queue);

icla_int_t
icla_dlahru_m(
    icla_int_t n, icla_int_t ihi, icla_int_t k, icla_int_t nb,
    double *A, icla_int_t lda,
    struct dgehrd_data *data);

#ifdef ICLA_REAL

icla_int_t
icla_dlaln2(
    icla_int_t trans, icla_int_t na, icla_int_t nw,
    double smin, double ca, const double *A, icla_int_t lda,
    double d1, double d2,   const double *B, icla_int_t ldb,
    double wr, double wi, double *X, icla_int_t ldx,
    double *scale, double *xnorm,
    icla_int_t *info);
#endif

icla_int_t
icla_dlaqps(
    icla_int_t m, icla_int_t n, icla_int_t offset,
    icla_int_t nb, icla_int_t *kb,
    double *A,  icla_int_t lda,
    iclaDouble_ptr dA, icla_int_t ldda,
    icla_int_t *jpvt, double *tau, double *vn1, double *vn2,
    double *auxv,
    double *F,  icla_int_t ldf,
    iclaDouble_ptr dF, icla_int_t lddf);

icla_int_t
icla_dlaqps_gpu(
    icla_int_t m, icla_int_t n, icla_int_t offset,
    icla_int_t nb, icla_int_t *kb,
    iclaDouble_ptr dA,  icla_int_t ldda,
    icla_int_t *jpvt, double *tau,
    double *vn1, double *vn2,
    iclaDouble_ptr dauxv,
    iclaDouble_ptr dF, icla_int_t lddf);

icla_int_t
icla_dlaqps2_gpu(
    icla_int_t m, icla_int_t n, icla_int_t offset,
    icla_int_t nb, icla_int_t *kb,
    iclaDouble_ptr dA,  icla_int_t ldda,
    icla_int_t *jpvt,
    iclaDouble_ptr dtau,
    iclaDouble_ptr dvn1, iclaDouble_ptr dvn2,
    iclaDouble_ptr dauxv,
    iclaDouble_ptr dF,  icla_int_t lddf,
    iclaDouble_ptr dlsticcs,
    icla_queue_t queue);

#ifdef ICLA_REAL

icla_int_t
icla_dlaqtrsd(
    icla_trans_t trans, icla_int_t n,
    const double *T, icla_int_t ldt,
    double *x,       icla_int_t ldx,
    const double *cnorm,
    icla_int_t *info);
#endif

icla_int_t
icla_dlarf_gpu(
    icla_int_t m,  icla_int_t n,
    iclaDouble_const_ptr dv, iclaDouble_const_ptr dtau,
    iclaDouble_ptr dC, icla_int_t lddc,
    icla_queue_t queue);

icla_int_t
icla_dlarfb2_gpu(
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDouble_const_ptr dV, icla_int_t lddv,
    iclaDouble_const_ptr dT, icla_int_t lddt,
    iclaDouble_ptr dC,       icla_int_t lddc,
    iclaDouble_ptr dwork,    icla_int_t ldwork,
    icla_queue_t queue);

icla_int_t
icla_dlatrd(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nb,
    double *A, icla_int_t lda,
    double *e, double *tau,
    double *W, icla_int_t ldw,
    double *work, icla_int_t lwork,
    iclaDouble_ptr dA, icla_int_t ldda,
    iclaDouble_ptr dW, icla_int_t lddw,
    icla_queue_t queue);

icla_int_t
icla_dlatrd2(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nb,
    double *A,  icla_int_t lda,
    double *e, double *tau,
    double *W,  icla_int_t ldw,
    double *work, icla_int_t lwork,
    iclaDouble_ptr dA, icla_int_t ldda,
    iclaDouble_ptr dW, icla_int_t lddw,
    iclaDouble_ptr dwork, icla_int_t ldwork,
    icla_queue_t queue);

icla_int_t
icla_dlatrd_mgpu(
    icla_int_t ngpu,
    icla_uplo_t uplo,
    icla_int_t n, icla_int_t nb, icla_int_t nb0,
    double *A,  icla_int_t lda,
    double *e, double *tau,
    double    *W,       icla_int_t ldw,
    iclaDouble_ptr dA[],    icla_int_t ldda, icla_int_t offset,
    iclaDouble_ptr dW[],    icla_int_t lddw,
    double    *hwork,   icla_int_t lhwork,
    iclaDouble_ptr dwork[], icla_int_t ldwork,
    icla_queue_t queues[]);

#ifdef ICLA_COMPLEX

icla_int_t
icla_dlatrsd(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_diag_t diag, icla_bool_t normin,
    icla_int_t n, const double *A, icla_int_t lda,
    double lambda,
    double *x,
    double *scale, double *cnorm,
    icla_int_t *info);
#endif

icla_int_t
icla_dlauum(
    icla_uplo_t uplo, icla_int_t n,
    double *A, icla_int_t lda,
    icla_int_t *info);

icla_int_t
icla_dlauum_gpu(
    icla_uplo_t uplo, icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    icla_int_t *info);

icla_int_t
icla_dposv(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nrhs,
    double *A, icla_int_t lda,
    double *B, icla_int_t ldb,
    icla_int_t *info);

icla_int_t
icla_dposv_gpu(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nrhs,
    iclaDouble_ptr dA, icla_int_t ldda,
    iclaDouble_ptr dB, icla_int_t lddb,
    icla_int_t *info);

icla_int_t
icla_dpotf2_gpu(
    icla_uplo_t uplo, icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    icla_queue_t queue,
    icla_int_t *info);

icla_int_t
icla_dpotf2_native(
    icla_uplo_t uplo, icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    icla_int_t step, icla_int_t *device_info,
    icla_queue_t queue );

icla_int_t
icla_dpotrf_rectile_native(
    icla_uplo_t uplo, icla_int_t n, icla_int_t recnb,
    double* dA,    icla_int_t ldda, icla_int_t gbstep,
    icla_int_t *dinfo,  icla_int_t *info, icla_queue_t queue);

icla_int_t
icla_dpotrf(
    icla_uplo_t uplo, icla_int_t n,
    double *A, icla_int_t lda,
    icla_int_t *info);

icla_int_t
icla_dpotrf_expert_gpu_work(
    icla_uplo_t uplo, icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    icla_int_t *info,
    icla_mode_t mode,
    icla_int_t nb, icla_int_t recnb,
    void* host_work,   icla_int_t *lwork_host,
    void* device_work, icla_int_t *lwork_device,
    icla_event_t events[2], icla_queue_t queues[2] );

icla_int_t
icla_dpotrf_expert_gpu(
    icla_uplo_t uplo, icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    icla_int_t *info,
    icla_int_t nb, icla_mode_t mode );

icla_int_t
icla_dpotrf_gpu(
    icla_uplo_t uplo, icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    icla_int_t *info);

icla_int_t
icla_dpotrf_native(
    icla_uplo_t uplo, icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    icla_int_t *info );

icla_int_t
icla_dpotrf_m(
    icla_int_t ngpu,
    icla_uplo_t uplo, icla_int_t n,
    double *A, icla_int_t lda,
    icla_int_t *info);

icla_int_t
icla_dpotrf_mgpu(
    icla_int_t ngpu,
    icla_uplo_t uplo, icla_int_t n,
    iclaDouble_ptr d_lA[], icla_int_t ldda,
    icla_int_t *info);

icla_int_t
icla_dpotrf_mgpu_right(
    icla_int_t ngpu,
    icla_uplo_t uplo, icla_int_t n,
    iclaDouble_ptr d_lA[], icla_int_t ldda,
    icla_int_t *info);

icla_int_t
icla_dpotrf3_mgpu(
    icla_int_t ngpu,
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    icla_int_t off_i, icla_int_t off_j, icla_int_t nb,
    iclaDouble_ptr d_lA[], icla_int_t ldda,
    iclaDouble_ptr d_lP[], icla_int_t lddp,
    double *A, icla_int_t lda, icla_int_t h,
    icla_queue_t queues[][3], icla_event_t events[][5],
    icla_int_t *info);

icla_int_t
icla_dpotri(
    icla_uplo_t uplo, icla_int_t n,
    double *A, icla_int_t lda,
    icla_int_t *info);

icla_int_t
icla_dpotri_gpu(
    icla_uplo_t uplo, icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    icla_int_t *info);

icla_int_t
icla_dpotrs_gpu(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nrhs,
    iclaDouble_ptr dA, icla_int_t ldda,
    iclaDouble_ptr dB, icla_int_t lddb,
    icla_int_t *info);

icla_int_t
icla_dpotrs_expert_gpu_work(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nrhs,
    iclaDouble_ptr dA, icla_int_t ldda,
    iclaDouble_ptr dB, icla_int_t lddb,
    icla_int_t *info,
    void* host_work,   icla_int_t *lwork_host,
    void* device_work, icla_int_t *lwork_device,
    icla_queue_t queue );

#ifdef ICLA_COMPLEX

icla_int_t
icla_dsysv_nopiv_gpu(
    icla_uplo_t uplo,  icla_int_t n, icla_int_t nrhs,
    iclaDouble_ptr dA, icla_int_t ldda,
    iclaDouble_ptr dB, icla_int_t lddb,
    icla_int_t *info);

icla_int_t
icla_dsytrf_nopiv_cpu(
    icla_uplo_t uplo, icla_int_t n, icla_int_t ib,
    double *A, icla_int_t lda,
    icla_int_t *info);

icla_int_t
icla_dsytrf_nopiv_gpu(
    icla_uplo_t uplo, icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    icla_int_t *info);

icla_int_t
icla_dsytrs_nopiv_gpu(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nrhs,
    iclaDouble_ptr dA, icla_int_t ldda,
    iclaDouble_ptr dB, icla_int_t lddb,
    icla_int_t *info);
#endif

icla_int_t
icla_dstedx(
    icla_range_t range, icla_int_t n, double vl, double vu,
    icla_int_t il, icla_int_t iu, double *d, double *e,
    double *Z, icla_int_t ldz,
    double *rwork, icla_int_t lrwork,
    icla_int_t *iwork, icla_int_t liwork,
    iclaDouble_ptr dwork,
    icla_int_t *info);

icla_int_t
icla_dstedx_m(
    icla_int_t ngpu,
    icla_range_t range, icla_int_t n, double vl, double vu,
    icla_int_t il, icla_int_t iu, double *d, double *e,
    double *Z, icla_int_t ldz,
    double *rwork, icla_int_t lrwork,
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

icla_int_t
icla_dtrevc3(
    icla_side_t side, icla_vec_t howmany,
    icla_int_t *select, icla_int_t n,
    double *T,  icla_int_t ldt,
    double *VL, icla_int_t ldvl,
    double *VR, icla_int_t ldvr,
    icla_int_t mm, icla_int_t *mout,
    double *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    double *rwork,
    #endif
    icla_int_t *info);

icla_int_t
icla_dtrevc3_mt(
    icla_side_t side, icla_vec_t howmany,
    icla_int_t *select, icla_int_t n,
    double *T,  icla_int_t ldt,
    double *VL, icla_int_t ldvl,
    double *VR, icla_int_t ldvr,
    icla_int_t mm, icla_int_t *mout,
    double *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    double *rwork,
    #endif
    icla_int_t *info);

icla_int_t
icla_dtrsm_m(
    icla_int_t ngpu,
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transa, icla_diag_t diag,
    icla_int_t m, icla_int_t n, double alpha,
    const double *A, icla_int_t lda,
    double       *B, icla_int_t ldb);

icla_int_t
icla_dtrtri(
    icla_uplo_t uplo, icla_diag_t diag, icla_int_t n,
    double *A, icla_int_t lda,
    icla_int_t *info);

icla_int_t
icla_dtrtri_gpu(
    icla_uplo_t uplo, icla_diag_t diag, icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    icla_int_t *info);

icla_int_t
icla_dtrtri_expert_gpu_work(
    icla_uplo_t uplo, icla_diag_t diag, icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    icla_int_t *info,
    void* host_work,   icla_int_t *lwork_host,
    void* device_work, icla_int_t *lwork_device,
    icla_queue_t queues[2] );

icla_int_t
icla_dorgbr(
    icla_vect_t vect, icla_int_t m, icla_int_t n, icla_int_t k,
    double *A, icla_int_t lda,
    double *tau,
    double *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_dorghr(
    icla_int_t n, icla_int_t ilo, icla_int_t ihi,
    double *A, icla_int_t lda,
    double *tau,
    iclaDouble_ptr dT, icla_int_t nb,
    icla_int_t *info);

icla_int_t
icla_dorghr_m(
    icla_int_t n, icla_int_t ilo, icla_int_t ihi,
    double *A, icla_int_t lda,
    double *tau,
    double *T, icla_int_t nb,
    icla_int_t *info);

icla_int_t
icla_dorglq(
    icla_int_t m, icla_int_t n, icla_int_t k,
    double *A, icla_int_t lda,
    double *tau,
    iclaDouble_ptr dT, icla_int_t nb,
    icla_int_t *info);

icla_int_t
icla_dorgqr(
    icla_int_t m, icla_int_t n, icla_int_t k,
    double *A, icla_int_t lda,
    double *tau,
    iclaDouble_ptr dT, icla_int_t nb,
    icla_int_t *info);

icla_int_t
icla_dorgqr_gpu(
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDouble_ptr dA, icla_int_t ldda,
    double *tau,
    iclaDouble_ptr dT, icla_int_t nb,
    icla_int_t *info);

icla_int_t
icla_dorgqr_m(
    icla_int_t m, icla_int_t n, icla_int_t k,
    double *A, icla_int_t lda,
    double *tau,
    double *T, icla_int_t nb,
    icla_int_t *info);

icla_int_t
icla_dorgqr2(
    icla_int_t m, icla_int_t n, icla_int_t k,
    double *A, icla_int_t lda,
    double *tau,
    icla_int_t *info);

icla_int_t
icla_dormbr(
    icla_vect_t vect, icla_side_t side, icla_trans_t trans,
    icla_int_t m, icla_int_t n, icla_int_t k,
    double *A, icla_int_t lda,
    double *tau,
    double *C, icla_int_t ldc,
    double *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_dormlq(
    icla_side_t side, icla_trans_t trans,
    icla_int_t m, icla_int_t n, icla_int_t k,
    double *A, icla_int_t lda,
    double *tau,
    double *C, icla_int_t ldc,
    double *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_dormrq(
    icla_side_t side, icla_trans_t trans,
    icla_int_t m, icla_int_t n, icla_int_t k,
    double *A, icla_int_t lda,
    double *tau,
    double *C, icla_int_t ldc,
    double *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_dormql(
    icla_side_t side, icla_trans_t trans,
    icla_int_t m, icla_int_t n, icla_int_t k,
    double *A, icla_int_t lda,
    double *tau,
    double *C, icla_int_t ldc,
    double *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_dormql2_gpu(
    icla_side_t side, icla_trans_t trans,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDouble_ptr dA, icla_int_t ldda,
    double *tau,
    iclaDouble_ptr dC, icla_int_t lddc,
    const double *wA, icla_int_t ldwa,
    icla_int_t *info);

icla_int_t
icla_dormqr(
    icla_side_t side, icla_trans_t trans,
    icla_int_t m, icla_int_t n, icla_int_t k,
    double *A, icla_int_t lda,
    double *tau,
    double *C, icla_int_t ldc,
    double *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_dormqr_gpu(
    icla_side_t side, icla_trans_t trans,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    double const   *tau,
    iclaDouble_ptr       dC, icla_int_t lddc,
    double       *hwork, icla_int_t lwork,
    iclaDouble_ptr       dT, icla_int_t nb,
    icla_int_t *info);

icla_int_t
icla_dormqr2_gpu(
    icla_side_t side, icla_trans_t trans,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDouble_ptr dA, icla_int_t ldda,
    double *tau,
    iclaDouble_ptr dC, icla_int_t lddc,
    const double *wA, icla_int_t ldwa,
    icla_int_t *info);

icla_int_t
icla_dormqr_m(
    icla_int_t ngpu,
    icla_side_t side, icla_trans_t trans,
    icla_int_t m, icla_int_t n, icla_int_t k,
    double *A,    icla_int_t lda,
    double *tau,
    double *C,    icla_int_t ldc,
    double *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_dormtr(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t m, icla_int_t n,
    double *A,    icla_int_t lda,
    double *tau,
    double *C,    icla_int_t ldc,
    double *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_dormtr_gpu(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t m, icla_int_t n,
    iclaDouble_ptr dA, icla_int_t ldda,
    double *tau,
    iclaDouble_ptr dC, icla_int_t lddc,
    const double *wA, icla_int_t ldwa,
    icla_int_t *info);

icla_int_t
icla_dormtr_m(
    icla_int_t ngpu,
    icla_side_t side, icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t m, icla_int_t n,
    double *A,    icla_int_t lda,
    double *tau,
    double *C,    icla_int_t ldc,
    double *work, icla_int_t lwork,
    icla_int_t *info);

extern const double ICLA_D_NAN;
extern const double ICLA_D_INF;

int icla_d_isnan( double x );
int icla_d_isinf( double x );
int icla_d_isnan_inf( double x );

double
icla_dmake_lwork( icla_int_t lwork );

icla_int_t
icla_dnan_inf(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    const double *A, icla_int_t lda,
    icla_int_t *cnt_nan,
    icla_int_t *cnt_inf);

icla_int_t
icla_dnan_inf_gpu(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    icla_int_t *cnt_nan,
    icla_int_t *cnt_inf,
    icla_queue_t queue);

void icla_dprint(
    icla_int_t m, icla_int_t n,
    const double *A, icla_int_t lda);

void icla_dprint_gpu(
    icla_int_t m, icla_int_t n,
    iclaDouble_const_ptr dA, icla_int_t ldda,
    icla_queue_t queue);

void icla_dpanel_to_q(
    icla_uplo_t uplo, icla_int_t ib,
    double *A, icla_int_t lda,
    double *work);

void icla_dq_to_panel(
    icla_uplo_t uplo, icla_int_t ib,
    double *A, icla_int_t lda,
    double *work);

void
iclablas_dextract_diag_sqrt(
    icla_int_t m, icla_int_t n,
    double* dA, icla_int_t ldda,
    double* dD, icla_int_t incd,
    icla_queue_t queue);

void
iclablas_dscal_shift_hpd(
    icla_uplo_t uplo, int n,
    double* dA, int ldda,
    double* dD, int incd,
    double miu, double cn, double eps,
    icla_queue_t queue);

void
iclablas_ddimv_invert(
    icla_int_t n,
    double alpha, double* dD, icla_int_t incd,
                              double* dx, icla_int_t incx,
    double beta,  double* dy, icla_int_t incy,
    icla_queue_t queue);

#ifdef __cplusplus
}
#endif

#undef ICLA_REAL

#endif

