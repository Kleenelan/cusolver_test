

#ifndef ICLA_S_H
#define ICLA_S_H

#include "icla_types.h"
#include "icla_sgehrd_m.h"

#define ICLA_REAL

#ifdef __cplusplus
extern "C" {
#endif


#ifdef ICLA_REAL
icla_int_t icla_get_slaex3_m_nb();
#endif


icla_int_t icla_get_spotrf_nb( icla_int_t n );
icla_int_t icla_get_sgetrf_nb( icla_int_t m, icla_int_t n );
icla_int_t icla_get_sgetrf_native_nb( icla_int_t m, icla_int_t n );
icla_int_t icla_get_sgetri_nb( icla_int_t n );
icla_int_t icla_get_ssytrf_nb( icla_int_t n );
icla_int_t icla_get_ssytrf_nopiv_nb( icla_int_t n );
icla_int_t icla_get_ssytrf_aasen_nb( icla_int_t n );


icla_int_t icla_get_sgeqp3_nb( icla_int_t m, icla_int_t n );
icla_int_t icla_get_sgeqrf_nb( icla_int_t m, icla_int_t n );
icla_int_t icla_get_sgeqlf_nb( icla_int_t m, icla_int_t n );
icla_int_t icla_get_sgelqf_nb( icla_int_t m, icla_int_t n );


icla_int_t icla_get_sgehrd_nb( icla_int_t n );
icla_int_t icla_get_ssytrd_nb( icla_int_t n );
icla_int_t icla_get_ssygst_nb( icla_int_t n );
icla_int_t icla_get_ssygst_m_nb( icla_int_t n );


icla_int_t icla_get_sgebrd_nb( icla_int_t m, icla_int_t n );
icla_int_t icla_get_sgesvd_nb( icla_int_t m, icla_int_t n );


icla_int_t icla_get_sbulge_nb( icla_int_t n, icla_int_t nbthreads );
icla_int_t icla_get_sbulge_nb_mgpu( icla_int_t n );
icla_int_t icla_get_sbulge_vblksiz( icla_int_t n, icla_int_t nb, icla_int_t nbthreads );
icla_int_t icla_get_sbulge_gcperf();


bool icla_sgetrf_gpu_recommend_cpu(icla_int_t m, icla_int_t n, icla_int_t nb);
bool icla_sgetrf_native_recommend_notrans(icla_int_t m, icla_int_t n, icla_int_t nb);


#ifdef ICLA_REAL

icla_int_t
icla_ssidi(
    icla_uplo_t uplo,
    float *A, icla_int_t lda, icla_int_t n, icla_int_t *ipiv,
    float *det, icla_int_t *inert,
    float *work, icla_int_t job,
    icla_int_t *info);

void
icla_smove_eig(
    icla_range_t range, icla_int_t n, float *w,
    icla_int_t *il, icla_int_t *iu, float vl, float vu, icla_int_t *mout);


void
icla_svrange(
    icla_int_t k, float *d, icla_int_t *il, icla_int_t *iu, float vl, float vu);

void
icla_sirange(
    icla_int_t k, icla_int_t *indxq, icla_int_t *iil, icla_int_t *iiu, icla_int_t il, icla_int_t iu);
#endif


icla_int_t
icla_sgbsv_native(
        icla_int_t n, icla_int_t kl, icla_int_t ku, icla_int_t nrhs,
        float* dA, icla_int_t ldda, icla_int_t* dipiv,
        float* dB, icla_int_t lddb,
        icla_int_t *info);

icla_int_t
icla_sgbtf2_native_v2(
    icla_int_t m, icla_int_t n, icla_int_t kl, icla_int_t ku,
    float* dA, icla_int_t ldda, icla_int_t* ipiv,
    icla_int_t* info, icla_queue_t queue);

icla_int_t
icla_sgbtf2_native_v2_work(
    icla_int_t m, icla_int_t n, icla_int_t kl, icla_int_t ku,
    float* dA, icla_int_t ldda, icla_int_t* ipiv,
    icla_int_t* info,
    void* device_work, icla_int_t* lwork,
    icla_queue_t queue);

void
icla_sgbsv_native_work(
        icla_int_t n, icla_int_t kl, icla_int_t ku, icla_int_t nrhs,
        float* dA, icla_int_t ldda, icla_int_t* dipiv,
        float* dB, icla_int_t lddb,
        icla_int_t *info, void* device_work, icla_int_t* lwork,
        icla_queue_t queue);

icla_int_t
icla_sgbtf2_native(
    icla_int_t m, icla_int_t n, icla_int_t kl, icla_int_t ku,
    float* dA, icla_int_t ldda, icla_int_t* ipiv,
    icla_int_t* info, icla_queue_t queue);

icla_int_t
icla_sgbtf2_native_work(
    icla_int_t m, icla_int_t n, icla_int_t kl, icla_int_t ku,
    float* dA, icla_int_t ldda, icla_int_t* ipiv,
    icla_int_t* info,
    void* device_work, icla_int_t* lwork,
    icla_queue_t queue);

icla_int_t
icla_sgbtrf_native(
    icla_int_t m, icla_int_t n,
    icla_int_t kl, icla_int_t ku,
    float* dAB, icla_int_t lddab, icla_int_t* dipiv,
    icla_int_t *info);

void
icla_sgbtrf_native_work(
    icla_int_t m, icla_int_t n,
    icla_int_t kl, icla_int_t ku,
    float* dAB, icla_int_t lddab,
    icla_int_t* dipiv, icla_int_t *info,
    void* device_work, icla_int_t* lwork,
    icla_queue_t queue);


icla_int_t
icla_sgebrd(
    icla_int_t m, icla_int_t n,
    float *A, icla_int_t lda,
    float *d, float *e,
    float *tauq, float *taup,
    float *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_sgeev(
    icla_vec_t jobvl, icla_vec_t jobvr, icla_int_t n,
    float *A, icla_int_t lda,
    #ifdef ICLA_COMPLEX
    float *w,
    #else
    float *wr, float *wi,
    #endif
    float *VL, icla_int_t ldvl,
    float *VR, icla_int_t ldvr,
    float *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    float *rwork,
    #endif
    icla_int_t *info);


icla_int_t
icla_sgeev_m(
    icla_vec_t jobvl, icla_vec_t jobvr, icla_int_t n,
    float *A, icla_int_t lda,
    #ifdef ICLA_COMPLEX
    float *w,
    #else
    float *wr, float *wi,
    #endif
    float *VL, icla_int_t ldvl,
    float *VR, icla_int_t ldvr,
    float *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    float *rwork,
    #endif
    icla_int_t *info);


icla_int_t
icla_sgegqr_gpu(
    icla_int_t ikind, icla_int_t m, icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    iclaFloat_ptr dwork, float *work,
    icla_int_t *info);

icla_int_t
icla_sgegqr_expert_gpu_work(
    icla_int_t ikind, icla_int_t m, icla_int_t n,
    iclaFloat_ptr dA,   icla_int_t ldda,
    void *host_work,   icla_int_t *lwork_host,
    void *device_work, icla_int_t *lwork_device,
    icla_int_t *info, icla_queue_t queue );

icla_int_t
icla_sgehrd(
    icla_int_t n, icla_int_t ilo, icla_int_t ihi,
    float *A, icla_int_t lda,
    float *tau,
    float *work, icla_int_t lwork,
    iclaFloat_ptr dT,
    icla_int_t *info);


icla_int_t
icla_sgehrd_m(
    icla_int_t n, icla_int_t ilo, icla_int_t ihi,
    float *A, icla_int_t lda,
    float *tau,
    float *work, icla_int_t lwork,
    float *T,
    icla_int_t *info);


icla_int_t
icla_sgehrd2(
    icla_int_t n, icla_int_t ilo, icla_int_t ihi,
    float *A, icla_int_t lda,
    float *tau,
    float *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_sgelqf(
    icla_int_t m, icla_int_t n,
    float *A,    icla_int_t lda,
    float *tau,
    float *work, icla_int_t lwork,
    icla_int_t *info);


icla_int_t
icla_sgelqf_gpu(
    icla_int_t m, icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    float *tau,
    float *work, icla_int_t lwork,
    icla_int_t *info);


icla_int_t
icla_sgels(
    icla_trans_t trans, icla_int_t m, icla_int_t n, icla_int_t nrhs,
    iclaFloat_ptr A, icla_int_t lda,
    iclaFloat_ptr B, icla_int_t ldb,
    float *hwork, icla_int_t lwork,
    icla_int_t *info);


icla_int_t
icla_sggrqf(
    icla_int_t m, icla_int_t p, icla_int_t n,
    float *A, icla_int_t lda,
    float *taua,
    float *B, icla_int_t ldb,
    float *taub,
    float *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_sgglse(
    icla_int_t m, icla_int_t n, icla_int_t p,
    float *A, icla_int_t lda,
    float *B, icla_int_t ldb,
    float *c, float *d, float *x,
    float *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_sgels_gpu(
    icla_trans_t trans, icla_int_t m, icla_int_t n, icla_int_t nrhs,
    iclaFloat_ptr dA, icla_int_t ldda,
    iclaFloat_ptr dB, icla_int_t lddb,
    float *hwork, icla_int_t lwork,
    icla_int_t *info);


icla_int_t
icla_sgels3_gpu(
    icla_trans_t trans, icla_int_t m, icla_int_t n, icla_int_t nrhs,
    iclaFloat_ptr dA, icla_int_t ldda,
    iclaFloat_ptr dB, icla_int_t lddb,
    float *hwork, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_sgeqlf(
    icla_int_t m, icla_int_t n,
    float *A,    icla_int_t lda,
    float *tau,
    float *work, icla_int_t lwork,
    icla_int_t *info);


icla_int_t
icla_sgeqp3(
    icla_int_t m, icla_int_t n,
    float *A, icla_int_t lda,
    icla_int_t *jpvt, float *tau,
    float *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    float *rwork,
    #endif
    icla_int_t *info);


icla_int_t
icla_sgeqp3_gpu(
    icla_int_t m, icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    icla_int_t *jpvt, float *tau,
    iclaFloat_ptr dwork, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    float *rwork,
    #endif
    icla_int_t *info);

icla_int_t
icla_sgeqp3_expert_gpu_work(
    icla_int_t m, icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    icla_int_t *jpvt, float *tau,
    void* host_work,   icla_int_t *lwork_host,
    void* device_work, icla_int_t *lwork_device,
    icla_int_t *info, icla_queue_t queue );


icla_int_t
icla_sgeqr2_gpu(
    icla_int_t m, icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    iclaFloat_ptr dtau,
    iclaFloat_ptr        dwork,
    icla_queue_t queue,
    icla_int_t *info);


icla_int_t
icla_sgeqr2x_gpu(
    icla_int_t m, icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    iclaFloat_ptr dtau,
    iclaFloat_ptr dT, iclaFloat_ptr ddA,
    iclaFloat_ptr        dwork,
    icla_int_t *info);


icla_int_t
icla_sgeqr2x2_gpu(
    icla_int_t m, icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    iclaFloat_ptr dtau,
    iclaFloat_ptr dT, iclaFloat_ptr ddA,
    iclaFloat_ptr        dwork,
    icla_int_t *info);

icla_int_t
icla_sgeqr2x3_gpu(
    icla_int_t m, icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    iclaFloat_ptr dtau,
    iclaFloat_ptr dT,
    iclaFloat_ptr ddA,
    iclaFloat_ptr        dwork,
    icla_int_t *info);


icla_int_t
icla_sgeqr2x4_gpu(
    icla_int_t m, icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    iclaFloat_ptr dtau,
    iclaFloat_ptr dT,
    iclaFloat_ptr ddA,
    iclaFloat_ptr        dwork,
    icla_queue_t queue,
    icla_int_t *info);

icla_int_t
icla_sgeqrf_m(
    icla_int_t ngpu,
    icla_int_t m, icla_int_t n,
    float *A,    icla_int_t lda,
    float *tau,
    float *work, icla_int_t lwork,
    icla_int_t *info);


icla_int_t
icla_sgeqrf_ooc(
    icla_int_t m, icla_int_t n,
    float *A, icla_int_t lda,
    float *tau,
    float *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_sgeqrf_gpu(
    icla_int_t m, icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    float *tau,
    icla_int_t *info, icla_queue_t the_queue,
    real_Double_t *gpu_time_to);

icla_int_t
icla_sgeqrf2_mgpu(
    icla_int_t ngpu,
    icla_int_t m, icla_int_t n,
    iclaFloat_ptr d_lA[], icla_int_t ldda,
    float *tau,
    icla_int_t *info);


icla_int_t
icla_sgeqrf3_gpu(
    icla_int_t m, icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    float *tau,
    iclaFloat_ptr dT,
    icla_int_t *info);

icla_int_t
icla_sgeqrs_gpu(
    icla_int_t m, icla_int_t n, icla_int_t nrhs,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    float const *tau,
    iclaFloat_ptr dT,
    iclaFloat_ptr dB, icla_int_t lddb,
    float *hwork, icla_int_t lwork,
    icla_int_t *info);


icla_int_t
icla_sgeqrs3_gpu(
    icla_int_t m, icla_int_t n, icla_int_t nrhs,
    iclaFloat_ptr dA, icla_int_t ldda,
    float const *tau,
    iclaFloat_ptr dT,
    iclaFloat_ptr dB, icla_int_t lddb,
    float *hwork, icla_int_t lwork,
    icla_int_t *info);


icla_int_t
icla_sgerbt_gpu(
    icla_bool_t gen, icla_int_t n, icla_int_t nrhs,
    iclaFloat_ptr dA, icla_int_t ldda,
    iclaFloat_ptr dB, icla_int_t lddb,
    float *U, float *V,
    icla_int_t *info);


icla_int_t
icla_sgerfs_nopiv_gpu(
    icla_trans_t trans, icla_int_t n, icla_int_t nrhs,
    iclaFloat_ptr dA, icla_int_t ldda,
    iclaFloat_ptr dB, icla_int_t lddb,
    iclaFloat_ptr dX, icla_int_t lddx,
    iclaFloat_ptr dworkd, iclaFloat_ptr dAF,
    icla_int_t *iter,
    icla_int_t *info);

icla_int_t
icla_sgesdd(
    icla_vec_t jobz, icla_int_t m, icla_int_t n,
    float *A, icla_int_t lda,
    float *s,
    float *U, icla_int_t ldu,
    float *VT, icla_int_t ldvt,
    float *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    float *rwork,
    #endif
    icla_int_t *iwork,
    icla_int_t *info);

icla_int_t
icla_sgesv(
    icla_int_t n, icla_int_t nrhs,
    float *A, icla_int_t lda,
    icla_int_t *ipiv,
    float *B, icla_int_t ldb,
    icla_int_t *info);

icla_int_t
icla_sgesv_gpu(
    icla_int_t n, icla_int_t nrhs,
    iclaFloat_ptr dA, icla_int_t ldda,
    icla_int_t *ipiv,
    iclaFloat_ptr dB, icla_int_t lddb,
    icla_int_t *info);

icla_int_t
icla_sgesv_nopiv_gpu(
    icla_int_t n, icla_int_t nrhs,
    iclaFloat_ptr dA, icla_int_t ldda,
    iclaFloat_ptr dB, icla_int_t lddb,
    icla_int_t *info);


icla_int_t
icla_sgesv_rbt(
    icla_bool_t ref, icla_int_t n, icla_int_t nrhs,
    float *A, icla_int_t lda,
    float *B, icla_int_t ldb,
    icla_int_t *info);

icla_int_t
icla_sgesvd(
    icla_vec_t jobu, icla_vec_t jobvt, icla_int_t m, icla_int_t n,
    float *A,    icla_int_t lda, float *s,
    float *U,    icla_int_t ldu,
    float *VT,   icla_int_t ldvt,
    float *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    float *rwork,
    #endif
    icla_int_t *info);


icla_int_t
icla_sgetf2_gpu(
    icla_int_t m, icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    icla_int_t *ipiv,
    icla_queue_t queue,
    icla_int_t *info);

icla_int_t
icla_sgetf2_native_fused(
    icla_int_t m, icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    icla_int_t *ipiv, icla_int_t gbstep,
    icla_int_t *flags,
    icla_int_t *info, icla_queue_t queue );

icla_int_t
icla_sgetf2_native(
    icla_int_t m, icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    icla_int_t *dipiv, icla_int_t* dipivinfo,
    icla_int_t *dinfo, icla_int_t gbstep,
    icla_event_t events[2],
    icla_queue_t queue, icla_queue_t update_queue);


icla_int_t
icla_sgetf2_nopiv(
    icla_int_t m, icla_int_t n,
    float *A, icla_int_t lda,
    icla_int_t *info);

icla_int_t
icla_sgetrf_recpanel_native(
    icla_int_t m, icla_int_t n, icla_int_t recnb,
    iclaFloat_ptr dA, icla_int_t ldda,
    icla_int_t* dipiv, icla_int_t* dipivinfo,
    icla_int_t *dinfo, icla_int_t gbstep,
    icla_event_t events[2], icla_queue_t queue, icla_queue_t update_queue);

icla_int_t
icla_sgetrf(
    icla_int_t m, icla_int_t n,
    float *A, icla_int_t lda,
    icla_int_t *ipiv,
    icla_int_t *info);

icla_int_t
icla_sgetrf_gpu(
    icla_int_t m, icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    icla_int_t *ipiv,
    icla_int_t *info, icla_queue_t the_queue, real_Double_t *gpu_time);

icla_int_t
icla_sgetrf_expert_gpu_work(
    icla_int_t m, icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    icla_int_t *ipiv,
    icla_int_t *info, icla_mode_t mode,
    icla_int_t nb, icla_int_t recnb,
    void* host_work,   icla_int_t *lwork_host,
    void* device_work, icla_int_t *lwork_device,
    icla_event_t events[2], icla_queue_t queues[2] );

icla_int_t
icla_sgetrf_gpu_expert(
    icla_int_t m, icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    icla_int_t *ipiv,
    icla_int_t *info,
    icla_int_t nb, icla_mode_t mode);

icla_int_t
icla_sgetrf_native(
    icla_int_t m, icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    icla_int_t *ipiv,
    icla_int_t *info );


icla_int_t
icla_sgetrf_m(
    icla_int_t ngpu,
    icla_int_t m, icla_int_t n,
    float *A, icla_int_t lda,
    icla_int_t *ipiv,
    icla_int_t *info);

icla_int_t
icla_sgetrf_mgpu(
    icla_int_t ngpu,
    icla_int_t m, icla_int_t n,
    iclaFloat_ptr d_lA[], icla_int_t ldda,
    icla_int_t *ipiv,
    icla_int_t *info);


icla_int_t
icla_sgetrf2(
    icla_int_t m, icla_int_t n,
    float *A, icla_int_t lda,
    icla_int_t *ipiv,
    icla_int_t *info);

icla_int_t
icla_sgetrf2_mgpu(
    icla_int_t ngpu,
    icla_int_t m, icla_int_t n, icla_int_t nb, icla_int_t offset,
    iclaFloat_ptr d_lAT[], icla_int_t lddat,
    icla_int_t *ipiv,
    iclaFloat_ptr d_lAP[],
    float *W, icla_int_t ldw,
    icla_queue_t queues[][2],
    icla_int_t *info);


icla_int_t
icla_sgetrf_nopiv(
    icla_int_t m, icla_int_t n,
    float *A, icla_int_t lda,
    icla_int_t *info);


icla_int_t
icla_sgetrf_nopiv_gpu(
    icla_int_t m, icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    icla_int_t *info);

icla_int_t
icla_sgetri_gpu(
    icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    icla_int_t *ipiv,
    iclaFloat_ptr dwork, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_sgetri_expert_gpu_work(
    icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda, icla_int_t *ipiv,
    icla_int_t *info,
    icla_mode_t mode,
    void* host_work,   icla_int_t *lwork_host,
    void* device_work, icla_int_t *lwork_device,
    icla_queue_t queues[2] );

icla_int_t
icla_sgetrs_gpu(
    icla_trans_t trans, icla_int_t n, icla_int_t nrhs,
    iclaFloat_ptr dA, icla_int_t ldda,
    icla_int_t *ipiv,
    iclaFloat_ptr dB, icla_int_t lddb,
    icla_int_t *info);

icla_int_t
icla_sgetrs_expert_gpu_work(
    icla_trans_t trans, icla_int_t n, icla_int_t nrhs,
    iclaFloat_ptr dA, icla_int_t ldda, icla_int_t *ipiv,
    iclaFloat_ptr dB, icla_int_t lddb,
    icla_int_t *info,
    icla_mode_t mode,
    void* host_work,   icla_int_t *lwork_host,
    void* device_work, icla_int_t *lwork_device,
    icla_queue_t queue );


icla_int_t
icla_sgetrs_nopiv_gpu(
    icla_trans_t trans, icla_int_t n, icla_int_t nrhs,
    iclaFloat_ptr dA, icla_int_t ldda,
    iclaFloat_ptr dB, icla_int_t lddb,
    icla_int_t *info);


icla_int_t
icla_ssyevd(
    icla_vec_t jobz, icla_uplo_t uplo, icla_int_t n,
    float *A, icla_int_t lda,
    float *w,
    float *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    float *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);


icla_int_t
icla_ssyevd_gpu(
    icla_vec_t jobz, icla_uplo_t uplo,
    icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    float *w,
    float *wA,  icla_int_t ldwa,
    float *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    float *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);


icla_int_t
icla_ssyevd_m(
    icla_int_t ngpu,
    icla_vec_t jobz, icla_uplo_t uplo,
    icla_int_t n,
    float *A, icla_int_t lda,
    float *w,
    float *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    float *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);


icla_int_t
icla_ssyevdx(
    icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo, icla_int_t n,
    float *A, icla_int_t lda,
    float vl, float vu, icla_int_t il, icla_int_t iu,
    icla_int_t *mout, float *w,
    float *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    float *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);


icla_int_t
icla_ssyevdx_gpu(
    icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo,
    icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    float vl, float vu,
    icla_int_t il, icla_int_t iu,
    icla_int_t *mout, float *w,
    float *wA,  icla_int_t ldwa,
    float *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    float *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);


icla_int_t
icla_ssyevdx_m(
    icla_int_t ngpu,
    icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo,
    icla_int_t n,
    float *A, icla_int_t lda,
    float vl, float vu, icla_int_t il, icla_int_t iu,
    icla_int_t *mout, float *w,
    float *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    float *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);


icla_int_t
icla_ssyevdx_2stage(
    icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo,
    icla_int_t n,
    float *A, icla_int_t lda,
    float vl, float vu, icla_int_t il, icla_int_t iu,
    icla_int_t *mout, float *w,
    float *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    float *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);


icla_int_t
icla_ssyevdx_2stage_m(
    icla_int_t ngpu,
    icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo,
    icla_int_t n,
    float *A, icla_int_t lda,
    float vl, float vu, icla_int_t il, icla_int_t iu,
    icla_int_t *mout, float *w,
    float *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    float *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

#ifdef ICLA_COMPLEX


icla_int_t
icla_ssyevr(
    icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo, icla_int_t n,
    float *A, icla_int_t lda,
    float vl, float vu,
    icla_int_t il, icla_int_t iu, float abstol, icla_int_t *mout,
    float *w,
    float *Z, icla_int_t ldz,
    icla_int_t *isuppz,
    float *work, icla_int_t lwork,
    float *rwork, icla_int_t lrwork,
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);


icla_int_t
icla_ssyevr_gpu(
    icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo, icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    float vl, float vu,
    icla_int_t il, icla_int_t iu, float abstol, icla_int_t *mout,
    float *w,
    iclaFloat_ptr dZ, icla_int_t lddz,
    icla_int_t *isuppz,
    float *wA, icla_int_t ldwa,
    float *wZ, icla_int_t ldwz,
    float *work, icla_int_t lwork,
    float *rwork, icla_int_t lrwork,
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);


icla_int_t
icla_ssyevx(
    icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo, icla_int_t n,
    float *A, icla_int_t lda,
    float vl, float vu,
    icla_int_t il, icla_int_t iu, float abstol, icla_int_t *mout,
    float *w,
    float *Z, icla_int_t ldz,
    float *work, icla_int_t lwork,
    float *rwork, icla_int_t *iwork,
    icla_int_t *ifail,
    icla_int_t *info);


icla_int_t
icla_ssyevx_gpu(
    icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo, icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    float vl, float vu, icla_int_t il, icla_int_t iu,
    float abstol, icla_int_t *mout,
    float *w,
    iclaFloat_ptr dZ, icla_int_t lddz,
    float *wA, icla_int_t ldwa,
    float *wZ, icla_int_t ldwz,
    float *work, icla_int_t lwork,
    float *rwork, icla_int_t *iwork,
    icla_int_t *ifail,
    icla_int_t *info);
#endif


icla_int_t
icla_ssygst(
    icla_int_t itype, icla_uplo_t uplo, icla_int_t n,
    float *A, icla_int_t lda,
    float *B, icla_int_t ldb,
    icla_int_t *info);


icla_int_t
icla_ssygst_gpu(
    icla_int_t itype, icla_uplo_t uplo, icla_int_t n,
    iclaFloat_ptr       dA, icla_int_t ldda,
    iclaFloat_const_ptr dB, icla_int_t lddb,
    icla_int_t *info);


icla_int_t
icla_ssygst_m(
    icla_int_t ngpu,
    icla_int_t itype, icla_uplo_t uplo, icla_int_t n,
    float *A, icla_int_t lda,
    float *B, icla_int_t ldb,
    icla_int_t *info);


icla_int_t
icla_ssygvd(
    icla_int_t itype, icla_vec_t jobz, icla_uplo_t uplo, icla_int_t n,
    float *A, icla_int_t lda,
    float *B, icla_int_t ldb,
    float *w, float *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    float *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);


icla_int_t
icla_ssygvd_m(
    icla_int_t ngpu,
    icla_int_t itype, icla_vec_t jobz, icla_uplo_t uplo,
    icla_int_t n,
    float *A, icla_int_t lda,
    float *B, icla_int_t ldb,
    float *w,
    float *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    float *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);


icla_int_t
icla_ssygvdx(
    icla_int_t itype, icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo,
    icla_int_t n, float *A, icla_int_t lda,
    float *B, icla_int_t ldb,
    float vl, float vu, icla_int_t il, icla_int_t iu,
    icla_int_t *mout, float *w,
    float *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    float *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);


icla_int_t
icla_ssygvdx_m(
    icla_int_t ngpu,
    icla_int_t itype, icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo,
    icla_int_t n,
    float *A, icla_int_t lda,
    float *B, icla_int_t ldb,
    float vl, float vu, icla_int_t il, icla_int_t iu,
    icla_int_t *mout, float *w,
    float *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    float *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);


icla_int_t
icla_ssygvdx_2stage(
    icla_int_t itype, icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo, icla_int_t n,
    float *A, icla_int_t lda,
    float *B, icla_int_t ldb,
    float vl, float vu, icla_int_t il, icla_int_t iu,
    icla_int_t *mout, float *w,
    float *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    float *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);


icla_int_t
icla_ssygvdx_2stage_m(
    icla_int_t ngpu,
    icla_int_t itype, icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo,
    icla_int_t n,
    float *A, icla_int_t lda,
    float *B, icla_int_t ldb,
    float vl, float vu, icla_int_t il, icla_int_t iu,
    icla_int_t *mout, float *w,
    float *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    float *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

#ifdef ICLA_COMPLEX


icla_int_t
icla_ssygvr(
    icla_int_t itype, icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo, icla_int_t n,
    float *A, icla_int_t lda,
    float *B, icla_int_t ldb,
    float vl, float vu, icla_int_t il, icla_int_t iu,
    float abstol, icla_int_t *mout, float *w,
    float *Z, icla_int_t ldz,
    icla_int_t *isuppz, float *work, icla_int_t lwork,
    float *rwork, icla_int_t lrwork,
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);



icla_int_t
icla_ssygvx(
    icla_int_t itype, icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo,
    icla_int_t n, float *A, icla_int_t lda,
    float *B, icla_int_t ldb,
    float vl, float vu, icla_int_t il, icla_int_t iu,
    float abstol, icla_int_t *mout, float *w,
    float *Z, icla_int_t ldz,
    float *work, icla_int_t lwork, float *rwork,
    icla_int_t *iwork, icla_int_t *ifail,
    icla_int_t *info);
#endif

icla_int_t
icla_ssysv(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nrhs,
    float *A, icla_int_t lda,
    icla_int_t *ipiv,
    float *B, icla_int_t ldb,
    icla_int_t *info);


icla_int_t
icla_ssysv_nopiv_gpu(
    icla_uplo_t uplo,  icla_int_t n, icla_int_t nrhs,
    iclaFloat_ptr dA, icla_int_t ldda,
    iclaFloat_ptr dB, icla_int_t lddb,
    icla_int_t *info);

icla_int_t
icla_ssytrd(
    icla_uplo_t uplo, icla_int_t n,
    float *A, icla_int_t lda,
    float *d, float *e, float *tau,
    float *work, icla_int_t lwork,
    icla_int_t *info);


icla_int_t
icla_ssytrd_gpu(
    icla_uplo_t uplo, icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    float *d, float *e, float *tau,
    float *wA,  icla_int_t ldwa,
    float *work, icla_int_t lwork,
    icla_int_t *info);


icla_int_t
icla_ssytrd2_gpu(
    icla_uplo_t uplo, icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    float *d, float *e, float *tau,
    float *wA,  icla_int_t ldwa,
    float *work, icla_int_t lwork,
    iclaFloat_ptr dwork, icla_int_t ldwork,
    icla_int_t *info);



icla_int_t
icla_ssytrd_mgpu(
    icla_int_t ngpu, icla_int_t nqueue,
    icla_uplo_t uplo, icla_int_t n,
    float *A, icla_int_t lda,
    float *d, float *e, float *tau,
    float *work, icla_int_t lwork,
    icla_int_t *info);


icla_int_t
icla_ssytrd_sb2st(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nb, icla_int_t Vblksiz,
    float *A, icla_int_t lda,
    float *d, float *e,
    float *V, icla_int_t ldv,
    float *TAU, icla_int_t compT,
    float *T, icla_int_t ldt);


icla_int_t
icla_ssytrd_sy2sb(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nb,
    float *A, icla_int_t lda,
    float *tau,
    float *work, icla_int_t lwork,
    iclaFloat_ptr dT,
    icla_int_t *info);


icla_int_t
icla_ssytrd_sy2sb_mgpu(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nb,
    float *A, icla_int_t lda,
    float *tau,
    float *work, icla_int_t lwork,
    iclaFloat_ptr dAmgpu[], icla_int_t ldda,
    iclaFloat_ptr dTmgpu[], icla_int_t lddt,
    icla_int_t ngpu, icla_int_t distblk,
    icla_queue_t queues[][20], icla_int_t nqueue,
    icla_int_t *info);

icla_int_t
icla_ssytrf(
    icla_uplo_t uplo, icla_int_t n,
    float *A, icla_int_t lda,
    icla_int_t *ipiv,
    icla_int_t *info);

icla_int_t
icla_ssytrf_gpu(
   icla_uplo_t uplo, icla_int_t n,
   float *dA, icla_int_t ldda,
   icla_int_t *ipiv,
   icla_int_t *info);


icla_int_t
icla_ssytrf_aasen(
    icla_uplo_t uplo, icla_int_t cpu_panel, icla_int_t n,
    float *A, icla_int_t lda,
    icla_int_t *ipiv, icla_int_t *info);

icla_int_t
icla_ssytrf_nopiv(
    icla_uplo_t uplo, icla_int_t n,
    float *A, icla_int_t lda,
    icla_int_t *info);

icla_int_t
icla_ssytrf_nopiv_cpu(
    icla_uplo_t uplo, icla_int_t n, icla_int_t ib,
    float *A, icla_int_t lda,
    icla_int_t *info);

icla_int_t
icla_ssytrf_nopiv_gpu(
    icla_uplo_t uplo, icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    icla_int_t *info);


icla_int_t
icla_ssytrs_nopiv_gpu(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nrhs,
    iclaFloat_ptr dA, icla_int_t ldda,
    iclaFloat_ptr dB, icla_int_t lddb,
    icla_int_t *info);


#ifdef ICLA_REAL

icla_int_t
icla_slaex0(
    icla_int_t n, float *d, float *e,
    float *Q, icla_int_t ldq,
    float *work, icla_int_t *iwork,
    iclaFloat_ptr dwork,
    icla_range_t range, float vl, float vu, icla_int_t il, icla_int_t iu,
    icla_int_t *info);


icla_int_t
icla_slaex0_m(
    icla_int_t ngpu,
    icla_int_t n, float *d, float *e,
    float *Q, icla_int_t ldq,
    float *work, icla_int_t *iwork,
    icla_range_t range, float vl, float vu,
    icla_int_t il, icla_int_t iu,
    icla_int_t *info);

icla_int_t
icla_slaex1(
    icla_int_t n, float *d,
    float *Q, icla_int_t ldq,
    icla_int_t *indxq, float rho, icla_int_t cutpnt,
    float *work, icla_int_t *iwork,
    iclaFloat_ptr dwork,
    icla_queue_t queue,
    icla_range_t range, float vl, float vu, icla_int_t il, icla_int_t iu,
    icla_int_t *info);


icla_int_t
icla_slaex1_m(
    icla_int_t ngpu,
    icla_int_t n, float *d,
    float *Q, icla_int_t ldq,
    icla_int_t *indxq, float rho, icla_int_t cutpnt,
    float *work, icla_int_t *iwork,
    iclaFloat_ptr dwork[],
    icla_queue_t queues[IclaMaxGPUs][2],
    icla_range_t range, float vl, float vu,
    icla_int_t il, icla_int_t iu, icla_int_t *info);

icla_int_t
icla_slaex3(
    icla_int_t k, icla_int_t n, icla_int_t n1, float *d,
    float *Q, icla_int_t ldq,
    float rho,
    float *dlamda, float *Q2, icla_int_t *indx,
    icla_int_t *ctot, float *w, float *s, icla_int_t *indxq,
    iclaFloat_ptr dwork,
    icla_queue_t queue,
    icla_range_t range, float vl, float vu, icla_int_t il, icla_int_t iu,
    icla_int_t *info);


icla_int_t
icla_slaex3_m(
    icla_int_t ngpu,
    icla_int_t k, icla_int_t n, icla_int_t n1, float *d,
    float *Q, icla_int_t ldq, float rho,
    float *dlamda, float *Q2, icla_int_t *indx,
    icla_int_t *ctot, float *w, float *s, icla_int_t *indxq,
    iclaFloat_ptr dwork[],
    icla_queue_t queues[IclaMaxGPUs][2],
    icla_range_t range, float vl, float vu, icla_int_t il, icla_int_t iu,
    icla_int_t *info);
#endif

icla_int_t
icla_slabrd_gpu(
    icla_int_t m, icla_int_t n, icla_int_t nb,
    float     *A, icla_int_t lda,
    iclaFloat_ptr dA, icla_int_t ldda,
    float *d, float *e, float *tauq, float *taup,
    float     *X, icla_int_t ldx,
    iclaFloat_ptr dX, icla_int_t lddx,
    float     *Y, icla_int_t ldy,
    iclaFloat_ptr dY, icla_int_t lddy,
    float  *work, icla_int_t lwork,
    icla_queue_t queue);

icla_int_t
icla_slasyf_gpu(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nb, icla_int_t *kb,
    iclaFloat_ptr dA, icla_int_t ldda,
    icla_int_t *ipiv,
    iclaFloat_ptr dW, icla_int_t lddw,
    icla_queue_t queues[],
    icla_int_t *info);

icla_int_t
icla_slahr2(
    icla_int_t n, icla_int_t k, icla_int_t nb,
    iclaFloat_ptr dA, icla_int_t ldda,
    iclaFloat_ptr dV, icla_int_t lddv,
    float *A,  icla_int_t lda,
    float *tau,
    float *T,  icla_int_t ldt,
    float *Y,  icla_int_t ldy,
    icla_queue_t queue);


icla_int_t
icla_slahr2_m(
    icla_int_t n, icla_int_t k, icla_int_t nb,
    float *A, icla_int_t lda,
    float *tau,
    float *T, icla_int_t ldt,
    float *Y, icla_int_t ldy,
    struct sgehrd_data *data);

icla_int_t
icla_slahru(
    icla_int_t n, icla_int_t ihi, icla_int_t k, icla_int_t nb,
    float     *A, icla_int_t lda,
    iclaFloat_ptr dA, icla_int_t ldda,
    iclaFloat_ptr dY, icla_int_t lddy,
    iclaFloat_ptr dV, icla_int_t lddv,
    iclaFloat_ptr dT,
    iclaFloat_ptr dwork,
    icla_queue_t queue);


icla_int_t
icla_slahru_m(
    icla_int_t n, icla_int_t ihi, icla_int_t k, icla_int_t nb,
    float *A, icla_int_t lda,
    struct sgehrd_data *data);

#ifdef ICLA_REAL

icla_int_t
icla_slaln2(
    icla_int_t trans, icla_int_t na, icla_int_t nw,
    float smin, float ca, const float *A, icla_int_t lda,
    float d1, float d2,   const float *B, icla_int_t ldb,
    float wr, float wi, float *X, icla_int_t ldx,
    float *scale, float *xnorm,
    icla_int_t *info);
#endif


icla_int_t
icla_slaqps(
    icla_int_t m, icla_int_t n, icla_int_t offset,
    icla_int_t nb, icla_int_t *kb,
    float *A,  icla_int_t lda,
    iclaFloat_ptr dA, icla_int_t ldda,
    icla_int_t *jpvt, float *tau, float *vn1, float *vn2,
    float *auxv,
    float *F,  icla_int_t ldf,
    iclaFloat_ptr dF, icla_int_t lddf);


icla_int_t
icla_slaqps_gpu(
    icla_int_t m, icla_int_t n, icla_int_t offset,
    icla_int_t nb, icla_int_t *kb,
    iclaFloat_ptr dA,  icla_int_t ldda,
    icla_int_t *jpvt, float *tau,
    float *vn1, float *vn2,
    iclaFloat_ptr dauxv,
    iclaFloat_ptr dF, icla_int_t lddf);


icla_int_t
icla_slaqps2_gpu(
    icla_int_t m, icla_int_t n, icla_int_t offset,
    icla_int_t nb, icla_int_t *kb,
    iclaFloat_ptr dA,  icla_int_t ldda,
    icla_int_t *jpvt,
    iclaFloat_ptr dtau,
    iclaFloat_ptr dvn1, iclaFloat_ptr dvn2,
    iclaFloat_ptr dauxv,
    iclaFloat_ptr dF,  icla_int_t lddf,
    iclaFloat_ptr dlsticcs,
    icla_queue_t queue);

#ifdef ICLA_REAL

icla_int_t
icla_slaqtrsd(
    icla_trans_t trans, icla_int_t n,
    const float *T, icla_int_t ldt,
    float *x,       icla_int_t ldx,
    const float *cnorm,
    icla_int_t *info);
#endif


icla_int_t
icla_slarf_gpu(
    icla_int_t m,  icla_int_t n,
    iclaFloat_const_ptr dv, iclaFloat_const_ptr dtau,
    iclaFloat_ptr dC, icla_int_t lddc,
    icla_queue_t queue);






icla_int_t
icla_slarfb2_gpu(
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaFloat_const_ptr dV, icla_int_t lddv,
    iclaFloat_const_ptr dT, icla_int_t lddt,
    iclaFloat_ptr dC,       icla_int_t lddc,
    iclaFloat_ptr dwork,    icla_int_t ldwork,
    icla_queue_t queue);

icla_int_t
icla_slatrd(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nb,
    float *A, icla_int_t lda,
    float *e, float *tau,
    float *W, icla_int_t ldw,
    float *work, icla_int_t lwork,
    iclaFloat_ptr dA, icla_int_t ldda,
    iclaFloat_ptr dW, icla_int_t lddw,
    icla_queue_t queue);


icla_int_t
icla_slatrd2(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nb,
    float *A,  icla_int_t lda,
    float *e, float *tau,
    float *W,  icla_int_t ldw,
    float *work, icla_int_t lwork,
    iclaFloat_ptr dA, icla_int_t ldda,
    iclaFloat_ptr dW, icla_int_t lddw,
    iclaFloat_ptr dwork, icla_int_t ldwork,
    icla_queue_t queue);


icla_int_t
icla_slatrd_mgpu(
    icla_int_t ngpu,
    icla_uplo_t uplo,
    icla_int_t n, icla_int_t nb, icla_int_t nb0,
    float *A,  icla_int_t lda,
    float *e, float *tau,
    float    *W,       icla_int_t ldw,
    iclaFloat_ptr dA[],    icla_int_t ldda, icla_int_t offset,
    iclaFloat_ptr dW[],    icla_int_t lddw,
    float    *hwork,   icla_int_t lhwork,
    iclaFloat_ptr dwork[], icla_int_t ldwork,
    icla_queue_t queues[]);

#ifdef ICLA_COMPLEX

icla_int_t
icla_slatrsd(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_diag_t diag, icla_bool_t normin,
    icla_int_t n, const float *A, icla_int_t lda,
    float lambda,
    float *x,
    float *scale, float *cnorm,
    icla_int_t *info);
#endif

icla_int_t
icla_slauum(
    icla_uplo_t uplo, icla_int_t n,
    float *A, icla_int_t lda,
    icla_int_t *info);

icla_int_t
icla_slauum_gpu(
    icla_uplo_t uplo, icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    icla_int_t *info);


icla_int_t
icla_sposv(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nrhs,
    float *A, icla_int_t lda,
    float *B, icla_int_t ldb,
    icla_int_t *info);

icla_int_t
icla_sposv_gpu(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nrhs,
    iclaFloat_ptr dA, icla_int_t ldda,
    iclaFloat_ptr dB, icla_int_t lddb,
    icla_int_t *info);


icla_int_t
icla_spotf2_gpu(
    icla_uplo_t uplo, icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    icla_queue_t queue,
    icla_int_t *info);

icla_int_t
icla_spotf2_native(
    icla_uplo_t uplo, icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    icla_int_t step, icla_int_t *device_info,
    icla_queue_t queue );

icla_int_t
icla_spotrf_rectile_native(
    icla_uplo_t uplo, icla_int_t n, icla_int_t recnb,
    float* dA,    icla_int_t ldda, icla_int_t gbstep,
    icla_int_t *dinfo,  icla_int_t *info, icla_queue_t queue);

icla_int_t
icla_spotrf(
    icla_uplo_t uplo, icla_int_t n,
    float *A, icla_int_t lda,
    icla_int_t *info);

icla_int_t
icla_spotrf_expert_gpu_work(
    icla_uplo_t uplo, icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    icla_int_t *info,
    icla_mode_t mode,
    icla_int_t nb, icla_int_t recnb,
    void* host_work,   icla_int_t *lwork_host,
    void* device_work, icla_int_t *lwork_device,
    icla_event_t events[2], icla_queue_t queues[2] );

icla_int_t
icla_spotrf_expert_gpu(
    icla_uplo_t uplo, icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    icla_int_t *info,
    icla_int_t nb, icla_mode_t mode );

icla_int_t
icla_spotrf_gpu(
    icla_uplo_t uplo, icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    icla_int_t *info, icla_queue_t queue, iclaDouble_ptr time);

icla_int_t
icla_spotrf_native(
    icla_uplo_t uplo, icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    icla_int_t *info );


icla_int_t
icla_spotrf_m(
    icla_int_t ngpu,
    icla_uplo_t uplo, icla_int_t n,
    float *A, icla_int_t lda,
    icla_int_t *info);

icla_int_t
icla_spotrf_mgpu(
    icla_int_t ngpu,
    icla_uplo_t uplo, icla_int_t n,
    iclaFloat_ptr d_lA[], icla_int_t ldda,
    icla_int_t *info);


icla_int_t
icla_spotrf_mgpu_right(
    icla_int_t ngpu,
    icla_uplo_t uplo, icla_int_t n,
    iclaFloat_ptr d_lA[], icla_int_t ldda,
    icla_int_t *info);


icla_int_t
icla_spotrf3_mgpu(
    icla_int_t ngpu,
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    icla_int_t off_i, icla_int_t off_j, icla_int_t nb,
    iclaFloat_ptr d_lA[], icla_int_t ldda,
    iclaFloat_ptr d_lP[], icla_int_t lddp,
    float *A, icla_int_t lda, icla_int_t h,
    icla_queue_t queues[][3], icla_event_t events[][5],
    icla_int_t *info);

icla_int_t
icla_spotri(
    icla_uplo_t uplo, icla_int_t n,
    float *A, icla_int_t lda,
    icla_int_t *info);

icla_int_t
icla_spotri_gpu(
    icla_uplo_t uplo, icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    icla_int_t *info);

icla_int_t
icla_spotrs_gpu(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nrhs,
    iclaFloat_ptr dA, icla_int_t ldda,
    iclaFloat_ptr dB, icla_int_t lddb,
    icla_int_t *info);

icla_int_t
icla_spotrs_expert_gpu_work(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nrhs,
    iclaFloat_ptr dA, icla_int_t ldda,
    iclaFloat_ptr dB, icla_int_t lddb,
    icla_int_t *info,
    void* host_work,   icla_int_t *lwork_host,
    void* device_work, icla_int_t *lwork_device,
    icla_queue_t queue );


#ifdef ICLA_COMPLEX

icla_int_t
icla_ssysv_nopiv_gpu(
    icla_uplo_t uplo,  icla_int_t n, icla_int_t nrhs,
    iclaFloat_ptr dA, icla_int_t ldda,
    iclaFloat_ptr dB, icla_int_t lddb,
    icla_int_t *info);


icla_int_t
icla_ssytrf_nopiv_cpu(
    icla_uplo_t uplo, icla_int_t n, icla_int_t ib,
    float *A, icla_int_t lda,
    icla_int_t *info);


icla_int_t
icla_ssytrf_nopiv_gpu(
    icla_uplo_t uplo, icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    icla_int_t *info);


icla_int_t
icla_ssytrs_nopiv_gpu(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nrhs,
    iclaFloat_ptr dA, icla_int_t ldda,
    iclaFloat_ptr dB, icla_int_t lddb,
    icla_int_t *info);
#endif


icla_int_t
icla_sstedx(
    icla_range_t range, icla_int_t n, float vl, float vu,
    icla_int_t il, icla_int_t iu, float *d, float *e,
    float *Z, icla_int_t ldz,
    float *rwork, icla_int_t lrwork,
    icla_int_t *iwork, icla_int_t liwork,
    iclaFloat_ptr dwork,
    icla_int_t *info);


icla_int_t
icla_sstedx_m(
    icla_int_t ngpu,
    icla_range_t range, icla_int_t n, float vl, float vu,
    icla_int_t il, icla_int_t iu, float *d, float *e,
    float *Z, icla_int_t ldz,
    float *rwork, icla_int_t lrwork,
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);



icla_int_t
icla_strevc3(
    icla_side_t side, icla_vec_t howmany,
    icla_int_t *select, icla_int_t n,
    float *T,  icla_int_t ldt,
    float *VL, icla_int_t ldvl,
    float *VR, icla_int_t ldvr,
    icla_int_t mm, icla_int_t *mout,
    float *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    float *rwork,
    #endif
    icla_int_t *info);


icla_int_t
icla_strevc3_mt(
    icla_side_t side, icla_vec_t howmany,
    icla_int_t *select, icla_int_t n,
    float *T,  icla_int_t ldt,
    float *VL, icla_int_t ldvl,
    float *VR, icla_int_t ldvr,
    icla_int_t mm, icla_int_t *mout,
    float *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    float *rwork,
    #endif
    icla_int_t *info);


icla_int_t
icla_strsm_m(
    icla_int_t ngpu,
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transa, icla_diag_t diag,
    icla_int_t m, icla_int_t n, float alpha,
    const float *A, icla_int_t lda,
    float       *B, icla_int_t ldb);

icla_int_t
icla_strtri(
    icla_uplo_t uplo, icla_diag_t diag, icla_int_t n,
    float *A, icla_int_t lda,
    icla_int_t *info);

icla_int_t
icla_strtri_gpu(
    icla_uplo_t uplo, icla_diag_t diag, icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    icla_int_t *info);

icla_int_t
icla_strtri_expert_gpu_work(
    icla_uplo_t uplo, icla_diag_t diag, icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    icla_int_t *info,
    void* host_work,   icla_int_t *lwork_host,
    void* device_work, icla_int_t *lwork_device,
    icla_queue_t queues[2] );



icla_int_t
icla_sorgbr(
    icla_vect_t vect, icla_int_t m, icla_int_t n, icla_int_t k,
    float *A, icla_int_t lda,
    float *tau,
    float *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_sorghr(
    icla_int_t n, icla_int_t ilo, icla_int_t ihi,
    float *A, icla_int_t lda,
    float *tau,
    iclaFloat_ptr dT, icla_int_t nb,
    icla_int_t *info);


icla_int_t
icla_sorghr_m(
    icla_int_t n, icla_int_t ilo, icla_int_t ihi,
    float *A, icla_int_t lda,
    float *tau,
    float *T, icla_int_t nb,
    icla_int_t *info);


icla_int_t
icla_sorglq(
    icla_int_t m, icla_int_t n, icla_int_t k,
    float *A, icla_int_t lda,
    float *tau,
    iclaFloat_ptr dT, icla_int_t nb,
    icla_int_t *info);

icla_int_t
icla_sorgqr(
    icla_int_t m, icla_int_t n, icla_int_t k,
    float *A, icla_int_t lda,
    float *tau,
    iclaFloat_ptr dT, icla_int_t nb,
    icla_int_t *info);


icla_int_t
icla_sorgqr_gpu(
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaFloat_ptr dA, icla_int_t ldda,
    float *tau,
    iclaFloat_ptr dT, icla_int_t nb,
    icla_int_t *info);


icla_int_t
icla_sorgqr_m(
    icla_int_t m, icla_int_t n, icla_int_t k,
    float *A, icla_int_t lda,
    float *tau,
    float *T, icla_int_t nb,
    icla_int_t *info);

icla_int_t
icla_sorgqr2(
    icla_int_t m, icla_int_t n, icla_int_t k,
    float *A, icla_int_t lda,
    float *tau,
    icla_int_t *info);

icla_int_t
icla_sormbr(
    icla_vect_t vect, icla_side_t side, icla_trans_t trans,
    icla_int_t m, icla_int_t n, icla_int_t k,
    float *A, icla_int_t lda,
    float *tau,
    float *C, icla_int_t ldc,
    float *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_sormlq(
    icla_side_t side, icla_trans_t trans,
    icla_int_t m, icla_int_t n, icla_int_t k,
    float *A, icla_int_t lda,
    float *tau,
    float *C, icla_int_t ldc,
    float *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_sormrq(
    icla_side_t side, icla_trans_t trans,
    icla_int_t m, icla_int_t n, icla_int_t k,
    float *A, icla_int_t lda,
    float *tau,
    float *C, icla_int_t ldc,
    float *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_sormql(
    icla_side_t side, icla_trans_t trans,
    icla_int_t m, icla_int_t n, icla_int_t k,
    float *A, icla_int_t lda,
    float *tau,
    float *C, icla_int_t ldc,
    float *work, icla_int_t lwork,
    icla_int_t *info);


icla_int_t
icla_sormql2_gpu(
    icla_side_t side, icla_trans_t trans,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaFloat_ptr dA, icla_int_t ldda,
    float *tau,
    iclaFloat_ptr dC, icla_int_t lddc,
    const float *wA, icla_int_t ldwa,
    icla_int_t *info);

icla_int_t
icla_sormqr(
    icla_side_t side, icla_trans_t trans,
    icla_int_t m, icla_int_t n, icla_int_t k,
    float *A, icla_int_t lda,
    float *tau,
    float *C, icla_int_t ldc,
    float *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_sormqr_gpu(
    icla_side_t side, icla_trans_t trans,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    float const   *tau,
    iclaFloat_ptr       dC, icla_int_t lddc,
    float       *hwork, icla_int_t lwork,
    iclaFloat_ptr       dT, icla_int_t nb,
    icla_int_t *info);


icla_int_t
icla_sormqr2_gpu(
    icla_side_t side, icla_trans_t trans,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaFloat_ptr dA, icla_int_t ldda,
    float *tau,
    iclaFloat_ptr dC, icla_int_t lddc,
    const float *wA, icla_int_t ldwa,
    icla_int_t *info);


icla_int_t
icla_sormqr_m(
    icla_int_t ngpu,
    icla_side_t side, icla_trans_t trans,
    icla_int_t m, icla_int_t n, icla_int_t k,
    float *A,    icla_int_t lda,
    float *tau,
    float *C,    icla_int_t ldc,
    float *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_sormtr(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t m, icla_int_t n,
    float *A,    icla_int_t lda,
    float *tau,
    float *C,    icla_int_t ldc,
    float *work, icla_int_t lwork,
    icla_int_t *info);


icla_int_t
icla_sormtr_gpu(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t m, icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    float *tau,
    iclaFloat_ptr dC, icla_int_t lddc,
    const float *wA, icla_int_t ldwa,
    icla_int_t *info);


icla_int_t
icla_sormtr_m(
    icla_int_t ngpu,
    icla_side_t side, icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t m, icla_int_t n,
    float *A,    icla_int_t lda,
    float *tau,
    float *C,    icla_int_t ldc,
    float *work, icla_int_t lwork,
    icla_int_t *info);




extern const float ICLA_S_NAN;
extern const float ICLA_S_INF;

int icla_s_isnan( float x );
int icla_s_isinf( float x );
int icla_s_isnan_inf( float x );

float
icla_smake_lwork( icla_int_t lwork );

icla_int_t
icla_snan_inf(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    const float *A, icla_int_t lda,
    icla_int_t *cnt_nan,
    icla_int_t *cnt_inf);

icla_int_t
icla_snan_inf_gpu(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    icla_int_t *cnt_nan,
    icla_int_t *cnt_inf,
    icla_queue_t queue);

void icla_sprint(
    icla_int_t m, icla_int_t n,
    const float *A, icla_int_t lda);

void icla_sprint_gpu(
    icla_int_t m, icla_int_t n,
    iclaFloat_const_ptr dA, icla_int_t ldda,
    icla_queue_t queue);

void icla_spanel_to_q(
    icla_uplo_t uplo, icla_int_t ib,
    float *A, icla_int_t lda,
    float *work);

void icla_sq_to_panel(
    icla_uplo_t uplo, icla_int_t ib,
    float *A, icla_int_t lda,
    float *work);



void
iclablas_sextract_diag_sqrt(
    icla_int_t m, icla_int_t n,
    float* dA, icla_int_t ldda,
    float* dD, icla_int_t incd,
    icla_queue_t queue);

void
iclablas_sscal_shift_hpd(
    icla_uplo_t uplo, int n,
    float* dA, int ldda,
    float* dD, int incd,
    float miu, float cn, float eps,
    icla_queue_t queue);

void
iclablas_sdimv_invert(
    icla_int_t n,
    float alpha, float* dD, icla_int_t incd,
                              float* dx, icla_int_t incx,
    float beta,  float* dy, icla_int_t incy,
    icla_queue_t queue);

#ifdef __cplusplus
}
#endif

#undef ICLA_REAL

#endif

