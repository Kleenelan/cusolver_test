
#ifndef ICLA_C_H
#define ICLA_C_H

#include "icla_types.h"
#include "icla_cgehrd_m.h"

#define ICLA_COMPLEX

#ifdef __cplusplus
extern "C" {
#endif

#ifdef ICLA_REAL
icla_int_t icla_get_slaex3_m_nb();

#endif

icla_int_t icla_get_cpotrf_nb( icla_int_t n );
icla_int_t icla_get_cgetrf_nb( icla_int_t m, icla_int_t n );
icla_int_t icla_get_cgetrf_native_nb( icla_int_t m, icla_int_t n );
icla_int_t icla_get_cgetri_nb( icla_int_t n );
icla_int_t icla_get_chetrf_nb( icla_int_t n );
icla_int_t icla_get_chetrf_nopiv_nb( icla_int_t n );
icla_int_t icla_get_chetrf_aasen_nb( icla_int_t n );

icla_int_t icla_get_cgeqp3_nb( icla_int_t m, icla_int_t n );
icla_int_t icla_get_cgeqrf_nb( icla_int_t m, icla_int_t n );
icla_int_t icla_get_cgeqlf_nb( icla_int_t m, icla_int_t n );
icla_int_t icla_get_cgelqf_nb( icla_int_t m, icla_int_t n );

icla_int_t icla_get_cgehrd_nb( icla_int_t n );
icla_int_t icla_get_chetrd_nb( icla_int_t n );
icla_int_t icla_get_chegst_nb( icla_int_t n );
icla_int_t icla_get_chegst_m_nb( icla_int_t n );

icla_int_t icla_get_cgebrd_nb( icla_int_t m, icla_int_t n );
icla_int_t icla_get_cgesvd_nb( icla_int_t m, icla_int_t n );

icla_int_t icla_get_cbulge_nb( icla_int_t n, icla_int_t nbthreads );
icla_int_t icla_get_cbulge_nb_mgpu( icla_int_t n );
icla_int_t icla_get_cbulge_vblksiz( icla_int_t n, icla_int_t nb, icla_int_t nbthreads );
icla_int_t icla_get_cbulge_gcperf();

bool icla_cgetrf_gpu_recommend_cpu(icla_int_t m, icla_int_t n, icla_int_t nb);
bool icla_cgetrf_native_recommend_notrans(icla_int_t m, icla_int_t n, icla_int_t nb);

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
icla_cvrange(
    icla_int_t k, float *d, icla_int_t *il, icla_int_t *iu, float vl, float vu);

void
icla_cirange(
    icla_int_t k, icla_int_t *indxq, icla_int_t *iil, icla_int_t *iiu, icla_int_t il, icla_int_t iu);
#endif

icla_int_t
icla_cgbsv_native(
        icla_int_t n, icla_int_t kl, icla_int_t ku, icla_int_t nrhs,
        iclaFloatComplex* dA, icla_int_t ldda, icla_int_t* dipiv,
        iclaFloatComplex* dB, icla_int_t lddb,
        icla_int_t *info);

icla_int_t
icla_cgbtf2_native_v2(
    icla_int_t m, icla_int_t n, icla_int_t kl, icla_int_t ku,
    iclaFloatComplex* dA, icla_int_t ldda, icla_int_t* ipiv,
    icla_int_t* info, icla_queue_t queue);

icla_int_t
icla_cgbtf2_native_v2_work(
    icla_int_t m, icla_int_t n, icla_int_t kl, icla_int_t ku,
    iclaFloatComplex* dA, icla_int_t ldda, icla_int_t* ipiv,
    icla_int_t* info,
    void* device_work, icla_int_t* lwork,
    icla_queue_t queue);

void
icla_cgbsv_native_work(
        icla_int_t n, icla_int_t kl, icla_int_t ku, icla_int_t nrhs,
        iclaFloatComplex* dA, icla_int_t ldda, icla_int_t* dipiv,
        iclaFloatComplex* dB, icla_int_t lddb,
        icla_int_t *info, void* device_work, icla_int_t* lwork,
        icla_queue_t queue);

icla_int_t
icla_cgbtf2_native(
    icla_int_t m, icla_int_t n, icla_int_t kl, icla_int_t ku,
    iclaFloatComplex* dA, icla_int_t ldda, icla_int_t* ipiv,
    icla_int_t* info, icla_queue_t queue);

icla_int_t
icla_cgbtf2_native_work(
    icla_int_t m, icla_int_t n, icla_int_t kl, icla_int_t ku,
    iclaFloatComplex* dA, icla_int_t ldda, icla_int_t* ipiv,
    icla_int_t* info,
    void* device_work, icla_int_t* lwork,
    icla_queue_t queue);

icla_int_t
icla_cgbtrf_native(
    icla_int_t m, icla_int_t n,
    icla_int_t kl, icla_int_t ku,
    iclaFloatComplex* dAB, icla_int_t lddab, icla_int_t* dipiv,
    icla_int_t *info);

void
icla_cgbtrf_native_work(
    icla_int_t m, icla_int_t n,
    icla_int_t kl, icla_int_t ku,
    iclaFloatComplex* dAB, icla_int_t lddab,
    icla_int_t* dipiv, icla_int_t *info,
    void* device_work, icla_int_t* lwork,
    icla_queue_t queue);

icla_int_t
icla_cgebrd(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex *A, icla_int_t lda,
    float *d, float *e,
    iclaFloatComplex *tauq, iclaFloatComplex *taup,
    iclaFloatComplex *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_cgeev(
    icla_vec_t jobvl, icla_vec_t jobvr, icla_int_t n,
    iclaFloatComplex *A, icla_int_t lda,
    #ifdef ICLA_COMPLEX
    iclaFloatComplex *w,
    #else
    float *wr, float *wi,
    #endif
    iclaFloatComplex *VL, icla_int_t ldvl,
    iclaFloatComplex *VR, icla_int_t ldvr,
    iclaFloatComplex *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    float *rwork,
    #endif
    icla_int_t *info);

icla_int_t
icla_cgeev_m(
    icla_vec_t jobvl, icla_vec_t jobvr, icla_int_t n,
    iclaFloatComplex *A, icla_int_t lda,
    #ifdef ICLA_COMPLEX
    iclaFloatComplex *w,
    #else
    float *wr, float *wi,
    #endif
    iclaFloatComplex *VL, icla_int_t ldvl,
    iclaFloatComplex *VR, icla_int_t ldvr,
    iclaFloatComplex *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    float *rwork,
    #endif
    icla_int_t *info);

icla_int_t
icla_cgegqr_gpu(
    icla_int_t ikind, icla_int_t m, icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr dwork, iclaFloatComplex *work,
    icla_int_t *info);

icla_int_t
icla_cgegqr_expert_gpu_work(
    icla_int_t ikind, icla_int_t m, icla_int_t n,
    iclaFloatComplex_ptr dA,   icla_int_t ldda,
    void *host_work,   icla_int_t *lwork_host,
    void *device_work, icla_int_t *lwork_device,
    icla_int_t *info, icla_queue_t queue );

icla_int_t
icla_cgehrd(
    icla_int_t n, icla_int_t ilo, icla_int_t ihi,
    iclaFloatComplex *A, icla_int_t lda,
    iclaFloatComplex *tau,
    iclaFloatComplex *work, icla_int_t lwork,
    iclaFloatComplex_ptr dT,
    icla_int_t *info);

icla_int_t
icla_cgehrd_m(
    icla_int_t n, icla_int_t ilo, icla_int_t ihi,
    iclaFloatComplex *A, icla_int_t lda,
    iclaFloatComplex *tau,
    iclaFloatComplex *work, icla_int_t lwork,
    iclaFloatComplex *T,
    icla_int_t *info);

icla_int_t
icla_cgehrd2(
    icla_int_t n, icla_int_t ilo, icla_int_t ihi,
    iclaFloatComplex *A, icla_int_t lda,
    iclaFloatComplex *tau,
    iclaFloatComplex *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_cgelqf(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex *A,    icla_int_t lda,
    iclaFloatComplex *tau,
    iclaFloatComplex *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_cgelqf_gpu(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    iclaFloatComplex *tau,
    iclaFloatComplex *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_cgels(
    icla_trans_t trans, icla_int_t m, icla_int_t n, icla_int_t nrhs,
    iclaFloatComplex_ptr A, icla_int_t lda,
    iclaFloatComplex_ptr B, icla_int_t ldb,
    iclaFloatComplex *hwork, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_cggrqf(
    icla_int_t m, icla_int_t p, icla_int_t n,
    iclaFloatComplex *A, icla_int_t lda,
    iclaFloatComplex *taua,
    iclaFloatComplex *B, icla_int_t ldb,
    iclaFloatComplex *taub,
    iclaFloatComplex *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_cgglse(
    icla_int_t m, icla_int_t n, icla_int_t p,
    iclaFloatComplex *A, icla_int_t lda,
    iclaFloatComplex *B, icla_int_t ldb,
    iclaFloatComplex *c, iclaFloatComplex *d, iclaFloatComplex *x,
    iclaFloatComplex *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_cgels_gpu(
    icla_trans_t trans, icla_int_t m, icla_int_t n, icla_int_t nrhs,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr dB, icla_int_t lddb,
    iclaFloatComplex *hwork, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_cgels3_gpu(
    icla_trans_t trans, icla_int_t m, icla_int_t n, icla_int_t nrhs,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr dB, icla_int_t lddb,
    iclaFloatComplex *hwork, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_cgeqlf(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex *A,    icla_int_t lda,
    iclaFloatComplex *tau,
    iclaFloatComplex *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_cgeqp3(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex *A, icla_int_t lda,
    icla_int_t *jpvt, iclaFloatComplex *tau,
    iclaFloatComplex *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    float *rwork,
    #endif
    icla_int_t *info);

icla_int_t
icla_cgeqp3_gpu(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    icla_int_t *jpvt, iclaFloatComplex *tau,
    iclaFloatComplex_ptr dwork, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    float *rwork,
    #endif
    icla_int_t *info);

icla_int_t
icla_cgeqp3_expert_gpu_work(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    icla_int_t *jpvt, iclaFloatComplex *tau,
    void* host_work,   icla_int_t *lwork_host,
    void* device_work, icla_int_t *lwork_device,
    icla_int_t *info, icla_queue_t queue );

icla_int_t
icla_cgeqr2_gpu(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr dtau,
    iclaFloat_ptr        dwork,
    icla_queue_t queue,
    icla_int_t *info);

icla_int_t
icla_cgeqr2x_gpu(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr dtau,
    iclaFloatComplex_ptr dT, iclaFloatComplex_ptr ddA,
    iclaFloat_ptr        dwork,
    icla_int_t *info);

icla_int_t
icla_cgeqr2x2_gpu(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr dtau,
    iclaFloatComplex_ptr dT, iclaFloatComplex_ptr ddA,
    iclaFloat_ptr        dwork,
    icla_int_t *info);

icla_int_t
icla_cgeqr2x3_gpu(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr dtau,
    iclaFloatComplex_ptr dT,
    iclaFloatComplex_ptr ddA,
    iclaFloat_ptr        dwork,
    icla_int_t *info);

icla_int_t
icla_cgeqr2x4_gpu(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr dtau,
    iclaFloatComplex_ptr dT,
    iclaFloatComplex_ptr ddA,
    iclaFloat_ptr        dwork,
    icla_queue_t queue,
    icla_int_t *info);

icla_int_t
icla_cgeqrf(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex *A, icla_int_t lda,
    iclaFloatComplex *tau,
    iclaFloatComplex *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_cgeqrf_gpu(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    iclaFloatComplex *tau,
    iclaFloatComplex_ptr dT,
    icla_int_t *info);

icla_int_t
icla_cgeqrf_expert_gpu_work(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    iclaFloatComplex *tau, iclaFloatComplex_ptr dT,
    icla_int_t *info,
    icla_mode_t mode, icla_int_t nb,
    void* host_work,   icla_int_t *lwork_host,
    void* device_work, icla_int_t *lwork_device,
    icla_queue_t queues[2] );

icla_int_t
icla_cgeqrf_m(
    icla_int_t ngpu,
    icla_int_t m, icla_int_t n,
    iclaFloatComplex *A,    icla_int_t lda,
    iclaFloatComplex *tau,
    iclaFloatComplex *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_cgeqrf_ooc(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex *A, icla_int_t lda,
    iclaFloatComplex *tau,
    iclaFloatComplex *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_cgeqrf2_gpu(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    iclaFloatComplex *tau,
    icla_int_t *info);

icla_int_t
icla_cgeqrf2_mgpu(
    icla_int_t ngpu,
    icla_int_t m, icla_int_t n,
    iclaFloatComplex_ptr d_lA[], icla_int_t ldda,
    iclaFloatComplex *tau,
    icla_int_t *info);

icla_int_t
icla_cgeqrf3_gpu(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    iclaFloatComplex *tau,
    iclaFloatComplex_ptr dT,
    icla_int_t *info);

icla_int_t
icla_cgeqrs_gpu(
    icla_int_t m, icla_int_t n, icla_int_t nrhs,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex const *tau,
    iclaFloatComplex_ptr dT,
    iclaFloatComplex_ptr dB, icla_int_t lddb,
    iclaFloatComplex *hwork, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_cgeqrs3_gpu(
    icla_int_t m, icla_int_t n, icla_int_t nrhs,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    iclaFloatComplex const *tau,
    iclaFloatComplex_ptr dT,
    iclaFloatComplex_ptr dB, icla_int_t lddb,
    iclaFloatComplex *hwork, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_cgerbt_gpu(
    icla_bool_t gen, icla_int_t n, icla_int_t nrhs,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr dB, icla_int_t lddb,
    iclaFloatComplex *U, iclaFloatComplex *V,
    icla_int_t *info);

icla_int_t
icla_cgerfs_nopiv_gpu(
    icla_trans_t trans, icla_int_t n, icla_int_t nrhs,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr dB, icla_int_t lddb,
    iclaFloatComplex_ptr dX, icla_int_t lddx,
    iclaFloatComplex_ptr dworkd, iclaFloatComplex_ptr dAF,
    icla_int_t *iter,
    icla_int_t *info);

icla_int_t
icla_cgesdd(
    icla_vec_t jobz, icla_int_t m, icla_int_t n,
    iclaFloatComplex *A, icla_int_t lda,
    float *s,
    iclaFloatComplex *U, icla_int_t ldu,
    iclaFloatComplex *VT, icla_int_t ldvt,
    iclaFloatComplex *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    float *rwork,
    #endif
    icla_int_t *iwork,
    icla_int_t *info);

icla_int_t
icla_cgesv(
    icla_int_t n, icla_int_t nrhs,
    iclaFloatComplex *A, icla_int_t lda,
    icla_int_t *ipiv,
    iclaFloatComplex *B, icla_int_t ldb,
    icla_int_t *info);

icla_int_t
icla_cgesv_gpu(
    icla_int_t n, icla_int_t nrhs,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    icla_int_t *ipiv,
    iclaFloatComplex_ptr dB, icla_int_t lddb,
    icla_int_t *info);

icla_int_t
icla_cgesv_nopiv_gpu(
    icla_int_t n, icla_int_t nrhs,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr dB, icla_int_t lddb,
    icla_int_t *info);

icla_int_t
icla_cgesv_rbt(
    icla_bool_t ref, icla_int_t n, icla_int_t nrhs,
    iclaFloatComplex *A, icla_int_t lda,
    iclaFloatComplex *B, icla_int_t ldb,
    icla_int_t *info);

icla_int_t
icla_cgesvd(
    icla_vec_t jobu, icla_vec_t jobvt, icla_int_t m, icla_int_t n,
    iclaFloatComplex *A,    icla_int_t lda, float *s,
    iclaFloatComplex *U,    icla_int_t ldu,
    iclaFloatComplex *VT,   icla_int_t ldvt,
    iclaFloatComplex *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    float *rwork,
    #endif
    icla_int_t *info);

icla_int_t
icla_cgetf2_gpu(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    icla_int_t *ipiv,
    icla_queue_t queue,
    icla_int_t *info);

icla_int_t
icla_cgetf2_native_fused(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    icla_int_t *ipiv, icla_int_t gbstep,
    icla_int_t *flags,
    icla_int_t *info, icla_queue_t queue );

icla_int_t
icla_cgetf2_native(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    icla_int_t *dipiv, icla_int_t* dipivinfo,
    icla_int_t *dinfo, icla_int_t gbstep,
    icla_event_t events[2],
    icla_queue_t queue, icla_queue_t update_queue);

icla_int_t
icla_cgetf2_nopiv(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex *A, icla_int_t lda,
    icla_int_t *info);

icla_int_t
icla_cgetrf_recpanel_native(
    icla_int_t m, icla_int_t n, icla_int_t recnb,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    icla_int_t* dipiv, icla_int_t* dipivinfo,
    icla_int_t *dinfo, icla_int_t gbstep,
    icla_event_t events[2], icla_queue_t queue, icla_queue_t update_queue);

icla_int_t
icla_cgetrf(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex *A, icla_int_t lda,
    icla_int_t *ipiv,
    icla_int_t *info);

icla_int_t
icla_cgetrf_gpu(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    icla_int_t *ipiv,
    icla_int_t *info);

icla_int_t
icla_cgetrf_expert_gpu_work(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    icla_int_t *ipiv,
    icla_int_t *info, icla_mode_t mode,
    icla_int_t nb, icla_int_t recnb,
    void* host_work,   icla_int_t *lwork_host,
    void* device_work, icla_int_t *lwork_device,
    icla_event_t events[2], icla_queue_t queues[2] );

icla_int_t
icla_cgetrf_gpu_expert(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    icla_int_t *ipiv,
    icla_int_t *info,
    icla_int_t nb, icla_mode_t mode);

icla_int_t
icla_cgetrf_native(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    icla_int_t *ipiv,
    icla_int_t *info );

icla_int_t
icla_cgetrf_m(
    icla_int_t ngpu,
    icla_int_t m, icla_int_t n,
    iclaFloatComplex *A, icla_int_t lda,
    icla_int_t *ipiv,
    icla_int_t *info);

icla_int_t
icla_cgetrf_mgpu(
    icla_int_t ngpu,
    icla_int_t m, icla_int_t n,
    iclaFloatComplex_ptr d_lA[], icla_int_t ldda,
    icla_int_t *ipiv,
    icla_int_t *info);

icla_int_t
icla_cgetrf2(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex *A, icla_int_t lda,
    icla_int_t *ipiv,
    icla_int_t *info);

icla_int_t
icla_cgetrf2_mgpu(
    icla_int_t ngpu,
    icla_int_t m, icla_int_t n, icla_int_t nb, icla_int_t offset,
    iclaFloatComplex_ptr d_lAT[], icla_int_t lddat,
    icla_int_t *ipiv,
    iclaFloatComplex_ptr d_lAP[],
    iclaFloatComplex *W, icla_int_t ldw,
    icla_queue_t queues[][2],
    icla_int_t *info);

icla_int_t
icla_cgetrf_nopiv(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex *A, icla_int_t lda,
    icla_int_t *info);

icla_int_t
icla_cgetrf_nopiv_gpu(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    icla_int_t *info);

icla_int_t
icla_cgetri_gpu(
    icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    icla_int_t *ipiv,
    iclaFloatComplex_ptr dwork, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_cgetri_expert_gpu_work(
    icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda, icla_int_t *ipiv,
    icla_int_t *info,
    icla_mode_t mode,
    void* host_work,   icla_int_t *lwork_host,
    void* device_work, icla_int_t *lwork_device,
    icla_queue_t queues[2] );

icla_int_t
icla_cgetrs_gpu(
    icla_trans_t trans, icla_int_t n, icla_int_t nrhs,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    icla_int_t *ipiv,
    iclaFloatComplex_ptr dB, icla_int_t lddb,
    icla_int_t *info);

icla_int_t
icla_cgetrs_expert_gpu_work(
    icla_trans_t trans, icla_int_t n, icla_int_t nrhs,
    iclaFloatComplex_ptr dA, icla_int_t ldda, icla_int_t *ipiv,
    iclaFloatComplex_ptr dB, icla_int_t lddb,
    icla_int_t *info,
    icla_mode_t mode,
    void* host_work,   icla_int_t *lwork_host,
    void* device_work, icla_int_t *lwork_device,
    icla_queue_t queue );

icla_int_t
icla_cgetrs_nopiv_gpu(
    icla_trans_t trans, icla_int_t n, icla_int_t nrhs,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr dB, icla_int_t lddb,
    icla_int_t *info);

icla_int_t
icla_cheevd(
    icla_vec_t jobz, icla_uplo_t uplo, icla_int_t n,
    iclaFloatComplex *A, icla_int_t lda,
    float *w,
    iclaFloatComplex *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    float *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

icla_int_t
icla_cheevd_gpu(
    icla_vec_t jobz, icla_uplo_t uplo,
    icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    float *w,
    iclaFloatComplex *wA,  icla_int_t ldwa,
    iclaFloatComplex *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    float *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

icla_int_t
icla_cheevd_m(
    icla_int_t ngpu,
    icla_vec_t jobz, icla_uplo_t uplo,
    icla_int_t n,
    iclaFloatComplex *A, icla_int_t lda,
    float *w,
    iclaFloatComplex *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    float *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

icla_int_t
icla_cheevdx(
    icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo, icla_int_t n,
    iclaFloatComplex *A, icla_int_t lda,
    float vl, float vu, icla_int_t il, icla_int_t iu,
    icla_int_t *mout, float *w,
    iclaFloatComplex *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    float *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

icla_int_t
icla_cheevdx_gpu(
    icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo,
    icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    float vl, float vu,
    icla_int_t il, icla_int_t iu,
    icla_int_t *mout, float *w,
    iclaFloatComplex *wA,  icla_int_t ldwa,
    iclaFloatComplex *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    float *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

icla_int_t
icla_cheevdx_m(
    icla_int_t ngpu,
    icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo,
    icla_int_t n,
    iclaFloatComplex *A, icla_int_t lda,
    float vl, float vu, icla_int_t il, icla_int_t iu,
    icla_int_t *mout, float *w,
    iclaFloatComplex *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    float *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

icla_int_t
icla_cheevdx_2stage(
    icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo,
    icla_int_t n,
    iclaFloatComplex *A, icla_int_t lda,
    float vl, float vu, icla_int_t il, icla_int_t iu,
    icla_int_t *mout, float *w,
    iclaFloatComplex *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    float *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

icla_int_t
icla_cheevdx_2stage_m(
    icla_int_t ngpu,
    icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo,
    icla_int_t n,
    iclaFloatComplex *A, icla_int_t lda,
    float vl, float vu, icla_int_t il, icla_int_t iu,
    icla_int_t *mout, float *w,
    iclaFloatComplex *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    float *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

#ifdef ICLA_COMPLEX

icla_int_t
icla_cheevr(
    icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo, icla_int_t n,
    iclaFloatComplex *A, icla_int_t lda,
    float vl, float vu,
    icla_int_t il, icla_int_t iu, float abstol, icla_int_t *mout,
    float *w,
    iclaFloatComplex *Z, icla_int_t ldz,
    icla_int_t *isuppz,
    iclaFloatComplex *work, icla_int_t lwork,
    float *rwork, icla_int_t lrwork,
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

icla_int_t
icla_cheevr_gpu(
    icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo, icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    float vl, float vu,
    icla_int_t il, icla_int_t iu, float abstol, icla_int_t *mout,
    float *w,
    iclaFloatComplex_ptr dZ, icla_int_t lddz,
    icla_int_t *isuppz,
    iclaFloatComplex *wA, icla_int_t ldwa,
    iclaFloatComplex *wZ, icla_int_t ldwz,
    iclaFloatComplex *work, icla_int_t lwork,
    float *rwork, icla_int_t lrwork,
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

icla_int_t
icla_cheevx(
    icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo, icla_int_t n,
    iclaFloatComplex *A, icla_int_t lda,
    float vl, float vu,
    icla_int_t il, icla_int_t iu, float abstol, icla_int_t *mout,
    float *w,
    iclaFloatComplex *Z, icla_int_t ldz,
    iclaFloatComplex *work, icla_int_t lwork,
    float *rwork, icla_int_t *iwork,
    icla_int_t *ifail,
    icla_int_t *info);

icla_int_t
icla_cheevx_gpu(
    icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo, icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    float vl, float vu, icla_int_t il, icla_int_t iu,
    float abstol, icla_int_t *mout,
    float *w,
    iclaFloatComplex_ptr dZ, icla_int_t lddz,
    iclaFloatComplex *wA, icla_int_t ldwa,
    iclaFloatComplex *wZ, icla_int_t ldwz,
    iclaFloatComplex *work, icla_int_t lwork,
    float *rwork, icla_int_t *iwork,
    icla_int_t *ifail,
    icla_int_t *info);
#endif

icla_int_t
icla_chegst(
    icla_int_t itype, icla_uplo_t uplo, icla_int_t n,
    iclaFloatComplex *A, icla_int_t lda,
    iclaFloatComplex *B, icla_int_t ldb,
    icla_int_t *info);

icla_int_t
icla_chegst_gpu(
    icla_int_t itype, icla_uplo_t uplo, icla_int_t n,
    iclaFloatComplex_ptr       dA, icla_int_t ldda,
    iclaFloatComplex_const_ptr dB, icla_int_t lddb,
    icla_int_t *info);

icla_int_t
icla_chegst_m(
    icla_int_t ngpu,
    icla_int_t itype, icla_uplo_t uplo, icla_int_t n,
    iclaFloatComplex *A, icla_int_t lda,
    iclaFloatComplex *B, icla_int_t ldb,
    icla_int_t *info);

icla_int_t
icla_chegvd(
    icla_int_t itype, icla_vec_t jobz, icla_uplo_t uplo, icla_int_t n,
    iclaFloatComplex *A, icla_int_t lda,
    iclaFloatComplex *B, icla_int_t ldb,
    float *w, iclaFloatComplex *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    float *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

icla_int_t
icla_chegvd_m(
    icla_int_t ngpu,
    icla_int_t itype, icla_vec_t jobz, icla_uplo_t uplo,
    icla_int_t n,
    iclaFloatComplex *A, icla_int_t lda,
    iclaFloatComplex *B, icla_int_t ldb,
    float *w,
    iclaFloatComplex *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    float *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

icla_int_t
icla_chegvdx(
    icla_int_t itype, icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo,
    icla_int_t n, iclaFloatComplex *A, icla_int_t lda,
    iclaFloatComplex *B, icla_int_t ldb,
    float vl, float vu, icla_int_t il, icla_int_t iu,
    icla_int_t *mout, float *w,
    iclaFloatComplex *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    float *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

icla_int_t
icla_chegvdx_m(
    icla_int_t ngpu,
    icla_int_t itype, icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo,
    icla_int_t n,
    iclaFloatComplex *A, icla_int_t lda,
    iclaFloatComplex *B, icla_int_t ldb,
    float vl, float vu, icla_int_t il, icla_int_t iu,
    icla_int_t *mout, float *w,
    iclaFloatComplex *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    float *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

icla_int_t
icla_chegvdx_2stage(
    icla_int_t itype, icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo, icla_int_t n,
    iclaFloatComplex *A, icla_int_t lda,
    iclaFloatComplex *B, icla_int_t ldb,
    float vl, float vu, icla_int_t il, icla_int_t iu,
    icla_int_t *mout, float *w,
    iclaFloatComplex *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    float *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

icla_int_t
icla_chegvdx_2stage_m(
    icla_int_t ngpu,
    icla_int_t itype, icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo,
    icla_int_t n,
    iclaFloatComplex *A, icla_int_t lda,
    iclaFloatComplex *B, icla_int_t ldb,
    float vl, float vu, icla_int_t il, icla_int_t iu,
    icla_int_t *mout, float *w,
    iclaFloatComplex *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    float *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

#ifdef ICLA_COMPLEX

icla_int_t
icla_chegvr(
    icla_int_t itype, icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo, icla_int_t n,
    iclaFloatComplex *A, icla_int_t lda,
    iclaFloatComplex *B, icla_int_t ldb,
    float vl, float vu, icla_int_t il, icla_int_t iu,
    float abstol, icla_int_t *mout, float *w,
    iclaFloatComplex *Z, icla_int_t ldz,
    icla_int_t *isuppz, iclaFloatComplex *work, icla_int_t lwork,
    float *rwork, icla_int_t lrwork,
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

icla_int_t
icla_chegvx(
    icla_int_t itype, icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo,
    icla_int_t n, iclaFloatComplex *A, icla_int_t lda,
    iclaFloatComplex *B, icla_int_t ldb,
    float vl, float vu, icla_int_t il, icla_int_t iu,
    float abstol, icla_int_t *mout, float *w,
    iclaFloatComplex *Z, icla_int_t ldz,
    iclaFloatComplex *work, icla_int_t lwork, float *rwork,
    icla_int_t *iwork, icla_int_t *ifail,
    icla_int_t *info);
#endif

icla_int_t
icla_chesv(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nrhs,
    iclaFloatComplex *A, icla_int_t lda,
    icla_int_t *ipiv,
    iclaFloatComplex *B, icla_int_t ldb,
    icla_int_t *info);

icla_int_t
icla_chesv_nopiv_gpu(
    icla_uplo_t uplo,  icla_int_t n, icla_int_t nrhs,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr dB, icla_int_t lddb,
    icla_int_t *info);

icla_int_t
icla_chetrd(
    icla_uplo_t uplo, icla_int_t n,
    iclaFloatComplex *A, icla_int_t lda,
    float *d, float *e, iclaFloatComplex *tau,
    iclaFloatComplex *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_chetrd_gpu(
    icla_uplo_t uplo, icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    float *d, float *e, iclaFloatComplex *tau,
    iclaFloatComplex *wA,  icla_int_t ldwa,
    iclaFloatComplex *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_chetrd2_gpu(
    icla_uplo_t uplo, icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    float *d, float *e, iclaFloatComplex *tau,
    iclaFloatComplex *wA,  icla_int_t ldwa,
    iclaFloatComplex *work, icla_int_t lwork,
    iclaFloatComplex_ptr dwork, icla_int_t ldwork,
    icla_int_t *info);

icla_int_t
icla_chetrd_mgpu(
    icla_int_t ngpu, icla_int_t nqueue,
    icla_uplo_t uplo, icla_int_t n,
    iclaFloatComplex *A, icla_int_t lda,
    float *d, float *e, iclaFloatComplex *tau,
    iclaFloatComplex *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_chetrd_hb2st(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nb, icla_int_t Vblksiz,
    iclaFloatComplex *A, icla_int_t lda,
    float *d, float *e,
    iclaFloatComplex *V, icla_int_t ldv,
    iclaFloatComplex *TAU, icla_int_t compT,
    iclaFloatComplex *T, icla_int_t ldt);

icla_int_t
icla_chetrd_he2hb(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nb,
    iclaFloatComplex *A, icla_int_t lda,
    iclaFloatComplex *tau,
    iclaFloatComplex *work, icla_int_t lwork,
    iclaFloatComplex_ptr dT,
    icla_int_t *info);

icla_int_t
icla_chetrd_he2hb_mgpu(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nb,
    iclaFloatComplex *A, icla_int_t lda,
    iclaFloatComplex *tau,
    iclaFloatComplex *work, icla_int_t lwork,
    iclaFloatComplex_ptr dAmgpu[], icla_int_t ldda,
    iclaFloatComplex_ptr dTmgpu[], icla_int_t lddt,
    icla_int_t ngpu, icla_int_t distblk,
    icla_queue_t queues[][20], icla_int_t nqueue,
    icla_int_t *info);

icla_int_t
icla_chetrf(
    icla_uplo_t uplo, icla_int_t n,
    iclaFloatComplex *A, icla_int_t lda,
    icla_int_t *ipiv,
    icla_int_t *info);

icla_int_t
icla_chetrf_gpu(
   icla_uplo_t uplo, icla_int_t n,
   iclaFloatComplex *dA, icla_int_t ldda,
   icla_int_t *ipiv,
   icla_int_t *info);

icla_int_t
icla_chetrf_aasen(
    icla_uplo_t uplo, icla_int_t cpu_panel, icla_int_t n,
    iclaFloatComplex *A, icla_int_t lda,
    icla_int_t *ipiv, icla_int_t *info);

icla_int_t
icla_chetrf_nopiv(
    icla_uplo_t uplo, icla_int_t n,
    iclaFloatComplex *A, icla_int_t lda,
    icla_int_t *info);

icla_int_t
icla_chetrf_nopiv_cpu(
    icla_uplo_t uplo, icla_int_t n, icla_int_t ib,
    iclaFloatComplex *A, icla_int_t lda,
    icla_int_t *info);

icla_int_t
icla_chetrf_nopiv_gpu(
    icla_uplo_t uplo, icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    icla_int_t *info);

icla_int_t
icla_chetrs_nopiv_gpu(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nrhs,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr dB, icla_int_t lddb,
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
    icla_queue_t queues[iclaMaxGPUs][2],
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
    icla_queue_t queues[iclaMaxGPUs][2],
    icla_range_t range, float vl, float vu, icla_int_t il, icla_int_t iu,
    icla_int_t *info);
#endif

icla_int_t
icla_clabrd_gpu(
    icla_int_t m, icla_int_t n, icla_int_t nb,
    iclaFloatComplex     *A, icla_int_t lda,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    float *d, float *e, iclaFloatComplex *tauq, iclaFloatComplex *taup,
    iclaFloatComplex     *X, icla_int_t ldx,
    iclaFloatComplex_ptr dX, icla_int_t lddx,
    iclaFloatComplex     *Y, icla_int_t ldy,
    iclaFloatComplex_ptr dY, icla_int_t lddy,
    iclaFloatComplex  *work, icla_int_t lwork,
    icla_queue_t queue);

icla_int_t
icla_clahef_gpu(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nb, icla_int_t *kb,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    icla_int_t *ipiv,
    iclaFloatComplex_ptr dW, icla_int_t lddw,
    icla_queue_t queues[],
    icla_int_t *info);

icla_int_t
icla_clahr2(
    icla_int_t n, icla_int_t k, icla_int_t nb,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr dV, icla_int_t lddv,
    iclaFloatComplex *A,  icla_int_t lda,
    iclaFloatComplex *tau,
    iclaFloatComplex *T,  icla_int_t ldt,
    iclaFloatComplex *Y,  icla_int_t ldy,
    icla_queue_t queue);

icla_int_t
icla_clahr2_m(
    icla_int_t n, icla_int_t k, icla_int_t nb,
    iclaFloatComplex *A, icla_int_t lda,
    iclaFloatComplex *tau,
    iclaFloatComplex *T, icla_int_t ldt,
    iclaFloatComplex *Y, icla_int_t ldy,
    struct cgehrd_data *data);

icla_int_t
icla_clahru(
    icla_int_t n, icla_int_t ihi, icla_int_t k, icla_int_t nb,
    iclaFloatComplex     *A, icla_int_t lda,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr dY, icla_int_t lddy,
    iclaFloatComplex_ptr dV, icla_int_t lddv,
    iclaFloatComplex_ptr dT,
    iclaFloatComplex_ptr dwork,
    icla_queue_t queue);

icla_int_t
icla_clahru_m(
    icla_int_t n, icla_int_t ihi, icla_int_t k, icla_int_t nb,
    iclaFloatComplex *A, icla_int_t lda,
    struct cgehrd_data *data);

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
icla_claqps(
    icla_int_t m, icla_int_t n, icla_int_t offset,
    icla_int_t nb, icla_int_t *kb,
    iclaFloatComplex *A,  icla_int_t lda,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    icla_int_t *jpvt, iclaFloatComplex *tau, float *vn1, float *vn2,
    iclaFloatComplex *auxv,
    iclaFloatComplex *F,  icla_int_t ldf,
    iclaFloatComplex_ptr dF, icla_int_t lddf);

icla_int_t
icla_claqps_gpu(
    icla_int_t m, icla_int_t n, icla_int_t offset,
    icla_int_t nb, icla_int_t *kb,
    iclaFloatComplex_ptr dA,  icla_int_t ldda,
    icla_int_t *jpvt, iclaFloatComplex *tau,
    float *vn1, float *vn2,
    iclaFloatComplex_ptr dauxv,
    iclaFloatComplex_ptr dF, icla_int_t lddf);

icla_int_t
icla_claqps2_gpu(
    icla_int_t m, icla_int_t n, icla_int_t offset,
    icla_int_t nb, icla_int_t *kb,
    iclaFloatComplex_ptr dA,  icla_int_t ldda,
    icla_int_t *jpvt,
    iclaFloatComplex_ptr dtau,
    iclaFloat_ptr dvn1, iclaFloat_ptr dvn2,
    iclaFloatComplex_ptr dauxv,
    iclaFloatComplex_ptr dF,  icla_int_t lddf,
    iclaFloat_ptr dlsticcs,
    icla_queue_t queue);

#ifdef ICLA_REAL

icla_int_t
icla_claqtrsd(
    icla_trans_t trans, icla_int_t n,
    const float *T, icla_int_t ldt,
    float *x,       icla_int_t ldx,
    const float *cnorm,
    icla_int_t *info);
#endif

icla_int_t
icla_clarf_gpu(
    icla_int_t m,  icla_int_t n,
    iclaFloatComplex_const_ptr dv, iclaFloatComplex_const_ptr dtau,
    iclaFloatComplex_ptr dC, icla_int_t lddc,
    icla_queue_t queue);

icla_int_t
icla_clarfb2_gpu(
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaFloatComplex_const_ptr dV, icla_int_t lddv,
    iclaFloatComplex_const_ptr dT, icla_int_t lddt,
    iclaFloatComplex_ptr dC,       icla_int_t lddc,
    iclaFloatComplex_ptr dwork,    icla_int_t ldwork,
    icla_queue_t queue);

icla_int_t
icla_clatrd(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nb,
    iclaFloatComplex *A, icla_int_t lda,
    float *e, iclaFloatComplex *tau,
    iclaFloatComplex *W, icla_int_t ldw,
    iclaFloatComplex *work, icla_int_t lwork,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr dW, icla_int_t lddw,
    icla_queue_t queue);

icla_int_t
icla_clatrd2(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nb,
    iclaFloatComplex *A,  icla_int_t lda,
    float *e, iclaFloatComplex *tau,
    iclaFloatComplex *W,  icla_int_t ldw,
    iclaFloatComplex *work, icla_int_t lwork,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr dW, icla_int_t lddw,
    iclaFloatComplex_ptr dwork, icla_int_t ldwork,
    icla_queue_t queue);

icla_int_t
icla_clatrd_mgpu(
    icla_int_t ngpu,
    icla_uplo_t uplo,
    icla_int_t n, icla_int_t nb, icla_int_t nb0,
    iclaFloatComplex *A,  icla_int_t lda,
    float *e, iclaFloatComplex *tau,
    iclaFloatComplex    *W,       icla_int_t ldw,
    iclaFloatComplex_ptr dA[],    icla_int_t ldda, icla_int_t offset,
    iclaFloatComplex_ptr dW[],    icla_int_t lddw,
    iclaFloatComplex    *hwork,   icla_int_t lhwork,
    iclaFloatComplex_ptr dwork[], icla_int_t ldwork,
    icla_queue_t queues[]);

#ifdef ICLA_COMPLEX

icla_int_t
icla_clatrsd(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_diag_t diag, icla_bool_t normin,
    icla_int_t n, const iclaFloatComplex *A, icla_int_t lda,
    iclaFloatComplex lambda,
    iclaFloatComplex *x,
    float *scale, float *cnorm,
    icla_int_t *info);
#endif

icla_int_t
icla_clauum(
    icla_uplo_t uplo, icla_int_t n,
    iclaFloatComplex *A, icla_int_t lda,
    icla_int_t *info);

icla_int_t
icla_clauum_gpu(
    icla_uplo_t uplo, icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    icla_int_t *info);

icla_int_t
icla_cposv(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nrhs,
    iclaFloatComplex *A, icla_int_t lda,
    iclaFloatComplex *B, icla_int_t ldb,
    icla_int_t *info);

icla_int_t
icla_cposv_gpu(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nrhs,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr dB, icla_int_t lddb,
    icla_int_t *info);

icla_int_t
icla_cpotf2_gpu(
    icla_uplo_t uplo, icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    icla_queue_t queue,
    icla_int_t *info);

icla_int_t
icla_cpotf2_native(
    icla_uplo_t uplo, icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    icla_int_t step, icla_int_t *device_info,
    icla_queue_t queue );

icla_int_t
icla_cpotrf_rectile_native(
    icla_uplo_t uplo, icla_int_t n, icla_int_t recnb,
    iclaFloatComplex* dA,    icla_int_t ldda, icla_int_t gbstep,
    icla_int_t *dinfo,  icla_int_t *info, icla_queue_t queue);

icla_int_t
icla_cpotrf(
    icla_uplo_t uplo, icla_int_t n,
    iclaFloatComplex *A, icla_int_t lda,
    icla_int_t *info);

icla_int_t
icla_cpotrf_expert_gpu_work(
    icla_uplo_t uplo, icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    icla_int_t *info,
    icla_mode_t mode,
    icla_int_t nb, icla_int_t recnb,
    void* host_work,   icla_int_t *lwork_host,
    void* device_work, icla_int_t *lwork_device,
    icla_event_t events[2], icla_queue_t queues[2] );

icla_int_t
icla_cpotrf_expert_gpu(
    icla_uplo_t uplo, icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    icla_int_t *info,
    icla_int_t nb, icla_mode_t mode );

icla_int_t
icla_cpotrf_gpu(
    icla_uplo_t uplo, icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    icla_int_t *info);

icla_int_t
icla_cpotrf_native(
    icla_uplo_t uplo, icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    icla_int_t *info );

icla_int_t
icla_cpotrf_m(
    icla_int_t ngpu,
    icla_uplo_t uplo, icla_int_t n,
    iclaFloatComplex *A, icla_int_t lda,
    icla_int_t *info);

icla_int_t
icla_cpotrf_mgpu(
    icla_int_t ngpu,
    icla_uplo_t uplo, icla_int_t n,
    iclaFloatComplex_ptr d_lA[], icla_int_t ldda,
    icla_int_t *info);

icla_int_t
icla_cpotrf_mgpu_right(
    icla_int_t ngpu,
    icla_uplo_t uplo, icla_int_t n,
    iclaFloatComplex_ptr d_lA[], icla_int_t ldda,
    icla_int_t *info);

icla_int_t
icla_cpotrf3_mgpu(
    icla_int_t ngpu,
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    icla_int_t off_i, icla_int_t off_j, icla_int_t nb,
    iclaFloatComplex_ptr d_lA[], icla_int_t ldda,
    iclaFloatComplex_ptr d_lP[], icla_int_t lddp,
    iclaFloatComplex *A, icla_int_t lda, icla_int_t h,
    icla_queue_t queues[][3], icla_event_t events[][5],
    icla_int_t *info);

icla_int_t
icla_cpotri(
    icla_uplo_t uplo, icla_int_t n,
    iclaFloatComplex *A, icla_int_t lda,
    icla_int_t *info);

icla_int_t
icla_cpotri_gpu(
    icla_uplo_t uplo, icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    icla_int_t *info);

icla_int_t
icla_cpotrs_gpu(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nrhs,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr dB, icla_int_t lddb,
    icla_int_t *info);

icla_int_t
icla_cpotrs_expert_gpu_work(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nrhs,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr dB, icla_int_t lddb,
    icla_int_t *info,
    void* host_work,   icla_int_t *lwork_host,
    void* device_work, icla_int_t *lwork_device,
    icla_queue_t queue );

#ifdef ICLA_COMPLEX

icla_int_t
icla_csysv_nopiv_gpu(
    icla_uplo_t uplo,  icla_int_t n, icla_int_t nrhs,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr dB, icla_int_t lddb,
    icla_int_t *info);

icla_int_t
icla_csytrf_nopiv_cpu(
    icla_uplo_t uplo, icla_int_t n, icla_int_t ib,
    iclaFloatComplex *A, icla_int_t lda,
    icla_int_t *info);

icla_int_t
icla_csytrf_nopiv_gpu(
    icla_uplo_t uplo, icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    icla_int_t *info);

icla_int_t
icla_csytrs_nopiv_gpu(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nrhs,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    iclaFloatComplex_ptr dB, icla_int_t lddb,
    icla_int_t *info);
#endif

icla_int_t
icla_cstedx(
    icla_range_t range, icla_int_t n, float vl, float vu,
    icla_int_t il, icla_int_t iu, float *d, float *e,
    iclaFloatComplex *Z, icla_int_t ldz,
    float *rwork, icla_int_t lrwork,
    icla_int_t *iwork, icla_int_t liwork,
    iclaFloat_ptr dwork,
    icla_int_t *info);

icla_int_t
icla_cstedx_m(
    icla_int_t ngpu,
    icla_range_t range, icla_int_t n, float vl, float vu,
    icla_int_t il, icla_int_t iu, float *d, float *e,
    iclaFloatComplex *Z, icla_int_t ldz,
    float *rwork, icla_int_t lrwork,
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

icla_int_t
icla_ctrevc3(
    icla_side_t side, icla_vec_t howmany,
    icla_int_t *select, icla_int_t n,
    iclaFloatComplex *T,  icla_int_t ldt,
    iclaFloatComplex *VL, icla_int_t ldvl,
    iclaFloatComplex *VR, icla_int_t ldvr,
    icla_int_t mm, icla_int_t *mout,
    iclaFloatComplex *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    float *rwork,
    #endif
    icla_int_t *info);

icla_int_t
icla_ctrevc3_mt(
    icla_side_t side, icla_vec_t howmany,
    icla_int_t *select, icla_int_t n,
    iclaFloatComplex *T,  icla_int_t ldt,
    iclaFloatComplex *VL, icla_int_t ldvl,
    iclaFloatComplex *VR, icla_int_t ldvr,
    icla_int_t mm, icla_int_t *mout,
    iclaFloatComplex *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    float *rwork,
    #endif
    icla_int_t *info);

icla_int_t
icla_ctrsm_m(
    icla_int_t ngpu,
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transa, icla_diag_t diag,
    icla_int_t m, icla_int_t n, iclaFloatComplex alpha,
    const iclaFloatComplex *A, icla_int_t lda,
    iclaFloatComplex       *B, icla_int_t ldb);

icla_int_t
icla_ctrtri(
    icla_uplo_t uplo, icla_diag_t diag, icla_int_t n,
    iclaFloatComplex *A, icla_int_t lda,
    icla_int_t *info);

icla_int_t
icla_ctrtri_gpu(
    icla_uplo_t uplo, icla_diag_t diag, icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    icla_int_t *info);

icla_int_t
icla_ctrtri_expert_gpu_work(
    icla_uplo_t uplo, icla_diag_t diag, icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    icla_int_t *info,
    void* host_work,   icla_int_t *lwork_host,
    void* device_work, icla_int_t *lwork_device,
    icla_queue_t queues[2] );

icla_int_t
icla_cungbr(
    icla_vect_t vect, icla_int_t m, icla_int_t n, icla_int_t k,
    iclaFloatComplex *A, icla_int_t lda,
    iclaFloatComplex *tau,
    iclaFloatComplex *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_cunghr(
    icla_int_t n, icla_int_t ilo, icla_int_t ihi,
    iclaFloatComplex *A, icla_int_t lda,
    iclaFloatComplex *tau,
    iclaFloatComplex_ptr dT, icla_int_t nb,
    icla_int_t *info);

icla_int_t
icla_cunghr_m(
    icla_int_t n, icla_int_t ilo, icla_int_t ihi,
    iclaFloatComplex *A, icla_int_t lda,
    iclaFloatComplex *tau,
    iclaFloatComplex *T, icla_int_t nb,
    icla_int_t *info);

icla_int_t
icla_cunglq(
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaFloatComplex *A, icla_int_t lda,
    iclaFloatComplex *tau,
    iclaFloatComplex_ptr dT, icla_int_t nb,
    icla_int_t *info);

icla_int_t
icla_cungqr(
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaFloatComplex *A, icla_int_t lda,
    iclaFloatComplex *tau,
    iclaFloatComplex_ptr dT, icla_int_t nb,
    icla_int_t *info);

icla_int_t
icla_cungqr_gpu(
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    iclaFloatComplex *tau,
    iclaFloatComplex_ptr dT, icla_int_t nb,
    icla_int_t *info);

icla_int_t
icla_cungqr_m(
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaFloatComplex *A, icla_int_t lda,
    iclaFloatComplex *tau,
    iclaFloatComplex *T, icla_int_t nb,
    icla_int_t *info);

icla_int_t
icla_cungqr2(
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaFloatComplex *A, icla_int_t lda,
    iclaFloatComplex *tau,
    icla_int_t *info);

icla_int_t
icla_cunmbr(
    icla_vect_t vect, icla_side_t side, icla_trans_t trans,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaFloatComplex *A, icla_int_t lda,
    iclaFloatComplex *tau,
    iclaFloatComplex *C, icla_int_t ldc,
    iclaFloatComplex *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_cunmlq(
    icla_side_t side, icla_trans_t trans,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaFloatComplex *A, icla_int_t lda,
    iclaFloatComplex *tau,
    iclaFloatComplex *C, icla_int_t ldc,
    iclaFloatComplex *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_cunmrq(
    icla_side_t side, icla_trans_t trans,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaFloatComplex *A, icla_int_t lda,
    iclaFloatComplex *tau,
    iclaFloatComplex *C, icla_int_t ldc,
    iclaFloatComplex *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_cunmql(
    icla_side_t side, icla_trans_t trans,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaFloatComplex *A, icla_int_t lda,
    iclaFloatComplex *tau,
    iclaFloatComplex *C, icla_int_t ldc,
    iclaFloatComplex *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_cunmql2_gpu(
    icla_side_t side, icla_trans_t trans,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    iclaFloatComplex *tau,
    iclaFloatComplex_ptr dC, icla_int_t lddc,
    const iclaFloatComplex *wA, icla_int_t ldwa,
    icla_int_t *info);

icla_int_t
icla_cunmqr(
    icla_side_t side, icla_trans_t trans,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaFloatComplex *A, icla_int_t lda,
    iclaFloatComplex *tau,
    iclaFloatComplex *C, icla_int_t ldc,
    iclaFloatComplex *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_cunmqr_gpu(
    icla_side_t side, icla_trans_t trans,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    iclaFloatComplex const   *tau,
    iclaFloatComplex_ptr       dC, icla_int_t lddc,
    iclaFloatComplex       *hwork, icla_int_t lwork,
    iclaFloatComplex_ptr       dT, icla_int_t nb,
    icla_int_t *info);

icla_int_t
icla_cunmqr2_gpu(
    icla_side_t side, icla_trans_t trans,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    iclaFloatComplex *tau,
    iclaFloatComplex_ptr dC, icla_int_t lddc,
    const iclaFloatComplex *wA, icla_int_t ldwa,
    icla_int_t *info);

icla_int_t
icla_cunmqr_m(
    icla_int_t ngpu,
    icla_side_t side, icla_trans_t trans,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaFloatComplex *A,    icla_int_t lda,
    iclaFloatComplex *tau,
    iclaFloatComplex *C,    icla_int_t ldc,
    iclaFloatComplex *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_cunmtr(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t m, icla_int_t n,
    iclaFloatComplex *A,    icla_int_t lda,
    iclaFloatComplex *tau,
    iclaFloatComplex *C,    icla_int_t ldc,
    iclaFloatComplex *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_cunmtr_gpu(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t m, icla_int_t n,
    iclaFloatComplex_ptr dA, icla_int_t ldda,
    iclaFloatComplex *tau,
    iclaFloatComplex_ptr dC, icla_int_t lddc,
    const iclaFloatComplex *wA, icla_int_t ldwa,
    icla_int_t *info);

icla_int_t
icla_cunmtr_m(
    icla_int_t ngpu,
    icla_side_t side, icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t m, icla_int_t n,
    iclaFloatComplex *A,    icla_int_t lda,
    iclaFloatComplex *tau,
    iclaFloatComplex *C,    icla_int_t ldc,
    iclaFloatComplex *work, icla_int_t lwork,
    icla_int_t *info);

extern const iclaFloatComplex ICLA_C_NAN;
extern const iclaFloatComplex ICLA_C_INF;

int icla_c_isnan( iclaFloatComplex x );
int icla_c_isinf( iclaFloatComplex x );
int icla_c_isnan_inf( iclaFloatComplex x );

iclaFloatComplex
icla_cmake_lwork( icla_int_t lwork );

icla_int_t
icla_cnan_inf(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    const iclaFloatComplex *A, icla_int_t lda,
    icla_int_t *cnt_nan,
    icla_int_t *cnt_inf);

icla_int_t
icla_cnan_inf_gpu(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    icla_int_t *cnt_nan,
    icla_int_t *cnt_inf,
    icla_queue_t queue);

void icla_cprint(
    icla_int_t m, icla_int_t n,
    const iclaFloatComplex *A, icla_int_t lda);

void icla_cprint_gpu(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex_const_ptr dA, icla_int_t ldda,
    icla_queue_t queue);

void icla_cpanel_to_q(
    icla_uplo_t uplo, icla_int_t ib,
    iclaFloatComplex *A, icla_int_t lda,
    iclaFloatComplex *work);

void icla_cq_to_panel(
    icla_uplo_t uplo, icla_int_t ib,
    iclaFloatComplex *A, icla_int_t lda,
    iclaFloatComplex *work);

void
iclablas_cextract_diag_sqrt(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex* dA, icla_int_t ldda,
    float* dD, icla_int_t incd,
    icla_queue_t queue);

void
iclablas_cscal_shift_hpd(
    icla_uplo_t uplo, int n,
    iclaFloatComplex* dA, int ldda,
    float* dD, int incd,
    float miu, float cn, float eps,
    icla_queue_t queue);

void
iclablas_cdimv_invert(
    icla_int_t n,
    iclaFloatComplex alpha, iclaFloatComplex* dD, icla_int_t incd,
                              iclaFloatComplex* dx, icla_int_t incx,
    iclaFloatComplex beta,  iclaFloatComplex* dy, icla_int_t incy,
    icla_queue_t queue);

#ifdef __cplusplus
}
#endif

#undef ICLA_COMPLEX

#endif

