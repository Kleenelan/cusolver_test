/*
    -- ICLA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
*/

#ifndef ICLA_Z_H
#define ICLA_Z_H

#include "icla_types.h"
#include "icla_zgehrd_m.h"

#define ICLA_COMPLEX

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// ICLA Auxiliary functions to get the NB used

#ifdef ICLA_REAL
icla_int_t icla_get_dlaex3_m_nb();       // defined in dlaex3_m.cpp
#endif

// Cholesky, LU, symmetric indefinite
icla_int_t icla_get_zpotrf_nb( icla_int_t n );
icla_int_t icla_get_zgetrf_nb( icla_int_t m, icla_int_t n );
icla_int_t icla_get_zgetrf_native_nb( icla_int_t m, icla_int_t n );
icla_int_t icla_get_zgetri_nb( icla_int_t n );
icla_int_t icla_get_zhetrf_nb( icla_int_t n );
icla_int_t icla_get_zhetrf_nopiv_nb( icla_int_t n );
icla_int_t icla_get_zhetrf_aasen_nb( icla_int_t n );

// QR
icla_int_t icla_get_zgeqp3_nb( icla_int_t m, icla_int_t n );
icla_int_t icla_get_zgeqrf_nb( icla_int_t m, icla_int_t n );
icla_int_t icla_get_zgeqlf_nb( icla_int_t m, icla_int_t n );
icla_int_t icla_get_zgelqf_nb( icla_int_t m, icla_int_t n );

// eigenvalues
icla_int_t icla_get_zgehrd_nb( icla_int_t n );
icla_int_t icla_get_zhetrd_nb( icla_int_t n );
icla_int_t icla_get_zhegst_nb( icla_int_t n );
icla_int_t icla_get_zhegst_m_nb( icla_int_t n );

// SVD
icla_int_t icla_get_zgebrd_nb( icla_int_t m, icla_int_t n );
icla_int_t icla_get_zgesvd_nb( icla_int_t m, icla_int_t n );

// 2-stage eigenvalues
icla_int_t icla_get_zbulge_nb( icla_int_t n, icla_int_t nbthreads );
icla_int_t icla_get_zbulge_nb_mgpu( icla_int_t n );
icla_int_t icla_get_zbulge_vblksiz( icla_int_t n, icla_int_t nb, icla_int_t nbthreads );
icla_int_t icla_get_zbulge_gcperf();

// =============================================================================
// Other auxiliary functions
bool icla_zgetrf_gpu_recommend_cpu(icla_int_t m, icla_int_t n, icla_int_t nb);
bool icla_zgetrf_native_recommend_notrans(icla_int_t m, icla_int_t n, icla_int_t nb);

// =============================================================================
// ICLA function definitions
//
// In alphabetical order of base name (ignoring precision).
// Keep different versions of the same routine together, sorted this way:
// cpu (no suffix), gpu (_gpu), cpu/multi-gpu (_m), multi-gpu (_mgpu). Ex:
// icla_zheevdx
// icla_zheevdx_gpu
// icla_zheevdx_m
// icla_zheevdx_2stage
// icla_zheevdx_2stage_m

#ifdef ICLA_REAL
// only applicable to real [sd] precisions
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

// defined in dlaex3.cpp
void
icla_zvrange(
    icla_int_t k, double *d, icla_int_t *il, icla_int_t *iu, double vl, double vu);

void
icla_zirange(
    icla_int_t k, icla_int_t *indxq, icla_int_t *iil, icla_int_t *iiu, icla_int_t il, icla_int_t iu);
#endif  // ICLA_REAL

// ---------------------------------------------------------------- zgb routines
icla_int_t
icla_zgbsv_native(
        icla_int_t n, icla_int_t kl, icla_int_t ku, icla_int_t nrhs,
        iclaDoubleComplex* dA, icla_int_t ldda, icla_int_t* dipiv,
        iclaDoubleComplex* dB, icla_int_t lddb,
        icla_int_t *info);

icla_int_t
icla_zgbtf2_native_v2(
    icla_int_t m, icla_int_t n, icla_int_t kl, icla_int_t ku,
    iclaDoubleComplex* dA, icla_int_t ldda, icla_int_t* ipiv,
    icla_int_t* info, icla_queue_t queue);

icla_int_t
icla_zgbtf2_native_v2_work(
    icla_int_t m, icla_int_t n, icla_int_t kl, icla_int_t ku,
    iclaDoubleComplex* dA, icla_int_t ldda, icla_int_t* ipiv,
    icla_int_t* info,
    void* device_work, icla_int_t* lwork,
    icla_queue_t queue);

void
icla_zgbsv_native_work(
        icla_int_t n, icla_int_t kl, icla_int_t ku, icla_int_t nrhs,
        iclaDoubleComplex* dA, icla_int_t ldda, icla_int_t* dipiv,
        iclaDoubleComplex* dB, icla_int_t lddb,
        icla_int_t *info, void* device_work, icla_int_t* lwork,
        icla_queue_t queue);

icla_int_t
icla_zgbtf2_native(
    icla_int_t m, icla_int_t n, icla_int_t kl, icla_int_t ku,
    iclaDoubleComplex* dA, icla_int_t ldda, icla_int_t* ipiv,
    icla_int_t* info, icla_queue_t queue);

icla_int_t
icla_zgbtf2_native_work(
    icla_int_t m, icla_int_t n, icla_int_t kl, icla_int_t ku,
    iclaDoubleComplex* dA, icla_int_t ldda, icla_int_t* ipiv,
    icla_int_t* info,
    void* device_work, icla_int_t* lwork,
    icla_queue_t queue);

icla_int_t
icla_zgbtrf_native(
    icla_int_t m, icla_int_t n,
    icla_int_t kl, icla_int_t ku,
    iclaDoubleComplex* dAB, icla_int_t lddab, icla_int_t* dipiv,
    icla_int_t *info);

void
icla_zgbtrf_native_work(
    icla_int_t m, icla_int_t n,
    icla_int_t kl, icla_int_t ku,
    iclaDoubleComplex* dAB, icla_int_t lddab,
    icla_int_t* dipiv, icla_int_t *info,
    void* device_work, icla_int_t* lwork,
    icla_queue_t queue);

// ---------------------------------------------------------------- zgb routines
icla_int_t
icla_zgebrd(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex *A, icla_int_t lda,
    double *d, double *e,
    iclaDoubleComplex *tauq, iclaDoubleComplex *taup,
    iclaDoubleComplex *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_zgeev(
    icla_vec_t jobvl, icla_vec_t jobvr, icla_int_t n,
    iclaDoubleComplex *A, icla_int_t lda,
    #ifdef ICLA_COMPLEX
    iclaDoubleComplex *w,
    #else
    double *wr, double *wi,
    #endif
    iclaDoubleComplex *VL, icla_int_t ldvl,
    iclaDoubleComplex *VR, icla_int_t ldvr,
    iclaDoubleComplex *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    double *rwork,
    #endif
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zgeev_m(
    icla_vec_t jobvl, icla_vec_t jobvr, icla_int_t n,
    iclaDoubleComplex *A, icla_int_t lda,
    #ifdef ICLA_COMPLEX
    iclaDoubleComplex *w,
    #else
    double *wr, double *wi,
    #endif
    iclaDoubleComplex *VL, icla_int_t ldvl,
    iclaDoubleComplex *VR, icla_int_t ldvr,
    iclaDoubleComplex *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    double *rwork,
    #endif
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zgegqr_gpu(
    icla_int_t ikind, icla_int_t m, icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr dwork, iclaDoubleComplex *work,
    icla_int_t *info);

icla_int_t
icla_zgegqr_expert_gpu_work(
    icla_int_t ikind, icla_int_t m, icla_int_t n,
    iclaDoubleComplex_ptr dA,   icla_int_t ldda,
    void *host_work,   icla_int_t *lwork_host,
    void *device_work, icla_int_t *lwork_device,
    icla_int_t *info, icla_queue_t queue );

icla_int_t
icla_zgehrd(
    icla_int_t n, icla_int_t ilo, icla_int_t ihi,
    iclaDoubleComplex *A, icla_int_t lda,
    iclaDoubleComplex *tau,
    iclaDoubleComplex *work, icla_int_t lwork,
    iclaDoubleComplex_ptr dT,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zgehrd_m(
    icla_int_t n, icla_int_t ilo, icla_int_t ihi,
    iclaDoubleComplex *A, icla_int_t lda,
    iclaDoubleComplex *tau,
    iclaDoubleComplex *work, icla_int_t lwork,
    iclaDoubleComplex *T,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zgehrd2(
    icla_int_t n, icla_int_t ilo, icla_int_t ihi,
    iclaDoubleComplex *A, icla_int_t lda,
    iclaDoubleComplex *tau,
    iclaDoubleComplex *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_zgelqf(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex *A,    icla_int_t lda,
    iclaDoubleComplex *tau,
    iclaDoubleComplex *work, icla_int_t lwork,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zgelqf_gpu(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    iclaDoubleComplex *tau,
    iclaDoubleComplex *work, icla_int_t lwork,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zgels(
    icla_trans_t trans, icla_int_t m, icla_int_t n, icla_int_t nrhs,
    iclaDoubleComplex_ptr A, icla_int_t lda,
    iclaDoubleComplex_ptr B, icla_int_t ldb,
    iclaDoubleComplex *hwork, icla_int_t lwork,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zggrqf(
    icla_int_t m, icla_int_t p, icla_int_t n,
    iclaDoubleComplex *A, icla_int_t lda,
    iclaDoubleComplex *taua,
    iclaDoubleComplex *B, icla_int_t ldb,
    iclaDoubleComplex *taub,
    iclaDoubleComplex *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_zgglse(
    icla_int_t m, icla_int_t n, icla_int_t p,
    iclaDoubleComplex *A, icla_int_t lda,
    iclaDoubleComplex *B, icla_int_t ldb,
    iclaDoubleComplex *c, iclaDoubleComplex *d, iclaDoubleComplex *x,
    iclaDoubleComplex *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_zgels_gpu(
    icla_trans_t trans, icla_int_t m, icla_int_t n, icla_int_t nrhs,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr dB, icla_int_t lddb,
    iclaDoubleComplex *hwork, icla_int_t lwork,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zgels3_gpu(
    icla_trans_t trans, icla_int_t m, icla_int_t n, icla_int_t nrhs,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr dB, icla_int_t lddb,
    iclaDoubleComplex *hwork, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_zgeqlf(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex *A,    icla_int_t lda,
    iclaDoubleComplex *tau,
    iclaDoubleComplex *work, icla_int_t lwork,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zgeqp3(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex *A, icla_int_t lda,
    icla_int_t *jpvt, iclaDoubleComplex *tau,
    iclaDoubleComplex *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    double *rwork,
    #endif
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zgeqp3_gpu(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    icla_int_t *jpvt, iclaDoubleComplex *tau,
    iclaDoubleComplex_ptr dwork, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    double *rwork,
    #endif
    icla_int_t *info);

icla_int_t
icla_zgeqp3_expert_gpu_work(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    icla_int_t *jpvt, iclaDoubleComplex *tau,
    void* host_work,   icla_int_t *lwork_host,
    void* device_work, icla_int_t *lwork_device,
    icla_int_t *info, icla_queue_t queue );

// CUDA ICLA only
icla_int_t
icla_zgeqr2_gpu(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr dtau,
    iclaDouble_ptr        dwork,
    icla_queue_t queue,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zgeqr2x_gpu(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr dtau,
    iclaDoubleComplex_ptr dT, iclaDoubleComplex_ptr ddA,
    iclaDouble_ptr        dwork,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zgeqr2x2_gpu(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr dtau,
    iclaDoubleComplex_ptr dT, iclaDoubleComplex_ptr ddA,
    iclaDouble_ptr        dwork,
    icla_int_t *info);

icla_int_t
icla_zgeqr2x3_gpu(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr dtau,
    iclaDoubleComplex_ptr dT,
    iclaDoubleComplex_ptr ddA,
    iclaDouble_ptr        dwork,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zgeqr2x4_gpu(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr dtau,
    iclaDoubleComplex_ptr dT,
    iclaDoubleComplex_ptr ddA,
    iclaDouble_ptr        dwork,
    icla_queue_t queue,
    icla_int_t *info);

icla_int_t
icla_zgeqrf(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex *A, icla_int_t lda,
    iclaDoubleComplex *tau,
    iclaDoubleComplex *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_zgeqrf_gpu(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    iclaDoubleComplex *tau,
    iclaDoubleComplex_ptr dT,
    icla_int_t *info);

icla_int_t
icla_zgeqrf_expert_gpu_work(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    iclaDoubleComplex *tau, iclaDoubleComplex_ptr dT,
    icla_int_t *info,
    icla_mode_t mode, icla_int_t nb,
    void* host_work,   icla_int_t *lwork_host,
    void* device_work, icla_int_t *lwork_device,
    icla_queue_t queues[2] );

// CUDA ICLA only
icla_int_t
icla_zgeqrf_m(
    icla_int_t ngpu,
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex *A,    icla_int_t lda,
    iclaDoubleComplex *tau,
    iclaDoubleComplex *work, icla_int_t lwork,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zgeqrf_ooc(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex *A, icla_int_t lda,
    iclaDoubleComplex *tau,
    iclaDoubleComplex *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_zgeqrf2_gpu(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    iclaDoubleComplex *tau,
    icla_int_t *info);

icla_int_t
icla_zgeqrf2_mgpu(
    icla_int_t ngpu,
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex_ptr d_lA[], icla_int_t ldda,
    iclaDoubleComplex *tau,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zgeqrf3_gpu(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    iclaDoubleComplex *tau,
    iclaDoubleComplex_ptr dT,
    icla_int_t *info);

icla_int_t
icla_zgeqrs_gpu(
    icla_int_t m, icla_int_t n, icla_int_t nrhs,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex const *tau,
    iclaDoubleComplex_ptr dT,
    iclaDoubleComplex_ptr dB, icla_int_t lddb,
    iclaDoubleComplex *hwork, icla_int_t lwork,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zgeqrs3_gpu(
    icla_int_t m, icla_int_t n, icla_int_t nrhs,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    iclaDoubleComplex const *tau,
    iclaDoubleComplex_ptr dT,
    iclaDoubleComplex_ptr dB, icla_int_t lddb,
    iclaDoubleComplex *hwork, icla_int_t lwork,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zgerbt_gpu(
    icla_bool_t gen, icla_int_t n, icla_int_t nrhs,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr dB, icla_int_t lddb,
    iclaDoubleComplex *U, iclaDoubleComplex *V,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zgerfs_nopiv_gpu(
    icla_trans_t trans, icla_int_t n, icla_int_t nrhs,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr dB, icla_int_t lddb,
    iclaDoubleComplex_ptr dX, icla_int_t lddx,
    iclaDoubleComplex_ptr dworkd, iclaDoubleComplex_ptr dAF,
    icla_int_t *iter,
    icla_int_t *info);

icla_int_t
icla_zgesdd(
    icla_vec_t jobz, icla_int_t m, icla_int_t n,
    iclaDoubleComplex *A, icla_int_t lda,
    double *s,
    iclaDoubleComplex *U, icla_int_t ldu,
    iclaDoubleComplex *VT, icla_int_t ldvt,
    iclaDoubleComplex *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    double *rwork,
    #endif
    icla_int_t *iwork,
    icla_int_t *info);

icla_int_t
icla_zgesv(
    icla_int_t n, icla_int_t nrhs,
    iclaDoubleComplex *A, icla_int_t lda,
    icla_int_t *ipiv,
    iclaDoubleComplex *B, icla_int_t ldb,
    icla_int_t *info);

icla_int_t
icla_zgesv_gpu(
    icla_int_t n, icla_int_t nrhs,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    icla_int_t *ipiv,
    iclaDoubleComplex_ptr dB, icla_int_t lddb,
    icla_int_t *info);

icla_int_t
icla_zgesv_nopiv_gpu(
    icla_int_t n, icla_int_t nrhs,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr dB, icla_int_t lddb,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zgesv_rbt(
    icla_bool_t ref, icla_int_t n, icla_int_t nrhs,
    iclaDoubleComplex *A, icla_int_t lda,
    iclaDoubleComplex *B, icla_int_t ldb,
    icla_int_t *info);

icla_int_t
icla_zgesvd(
    icla_vec_t jobu, icla_vec_t jobvt, icla_int_t m, icla_int_t n,
    iclaDoubleComplex *A,    icla_int_t lda, double *s,
    iclaDoubleComplex *U,    icla_int_t ldu,
    iclaDoubleComplex *VT,   icla_int_t ldvt,
    iclaDoubleComplex *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    double *rwork,
    #endif
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zgetf2_gpu(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    icla_int_t *ipiv,
    icla_queue_t queue,
    icla_int_t *info);

icla_int_t
icla_zgetf2_native_fused(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    icla_int_t *ipiv, icla_int_t gbstep,
    icla_int_t *flags,
    icla_int_t *info, icla_queue_t queue );

icla_int_t
icla_zgetf2_native(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    icla_int_t *dipiv, icla_int_t* dipivinfo,
    icla_int_t *dinfo, icla_int_t gbstep,
    icla_event_t events[2],
    icla_queue_t queue, icla_queue_t update_queue);

// CUDA ICLA only
icla_int_t
icla_zgetf2_nopiv(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex *A, icla_int_t lda,
    icla_int_t *info);

icla_int_t
icla_zgetrf_recpanel_native(
    icla_int_t m, icla_int_t n, icla_int_t recnb,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    icla_int_t* dipiv, icla_int_t* dipivinfo,
    icla_int_t *dinfo, icla_int_t gbstep,
    icla_event_t events[2], icla_queue_t queue, icla_queue_t update_queue);

icla_int_t
icla_zgetrf(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex *A, icla_int_t lda,
    icla_int_t *ipiv,
    icla_int_t *info);

icla_int_t
icla_zgetrf_gpu(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    icla_int_t *ipiv,
    icla_int_t *info);

icla_int_t
icla_zgetrf_expert_gpu_work(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    icla_int_t *ipiv,
    icla_int_t *info, icla_mode_t mode,
    icla_int_t nb, icla_int_t recnb,
    void* host_work,   icla_int_t *lwork_host,
    void* device_work, icla_int_t *lwork_device,
    icla_event_t events[2], icla_queue_t queues[2] );

icla_int_t
icla_zgetrf_gpu_expert(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    icla_int_t *ipiv,
    icla_int_t *info,
    icla_int_t nb, icla_mode_t mode);

icla_int_t
icla_zgetrf_native(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    icla_int_t *ipiv,
    icla_int_t *info );

// CUDA ICLA only
icla_int_t
icla_zgetrf_m(
    icla_int_t ngpu,
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex *A, icla_int_t lda,
    icla_int_t *ipiv,
    icla_int_t *info);

icla_int_t
icla_zgetrf_mgpu(
    icla_int_t ngpu,
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex_ptr d_lA[], icla_int_t ldda,
    icla_int_t *ipiv,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zgetrf2(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex *A, icla_int_t lda,
    icla_int_t *ipiv,
    icla_int_t *info);

icla_int_t
icla_zgetrf2_mgpu(
    icla_int_t ngpu,
    icla_int_t m, icla_int_t n, icla_int_t nb, icla_int_t offset,
    iclaDoubleComplex_ptr d_lAT[], icla_int_t lddat,
    icla_int_t *ipiv,
    iclaDoubleComplex_ptr d_lAP[],
    iclaDoubleComplex *W, icla_int_t ldw,
    icla_queue_t queues[][2],
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zgetrf_nopiv(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex *A, icla_int_t lda,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zgetrf_nopiv_gpu(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    icla_int_t *info);

icla_int_t
icla_zgetri_gpu(
    icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    icla_int_t *ipiv,
    iclaDoubleComplex_ptr dwork, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_zgetri_expert_gpu_work(
    icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda, icla_int_t *ipiv,
    icla_int_t *info,
    icla_mode_t mode,
    void* host_work,   icla_int_t *lwork_host,
    void* device_work, icla_int_t *lwork_device,
    icla_queue_t queues[2] );

icla_int_t
icla_zgetrs_gpu(
    icla_trans_t trans, icla_int_t n, icla_int_t nrhs,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    icla_int_t *ipiv,
    iclaDoubleComplex_ptr dB, icla_int_t lddb,
    icla_int_t *info);

icla_int_t
icla_zgetrs_expert_gpu_work(
    icla_trans_t trans, icla_int_t n, icla_int_t nrhs,
    iclaDoubleComplex_ptr dA, icla_int_t ldda, icla_int_t *ipiv,
    iclaDoubleComplex_ptr dB, icla_int_t lddb,
    icla_int_t *info,
    icla_mode_t mode,
    void* host_work,   icla_int_t *lwork_host,
    void* device_work, icla_int_t *lwork_device,
    icla_queue_t queue );

// CUDA ICLA only
icla_int_t
icla_zgetrs_nopiv_gpu(
    icla_trans_t trans, icla_int_t n, icla_int_t nrhs,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr dB, icla_int_t lddb,
    icla_int_t *info);

// ------------------------------------------------------------ zhe routines
icla_int_t
icla_zheevd(
    icla_vec_t jobz, icla_uplo_t uplo, icla_int_t n,
    iclaDoubleComplex *A, icla_int_t lda,
    double *w,
    iclaDoubleComplex *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    double *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zheevd_gpu(
    icla_vec_t jobz, icla_uplo_t uplo,
    icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    double *w,
    iclaDoubleComplex *wA,  icla_int_t ldwa,
    iclaDoubleComplex *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    double *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zheevd_m(
    icla_int_t ngpu,
    icla_vec_t jobz, icla_uplo_t uplo,
    icla_int_t n,
    iclaDoubleComplex *A, icla_int_t lda,
    double *w,
    iclaDoubleComplex *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    double *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zheevdx(
    icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo, icla_int_t n,
    iclaDoubleComplex *A, icla_int_t lda,
    double vl, double vu, icla_int_t il, icla_int_t iu,
    icla_int_t *mout, double *w,
    iclaDoubleComplex *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    double *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zheevdx_gpu(
    icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo,
    icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    double vl, double vu,
    icla_int_t il, icla_int_t iu,
    icla_int_t *mout, double *w,
    iclaDoubleComplex *wA,  icla_int_t ldwa,
    iclaDoubleComplex *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    double *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zheevdx_m(
    icla_int_t ngpu,
    icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo,
    icla_int_t n,
    iclaDoubleComplex *A, icla_int_t lda,
    double vl, double vu, icla_int_t il, icla_int_t iu,
    icla_int_t *mout, double *w,
    iclaDoubleComplex *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    double *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zheevdx_2stage(
    icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo,
    icla_int_t n,
    iclaDoubleComplex *A, icla_int_t lda,
    double vl, double vu, icla_int_t il, icla_int_t iu,
    icla_int_t *mout, double *w,
    iclaDoubleComplex *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    double *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zheevdx_2stage_m(
    icla_int_t ngpu,
    icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo,
    icla_int_t n,
    iclaDoubleComplex *A, icla_int_t lda,
    double vl, double vu, icla_int_t il, icla_int_t iu,
    icla_int_t *mout, double *w,
    iclaDoubleComplex *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    double *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

#ifdef ICLA_COMPLEX
// no real [sd] precisions available
// CUDA ICLA only
icla_int_t
icla_zheevr(
    icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo, icla_int_t n,
    iclaDoubleComplex *A, icla_int_t lda,
    double vl, double vu,
    icla_int_t il, icla_int_t iu, double abstol, icla_int_t *mout,
    double *w,
    iclaDoubleComplex *Z, icla_int_t ldz,
    icla_int_t *isuppz,
    iclaDoubleComplex *work, icla_int_t lwork,
    double *rwork, icla_int_t lrwork,
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zheevr_gpu(
    icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo, icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    double vl, double vu,
    icla_int_t il, icla_int_t iu, double abstol, icla_int_t *mout,
    double *w,
    iclaDoubleComplex_ptr dZ, icla_int_t lddz,
    icla_int_t *isuppz,
    iclaDoubleComplex *wA, icla_int_t ldwa,
    iclaDoubleComplex *wZ, icla_int_t ldwz,
    iclaDoubleComplex *work, icla_int_t lwork,
    double *rwork, icla_int_t lrwork,
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zheevx(
    icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo, icla_int_t n,
    iclaDoubleComplex *A, icla_int_t lda,
    double vl, double vu,
    icla_int_t il, icla_int_t iu, double abstol, icla_int_t *mout,
    double *w,
    iclaDoubleComplex *Z, icla_int_t ldz,
    iclaDoubleComplex *work, icla_int_t lwork,
    double *rwork, icla_int_t *iwork,
    icla_int_t *ifail,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zheevx_gpu(
    icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo, icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    double vl, double vu, icla_int_t il, icla_int_t iu,
    double abstol, icla_int_t *mout,
    double *w,
    iclaDoubleComplex_ptr dZ, icla_int_t lddz,
    iclaDoubleComplex *wA, icla_int_t ldwa,
    iclaDoubleComplex *wZ, icla_int_t ldwz,
    iclaDoubleComplex *work, icla_int_t lwork,
    double *rwork, icla_int_t *iwork,
    icla_int_t *ifail,
    icla_int_t *info);
#endif  // ICLA_COMPLEX

// CUDA ICLA only
icla_int_t
icla_zhegst(
    icla_int_t itype, icla_uplo_t uplo, icla_int_t n,
    iclaDoubleComplex *A, icla_int_t lda,
    iclaDoubleComplex *B, icla_int_t ldb,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zhegst_gpu(
    icla_int_t itype, icla_uplo_t uplo, icla_int_t n,
    iclaDoubleComplex_ptr       dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dB, icla_int_t lddb,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zhegst_m(
    icla_int_t ngpu,
    icla_int_t itype, icla_uplo_t uplo, icla_int_t n,
    iclaDoubleComplex *A, icla_int_t lda,
    iclaDoubleComplex *B, icla_int_t ldb,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zhegvd(
    icla_int_t itype, icla_vec_t jobz, icla_uplo_t uplo, icla_int_t n,
    iclaDoubleComplex *A, icla_int_t lda,
    iclaDoubleComplex *B, icla_int_t ldb,
    double *w, iclaDoubleComplex *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    double *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zhegvd_m(
    icla_int_t ngpu,
    icla_int_t itype, icla_vec_t jobz, icla_uplo_t uplo,
    icla_int_t n,
    iclaDoubleComplex *A, icla_int_t lda,
    iclaDoubleComplex *B, icla_int_t ldb,
    double *w,
    iclaDoubleComplex *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    double *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zhegvdx(
    icla_int_t itype, icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo,
    icla_int_t n, iclaDoubleComplex *A, icla_int_t lda,
    iclaDoubleComplex *B, icla_int_t ldb,
    double vl, double vu, icla_int_t il, icla_int_t iu,
    icla_int_t *mout, double *w,
    iclaDoubleComplex *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    double *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zhegvdx_m(
    icla_int_t ngpu,
    icla_int_t itype, icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo,
    icla_int_t n,
    iclaDoubleComplex *A, icla_int_t lda,
    iclaDoubleComplex *B, icla_int_t ldb,
    double vl, double vu, icla_int_t il, icla_int_t iu,
    icla_int_t *mout, double *w,
    iclaDoubleComplex *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    double *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zhegvdx_2stage(
    icla_int_t itype, icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo, icla_int_t n,
    iclaDoubleComplex *A, icla_int_t lda,
    iclaDoubleComplex *B, icla_int_t ldb,
    double vl, double vu, icla_int_t il, icla_int_t iu,
    icla_int_t *mout, double *w,
    iclaDoubleComplex *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    double *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zhegvdx_2stage_m(
    icla_int_t ngpu,
    icla_int_t itype, icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo,
    icla_int_t n,
    iclaDoubleComplex *A, icla_int_t lda,
    iclaDoubleComplex *B, icla_int_t ldb,
    double vl, double vu, icla_int_t il, icla_int_t iu,
    icla_int_t *mout, double *w,
    iclaDoubleComplex *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    double *rwork, icla_int_t lrwork,
    #endif
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

#ifdef ICLA_COMPLEX
// no real [sd] precisions available
// CUDA ICLA only
icla_int_t
icla_zhegvr(
    icla_int_t itype, icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo, icla_int_t n,
    iclaDoubleComplex *A, icla_int_t lda,
    iclaDoubleComplex *B, icla_int_t ldb,
    double vl, double vu, icla_int_t il, icla_int_t iu,
    double abstol, icla_int_t *mout, double *w,
    iclaDoubleComplex *Z, icla_int_t ldz,
    icla_int_t *isuppz, iclaDoubleComplex *work, icla_int_t lwork,
    double *rwork, icla_int_t lrwork,
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

// no real [sd] precisions available
// CUDA ICLA only
icla_int_t
icla_zhegvx(
    icla_int_t itype, icla_vec_t jobz, icla_range_t range, icla_uplo_t uplo,
    icla_int_t n, iclaDoubleComplex *A, icla_int_t lda,
    iclaDoubleComplex *B, icla_int_t ldb,
    double vl, double vu, icla_int_t il, icla_int_t iu,
    double abstol, icla_int_t *mout, double *w,
    iclaDoubleComplex *Z, icla_int_t ldz,
    iclaDoubleComplex *work, icla_int_t lwork, double *rwork,
    icla_int_t *iwork, icla_int_t *ifail,
    icla_int_t *info);
#endif

icla_int_t
icla_zhesv(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nrhs,
    iclaDoubleComplex *A, icla_int_t lda,
    icla_int_t *ipiv,
    iclaDoubleComplex *B, icla_int_t ldb,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zhesv_nopiv_gpu(
    icla_uplo_t uplo,  icla_int_t n, icla_int_t nrhs,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr dB, icla_int_t lddb,
    icla_int_t *info);

icla_int_t
icla_zhetrd(
    icla_uplo_t uplo, icla_int_t n,
    iclaDoubleComplex *A, icla_int_t lda,
    double *d, double *e, iclaDoubleComplex *tau,
    iclaDoubleComplex *work, icla_int_t lwork,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zhetrd_gpu(
    icla_uplo_t uplo, icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    double *d, double *e, iclaDoubleComplex *tau,
    iclaDoubleComplex *wA,  icla_int_t ldwa,
    iclaDoubleComplex *work, icla_int_t lwork,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zhetrd2_gpu(
    icla_uplo_t uplo, icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    double *d, double *e, iclaDoubleComplex *tau,
    iclaDoubleComplex *wA,  icla_int_t ldwa,
    iclaDoubleComplex *work, icla_int_t lwork,
    iclaDoubleComplex_ptr dwork, icla_int_t ldwork,
    icla_int_t *info);

// TODO: rename icla_zhetrd_m?
// CUDA ICLA only
icla_int_t
icla_zhetrd_mgpu(
    icla_int_t ngpu, icla_int_t nqueue,
    icla_uplo_t uplo, icla_int_t n,
    iclaDoubleComplex *A, icla_int_t lda,
    double *d, double *e, iclaDoubleComplex *tau,
    iclaDoubleComplex *work, icla_int_t lwork,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zhetrd_hb2st(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nb, icla_int_t Vblksiz,
    iclaDoubleComplex *A, icla_int_t lda,
    double *d, double *e,
    iclaDoubleComplex *V, icla_int_t ldv,
    iclaDoubleComplex *TAU, icla_int_t compT,
    iclaDoubleComplex *T, icla_int_t ldt);

// CUDA ICLA only
icla_int_t
icla_zhetrd_he2hb(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nb,
    iclaDoubleComplex *A, icla_int_t lda,
    iclaDoubleComplex *tau,
    iclaDoubleComplex *work, icla_int_t lwork,
    iclaDoubleComplex_ptr dT,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zhetrd_he2hb_mgpu(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nb,
    iclaDoubleComplex *A, icla_int_t lda,
    iclaDoubleComplex *tau,
    iclaDoubleComplex *work, icla_int_t lwork,
    iclaDoubleComplex_ptr dAmgpu[], icla_int_t ldda,
    iclaDoubleComplex_ptr dTmgpu[], icla_int_t lddt,
    icla_int_t ngpu, icla_int_t distblk,
    icla_queue_t queues[][20], icla_int_t nqueue,
    icla_int_t *info);

icla_int_t
icla_zhetrf(
    icla_uplo_t uplo, icla_int_t n,
    iclaDoubleComplex *A, icla_int_t lda,
    icla_int_t *ipiv,
    icla_int_t *info);

icla_int_t
icla_zhetrf_gpu(
   icla_uplo_t uplo, icla_int_t n,
   iclaDoubleComplex *dA, icla_int_t ldda,
   icla_int_t *ipiv,
   icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zhetrf_aasen(
    icla_uplo_t uplo, icla_int_t cpu_panel, icla_int_t n,
    iclaDoubleComplex *A, icla_int_t lda,
    icla_int_t *ipiv, icla_int_t *info);

icla_int_t
icla_zhetrf_nopiv(
    icla_uplo_t uplo, icla_int_t n,
    iclaDoubleComplex *A, icla_int_t lda,
    icla_int_t *info);

icla_int_t
icla_zhetrf_nopiv_cpu(
    icla_uplo_t uplo, icla_int_t n, icla_int_t ib,
    iclaDoubleComplex *A, icla_int_t lda,
    icla_int_t *info);

icla_int_t
icla_zhetrf_nopiv_gpu(
    icla_uplo_t uplo, icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zhetrs_nopiv_gpu(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nrhs,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr dB, icla_int_t lddb,
    icla_int_t *info);

// ------------------------------------------------------------ [dz]la routines
#ifdef ICLA_REAL
// only applicable to real [sd] precisions
icla_int_t
icla_dlaex0(
    icla_int_t n, double *d, double *e,
    double *Q, icla_int_t ldq,
    double *work, icla_int_t *iwork,
    iclaDouble_ptr dwork,
    icla_range_t range, double vl, double vu, icla_int_t il, icla_int_t iu,
    icla_int_t *info);

// CUDA ICLA only
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

// CUDA ICLA only
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

// CUDA ICLA only
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
#endif  // ICLA_REAL

icla_int_t
icla_zlabrd_gpu(
    icla_int_t m, icla_int_t n, icla_int_t nb,
    iclaDoubleComplex     *A, icla_int_t lda,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    double *d, double *e, iclaDoubleComplex *tauq, iclaDoubleComplex *taup,
    iclaDoubleComplex     *X, icla_int_t ldx,
    iclaDoubleComplex_ptr dX, icla_int_t lddx,
    iclaDoubleComplex     *Y, icla_int_t ldy,
    iclaDoubleComplex_ptr dY, icla_int_t lddy,
    iclaDoubleComplex  *work, icla_int_t lwork,
    icla_queue_t queue);

icla_int_t
icla_zlahef_gpu(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nb, icla_int_t *kb,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    icla_int_t *ipiv,
    iclaDoubleComplex_ptr dW, icla_int_t lddw,
    icla_queue_t queues[],
    icla_int_t *info);

icla_int_t
icla_zlahr2(
    icla_int_t n, icla_int_t k, icla_int_t nb,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr dV, icla_int_t lddv,
    iclaDoubleComplex *A,  icla_int_t lda,
    iclaDoubleComplex *tau,
    iclaDoubleComplex *T,  icla_int_t ldt,
    iclaDoubleComplex *Y,  icla_int_t ldy,
    icla_queue_t queue);

// CUDA ICLA only
icla_int_t
icla_zlahr2_m(
    icla_int_t n, icla_int_t k, icla_int_t nb,
    iclaDoubleComplex *A, icla_int_t lda,
    iclaDoubleComplex *tau,
    iclaDoubleComplex *T, icla_int_t ldt,
    iclaDoubleComplex *Y, icla_int_t ldy,
    struct zgehrd_data *data);

icla_int_t
icla_zlahru(
    icla_int_t n, icla_int_t ihi, icla_int_t k, icla_int_t nb,
    iclaDoubleComplex     *A, icla_int_t lda,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr dY, icla_int_t lddy,
    iclaDoubleComplex_ptr dV, icla_int_t lddv,
    iclaDoubleComplex_ptr dT,
    iclaDoubleComplex_ptr dwork,
    icla_queue_t queue);

// CUDA ICLA only
icla_int_t
icla_zlahru_m(
    icla_int_t n, icla_int_t ihi, icla_int_t k, icla_int_t nb,
    iclaDoubleComplex *A, icla_int_t lda,
    struct zgehrd_data *data);

#ifdef ICLA_REAL
// CUDA ICLA only
icla_int_t
icla_dlaln2(
    icla_int_t trans, icla_int_t na, icla_int_t nw,
    double smin, double ca, const double *A, icla_int_t lda,
    double d1, double d2,   const double *B, icla_int_t ldb,
    double wr, double wi, double *X, icla_int_t ldx,
    double *scale, double *xnorm,
    icla_int_t *info);
#endif

// CUDA ICLA only
icla_int_t
icla_zlaqps(
    icla_int_t m, icla_int_t n, icla_int_t offset,
    icla_int_t nb, icla_int_t *kb,
    iclaDoubleComplex *A,  icla_int_t lda,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    icla_int_t *jpvt, iclaDoubleComplex *tau, double *vn1, double *vn2,
    iclaDoubleComplex *auxv,
    iclaDoubleComplex *F,  icla_int_t ldf,
    iclaDoubleComplex_ptr dF, icla_int_t lddf);

// CUDA ICLA only
icla_int_t
icla_zlaqps_gpu(
    icla_int_t m, icla_int_t n, icla_int_t offset,
    icla_int_t nb, icla_int_t *kb,
    iclaDoubleComplex_ptr dA,  icla_int_t ldda,
    icla_int_t *jpvt, iclaDoubleComplex *tau,
    double *vn1, double *vn2,
    iclaDoubleComplex_ptr dauxv,
    iclaDoubleComplex_ptr dF, icla_int_t lddf);

// CUDA ICLA only
icla_int_t
icla_zlaqps2_gpu(
    icla_int_t m, icla_int_t n, icla_int_t offset,
    icla_int_t nb, icla_int_t *kb,
    iclaDoubleComplex_ptr dA,  icla_int_t ldda,
    icla_int_t *jpvt,
    iclaDoubleComplex_ptr dtau,
    iclaDouble_ptr dvn1, iclaDouble_ptr dvn2,
    iclaDoubleComplex_ptr dauxv,
    iclaDoubleComplex_ptr dF,  icla_int_t lddf,
    iclaDouble_ptr dlsticcs,
    icla_queue_t queue);

#ifdef ICLA_REAL
// CUDA ICLA only
icla_int_t
icla_zlaqtrsd(
    icla_trans_t trans, icla_int_t n,
    const double *T, icla_int_t ldt,
    double *x,       icla_int_t ldx,
    const double *cnorm,
    icla_int_t *info);
#endif

// CUDA ICLA only
icla_int_t
icla_zlarf_gpu(
    icla_int_t m,  icla_int_t n,
    iclaDoubleComplex_const_ptr dv, iclaDoubleComplex_const_ptr dtau,
    iclaDoubleComplex_ptr dC, icla_int_t lddc,
    icla_queue_t queue);

// icla_zlarfb_gpu
// see iclablas_q.h

// in zgeqr2x_gpu-v3.cpp
// CUDA ICLA only
icla_int_t
icla_zlarfb2_gpu(
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDoubleComplex_const_ptr dV, icla_int_t lddv,
    iclaDoubleComplex_const_ptr dT, icla_int_t lddt,
    iclaDoubleComplex_ptr dC,       icla_int_t lddc,
    iclaDoubleComplex_ptr dwork,    icla_int_t ldwork,
    icla_queue_t queue);

icla_int_t
icla_zlatrd(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nb,
    iclaDoubleComplex *A, icla_int_t lda,
    double *e, iclaDoubleComplex *tau,
    iclaDoubleComplex *W, icla_int_t ldw,
    iclaDoubleComplex *work, icla_int_t lwork,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr dW, icla_int_t lddw,
    icla_queue_t queue);

// CUDA ICLA only
icla_int_t
icla_zlatrd2(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nb,
    iclaDoubleComplex *A,  icla_int_t lda,
    double *e, iclaDoubleComplex *tau,
    iclaDoubleComplex *W,  icla_int_t ldw,
    iclaDoubleComplex *work, icla_int_t lwork,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr dW, icla_int_t lddw,
    iclaDoubleComplex_ptr dwork, icla_int_t ldwork,
    icla_queue_t queue);

// CUDA ICLA only
icla_int_t
icla_zlatrd_mgpu(
    icla_int_t ngpu,
    icla_uplo_t uplo,
    icla_int_t n, icla_int_t nb, icla_int_t nb0,
    iclaDoubleComplex *A,  icla_int_t lda,
    double *e, iclaDoubleComplex *tau,
    iclaDoubleComplex    *W,       icla_int_t ldw,
    iclaDoubleComplex_ptr dA[],    icla_int_t ldda, icla_int_t offset,
    iclaDoubleComplex_ptr dW[],    icla_int_t lddw,
    iclaDoubleComplex    *hwork,   icla_int_t lhwork,
    iclaDoubleComplex_ptr dwork[], icla_int_t ldwork,
    icla_queue_t queues[]);

#ifdef ICLA_COMPLEX
// CUDA ICLA only
icla_int_t
icla_zlatrsd(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_diag_t diag, icla_bool_t normin,
    icla_int_t n, const iclaDoubleComplex *A, icla_int_t lda,
    iclaDoubleComplex lambda,
    iclaDoubleComplex *x,
    double *scale, double *cnorm,
    icla_int_t *info);
#endif

icla_int_t
icla_zlauum(
    icla_uplo_t uplo, icla_int_t n,
    iclaDoubleComplex *A, icla_int_t lda,
    icla_int_t *info);

icla_int_t
icla_zlauum_gpu(
    icla_uplo_t uplo, icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    icla_int_t *info);

// ------------------------------------------------------------ zpo routines
icla_int_t
icla_zposv(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nrhs,
    iclaDoubleComplex *A, icla_int_t lda,
    iclaDoubleComplex *B, icla_int_t ldb,
    icla_int_t *info);

icla_int_t
icla_zposv_gpu(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nrhs,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr dB, icla_int_t lddb,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zpotf2_gpu(
    icla_uplo_t uplo, icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    icla_queue_t queue,
    icla_int_t *info);

icla_int_t
icla_zpotf2_native(
    icla_uplo_t uplo, icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    icla_int_t step, icla_int_t *device_info,
    icla_queue_t queue );

icla_int_t
icla_zpotrf_rectile_native(
    icla_uplo_t uplo, icla_int_t n, icla_int_t recnb,
    iclaDoubleComplex* dA,    icla_int_t ldda, icla_int_t gbstep,
    icla_int_t *dinfo,  icla_int_t *info, icla_queue_t queue);

icla_int_t
icla_zpotrf(
    icla_uplo_t uplo, icla_int_t n,
    iclaDoubleComplex *A, icla_int_t lda,
    icla_int_t *info);

icla_int_t
icla_zpotrf_expert_gpu_work(
    icla_uplo_t uplo, icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    icla_int_t *info,
    icla_mode_t mode,
    icla_int_t nb, icla_int_t recnb,
    void* host_work,   icla_int_t *lwork_host,
    void* device_work, icla_int_t *lwork_device,
    icla_event_t events[2], icla_queue_t queues[2] );

icla_int_t
icla_zpotrf_expert_gpu(
    icla_uplo_t uplo, icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    icla_int_t *info,
    icla_int_t nb, icla_mode_t mode );

icla_int_t
icla_zpotrf_gpu(
    icla_uplo_t uplo, icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    icla_int_t *info);

icla_int_t
icla_zpotrf_native(
    icla_uplo_t uplo, icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    icla_int_t *info );

// CUDA ICLA only
icla_int_t
icla_zpotrf_m(
    icla_int_t ngpu,
    icla_uplo_t uplo, icla_int_t n,
    iclaDoubleComplex *A, icla_int_t lda,
    icla_int_t *info);

icla_int_t
icla_zpotrf_mgpu(
    icla_int_t ngpu,
    icla_uplo_t uplo, icla_int_t n,
    iclaDoubleComplex_ptr d_lA[], icla_int_t ldda,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zpotrf_mgpu_right(
    icla_int_t ngpu,
    icla_uplo_t uplo, icla_int_t n,
    iclaDoubleComplex_ptr d_lA[], icla_int_t ldda,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zpotrf3_mgpu(
    icla_int_t ngpu,
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    icla_int_t off_i, icla_int_t off_j, icla_int_t nb,
    iclaDoubleComplex_ptr d_lA[], icla_int_t ldda,
    iclaDoubleComplex_ptr d_lP[], icla_int_t lddp,
    iclaDoubleComplex *A, icla_int_t lda, icla_int_t h,
    icla_queue_t queues[][3], icla_event_t events[][5],
    icla_int_t *info);

icla_int_t
icla_zpotri(
    icla_uplo_t uplo, icla_int_t n,
    iclaDoubleComplex *A, icla_int_t lda,
    icla_int_t *info);

icla_int_t
icla_zpotri_gpu(
    icla_uplo_t uplo, icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    icla_int_t *info);

icla_int_t
icla_zpotrs_gpu(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nrhs,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr dB, icla_int_t lddb,
    icla_int_t *info);

icla_int_t
icla_zpotrs_expert_gpu_work(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nrhs,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr dB, icla_int_t lddb,
    icla_int_t *info,
    void* host_work,   icla_int_t *lwork_host,
    void* device_work, icla_int_t *lwork_device,
    icla_queue_t queue );

// ------------------------------------------------------------ zsy routines
#ifdef ICLA_COMPLEX
// CUDA ICLA only
icla_int_t
icla_zsysv_nopiv_gpu(
    icla_uplo_t uplo,  icla_int_t n, icla_int_t nrhs,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr dB, icla_int_t lddb,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zsytrf_nopiv_cpu(
    icla_uplo_t uplo, icla_int_t n, icla_int_t ib,
    iclaDoubleComplex *A, icla_int_t lda,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zsytrf_nopiv_gpu(
    icla_uplo_t uplo, icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zsytrs_nopiv_gpu(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nrhs,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr dB, icla_int_t lddb,
    icla_int_t *info);
#endif

// ------------------------------------------------------------ zst routines
icla_int_t
icla_zstedx(
    icla_range_t range, icla_int_t n, double vl, double vu,
    icla_int_t il, icla_int_t iu, double *d, double *e,
    iclaDoubleComplex *Z, icla_int_t ldz,
    double *rwork, icla_int_t lrwork,
    icla_int_t *iwork, icla_int_t liwork,
    iclaDouble_ptr dwork,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zstedx_m(
    icla_int_t ngpu,
    icla_range_t range, icla_int_t n, double vl, double vu,
    icla_int_t il, icla_int_t iu, double *d, double *e,
    iclaDoubleComplex *Z, icla_int_t ldz,
    double *rwork, icla_int_t lrwork,
    icla_int_t *iwork, icla_int_t liwork,
    icla_int_t *info);

// ------------------------------------------------------------ ztr routines
// CUDA ICLA only
icla_int_t
icla_ztrevc3(
    icla_side_t side, icla_vec_t howmany,
    icla_int_t *select, icla_int_t n,
    iclaDoubleComplex *T,  icla_int_t ldt,
    iclaDoubleComplex *VL, icla_int_t ldvl,
    iclaDoubleComplex *VR, icla_int_t ldvr,
    icla_int_t mm, icla_int_t *mout,
    iclaDoubleComplex *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    double *rwork,
    #endif
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_ztrevc3_mt(
    icla_side_t side, icla_vec_t howmany,
    icla_int_t *select, icla_int_t n,
    iclaDoubleComplex *T,  icla_int_t ldt,
    iclaDoubleComplex *VL, icla_int_t ldvl,
    iclaDoubleComplex *VR, icla_int_t ldvr,
    icla_int_t mm, icla_int_t *mout,
    iclaDoubleComplex *work, icla_int_t lwork,
    #ifdef ICLA_COMPLEX
    double *rwork,
    #endif
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_ztrsm_m(
    icla_int_t ngpu,
    icla_side_t side, icla_uplo_t uplo, icla_trans_t transa, icla_diag_t diag,
    icla_int_t m, icla_int_t n, iclaDoubleComplex alpha,
    const iclaDoubleComplex *A, icla_int_t lda,
    iclaDoubleComplex       *B, icla_int_t ldb);

icla_int_t
icla_ztrtri(
    icla_uplo_t uplo, icla_diag_t diag, icla_int_t n,
    iclaDoubleComplex *A, icla_int_t lda,
    icla_int_t *info);

icla_int_t
icla_ztrtri_gpu(
    icla_uplo_t uplo, icla_diag_t diag, icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    icla_int_t *info);

icla_int_t
icla_ztrtri_expert_gpu_work(
    icla_uplo_t uplo, icla_diag_t diag, icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    icla_int_t *info,
    void* host_work,   icla_int_t *lwork_host,
    void* device_work, icla_int_t *lwork_device,
    icla_queue_t queues[2] );

// ------------------------------------------------------------ zun routines
// CUDA ICLA only
icla_int_t
icla_zungbr(
    icla_vect_t vect, icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDoubleComplex *A, icla_int_t lda,
    iclaDoubleComplex *tau,
    iclaDoubleComplex *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_zunghr(
    icla_int_t n, icla_int_t ilo, icla_int_t ihi,
    iclaDoubleComplex *A, icla_int_t lda,
    iclaDoubleComplex *tau,
    iclaDoubleComplex_ptr dT, icla_int_t nb,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zunghr_m(
    icla_int_t n, icla_int_t ilo, icla_int_t ihi,
    iclaDoubleComplex *A, icla_int_t lda,
    iclaDoubleComplex *tau,
    iclaDoubleComplex *T, icla_int_t nb,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zunglq(
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDoubleComplex *A, icla_int_t lda,
    iclaDoubleComplex *tau,
    iclaDoubleComplex_ptr dT, icla_int_t nb,
    icla_int_t *info);

icla_int_t
icla_zungqr(
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDoubleComplex *A, icla_int_t lda,
    iclaDoubleComplex *tau,
    iclaDoubleComplex_ptr dT, icla_int_t nb,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zungqr_gpu(
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    iclaDoubleComplex *tau,
    iclaDoubleComplex_ptr dT, icla_int_t nb,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zungqr_m(
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDoubleComplex *A, icla_int_t lda,
    iclaDoubleComplex *tau,
    iclaDoubleComplex *T, icla_int_t nb,
    icla_int_t *info);

icla_int_t
icla_zungqr2(
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDoubleComplex *A, icla_int_t lda,
    iclaDoubleComplex *tau,
    icla_int_t *info);

icla_int_t
icla_zunmbr(
    icla_vect_t vect, icla_side_t side, icla_trans_t trans,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDoubleComplex *A, icla_int_t lda,
    iclaDoubleComplex *tau,
    iclaDoubleComplex *C, icla_int_t ldc,
    iclaDoubleComplex *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_zunmlq(
    icla_side_t side, icla_trans_t trans,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDoubleComplex *A, icla_int_t lda,
    iclaDoubleComplex *tau,
    iclaDoubleComplex *C, icla_int_t ldc,
    iclaDoubleComplex *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_zunmrq(
    icla_side_t side, icla_trans_t trans,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDoubleComplex *A, icla_int_t lda,
    iclaDoubleComplex *tau,
    iclaDoubleComplex *C, icla_int_t ldc,
    iclaDoubleComplex *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_zunmql(
    icla_side_t side, icla_trans_t trans,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDoubleComplex *A, icla_int_t lda,
    iclaDoubleComplex *tau,
    iclaDoubleComplex *C, icla_int_t ldc,
    iclaDoubleComplex *work, icla_int_t lwork,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zunmql2_gpu(
    icla_side_t side, icla_trans_t trans,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    iclaDoubleComplex *tau,
    iclaDoubleComplex_ptr dC, icla_int_t lddc,
    const iclaDoubleComplex *wA, icla_int_t ldwa,
    icla_int_t *info);

icla_int_t
icla_zunmqr(
    icla_side_t side, icla_trans_t trans,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDoubleComplex *A, icla_int_t lda,
    iclaDoubleComplex *tau,
    iclaDoubleComplex *C, icla_int_t ldc,
    iclaDoubleComplex *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_zunmqr_gpu(
    icla_side_t side, icla_trans_t trans,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex const   *tau,
    iclaDoubleComplex_ptr       dC, icla_int_t lddc,
    iclaDoubleComplex       *hwork, icla_int_t lwork,
    iclaDoubleComplex_ptr       dT, icla_int_t nb,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zunmqr2_gpu(
    icla_side_t side, icla_trans_t trans,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    iclaDoubleComplex *tau,
    iclaDoubleComplex_ptr dC, icla_int_t lddc,
    const iclaDoubleComplex *wA, icla_int_t ldwa,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zunmqr_m(
    icla_int_t ngpu,
    icla_side_t side, icla_trans_t trans,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDoubleComplex *A,    icla_int_t lda,
    iclaDoubleComplex *tau,
    iclaDoubleComplex *C,    icla_int_t ldc,
    iclaDoubleComplex *work, icla_int_t lwork,
    icla_int_t *info);

icla_int_t
icla_zunmtr(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex *A,    icla_int_t lda,
    iclaDoubleComplex *tau,
    iclaDoubleComplex *C,    icla_int_t ldc,
    iclaDoubleComplex *work, icla_int_t lwork,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zunmtr_gpu(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    iclaDoubleComplex *tau,
    iclaDoubleComplex_ptr dC, icla_int_t lddc,
    const iclaDoubleComplex *wA, icla_int_t ldwa,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zunmtr_m(
    icla_int_t ngpu,
    icla_side_t side, icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex *A,    icla_int_t lda,
    iclaDoubleComplex *tau,
    iclaDoubleComplex *C,    icla_int_t ldc,
    iclaDoubleComplex *work, icla_int_t lwork,
    icla_int_t *info);

// =============================================================================
// ICLA utility function definitions

extern const iclaDoubleComplex ICLA_Z_NAN;
extern const iclaDoubleComplex ICLA_Z_INF;

int icla_z_isnan( iclaDoubleComplex x );
int icla_z_isinf( iclaDoubleComplex x );
int icla_z_isnan_inf( iclaDoubleComplex x );

iclaDoubleComplex
icla_zmake_lwork( icla_int_t lwork );

icla_int_t
icla_znan_inf(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    const iclaDoubleComplex *A, icla_int_t lda,
    icla_int_t *cnt_nan,
    icla_int_t *cnt_inf);

icla_int_t
icla_znan_inf_gpu(
    icla_uplo_t uplo, icla_int_t m, icla_int_t n,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    icla_int_t *cnt_nan,
    icla_int_t *cnt_inf,
    icla_queue_t queue);

void icla_zprint(
    icla_int_t m, icla_int_t n,
    const iclaDoubleComplex *A, icla_int_t lda);

void icla_zprint_gpu(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    icla_queue_t queue);

void icla_zpanel_to_q(
    icla_uplo_t uplo, icla_int_t ib,
    iclaDoubleComplex *A, icla_int_t lda,
    iclaDoubleComplex *work);

void icla_zq_to_panel(
    icla_uplo_t uplo, icla_int_t ib,
    iclaDoubleComplex *A, icla_int_t lda,
    iclaDoubleComplex *work);

/* auxiliary routines for posv-irgmres  */
void
iclablas_zextract_diag_sqrt(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex* dA, icla_int_t ldda,
    double* dD, icla_int_t incd,
    icla_queue_t queue);

void
iclablas_zscal_shift_hpd(
    icla_uplo_t uplo, int n,
    iclaDoubleComplex* dA, int ldda,
    double* dD, int incd,
    double miu, double cn, double eps,
    icla_queue_t queue);

void
iclablas_zdimv_invert(
    icla_int_t n,
    iclaDoubleComplex alpha, iclaDoubleComplex* dD, icla_int_t incd,
                              iclaDoubleComplex* dx, icla_int_t incx,
    iclaDoubleComplex beta,  iclaDoubleComplex* dy, icla_int_t incy,
    icla_queue_t queue);

#ifdef __cplusplus
}
#endif

#undef ICLA_COMPLEX

#endif /* ICLA_Z_H */
