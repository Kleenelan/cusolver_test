

#ifndef ICLA_VBATCHED_H
#define ICLA_VBATCHED_H

#include "icla_types.h"




#include "icla_zvbatched.h"
#include "icla_cvbatched.h"
#include "icla_dvbatched.h"
#include "icla_svbatched.h"

#ifdef __cplusplus
extern "C" {
#endif


void icla_getrf_vbatched_setup(
    icla_int_t *m, icla_int_t *n, icla_int_t *stats,
    icla_int_t batchCount, icla_queue_t queue );


void
setup_pivinfo_vbatched(
    icla_int_t **pivinfo_array, icla_int_t pivinfo_offset,
    icla_int_t **ipiv_array,    icla_int_t ipiv_offset,
    icla_int_t* m, icla_int_t* n,
    icla_int_t max_m, icla_int_t nb, icla_int_t batchCount,
    icla_queue_t queue);


void
adjust_ipiv_vbatched(
    icla_int_t **ipiv_array, icla_int_t ipiv_offset,
    icla_int_t *minmn, icla_int_t max_minmn, icla_int_t offset,
    icla_int_t batchCount, icla_queue_t queue);


icla_int_t
icla_getrf_vbatched_checker(
        icla_int_t* m, icla_int_t* n, icla_int_t* ldda,
        icla_int_t* errors, icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_potrf_vbatched_checker(
        icla_uplo_t uplo,
        icla_int_t* n, icla_int_t* ldda,
        icla_int_t batchCount, icla_queue_t queue );


icla_int_t
icla_gemm_vbatched_checker(
        icla_trans_t transA, icla_trans_t transB,
        icla_int_t* m, icla_int_t* n, icla_int_t* k,
        icla_int_t* ldda, icla_int_t* lddb, icla_int_t* lddc,
        icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_trsm_vbatched_checker(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t* m, icla_int_t* n,
        icla_int_t* ldda, icla_int_t* lddb,
        icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_syrk_vbatched_checker(
        icla_int_t icomplex,
        icla_uplo_t uplo, icla_trans_t trans,
        icla_int_t *n, icla_int_t *k,
        icla_int_t *ldda, icla_int_t *lddc,
        icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_herk_vbatched_checker(
        icla_uplo_t uplo, icla_trans_t trans,
        icla_int_t *n, icla_int_t *k,
        icla_int_t *ldda, icla_int_t *lddc,
        icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_syr2k_vbatched_checker(
        icla_int_t icomplex,
        icla_uplo_t uplo, icla_trans_t trans,
        icla_int_t *n, icla_int_t *k,
        icla_int_t *ldda, icla_int_t *lddb, icla_int_t *lddc,
        icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_her2k_vbatched_checker(
        icla_uplo_t uplo, icla_trans_t trans,
        icla_int_t *n, icla_int_t *k,
        icla_int_t *ldda, icla_int_t *lddb, icla_int_t *lddc,
        icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_trmm_vbatched_checker(
        icla_side_t side, icla_uplo_t uplo, icla_trans_t transA, icla_diag_t diag,
        icla_int_t* m, icla_int_t* n,
        icla_int_t* ldda, icla_int_t* lddb,
        icla_int_t batchCount, icla_queue_t queue);

icla_int_t
icla_hemm_vbatched_checker(
        icla_side_t side, icla_uplo_t uplo,
        icla_int_t* m, icla_int_t* n,
        icla_int_t* ldda, icla_int_t* lddb, icla_int_t* lddc,
        icla_int_t batchCount, icla_queue_t queue );


icla_int_t
icla_gemv_vbatched_checker(
        icla_trans_t trans,
        icla_int_t* m, icla_int_t* n,
        icla_int_t* ldda, icla_int_t* incx, icla_int_t* incy,
        icla_int_t batchCount, icla_queue_t queue );

icla_int_t
icla_hemv_vbatched_checker(
        icla_uplo_t uplo,
        icla_int_t* n, icla_int_t* ldda, icla_int_t* incx, icla_int_t* incy,
        icla_int_t batchCount, icla_queue_t queue );


icla_int_t
icla_axpy_vbatched_checker(
        icla_int_t *n,
        icla_int_t *incx, icla_int_t *incy,
        icla_int_t batchCount, icla_queue_t queue);


void icla_imax_size_1(icla_int_t *n, icla_int_t l, icla_queue_t queue);

void icla_imax_size_2(icla_int_t *m, icla_int_t *n, icla_int_t l, icla_queue_t queue);

void icla_imax_size_3(icla_int_t *m, icla_int_t *n, icla_int_t *k, icla_int_t l, icla_queue_t queue);


icla_int_t
icla_ivec_max( icla_int_t vecsize,
                  icla_int_t* x,
                  icla_int_t* work, icla_int_t lwork, icla_queue_t queue);


icla_int_t
icla_isum_reduce( icla_int_t vecsize,
                   icla_int_t* x,
                   icla_int_t* work, icla_int_t lwork, icla_queue_t queue);

void
icla_ivec_add( icla_int_t vecsize,
                      icla_int_t a1, icla_int_t *x1,
                      icla_int_t a2, icla_int_t *x2,
                      icla_int_t *y, icla_queue_t queue);

void
icla_ivec_mul( icla_int_t vecsize,
                      icla_int_t *x1, icla_int_t *x2,
                      icla_int_t *y, icla_queue_t queue);

void
icla_ivec_ceildiv( icla_int_t vecsize,
                   icla_int_t *x,
                   icla_int_t nb,
                   icla_int_t *y, icla_queue_t queue);

void
icla_ivec_roundup( icla_int_t vecsize,
                   icla_int_t *x,
                   icla_int_t nb,
                   icla_int_t *y, icla_queue_t queue);

void
icla_ivec_setc( icla_int_t vecsize,
                           icla_int_t *x,
                           icla_int_t value,
                           icla_queue_t queue);

void
icla_zsetvector_const( icla_int_t vecsize,
                           iclaDoubleComplex *x,
                           iclaDoubleComplex value,
                           icla_queue_t queue);

void
icla_csetvector_const( icla_int_t vecsize,
                           iclaFloatComplex *x,
                           iclaFloatComplex value,
                           icla_queue_t queue);

void
icla_dsetvector_const( icla_int_t vecsize,
                           double *x,
                           double value,
                           icla_queue_t queue);

void
icla_ssetvector_const( icla_int_t vecsize,
                           float *x,
                           float value,
                           icla_queue_t queue);

void
icla_ivec_addc( icla_int_t vecsize,
                     icla_int_t *x, icla_int_t value,
                     icla_int_t *y, icla_queue_t queue);

void
icla_ivec_mulc( icla_int_t vecsize,
                     icla_int_t *x, icla_int_t value,
                     icla_int_t *y, icla_queue_t queue);

void
icla_ivec_minc( icla_int_t vecsize,
                     icla_int_t *x, icla_int_t value,
                     icla_int_t *y, icla_queue_t queue);

void
icla_ivec_maxc( icla_int_t vecsize,
                     icla_int_t* x, icla_int_t value,
                     icla_int_t* y, icla_queue_t queue);

void
icla_ivec_min_vv( icla_int_t vecsize,
                   icla_int_t *v1, icla_int_t *v2, icla_int_t *y,
                   icla_queue_t queue);

void
icla_compute_trsm_jb(
    icla_int_t vecsize, icla_int_t* m,
    icla_int_t tri_nb, icla_int_t* jbv,
    icla_queue_t queue);

void
icla_prefix_sum_inplace(icla_int_t* ivec, icla_int_t length, icla_queue_t queue);

void
icla_prefix_sum_outofplace(icla_int_t* ivec, icla_int_t* ovec, icla_int_t length, icla_queue_t queue);

void
icla_prefix_sum_inplace_w(icla_int_t* ivec, icla_int_t length, icla_int_t* workspace, icla_int_t lwork, icla_queue_t queue);

void
icla_prefix_sum_outofplace_w(icla_int_t* ivec, icla_int_t* ovec, icla_int_t length, icla_int_t* workspace, icla_int_t lwork, icla_queue_t queue);

void
icla_imax_size_1(icla_int_t *n, icla_int_t l, icla_queue_t queue);

void
icla_imax_size_2(icla_int_t *m, icla_int_t *n, icla_int_t l, icla_queue_t queue);

void
icla_imax_size_3(icla_int_t *m, icla_int_t *n, icla_int_t *k, icla_int_t l, icla_queue_t queue);

#ifdef __cplusplus
}
#endif


#endif

