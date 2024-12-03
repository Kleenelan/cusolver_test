
#ifndef ICLABLAS_DS_H
#define ICLABLAS_DS_H

#include "icla_types.h"

#ifdef __cplusplus
extern "C" {
#endif

void
iclablas_dsaxpycp(
    icla_int_t m,
    iclaFloat_ptr        r,
    iclaDouble_ptr       x,
    iclaDouble_const_ptr b,
    iclaDouble_ptr       w,
    icla_queue_t queue );

void
iclablas_dslaswp(
    icla_int_t n,
    iclaDouble_ptr A, icla_int_t lda,
    iclaFloat_ptr SA, icla_int_t ldsa,
    icla_int_t m,
    const icla_int_t *ipiv, icla_int_t incx,
    icla_queue_t queue );

void
iclablas_dlag2s(
    icla_int_t m, icla_int_t n,
    iclaDouble_const_ptr A,  icla_int_t lda,
    iclaFloat_ptr       SA, icla_int_t ldsa,
    icla_queue_t queue,
    icla_int_t *info );

void
iclablas_slag2d(
    icla_int_t m, icla_int_t n,
    iclaFloat_const_ptr SA, icla_int_t ldsa,
    iclaDouble_ptr       A,  icla_int_t lda,
    icla_queue_t queue,
    icla_int_t *info );

void
iclablas_dlat2s(
    icla_uplo_t uplo, icla_int_t n,
    iclaDouble_const_ptr A,  icla_int_t lda,
    iclaFloat_ptr       SA, icla_int_t ldsa,
    icla_queue_t queue,
    icla_int_t *info );

void
iclablas_slat2d(
    icla_uplo_t uplo, icla_int_t n,
    iclaFloat_const_ptr SA, icla_int_t ldsa,
    iclaDouble_ptr       A,  icla_int_t lda,
    icla_queue_t queue,
    icla_int_t *info );

#ifdef __cplusplus
}
#endif

#endif

