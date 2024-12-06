

#ifndef ICLABLAS_ZC_H
#define ICLABLAS_ZC_H

#include "icla_types.h"

#ifdef __cplusplus
extern "C" {
#endif



void
iclablas_zcaxpycp(
    icla_int_t m,
    iclaFloatComplex_ptr        r,
    iclaDoubleComplex_ptr       x,
    iclaDoubleComplex_const_ptr b,
    iclaDoubleComplex_ptr       w,
    icla_queue_t queue );

void
iclablas_zclaswp(
    icla_int_t n,
    iclaDoubleComplex_ptr A, icla_int_t lda,
    iclaFloatComplex_ptr SA, icla_int_t ldsa,
    icla_int_t m,
    const icla_int_t *ipiv, icla_int_t incx,
    icla_queue_t queue );

void
iclablas_zlag2c(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex_const_ptr A,  icla_int_t lda,
    iclaFloatComplex_ptr       SA, icla_int_t ldsa,
    icla_queue_t queue,
    icla_int_t *info );

void
iclablas_clag2z(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex_const_ptr SA, icla_int_t ldsa,
    iclaDoubleComplex_ptr       A,  icla_int_t lda,
    icla_queue_t queue,
    icla_int_t *info );

void
iclablas_zlat2c(
    icla_uplo_t uplo, icla_int_t n,
    iclaDoubleComplex_const_ptr A,  icla_int_t lda,
    iclaFloatComplex_ptr       SA, icla_int_t ldsa,
    icla_queue_t queue,
    icla_int_t *info );

void
iclablas_clat2z(
    icla_uplo_t uplo, icla_int_t n,
    iclaFloatComplex_const_ptr SA, icla_int_t ldsa,
    iclaDoubleComplex_ptr       A,  icla_int_t lda,
    icla_queue_t queue,
    icla_int_t *info );

#ifdef __cplusplus
}
#endif

#endif
