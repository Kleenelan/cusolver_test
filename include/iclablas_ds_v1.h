
#ifndef ICLABLAS_DS_V1_H
#define ICLABLAS_DS_V1_H

#ifdef ICLA_NO_V1
#error "Since ICLA_NO_V1 is defined, icla.h is invalid; use icla_v2.h"
#endif

#include "icla_types.h"

#ifdef __cplusplus
extern "C" {
#endif

void
iclablas_dsaxpycp_v1(
    icla_int_t m,
    iclaFloat_ptr  r,
    iclaDouble_ptr x,
    iclaDouble_const_ptr b,
    iclaDouble_ptr w );

void
iclablas_dslaswp_v1(
    icla_int_t n,
    iclaDouble_ptr  A, icla_int_t lda,
    iclaFloat_ptr  SA,
    icla_int_t m,
    const icla_int_t *ipiv, icla_int_t incx );

void
iclablas_dlag2s_v1(
    icla_int_t m, icla_int_t n,
    iclaDouble_const_ptr  A, icla_int_t lda,
    iclaFloat_ptr        SA, icla_int_t ldsa,
    icla_int_t *info );

void
iclablas_slag2d_v1(
    icla_int_t m, icla_int_t n,
    iclaFloat_const_ptr  SA, icla_int_t ldsa,
    iclaDouble_ptr        A, icla_int_t lda,
    icla_int_t *info );

void
iclablas_dlat2s_v1(
    icla_uplo_t uplo, icla_int_t n,
    iclaDouble_const_ptr  A, icla_int_t lda,
    iclaFloat_ptr        SA, icla_int_t ldsa,
    icla_int_t *info );

void
iclablas_slat2d_v1(
    icla_uplo_t uplo, icla_int_t n,
    iclaFloat_const_ptr  SA, icla_int_t ldsa,
    iclaDouble_ptr        A, icla_int_t lda,
    icla_int_t *info );

#ifdef __cplusplus
}
#endif

#endif

