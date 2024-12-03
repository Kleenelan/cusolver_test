
#ifndef ICLABLAS_ZC_V1_H
#define ICLABLAS_ZC_V1_H

#ifdef ICLA_NO_V1
#error "Since ICLA_NO_V1 is defined, icla.h is invalid; use icla_v2.h"
#endif

#include "icla_types.h"

#ifdef __cplusplus
extern "C" {
#endif

void
iclablas_zcaxpycp_v1(
    icla_int_t m,
    iclaFloatComplex_ptr  r,
    iclaDoubleComplex_ptr x,
    iclaDoubleComplex_const_ptr b,
    iclaDoubleComplex_ptr w );

void
iclablas_zclaswp_v1(
    icla_int_t n,
    iclaDoubleComplex_ptr  A, icla_int_t lda,
    iclaFloatComplex_ptr  SA,
    icla_int_t m,
    const icla_int_t *ipiv, icla_int_t incx );

void
iclablas_zlag2c_v1(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex_const_ptr  A, icla_int_t lda,
    iclaFloatComplex_ptr        SA, icla_int_t ldsa,
    icla_int_t *info );

void
iclablas_clag2z_v1(
    icla_int_t m, icla_int_t n,
    iclaFloatComplex_const_ptr  SA, icla_int_t ldsa,
    iclaDoubleComplex_ptr        A, icla_int_t lda,
    icla_int_t *info );

void
iclablas_zlat2c_v1(
    icla_uplo_t uplo, icla_int_t n,
    iclaDoubleComplex_const_ptr  A, icla_int_t lda,
    iclaFloatComplex_ptr        SA, icla_int_t ldsa,
    icla_int_t *info );

void
iclablas_clat2z_v1(
    icla_uplo_t uplo, icla_int_t n,
    iclaFloatComplex_const_ptr  SA, icla_int_t ldsa,
    iclaDoubleComplex_ptr        A, icla_int_t lda,
    icla_int_t *info );

#ifdef __cplusplus
}
#endif

#endif

