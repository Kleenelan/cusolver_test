/*
    -- ICLA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions mixed zc -> ds
*/

#ifndef ICLA_ZC_H
#define ICLA_ZC_H

#include "icla_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// ICLA mixed precision function definitions
//
// In alphabetical order of base name (ignoring precision).
icla_int_t
icla_zcgeqrsv_gpu(
    icla_int_t m, icla_int_t n, icla_int_t nrhs,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr dB, icla_int_t lddb,
    iclaDoubleComplex_ptr dX, icla_int_t lddx,
    icla_int_t *iter,
    icla_int_t *info);

icla_int_t
icla_zcgesv_gpu(
    icla_trans_t trans, icla_int_t n, icla_int_t nrhs,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    icla_int_t *ipiv,
    iclaInt_ptr dipiv,
    iclaDoubleComplex_ptr dB, icla_int_t lddb,
    iclaDoubleComplex_ptr dX, icla_int_t lddx,
    iclaDoubleComplex_ptr dworkd,
    iclaFloatComplex_ptr  dworks,
    icla_int_t *iter,
    icla_int_t *info);

icla_int_t
icla_zcgetrs_gpu(
    icla_trans_t trans, icla_int_t n, icla_int_t nrhs,
    iclaFloatComplex_ptr  dA, icla_int_t ldda,
    iclaInt_ptr        dipiv,
    iclaDoubleComplex_ptr dB, icla_int_t lddb,
    iclaDoubleComplex_ptr dX, icla_int_t lddx,
    iclaFloatComplex_ptr dSX,
    icla_int_t *info);

// CUDA ICLA only
icla_int_t
icla_zchesv_gpu(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nrhs,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr dB, icla_int_t lddb,
    iclaDoubleComplex_ptr dX, icla_int_t lddx,
    iclaDoubleComplex_ptr dworkd,
    iclaFloatComplex_ptr  dworks,
    icla_int_t *iter,
    icla_int_t *info);

icla_int_t
icla_zcposv_gpu(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nrhs,
    iclaDoubleComplex_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr dB, icla_int_t lddb,
    iclaDoubleComplex_ptr dX, icla_int_t lddx,
    iclaDoubleComplex_ptr dworkd,
    iclaFloatComplex_ptr  dworks,
    icla_int_t *iter,
    icla_int_t *info);

#ifdef __cplusplus
}
#endif

#endif /* ICLA_ZC_H */
