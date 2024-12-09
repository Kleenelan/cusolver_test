
#ifndef ICLA_DS_H
#define ICLA_DS_H

#include "icla_types.h"

#ifdef __cplusplus
extern "C" {
#endif

icla_int_t
icla_dsgeqrsv_gpu(
    icla_int_t m, icla_int_t n, icla_int_t nrhs,
    iclaDouble_ptr dA, icla_int_t ldda,
    iclaDouble_ptr dB, icla_int_t lddb,
    iclaDouble_ptr dX, icla_int_t lddx,
    icla_int_t *iter,
    icla_int_t *info);

icla_int_t
icla_dsgesv_gpu(
    icla_trans_t trans, icla_int_t n, icla_int_t nrhs,
    iclaDouble_ptr dA, icla_int_t ldda,
    icla_int_t *ipiv,
    iclaInt_ptr dipiv,
    iclaDouble_ptr dB, icla_int_t lddb,
    iclaDouble_ptr dX, icla_int_t lddx,
    iclaDouble_ptr dworkd,
    iclaFloat_ptr  dworks,
    icla_int_t *iter,
    icla_int_t *info);

icla_int_t
icla_dsgetrs_gpu(
    icla_trans_t trans, icla_int_t n, icla_int_t nrhs,
    iclaFloat_ptr  dA, icla_int_t ldda,
    iclaInt_ptr        dipiv,
    iclaDouble_ptr dB, icla_int_t lddb,
    iclaDouble_ptr dX, icla_int_t lddx,
    iclaFloat_ptr dSX,
    icla_int_t *info);


icla_int_t
icla_dssysv_gpu(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nrhs,
    iclaDouble_ptr dA, icla_int_t ldda,
    iclaDouble_ptr dB, icla_int_t lddb,
    iclaDouble_ptr dX, icla_int_t lddx,
    iclaDouble_ptr dworkd,
    iclaFloat_ptr  dworks,
    icla_int_t *iter,
    icla_int_t *info);

icla_int_t
icla_dsposv_gpu(
    icla_uplo_t uplo, icla_int_t n, icla_int_t nrhs,
    iclaDouble_ptr dA, icla_int_t ldda,
    iclaDouble_ptr dB, icla_int_t lddb,
    iclaDouble_ptr dX, icla_int_t lddx,
    iclaDouble_ptr dworkd,
    iclaFloat_ptr  dworks,
    icla_int_t *iter,
    icla_int_t *info);

#ifdef __cplusplus
}
#endif

#endif

