

#ifndef ICLABLAS_H_H
#define ICLABLAS_H_H

#include "icla_types.h"
#include "icla_copy.h"


#ifdef __cplusplus
extern "C" {



void
iclablas_slag2h(
    icla_int_t m, icla_int_t n,
    float const * dA, icla_int_t lda,
    iclaHalf* dHA, icla_int_t ldha,
    icla_int_t *info, icla_queue_t queue);

void
iclablas_hlag2s(
    icla_int_t m, icla_int_t n,
    iclaHalf_const_ptr dA, icla_int_t lda,
    float             *dSA, icla_int_t ldsa,
    icla_queue_t queue );

void
iclablas_slag2h_batched(
    icla_int_t m, icla_int_t n,
    float const * const * dAarray, icla_int_t lda,
    iclaHalf** dHAarray, icla_int_t ldha,
    icla_int_t *info_array, icla_int_t batchCount,
    icla_queue_t queue);

void
iclablas_hlag2s_batched(
    icla_int_t m, icla_int_t n,
    iclaHalf const * const * dAarray, icla_int_t lda,
    float               **dSAarray, icla_int_t ldsa,
    icla_int_t batchCount, icla_queue_t queue );



void
icla_hgemm(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaHalf alpha,
    iclaHalf_const_ptr dA, icla_int_t ldda,
    iclaHalf_const_ptr dB, icla_int_t lddb,
    iclaHalf beta,
    iclaHalf_ptr       dC, icla_int_t lddc,
    icla_queue_t queue );

void
icla_hgemmx(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    float alpha,
    iclaHalf_const_ptr dA, icla_int_t ldda,
    iclaHalf_const_ptr dB, icla_int_t lddb,
    float beta,
    float *dC, icla_int_t lddc,
    icla_queue_t queue );
}

#endif
#endif
