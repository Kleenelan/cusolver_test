
#ifndef ICLA_HBATCHED_H
#define ICLA_HBATCHED_H

#include "icla_types.h"

#ifdef __cplusplus
extern "C" {
#endif

void icla_hset_pointer(
    iclaHalf **output_array,
    iclaHalf *input,
    icla_int_t lda,
    icla_int_t row, icla_int_t column,
    icla_int_t batch_offset,
    icla_int_t batchCount,
    icla_queue_t queue);

icla_int_t
iclablas_hgemm_batched(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaHalf alpha,
    iclaHalf const * const * dAarray, icla_int_t ldda,
    iclaHalf const * const * dBarray, icla_int_t lddb,
    iclaHalf beta,
    iclaHalf **dCarray, icla_int_t lddc,
    icla_int_t batchCount, icla_queue_t queue );

#ifdef __cplusplus
}
#endif


#endif

