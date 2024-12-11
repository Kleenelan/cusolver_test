
#ifndef ICLA_BATCHED_H
#define ICLA_BATCHED_H

#include "icla_types.h"


#include "icla_zbatched.h"
#include "icla_cbatched.h"
#include "icla_dbatched.h"
#include "icla_sbatched.h"
#include "icla_hbatched.h"



#ifdef __cplusplus
extern "C" {
#endif


void
setup_pivinfo_batched( icla_int_t **pivinfo_array, icla_int_t **ipiv_array, icla_int_t ipiv_offset,
    icla_int_t m, icla_int_t nb,
    icla_int_t batchCount,  icla_queue_t queue);


void
adjust_ipiv_batched( icla_int_t **ipiv_array, icla_int_t ipiv_offset,
    icla_int_t m, icla_int_t offset,
    icla_int_t batchCount, icla_queue_t queue);

void
icla_idisplace_pointers(icla_int_t **output_array,
    icla_int_t **input_array, icla_int_t lda,
    icla_int_t row, icla_int_t column,
    icla_int_t batchCount, icla_queue_t queue);

void
stepinit_ipiv(icla_int_t **ipiv_array,
    icla_int_t pm,
    icla_int_t batchCount, icla_queue_t queue);

void
setup_pivinfo(
    icla_int_t *pivinfo, icla_int_t *ipiv,
    icla_int_t m, icla_int_t nb,
    icla_queue_t queue);

void
adjust_ipiv( icla_int_t *ipiv,
                 icla_int_t m, icla_int_t offset,
                 icla_queue_t queue);

void icla_iset_pointer(
    icla_int_t **output_array,
    icla_int_t *input,
    icla_int_t lda,
    icla_int_t row, icla_int_t column,
    icla_int_t batchSize,
    icla_int_t batchCount, icla_queue_t queue);

void icla_gbtrf_adjust_ju(
    icla_int_t n, icla_int_t ku,
    icla_int_t** dipiv_array, int* ju_array,
    icla_int_t gbstep, icla_int_t batchCount,
    icla_queue_t queue);

#ifdef __cplusplus
}
#endif


#endif

