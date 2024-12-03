/*
    -- ICLA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Mark Gates
*/

#ifndef ICLA_ZGEHRD_H
#define ICLA_ZGEHRD_H

#include "icla_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/***************************************************************************//**
    Structure containing matrices for multi-GPU zgehrd.

    - dA  is distributed column block-cyclic across GPUs.
    - dV  is duplicated on all GPUs.
    - dVd is distributed row block-cyclic across GPUs (TODO: verify).
    - dY  is partial results on each GPU in zlahr2,
          then complete results are duplicated on all GPUs for zlahru.
    - dW  is local to each GPU (workspace).
    - dTi is duplicated on all GPUs.

    @ingroup icla_gehrd
*******************************************************************************/
struct zgehrd_data
{
    icla_int_t ngpu;

    icla_int_t ldda;
    icla_int_t ldv;
    icla_int_t ldvd;

    iclaDoubleComplex_ptr dA [ iclaMaxGPUs ];  // ldda*nlocal
    iclaDoubleComplex_ptr dV [ iclaMaxGPUs ];  // ldv *nb, whole panel
    iclaDoubleComplex_ptr dVd[ iclaMaxGPUs ];  // ldvd*nb, block-cyclic
    iclaDoubleComplex_ptr dY [ iclaMaxGPUs ];  // ldda*nb
    iclaDoubleComplex_ptr dW [ iclaMaxGPUs ];  // ldda*nb
    iclaDoubleComplex_ptr dTi[ iclaMaxGPUs ];  // nb*nb

    icla_queue_t queues[ iclaMaxGPUs ];
};

#ifdef __cplusplus
}
#endif

#endif        //  #ifndef ICLA_ZGEHRD_H
