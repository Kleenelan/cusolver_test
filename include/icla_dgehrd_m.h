/*
    -- ICLA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from include/icla_zgehrd_m.h, normal z -> d, Fri Nov 29 12:16:14 2024
       @author Mark Gates
*/

#ifndef ICLA_DGEHRD_H
#define ICLA_DGEHRD_H

#include "icla_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/***************************************************************************//**
    Structure containing matrices for multi-GPU dgehrd.

    - dA  is distributed column block-cyclic across GPUs.
    - dV  is duplicated on all GPUs.
    - dVd is distributed row block-cyclic across GPUs (TODO: verify).
    - dY  is partial results on each GPU in dlahr2,
          then complete results are duplicated on all GPUs for dlahru.
    - dW  is local to each GPU (workspace).
    - dTi is duplicated on all GPUs.

    @ingroup icla_gehrd
*******************************************************************************/
struct dgehrd_data
{
    icla_int_t ngpu;

    icla_int_t ldda;
    icla_int_t ldv;
    icla_int_t ldvd;

    iclaDouble_ptr dA [ iclaMaxGPUs ];  // ldda*nlocal
    iclaDouble_ptr dV [ iclaMaxGPUs ];  // ldv *nb, whole panel
    iclaDouble_ptr dVd[ iclaMaxGPUs ];  // ldvd*nb, block-cyclic
    iclaDouble_ptr dY [ iclaMaxGPUs ];  // ldda*nb
    iclaDouble_ptr dW [ iclaMaxGPUs ];  // ldda*nb
    iclaDouble_ptr dTi[ iclaMaxGPUs ];  // nb*nb

    icla_queue_t queues[ iclaMaxGPUs ];
};

#ifdef __cplusplus
}
#endif

#endif        //  #ifndef ICLA_DGEHRD_H
