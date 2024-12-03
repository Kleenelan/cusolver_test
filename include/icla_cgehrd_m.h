/*
    -- ICLA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from include/icla_zgehrd_m.h, normal z -> c, Fri Nov 29 12:16:14 2024
       @author Mark Gates
*/

#ifndef ICLA_CGEHRD_H
#define ICLA_CGEHRD_H

#include "icla_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/***************************************************************************//**
    Structure containing matrices for multi-GPU cgehrd.

    - dA  is distributed column block-cyclic across GPUs.
    - dV  is duplicated on all GPUs.
    - dVd is distributed row block-cyclic across GPUs (TODO: verify).
    - dY  is partial results on each GPU in clahr2,
          then complete results are duplicated on all GPUs for clahru.
    - dW  is local to each GPU (workspace).
    - dTi is duplicated on all GPUs.

    @ingroup icla_gehrd
*******************************************************************************/
struct cgehrd_data
{
    icla_int_t ngpu;

    icla_int_t ldda;
    icla_int_t ldv;
    icla_int_t ldvd;

    iclaFloatComplex_ptr dA [ iclaMaxGPUs ];  // ldda*nlocal
    iclaFloatComplex_ptr dV [ iclaMaxGPUs ];  // ldv *nb, whole panel
    iclaFloatComplex_ptr dVd[ iclaMaxGPUs ];  // ldvd*nb, block-cyclic
    iclaFloatComplex_ptr dY [ iclaMaxGPUs ];  // ldda*nb
    iclaFloatComplex_ptr dW [ iclaMaxGPUs ];  // ldda*nb
    iclaFloatComplex_ptr dTi[ iclaMaxGPUs ];  // nb*nb

    icla_queue_t queues[ iclaMaxGPUs ];
};

#ifdef __cplusplus
}
#endif

#endif        //  #ifndef ICLA_CGEHRD_H
