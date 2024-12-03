
#ifndef ICLA_ZGEHRD_H
#define ICLA_ZGEHRD_H

#include "icla_types.h"

#ifdef __cplusplus
extern "C" {
#endif

struct zgehrd_data
{
    icla_int_t ngpu;

    icla_int_t ldda;
    icla_int_t ldv;
    icla_int_t ldvd;

    iclaDoubleComplex_ptr dA [ iclaMaxGPUs ];

    iclaDoubleComplex_ptr dV [ iclaMaxGPUs ];

    iclaDoubleComplex_ptr dVd[ iclaMaxGPUs ];

    iclaDoubleComplex_ptr dY [ iclaMaxGPUs ];

    iclaDoubleComplex_ptr dW [ iclaMaxGPUs ];

    iclaDoubleComplex_ptr dTi[ iclaMaxGPUs ];

    icla_queue_t queues[ iclaMaxGPUs ];
};

#ifdef __cplusplus
}
#endif

#endif

