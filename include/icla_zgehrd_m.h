

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

    iclaDoubleComplex_ptr dA [ IclaMaxGPUs ];
    iclaDoubleComplex_ptr dV [ IclaMaxGPUs ];
    iclaDoubleComplex_ptr dVd[ IclaMaxGPUs ];
    iclaDoubleComplex_ptr dY [ IclaMaxGPUs ];
    iclaDoubleComplex_ptr dW [ IclaMaxGPUs ];
    iclaDoubleComplex_ptr dTi[ IclaMaxGPUs ];

    icla_queue_t queues[ IclaMaxGPUs ];
};

#ifdef __cplusplus
}
#endif

#endif
