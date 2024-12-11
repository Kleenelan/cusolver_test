

#ifndef ICLA_CGEHRD_H
#define ICLA_CGEHRD_H

#include "icla_types.h"

#ifdef __cplusplus
extern "C" {
#endif


struct cgehrd_data
{
    icla_int_t ngpu;

    icla_int_t ldda;
    icla_int_t ldv;
    icla_int_t ldvd;

    iclaFloatComplex_ptr dA [ IclaMaxGPUs ];
    iclaFloatComplex_ptr dV [ IclaMaxGPUs ];
    iclaFloatComplex_ptr dVd[ IclaMaxGPUs ];
    iclaFloatComplex_ptr dY [ IclaMaxGPUs ];
    iclaFloatComplex_ptr dW [ IclaMaxGPUs ];
    iclaFloatComplex_ptr dTi[ IclaMaxGPUs ];

    icla_queue_t queues[ IclaMaxGPUs ];
};

#ifdef __cplusplus
}
#endif

#endif
