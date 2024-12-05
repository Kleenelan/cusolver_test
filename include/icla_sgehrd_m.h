
#ifndef ICLA_SGEHRD_H
#define ICLA_SGEHRD_H

#include "icla_types.h"

#ifdef __cplusplus
extern "C" {
#endif

struct sgehrd_data
{
    icla_int_t ngpu;

    icla_int_t ldda;
    icla_int_t ldv;
    icla_int_t ldvd;

    iclaFloat_ptr dA [ IclaMaxGPUs ];
    iclaFloat_ptr dV [ IclaMaxGPUs ];
    iclaFloat_ptr dVd[ IclaMaxGPUs ];
    iclaFloat_ptr dY [ IclaMaxGPUs ];
    iclaFloat_ptr dW [ IclaMaxGPUs ];
    iclaFloat_ptr dTi[ IclaMaxGPUs ];

    icla_queue_t queues[ IclaMaxGPUs ];
};

#ifdef __cplusplus
}
#endif

#endif
