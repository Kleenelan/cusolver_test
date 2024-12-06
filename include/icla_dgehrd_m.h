

#ifndef ICLA_DGEHRD_H
#define ICLA_DGEHRD_H

#include "icla_types.h"

#ifdef __cplusplus
extern "C" {
#endif

struct dgehrd_data
{
    icla_int_t ngpu;

    icla_int_t ldda;
    icla_int_t ldv;
    icla_int_t ldvd;

    iclaDouble_ptr dA [ IclaMaxGPUs ];
    iclaDouble_ptr dV [ IclaMaxGPUs ];
    iclaDouble_ptr dVd[ IclaMaxGPUs ];
    iclaDouble_ptr dY [ IclaMaxGPUs ];
    iclaDouble_ptr dW [ IclaMaxGPUs ];
    iclaDouble_ptr dTi[ IclaMaxGPUs ];

    icla_queue_t queues[ IclaMaxGPUs ];
};

#ifdef __cplusplus
}
#endif

#endif
