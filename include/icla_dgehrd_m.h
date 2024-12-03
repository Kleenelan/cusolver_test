
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

    iclaDouble_ptr dA [ iclaMaxGPUs ];

    iclaDouble_ptr dV [ iclaMaxGPUs ];

    iclaDouble_ptr dVd[ iclaMaxGPUs ];

    iclaDouble_ptr dY [ iclaMaxGPUs ];

    iclaDouble_ptr dW [ iclaMaxGPUs ];

    iclaDouble_ptr dTi[ iclaMaxGPUs ];

    icla_queue_t queues[ iclaMaxGPUs ];
};

#ifdef __cplusplus
}
#endif

#endif

