
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

    iclaFloat_ptr dA [ iclaMaxGPUs ];

    iclaFloat_ptr dV [ iclaMaxGPUs ];

    iclaFloat_ptr dVd[ iclaMaxGPUs ];

    iclaFloat_ptr dY [ iclaMaxGPUs ];

    iclaFloat_ptr dW [ iclaMaxGPUs ];

    iclaFloat_ptr dTi[ iclaMaxGPUs ];

    icla_queue_t queues[ iclaMaxGPUs ];
};

#ifdef __cplusplus
}
#endif

#endif

