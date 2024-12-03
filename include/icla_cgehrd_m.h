
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

    iclaFloatComplex_ptr dA [ iclaMaxGPUs ];

    iclaFloatComplex_ptr dV [ iclaMaxGPUs ];

    iclaFloatComplex_ptr dVd[ iclaMaxGPUs ];

    iclaFloatComplex_ptr dY [ iclaMaxGPUs ];

    iclaFloatComplex_ptr dW [ iclaMaxGPUs ];

    iclaFloatComplex_ptr dTi[ iclaMaxGPUs ];

    icla_queue_t queues[ iclaMaxGPUs ];
};

#ifdef __cplusplus
}
#endif

#endif

