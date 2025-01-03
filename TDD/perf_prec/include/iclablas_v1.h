

#ifndef ICLABLAS_V1_H
#define ICLABLAS_V1_H

#ifdef ICLA_NO_V1
#error "Since ICLA_NO_V1 is defined, icla.h is invalid; use icla_v2.h"
#endif

#include "icla_copy_v1.h"
#include "iclablas_z.h"
#include "iclablas_z_v1.h"
#include "iclablas_c_v1.h"
#include "iclablas_d_v1.h"
#include "iclablas_s_v1.h"
#include "iclablas_zc_v1.h"
#include "iclablas_ds_v1.h"

#ifdef __cplusplus
extern "C" {
#endif




#define icla_queue_create_v1( queue_ptr ) \
        icla_queue_create_v1_internal( queue_ptr, __func__, __FILE__, __LINE__ )

void icla_queue_create_v1_internal(
    icla_queue_t* queue_ptr,
    const char* func, const char* file, int line );





#define IclaUpperLower     IclaFull
#define IclaUpperLowerStr  IclaFullStr

#define ICLA_Z_CNJG(a)     ICLA_Z_CONJ(a)
#define ICLA_C_CNJG(a)     ICLA_C_CONJ(a)
#define ICLA_D_CNJG(a)     ICLA_D_CONJ(a)
#define ICLA_S_CNJG(a)     ICLA_S_CONJ(a)


void icla_device_sync();





icla_int_t iclablasSetKernelStream( icla_queue_t queue );
icla_int_t iclablasGetKernelStream( icla_queue_t *queue );
icla_queue_t iclablasGetQueue();

#ifdef __cplusplus
}
#endif

#endif
