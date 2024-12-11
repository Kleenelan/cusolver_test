#ifndef ERROR_H
#define ERROR_H

#include "icla_types.h"

#ifdef ICLA_HAVE_CUDA
void icla_xerror( cudaError_t    err, const char* func, const char* file, int line );
#endif

#ifdef ICLA_HAVE_CUDA
void icla_xerror( cublasStatus_t err, const char* func, const char* file, int line );
#endif

#ifdef ICLA_HAVE_HIP
void icla_xerror( hipError_t     err, const char* func, const char* file, int line );
#endif

void icla_xerror( icla_int_t    err, const char* func, const char* file, int line );

#ifdef __cplusplus
extern "C" {
#endif

#ifdef ICLA_HAVE_CUDA
const char* icla_cublasGetErrorString( cublasStatus_t error );
#endif

#ifdef __cplusplus
}
#endif

#ifdef NDEBUG
#define check_error( err )                     ((void)0)
#define check_xerror( err, func, file, line )  ((void)0)
#else

#define check_error( err ) \
        icla_xerror( err, __func__, __FILE__, __LINE__ )

#define check_xerror( err, func, file, line ) \
        icla_xerror( err, func, file, line )

#endif

#endif

