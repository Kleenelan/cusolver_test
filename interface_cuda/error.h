#ifndef ERROR_H
#define ERROR_H

#include "icla_types.h"

// overloaded C++ functions to deal with errors
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

// cuda provides cudaGetErrorString,
// but not cublasGetErrorString, so provide our own.
// In icla.h, we also provide icla_strerror.
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

/***************************************************************************//**
    Checks if err is not success, and prints an error message.
    Similar to assert(), if NDEBUG is defined, this does nothing.
    This version adds the current func, file, and line to the error message.

    @param[in]
    err     Error code.
    @ingroup icla_error_internal
*******************************************************************************/
#define check_error( err ) \
        icla_xerror( err, __func__, __FILE__, __LINE__ )

/***************************************************************************//**
    Checks if err is not success, and prints an error message.
    Similar to assert(), if NDEBUG is defined, this does nothing.
    This version takes func, file, and line as arguments to add to error message.

    @param[in]
    err     Error code.

    @param[in]
    func    Function where error occurred.

    @param[in]
    file    File     where error occurred.

    @param[in]
    line    Line     where error occurred.

    @ingroup icla_error_internal
*******************************************************************************/
#define check_xerror( err, func, file, line ) \
        icla_xerror( err, func, file, line )

#endif  // not NDEBUG

#endif // ERROR_H
