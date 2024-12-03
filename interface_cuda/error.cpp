#include <cuda_runtime.h>

#include "icla_internal.h"
#include "error.h"


/***************************************************************************//**
    @return String describing cuBLAS errors (cublasStatus_t).
    CUDA provides cudaGetErrorString, but not cublasGetErrorString.

    @param[in]
    err     Error code.

    @ingroup icla_error_internal
*******************************************************************************/
#if defined(ICLA_HAVE_CUDA) || defined(ICLA_HAVE_HIP)
extern "C"
const char* icla_cublasGetErrorString( cublasStatus_t err )
{
    switch( err ) {
        case CUBLAS_STATUS_SUCCESS:
            return "success";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "not initialized";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "out of memory";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "invalid value";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "architecture mismatch";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "memory mapping error";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "execution failed";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "internal error";

        default:
            return "unknown CUBLAS error code";
    }
}
#endif

/***************************************************************************//**
    Prints error message to stderr.
    C++ function overloaded for different error types (CUDA,
    cuBLAS, ICLA errors). Note CUDA and cuBLAS errors are enums,
    so can be differentiated.
    Used by the check_error() and check_xerror() macros.

    @param[in]
    err     Error code.

    @param[in]
    func    Function where error occurred; inserted by check_error().

    @param[in]
    file    File     where error occurred; inserted by check_error().

    @param[in]
    line    Line     where error occurred; inserted by check_error().

    @ingroup icla_error_internal
*******************************************************************************/
#ifdef ICLA_HAVE_CUDA
void icla_xerror( cudaError_t err, const char* func, const char* file, int line )
{
    if ( err != cudaSuccess ) {
        fprintf( stderr, "CUDA runtime error: %s (%d) in %s at %s:%d\n",
                 cudaGetErrorString( err ), err, func, file, line );
    }
}
#endif


/******************************************************************************/
/// @see icla_xerror
/// @ingroup icla_error_internal
#if defined(ICLA_HAVE_CUDA) || defined(ICLA_HAVE_HIP)
void icla_xerror( cublasStatus_t err, const char* func, const char* file, int line )
{
    if ( err != CUBLAS_STATUS_SUCCESS ) {
        fprintf( stderr, "CUBLAS error: %s (%d) in %s at %s:%d\n",
                 icla_cublasGetErrorString( err ), err, func, file, line );
    }
}
#endif



/******************************************************************************/
/// @see icla_xerror
/// @ingroup icla_error_internal
#ifdef ICLA_HAVE_HIP
void icla_xerror( hipError_t err, const char* func, const char* file, int line)
{

    if (err != HIP_SUCCESS) {
        fprintf( stderr, "HIP error: %s (%d) in %s at %s:%d\n",
                hipGetErrorString( err ), err, func, file, line );
    }

}
#endif


/******************************************************************************/
/// @see icla_xerror
/// @ingroup icla_error_internal
void icla_xerror( icla_int_t err, const char* func, const char* file, int line )
{
    if ( err != ICLA_SUCCESS ) {
        fprintf( stderr, "ICLA error: %s (%lld) in %s at %s:%d\n",
                 icla_strerror( err ), (long long) err, func, file, line );
    }
}



/***************************************************************************//**
    @return String describing ICLA errors (icla_int_t).

    @param[in]
    err     Error code.

    @ingroup icla_error
*******************************************************************************/
extern "C"
const char* icla_strerror( icla_int_t err )
{
    // LAPACK-compliant errors
    if ( err > 0 ) {
        return "function-specific error, see documentation";
    }
    else if ( err < 0 && err > ICLA_ERR ) {
        return "invalid argument";
    }
    // ICLA-specific errors
    switch( err ) {
        case ICLA_SUCCESS:
            return "success";

        case ICLA_ERR:
            return "unknown error";

        case ICLA_ERR_NOT_INITIALIZED:
            return "not initialized";

        case ICLA_ERR_REINITIALIZED:
            return "reinitialized";

        case ICLA_ERR_NOT_SUPPORTED:
            return "not supported";

        case ICLA_ERR_ILLEGAL_VALUE:
            return "illegal value";

        case ICLA_ERR_NOT_FOUND:
            return "not found";

        case ICLA_ERR_ALLOCATION:
            return "allocation";

        case ICLA_ERR_INTERNAL_LIMIT:
            return "internal limit";

        case ICLA_ERR_UNALLOCATED:
            return "unallocated error";

        case ICLA_ERR_FILESYSTEM:
            return "filesystem error";

        case ICLA_ERR_UNEXPECTED:
            return "unexpected error";

        case ICLA_ERR_SEQUENCE_FLUSHED:
            return "sequence flushed";

        case ICLA_ERR_HOST_ALLOC:
            return "cannot allocate memory on CPU host";

        case ICLA_ERR_DEVICE_ALLOC:
            return "cannot allocate memory on GPU device";

        case ICLA_ERR_CUDASTREAM:
            return "CUDA stream error";

        case ICLA_ERR_INVALID_PTR:
            return "invalid pointer";

        case ICLA_ERR_UNKNOWN:
            return "unknown error";

        case ICLA_ERR_NOT_IMPLEMENTED:
            return "not implemented";

        case ICLA_ERR_NAN:
            return "NaN detected";

        // some ICLA-sparse errors
        case ICLA_SLOW_CONVERGENCE:
            return "stopping criterion not reached within iterations";

        case ICLA_DIVERGENCE:
            return "divergence";

        case ICLA_NOTCONVERGED :
            return "stopping criterion not reached within iterations";

        case ICLA_NONSPD:
            return "not positive definite (SPD/HPD)";

        case ICLA_ERR_BADPRECOND:
            return "bad preconditioner";

        // map cusparse errors to icla errors
        case ICLA_ERR_CUSPARSE_NOT_INITIALIZED:
            return "cusparse: not initialized";

        case ICLA_ERR_CUSPARSE_ALLOC_FAILED:
            return "cusparse: allocation failed";

        case ICLA_ERR_CUSPARSE_INVALID_VALUE:
            return "cusparse: invalid value";

        case ICLA_ERR_CUSPARSE_ARCH_MISMATCH:
            return "cusparse: architecture mismatch";

        case ICLA_ERR_CUSPARSE_MAPPING_ERROR:
            return "cusparse: mapping error";

        case ICLA_ERR_CUSPARSE_EXECUTION_FAILED:
            return "cusparse: execution failed";

        case ICLA_ERR_CUSPARSE_INTERNAL_ERROR:
            return "cusparse: internal error";

        case ICLA_ERR_CUSPARSE_MATRIX_TYPE_NOT_SUPPORTED:
            return "cusparse: matrix type not supported";

        case ICLA_ERR_CUSPARSE_ZERO_PIVOT:
            return "cusparse: zero pivot";

        default:
            return "unknown ICLA error code";
    }
}
