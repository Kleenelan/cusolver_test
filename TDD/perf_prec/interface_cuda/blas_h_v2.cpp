
#include <cuda.h>

#include "icla_internal.h"
#include "error.h"

extern "C" void
icla_hgemm(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaHalf alpha,
    iclaHalf_const_ptr dA, icla_int_t ldda,
    iclaHalf_const_ptr dB, icla_int_t lddb,
    iclaHalf beta,
    iclaHalf_ptr       dC, icla_int_t lddc,
    icla_queue_t queue )
{
#if CUDA_VERSION >= 7500
    icla_int_t arch = icla_getdevice_arch();
    if( arch >= 530 ) {
        #if CUDA_VERSION >= 9000

        cublasSetMathMode(queue->cublas_handle(), CUBLAS_TENSOR_OP_MATH);
        #endif

        cublasHgemm(
            queue->cublas_handle(),
            cublas_trans_const( transA ),
            cublas_trans_const( transB ),
            int(m), int(n), int(k),
            &alpha, dA, int(ldda),
                    dB, int(lddb),
            &beta,  dC, int(lddc) );

        #if CUDA_VERSION >= 9000

        cublasSetMathMode(queue->cublas_handle(), CUBLAS_DEFAULT_MATH);
        #endif
    }
    else {
        printf("ERROR: unsupported architecture for %s \n", __func__ );
    }
#elif defined(ICLA_HAVE_HIP)
    icla_int_t arch = icla_getdevice_arch();
    if( arch >= 330 ) {
        hipblasGemmEx(
		      queue->hipblas_handle(),
		      hipblas_trans_const( transA ),
		      hipblas_trans_const( transB ),
		      int(m), int(n), int(k),
		      (void*)&alpha, (void*)dA, HIPBLAS_R_16F, int(ldda),
		      (void*)dB, HIPBLAS_R_16F, int(lddb),
		      (void *)&beta,  (void*)dC, HIPBLAS_R_16F, int(lddc),
		      HIPBLAS_R_16F,
		      HIPBLAS_GEMM_DEFAULT);
    }
    else {
        printf("ERROR: unsupported architecture for %s \n", __func__ );
    }
#else
    printf("ERROR: unsupported architecture version for %s \n", __func__ );
#endif
}

extern "C" void
icla_hgemmx(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    float alpha,
    iclaHalf_const_ptr dA, icla_int_t ldda,
    iclaHalf_const_ptr dB, icla_int_t lddb,
    float beta,
    float *dC, icla_int_t lddc,
    icla_queue_t queue )
{
#if defined(ICLA_HAVE_HIP)
    icla_int_t arch = icla_getdevice_arch();
    if( arch >= 330 ) {
        hipblasGemmEx(
		      queue->hipblas_handle(),
		      hipblas_trans_const( transA ),
		      hipblas_trans_const( transB ),
		      int(m), int(n), int(k),
		      (void*)&alpha, (void*)dA, HIPBLAS_R_16F, int(ldda),
                                     (void*)dB, HIPBLAS_R_16F, int(lddb),
		      (void*)&beta,  (void*)dC, HIPBLAS_R_32F, int(lddc),
		      HIPBLAS_R_32F,
		      HIPBLAS_GEMM_DEFAULT);
    }
    else {
        printf("ERROR: unsupported architecture for %s \n", __func__ );
    }
#else
    #if CUDA_VERSION >= 7500
    icla_int_t arch = icla_getdevice_arch();
    if( arch >= 530 ) {
        #if CUDA_VERSION >= 9000

        cublasSetMathMode(queue->cublas_handle(), CUBLAS_TENSOR_OP_MATH);
        #endif
        cublasGemmEx( queue->cublas_handle(),
                      cublas_trans_const( transA ), cublas_trans_const( transB ),
                      int(m), int(n), int(k),
                      &alpha, dA,        CUDA_R_16F, int(ldda),
                              dB,        CUDA_R_16F, int(lddb),
                      &beta,  dC,        CUDA_R_32F, int(lddc),
                      CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        #if CUDA_VERSION >= 9000

        cublasSetMathMode(queue->cublas_handle(), CUBLAS_DEFAULT_MATH);
        #endif
    }
    #endif
#endif
}

