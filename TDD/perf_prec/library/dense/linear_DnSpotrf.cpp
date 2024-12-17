

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "flops.h"
#include "icla_v2.h"
#include "icla_lapack.h"
#include "testings.h"

#include "icla_internal.h"



icla_int_t
icla_spotrf_gpu(
    icla_uplo_t uplo, icla_int_t N,
    iclaFloat_ptr dA, icla_int_t ldda,
    icla_int_t *info, icla_queue_t the_queue,
    iclaDouble_ptr gpu_time_to)
{
//    icla_int_t      min_mn = 0;
    icla_int_t     *devIpiv = nullptr;
    icla_int_t     *devInfo = nullptr;
    iclaFloat_ptr   dwork = NULL;
    icla_int_t      lwork = -1;

    //min_mn = min__(M, N);
#ifdef ICLA_HAVE_CUDA
    cusolverDnHandle_t cusolverH = NULL;
    cusolverH = the_queue->cusolverdn_handle();

    cusolverStatus_t cusolverS = CUSOLVER_STATUS_SUCCESS;
    //cusolverStatus_t cusolverDnSpotrf_bufferSize(cusolverDnHandle_t handle,cublasFillMode_t uplo,int n,float *A,int lda,int *Lwork );
    cublasFillMode_t upperLower;

    if(uplo == IclaUpper)
        upperLower = CUBLAS_FILL_MODE_UPPER;
    else if(uplo == IclaLower)
        upperLower = CUBLAS_FILL_MODE_LOWER;
    else
        exit(-1);

    cusolverS = cusolverDnSpotrf_bufferSize(cusolverH, upperLower, N, dA, ldda, &lwork);
#endif

    if( lwork > 0 ) {
        icla_malloc( (void**)&dwork, lwork*sizeof(float) );
    }

    icla_malloc((void**)&devInfo, sizeof(int));

#ifdef ICLA_HAVE_CUDA
    cudaEvent_t start, end;

    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    //cusolverStatus_t cusolverDnSpotrf(cusolverDnHandle_t handle,cublasFillMode_t uplo,int n,float *A,int lda,float *Workspace,int Lwork,int *devInfo );
    cusolverS = cusolverDnSpotrf(cusolverH, upperLower, N, dA, ldda, dwork, lwork, devInfo);
    //cusolverS = cusolverDnSgetrf(cusolverH, M, N, d_A, ldda, dwork, devIpiv, devInfo);

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float time_ms = 0.f;
    cudaEventElapsedTime(&time_ms, start, end);

#endif
    *gpu_time_to = time_ms/1000.0;

    *info = 1;
    icla_getvector(1, sizeof(int), devInfo, 1, info, 1, the_queue);

    if( dwork != NULL ) {
        icla_free( dwork );
    }
    if( devInfo != NULL ){
        icla_free( devInfo );
    }

    return *info;
}
