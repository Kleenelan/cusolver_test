#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>


#include "flops.h"
#include "icla_v2.h"
#include "icla_lapack.h"
#include "testings.h"

#include "icla_internal.h"

extern "C" icla_int_t
icla_sgetrf_gpu(
    icla_int_t M, icla_int_t N,
    iclaFloat_ptr d_A, icla_int_t ldda,
    icla_int_t *ipiv,
    icla_int_t *info, icla_queue_t the_queue, real_Double_t *gpu_time_to )
{
    real_Double_t gpu_time = 0;
    icla_int_t min_mn = 0;
    icla_int_t     *devIpiv = nullptr;
    icla_int_t     *devInfo = nullptr;
    iclaFloat_ptr   dwork = NULL;
    icla_int_t      lwork = -1;

    min_mn = min__(M, N);
#ifdef ICLA_HAVE_CUDA
    cusolverDnHandle_t cusolverH = NULL;
    cusolverH = the_queue->cusolverdn_handle();

    cusolverStatus_t cusolverS = CUSOLVER_STATUS_SUCCESS;
    cusolverS = cusolverDnSgetrf_bufferSize(cusolverH, M, N, d_A, ldda, &lwork);
#endif

    if( lwork > 0 ) {
        icla_malloc( (void**)&dwork, lwork*sizeof(float) );
    }

    icla_malloc( (void**)&devIpiv, min_mn*sizeof(int) );
    icla_malloc((void**)&devInfo, sizeof(int));

#ifdef ICLA_HAVE_CUDA
    cudaEvent_t start, end;

    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    cusolverS = cusolverDnSgetrf(cusolverH, M, N, d_A, ldda, dwork, devIpiv, devInfo);
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float time_ms = 0.f;
    cudaEventElapsedTime(&time_ms, start, end);

#endif
    *gpu_time_to = time_ms/1000.0;

    *info = 1;
    icla_getvector(1, sizeof(int), devInfo, 1, info, 1, the_queue);
    icla_getvector(min_mn, sizeof(int), devIpiv, 1, ipiv, 1, the_queue);

    if( dwork != NULL ) {
        icla_free( dwork );
    }
    if( devInfo != NULL ){
        icla_free( devInfo );
    }
    if( devIpiv != NULL ){
        icla_free( devIpiv );
    }

    return *info;
}

