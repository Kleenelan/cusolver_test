#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>


#include "flops.h"
#include "icla_v2.h"
#include "icla_lapack.h"
#include "testings.h"

#include "icla_internal.h"
icla_int_t
icla_sgeqrf_gpu(
    icla_int_t m, icla_int_t n,
    iclaFloat_ptr dA, icla_int_t ldda,
    float *tau,
    icla_int_t *info, icla_queue_t the_queue,
    real_Double_t *gpu_time_to)
{
//ldwork
//dtau
// call cusolverDnSgeqrf();
//set tau
//time

    icla_int_t      min_mn = 0;
    icla_int_t     *devIpiv = nullptr;
    icla_int_t     *devInfo = nullptr;
    iclaFloat_ptr   dwork = NULL;
    icla_int_t      lwork = -1;
    iclaFloat_ptr   dtau = NULL;

    min_mn = min__(m, n);
#ifdef ICLA_HAVE_CUDA
    cusolverDnHandle_t cusolverH = NULL;
    cusolverH = the_queue->cusolverdn_handle();

    cusolverStatus_t cusolverS = CUSOLVER_STATUS_SUCCESS;
    //cusolverStatus_t cusolverDnSgeqrf_bufferSize(cusolverDnHandle_t handle, int m, int n, float *A, int lda, int *Lwork );
    //cusolverStatus_t cusolverDnSpotrf_bufferSize(cusolverDnHandle_t handle,cublasFillMode_t uplo,int n,float *A,int lda,int *Lwork );
    cublasFillMode_t upperLower;

    //cusolverS = cusolverDnSpotrf_bufferSize(cusolverH, upperLower, N, dA, ldda, &lwork);
    cusolverS = cusolverDnSgeqrf_bufferSize(cusolverH, m, n, dA, ldda, &lwork );
#endif

    if( lwork > 0 ) {
        icla_malloc( (void**)&dwork, lwork*sizeof(float) );
    }
    icla_malloc((void**)&dtau, min_mn*sizeof(float));

    icla_malloc((void**)&devInfo, sizeof(int));

#ifdef ICLA_HAVE_CUDA
    cudaEvent_t start, end;

    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    //cusolverStatus_t cusolverDnSpotrf(cusolverDnHandle_t handle,cublasFillMode_t uplo,int n,float *A,int lda,float *Workspace,int Lwork,int *devInfo );
//    cusolverS = cusolverDnSpotrf(cusolverH, upperLower, N, dA, ldda, dwork, lwork, devInfo);
    //cusolverS = cusolverDnSgetrf(cusolverH, M, N, d_A, ldda, dwork, devIpiv, devInfo);
    //cusolverStatus_t cusolverDnSgeqrf(cusolverDnHandle_t handle, int m, int n, float *A, int lda, float *TAU, float *Workspace, int Lwork, int *devInfo );

    cusolverS = cusolverDnSgeqrf(cusolverH, m, n, dA, ldda, dtau, dwork, lwork, devInfo);
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float time_ms = 0.f;
    cudaEventElapsedTime(&time_ms, start, end);

#endif
    *gpu_time_to = time_ms/1000.0;

    *info = 1;
    icla_getvector(1, sizeof(int), devInfo, 1, info, 1, the_queue);

    icla_getvector(min_mn, sizeof(float), dtau, 1, tau, 1, the_queue);

    if( dwork != NULL ) {
        icla_free( dwork );
    }
    if( dtau != NULL){
        icla_free( dtau );
    }
    if( devInfo != NULL ){
        icla_free( devInfo );
    }

    return *info;
}
