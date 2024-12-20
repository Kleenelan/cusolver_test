

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
icla_ssyevd_gpu(
    icla_vec_t jobz_, icla_uplo_t uplo_,
    icla_int_t n,
    iclaFloat_ptr d_A, icla_int_t lda,
    float *w, icla_int_t *info,
    icla_queue_t the_queue, real_Double_t *gpu_time_to)
{
    float *d_W = nullptr;
    int *devInfo = nullptr;
    int lwork = 0;            /* size of workspace */
    float *d_work = nullptr; /* device workspace*/

    icla_malloc( (void**)&d_W, n*sizeof(float) );
    icla_malloc(reinterpret_cast<void **>(&devInfo), sizeof(int));

    cusolverEigMode_t jobz;

    if(jobz_ == IclaVec)
        jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
    else if(jobz_ == IclaNoVec)
        jobz = CUSOLVER_EIG_MODE_NOVECTOR;
    else
        exit(1);

    cublasFillMode_t uplo;

    if(uplo_ == IclaUpper)
        uplo = CUBLAS_FILL_MODE_UPPER;
    else if(uplo_ == IclaLower)
        uplo = CUBLAS_FILL_MODE_LOWER;
    else
        exit(1);
   // query working space of syevd
#ifdef ICLA_HAVE_CUDA
    cusolverDnHandle_t cusolverDnH = NULL;

    cusolverDnH = the_queue->cusolverdn_handle();
    cusolverDnSsyevd_bufferSize(cusolverDnH, jobz, uplo, n, (float*)d_A, lda, d_W, &lwork);
#endif

    if(lwork>0)
        icla_malloc(reinterpret_cast<void **>(&d_work), sizeof(float) * lwork);

#ifdef ICLA_HAVE_CUDA
    cudaEvent_t start, end;

    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    // compute spectrum
    cusolverDnSsyevd(cusolverDnH, jobz, uplo, n, (float*)d_A, lda, d_W, d_work, lwork, devInfo);

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float time_ms = 0.f;
    cudaEventElapsedTime(&time_ms, start, end);
#endif

    *gpu_time_to = time_ms/1000.0;

    icla_getvector(n, sizeof(float), d_W, 1, w, 1, the_queue);
    *info = -1;
    icla_getvector(1, sizeof(int), devInfo, 1, info, 1, the_queue);

    if (0 > *info) {
        std::printf("%d-th parameter is wrong \n", -(*info));
        exit(1);
    }

    /* free resources */
    if(d_W != NULL)
        icla_free(d_W);
    if(devInfo != NULL)
        icla_free(devInfo);
    if(d_work != NULL)
        icla_free(d_work);

    return *info;
}
