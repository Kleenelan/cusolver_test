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

#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cusolverDn.h"


// cusolverStatus_t cusolverDnSgetrf_bufferSize(cusolverDnHandle_t handle, int m, int n, float *A, int lda, int *Lwork );
// cusolverStatus_t cusolverDnSgetrf(cusolverDnHandle_t handle, int m, int n, float *A, int lda, float *Workspace, int *devIpiv, int *devInfo );

// icla_sgetrf( M, N, h_A, lda, ipiv, &info );
real_Double_t icla_sgetrf( icla_int_t _M, icla_int_t _N, float* _h_A, icla_int_t _lda, icla_int_t* _ipiv, icla_int_t* _info )
{
#if defined(ICLA_HAVE_CUDA)
    int M = _M, N = _N, lda = _lda;
    int mn_min = min(M, N);
    int* devIpiv = nullptr;
    int* devInfo = nullptr;
    float* Workspace = nullptr;
    float* A_d = nullptr;

    cusolverDnHandle_t handle = nullptr;
    cusolverDnCreate(&handle);

    int Lwork = 0;
    cusolverDnSgetrf_bufferSize(handle, M, N, A_d, lda, &Lwork);
    cudaMalloc((void**)&A_d, lda*N*sizeof(float));
    cudaMalloc((void**)&devIpiv, mn_min*sizeof(float));
    cudaMalloc((void**)&devInfo, sizeof(int));
    cudaMalloc((void**)&Workspace, Lwork*sizeof(float));

    cudaMemcpy(A_d, _h_A, lda*N*sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    cusolverDnSgetrf(handle, M, N, A_d, lda, Workspace, devIpiv, devInfo);
    cudaEventRecord(end);

    cudaEventSynchronize(end);
    float time_ms = 0.f;
    cudaEventElapsedTime(&time_ms, start, end);

    cudaMemcpy(_h_A, A_d, lda*N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(_ipiv, devIpiv, mn_min*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(_info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    A_d = nullptr;
    cudaFree(devIpiv);
    devIpiv = nullptr;
    cudaFree(devInfo);
    devInfo = nullptr;
    cudaFree(Workspace);
    Workspace = nullptr;
#endif
    return time_ms;
}

real_Double_t icla_sgetrf_nopiv( icla_int_t _M, icla_int_t _N, float* _h_A, icla_int_t _lda, icla_int_t* _info )
{
#if defined(ICLA_HAVE_CUDA)
    int M = _M, N = _N, lda = _lda;
    int mn_min = min(M, N);
    int* devIpiv = nullptr;
    int* devInfo = nullptr;
    float* Workspace = nullptr;
    float* A_d = nullptr;

    cusolverDnHandle_t handle = nullptr;
    cusolverDnCreate(&handle);

    int Lwork = 0;
    cusolverDnSgetrf_bufferSize(handle, M, N, A_d, lda, &Lwork);
    cudaMalloc((void**)&A_d, lda*N*sizeof(float));
    //cudaMalloc((void**)&devIpiv, mn_min*sizeof(float));
    cudaMalloc((void**)&devInfo, sizeof(int));
    cudaMalloc((void**)&Workspace, Lwork*sizeof(float));

    cudaMemcpy(A_d, _h_A, lda*N*sizeof(float), cudaMemcpyHostToDevice);


    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    cusolverDnSgetrf(handle, M, N, A_d, lda, Workspace, devIpiv, devInfo);
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float time_ms = 0.f;
    cudaEventElapsedTime(&time_ms, start, end);

    cudaMemcpy(_h_A, A_d, lda*N*sizeof(float), cudaMemcpyDeviceToHost);
    //cudaMemcpy(_ipiv, devIpiv, mn_min*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(_info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    A_d = nullptr;
    //cudaFree(devIpiv);
    devIpiv = nullptr;
    cudaFree(devInfo);
    devInfo = nullptr;
    cudaFree(Workspace);
    Workspace = nullptr;
#endif
    return time_ms;
}



// Initialize matrix to random.
// This ensures the same ISEED is always used,
// so we can re-generate the identical matrix.
void init_matrix(
    icla_opts &opts,
    icla_int_t m, icla_int_t n,
    float *A, icla_int_t lda )
{
    icla_int_t iseed_save[4];
    for (icla_int_t i = 0; i < 4; ++i) {
        iseed_save[i] = opts.iseed[i];
    }

    icla_generate_matrix( opts, m, n, A, lda );

    // restore iseed
    for (icla_int_t i = 0; i < 4; ++i) {
        opts.iseed[i] = iseed_save[i];
    }
}


// On input, A and ipiv is LU factorization of A. On output, A is overwritten.
// Requires m == n.
// Uses init_matrix() to re-generate original A as needed.
// Generates random RHS b and solves Ax=b.
// Returns residual, |Ax - b| / (n |A| |x|).
float get_residual(
    icla_opts &opts,
    icla_int_t m, icla_int_t n,
    float *A, icla_int_t lda,
    icla_int_t *ipiv )
{
    if ( m != n ) {
        printf( "\nERROR: residual check defined only for square matrices\n" );
        return -1;
    }

    const float c_one     = ICLA_S_ONE;
    const float c_neg_one = ICLA_S_NEG_ONE;
    const icla_int_t ione = 1;

    // this seed should be DIFFERENT than used in init_matrix
    // (else x is column of A, so residual can be exactly zero)
    icla_int_t ISEED[4] = {0,0,0,1};
    icla_int_t info = 0;
    float *x, *b;

    // initialize RHS
    TESTING_CHECK( icla_smalloc_cpu( &x, n ));
    TESTING_CHECK( icla_smalloc_cpu( &b, n ));
    lapackf77_slarnv( &ione, ISEED, &n, b );
    blasf77_scopy( &n, b, &ione, x, &ione );

    // solve Ax = b
    lapackf77_sgetrs( "Notrans", &n, &ione, A, &lda, ipiv, x, &n, &info );
    if (info != 0) {
        printf("lapackf77_sgetrs returned error %lld: %s.\n",
               (long long) info, icla_strerror( info ));
    }

    // reset to original A
    init_matrix( opts, m, n, A, lda );

    // compute r = Ax - b, saved in b
    blasf77_sgemv( "Notrans", &m, &n, &c_one, A, &lda, x, &ione, &c_neg_one, b, &ione );

    // compute residual |Ax - b| / (n*|A|*|x|)
    float norm_x, norm_A, norm_r, work[1];
    norm_A = lapackf77_slange( "F", &m, &n, A, &lda, work );
    norm_r = lapackf77_slange( "F", &n, &ione, b, &n, work );
    norm_x = lapackf77_slange( "F", &n, &ione, x, &n, work );

    //printf( "r=\n" ); icla_sprint( 1, n, b, 1 );

    icla_free_cpu( x );
    icla_free_cpu( b );

    //printf( "r=%.2e, A=%.2e, x=%.2e, n=%lld\n", norm_r, norm_A, norm_x, (long long) n );
    return norm_r / (n * norm_A * norm_x);
}


// On input, LU and ipiv is LU factorization of A. On output, LU is overwritten.
// Works for any m, n.
// Uses init_matrix() to re-generate original A as needed.
// Returns error in factorization, |PA - LU| / (n |A|)
// This allocates 3 more matrices to store A, L, and U.
float get_LU_error(
    icla_opts &opts,
    icla_int_t M, icla_int_t N,
    float *LU, icla_int_t lda,
    icla_int_t *ipiv)
{
    icla_int_t min_mn = min(M,N);
    icla_int_t ione   = 1;
    icla_int_t i, j;
    float alpha = ICLA_S_ONE;
    float beta  = ICLA_S_ZERO;
    float *A, *L, *U;
    float work[1], matnorm, residual;

    TESTING_CHECK( icla_smalloc_cpu( &A, lda*N    ));
    TESTING_CHECK( icla_smalloc_cpu( &L, M*min_mn ));
    TESTING_CHECK( icla_smalloc_cpu( &U, min_mn*N ));
    memset( L, 0, M*min_mn*sizeof(float) );
    memset( U, 0, min_mn*N*sizeof(float) );

    // set to original A
    init_matrix( opts, M, N, A, lda );
    lapackf77_slaswp( &N, A, &lda, &ione, &min_mn, ipiv, &ione);

    // copy LU to L and U, and set diagonal to 1
    lapackf77_slacpy( iclaLowerStr, &M, &min_mn, LU, &lda, L, &M      );
    lapackf77_slacpy( iclaUpperStr, &min_mn, &N, LU, &lda, U, &min_mn );
    for (j=0; j < min_mn; j++)
        L[j+j*M] = ICLA_S_MAKE( 1., 0. );

    matnorm = lapackf77_slange("f", &M, &N, A, &lda, work);

    blasf77_sgemm("N", "N", &M, &N, &min_mn,
                  &alpha, L, &M, U, &min_mn, &beta, LU, &lda);

    for( j = 0; j < N; j++ ) {
        for( i = 0; i < M; i++ ) {
            LU[i+j*lda] = ICLA_S_SUB( LU[i+j*lda], A[i+j*lda] );
        }
    }
    residual = lapackf77_slange("f", &M, &N, LU, &lda, work);

    icla_free_cpu( A );
    icla_free_cpu( L );
    icla_free_cpu( U );

    return residual / (matnorm * N);
}


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing sgetrf
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( icla_init() );
    icla_print_environment();

    real_Double_t   gflops, gpu_perf, gpu_time, cpu_perf=0, cpu_time=0;
    float          error;
    float *h_A;
    icla_int_t     *ipiv;
    icla_int_t     M, N, n2, lda, info, min_mn;
    int status = 0;

    icla_opts opts;
    opts.parse_opts( argc, argv );

    float tol = opts.tolerance * lapackf77_slamch("E");

    printf("%% ngpu %lld, version %lld\n", (long long) opts.ngpu, (long long) opts.version );
    if ( opts.check == 2 ) {
        printf("%%   M     N   CPU Gflop/s (sec)   GPU Gflop/s (sec)   |Ax-b|/(N*|A|*|x|)\n");
    }
    else {
        printf("%%   M     N   CPU Gflop/s (sec)   GPU Gflop/s (sec)   |PA-LU|/(N*|A|)\n");
    }
    printf("%%========================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            min_mn = min(M, N);
            lda    = M;
            n2     = lda*N;
            gflops = FLOPS_SGETRF( M, N ) / 1e9;

            TESTING_CHECK( icla_imalloc_cpu( &ipiv, min_mn ));
            TESTING_CHECK( icla_smalloc_pinned( &h_A,  n2 ));

            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                init_matrix( opts, M, N, h_A, lda );

                cpu_time = icla_wtime();
                lapackf77_sgetrf( &M, &N, h_A, &lda, ipiv, &info );
                cpu_time = icla_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0) {
                    printf("lapackf77_sgetrf returned error %lld: %s.\n",
                           (long long) info, icla_strerror( info ));
                }
            }

            /* ====================================================================
               Performs operation using ICLA
               =================================================================== */
            init_matrix( opts, M, N, h_A, lda );
            if ( opts.version == 2 || opts.version == 3 ) {
                // no pivoting versions, so set ipiv to identity
                for (icla_int_t i=0; i < min_mn; ++i ) {
                    ipiv[i] = i+1;
                }
            }

            //gpu_time = icla_wtime();
            if ( opts.version == 1 ) {
                gpu_time = icla_sgetrf( M, N, h_A, lda, ipiv, &info );
            }
            else if ( opts.version == 2 ) {
                gpu_time = icla_sgetrf_nopiv( M, N, h_A, lda, &info );
            }
            //gpu_time = icla_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0) {
                printf("icla_sgetrf returned error %lld: %s.\n",
                       (long long) info, icla_strerror( info ));
            }

            /* =====================================================================
               Check the factorization
               =================================================================== */
            if ( opts.lapack ) {
                printf("%5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)",
                       (long long) M, (long long) N, cpu_perf, cpu_time, gpu_perf, gpu_time );
            }
            else {
                printf("%5lld %5lld     ---   (  ---  )   %7.2f (%7.2f)",
                       (long long) M, (long long) N, gpu_perf, gpu_time );
            }
            if ( opts.check == 2 ) {
                error = get_residual( opts, M, N, h_A, lda, ipiv );
                printf("   %8.2e   %s\n", error, (error < tol ? "ok" : "failed"));
                status += ! (error < tol);
            }
            else if ( opts.check ) {
                error = get_LU_error( opts, M, N, h_A, lda, ipiv );
                printf("   %8.2e   %s\n", error, (error < tol ? "ok" : "failed"));
                status += ! (error < tol);
            }
            else {
                printf("     ---   \n");
            }

            icla_free_cpu( ipiv );
            icla_free_pinned( h_A  );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    opts.cleanup();
    TESTING_CHECK( icla_finalize() );
    return status;
}
