
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "flops.h"
#include "icla_v2.h"
#include "icla_lapack.h"
#include "testings.h"


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

    for (icla_int_t i = 0; i < 4; ++i) {
        opts.iseed[i] = iseed_save[i];
    }
}

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

    icla_int_t ISEED[4] = {0,0,0,1};
    icla_int_t info = 0;
    float *x, *b;

    TESTING_CHECK( icla_smalloc_cpu( &x, n ));
    TESTING_CHECK( icla_smalloc_cpu( &b, n ));
    lapackf77_slarnv( &ione, ISEED, &n, b );
    blasf77_scopy( &n, b, &ione, x, &ione );

    lapackf77_sgetrs( "Notrans", &n, &ione, A, &lda, ipiv, x, &n, &info );
    if (info != 0) {
        printf("lapackf77_sgetrs returned error %lld: %s.\n",
               (long long) info, icla_strerror( info ));
    }

    init_matrix( opts, m, n, A, lda );

    blasf77_sgemv( "Notrans", &m, &n, &c_one, A, &lda, x, &ione, &c_neg_one, b, &ione );

    float norm_x, norm_A, norm_r, work[1];
    norm_A = lapackf77_slange( "F", &m, &n, A, &lda, work );
    norm_r = lapackf77_slange( "F", &n, &ione, b, &n, work );
    norm_x = lapackf77_slange( "F", &n, &ione, x, &n, work );

    icla_free_cpu( x );
    icla_free_cpu( b );

    return norm_r / (n * norm_A * norm_x);
}

float get_LU_error(
    icla_opts &opts,
    icla_int_t M, icla_int_t N,
    float *LU, icla_int_t lda,
    icla_int_t *ipiv)
{
    icla_int_t min_mn = min__(M,N);
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

    init_matrix( opts, M, N, A, lda );
    lapackf77_slaswp( &N, A, &lda, &ione, &min_mn, ipiv, &ione);

    lapackf77_slacpy( IclaLowerStr, &M, &min_mn, LU, &lda, L, &M      );
    lapackf77_slacpy( IclaUpperStr, &min_mn, &N, LU, &lda, U, &min_mn );
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

//#define BUILD_MAIN_ 1
#ifdef BUILD_MAIN_

int main( int argc, char** argv)
{
	printf("argc = %d\n", argc);
    TESTING_CHECK( icla_init() );
    icla_print_environment();

    icla_opts opts;
    opts.parse_opts( argc, argv );

    iclaInt_ptr success_array = NULL;
    icla_imalloc_cpu( &success_array, opts.ntest*opts.niter );
#else

int gtest_sgetrf_gpu( iclaInt_ptr success_array, iclaInt_ptr len)
{
    int argc;
    char** argv;

    argc = 5;
    argv = (char**)malloc(argc*sizeof(char*));
    argv[0] = "helloWorld";
    argv[1] = "-l";
    argv[2] = "-c";
    argv[3] = "--niter";
    argv[4] = "1";

    TESTING_CHECK( icla_init() );
    icla_print_environment();

    icla_opts opts;
    opts.parse_opts( argc, argv );

    *len = opts.ntest * opts.niter;
    printf("len = %d\n", len);
#endif
    //TESTING_CHECK( icla_init() );
    //icla_print_environment();

    real_Double_t   gflops=0, gpu_perf=0, gpu_time=0, cpu_perf=0, cpu_time=0;
    float          error;
    float *h_A;
    iclaFloat_ptr d_A;
    icla_int_t     *ipiv = nullptr;

    icla_int_t M, N, n2, lda, ldda, info, min_mn;
    int status = 0;

//    icla_opts opts;
//    opts.parse_opts( argc, argv );

    float tol = opts.tolerance * lapackf77_slamch("E");

    icla_device_t cdev;
    icla_queue_t the_queue;
    icla_getdevice( &cdev );
    icla_queue_create( cdev, &the_queue );

//    iclaInt_ptr success_array = NULL;
//    icla_imalloc_cpu( &success_array, opts.ntest*opts.niter );
    memset( success_array, -1, opts.ntest*opts.niter*sizeof(icla_int_t) );

    printf("%% version %lld\n", (long long) opts.version );
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
            min_mn = min__(M, N);
            lda    = M;
            n2     = lda*N;
            ldda   = icla_roundup( M, opts.align );
            gflops = FLOPS_SGETRF( M, N ) / 1e9;

            TESTING_CHECK( icla_imalloc_cpu( &ipiv, min_mn ));
            TESTING_CHECK( icla_smalloc_cpu( &h_A,  n2     ));
            TESTING_CHECK( icla_smalloc( &d_A,  ldda*N ));

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

            init_matrix( opts, M, N, h_A, lda );
            if ( opts.version == 2 ) {

                for (icla_int_t i=0; i < min_mn; ++i ) {
                    ipiv[i] = i+1;
                }
            }
            icla_ssetmatrix( M, N, h_A, lda, d_A, ldda, opts.queue );

            if (opts.version == 1) {
                icla_sgetrf_gpu( M, N, d_A, ldda, ipiv, &info, the_queue, &gpu_time);
            }
            gpu_perf = gflops / gpu_time;

            if (info != 0) {
                printf("icla_sgetrf_gpu returned error %lld: %s.\n",
                       (long long) info, icla_strerror( info ));
            }

            if ( opts.lapack ) {
                printf("%5lld %5lld   %7.2f (%7.2f)   %9.2f (%9.6f)",
                       (long long) M, (long long) N, cpu_perf, cpu_time, gpu_perf, gpu_time );
            }
            else {
                printf("%5lld %5lld     ---   (  ---  )   %7.2f (%7.4f)",
                       (long long) M, (long long) N, gpu_perf, gpu_time );
            }

            if ( opts.check == 2 ) {
                icla_sgetmatrix( M, N, d_A, ldda, h_A, lda, opts.queue );
                error = get_residual( opts, M, N, h_A, lda, ipiv );
                printf("   %8.2e   %s\n", error, (error < tol ? (success_array[itest*opts.niter + iter] = 0, "ok") : (success_array[itest*opts.niter + iter] = -1, "failed")));
                status += ! (error < tol);
            }
            else if ( opts.check ) {
                icla_sgetmatrix( M, N, d_A, ldda, h_A, lda, opts.queue );
                error = get_LU_error( opts, M, N, h_A, lda, ipiv );
                printf("   %8.2e   %s\n", error, (error < tol ? (success_array[itest*opts.niter + iter] = 0, "ok") : (success_array[itest*opts.niter + iter] = -1, "failed")));
                status += ! (error < tol);
            }
            else {
                printf("     ---  \n");
            }

            icla_free_cpu( ipiv );
            icla_free_cpu( h_A );
            icla_free( d_A );

            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }
    if ( opts.check || opts.check == 2) {
        printf("\n\nTest status:\n");
        for( int itest = 0; itest < opts.ntest; ++itest ) {
            for( int iter = 0; iter < opts.niter; ++iter ) {
                printf("(%d, %s) ", (itest*opts.niter + iter), (success_array[itest*opts.niter + iter]==0? "OK" : "FAILED"));
            }
            printf(" ");
        }
        printf("\n\n");
    }

    icla_queue_destroy( the_queue );

    opts.cleanup();
    TESTING_CHECK( icla_finalize() );
    return status;
}


#ifndef BUILD_MAIN_
#include "gtest/gtest.h"

//int main()
TEST(DnSgetrf, perf_pricision)
{

//int gtest_sgetrf_gpu( iclaInt_ptr success_array )
	iclaInt_ptr sa;
    icla_int_t len = 0;
	sa = (iclaInt_ptr)malloc(20*sizeof(icla_int_t));
	gtest_sgetrf_gpu(sa, &len);
    printf("lenlenlen=%d\n", len);
    printf("\n");
    for(int i=0; i<len; i++){
        printf("%d ", sa[i]);
        EXPECT_EQ(sa[i], 0);
    }
    printf("\n");



//	return 0;

}
#endif