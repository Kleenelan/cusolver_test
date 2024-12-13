
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

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing spotrf
*/
//#define BUILD_MAIN_ 1
#ifdef BUILD_MAIN_
int main( int argc, char** argv)
{
    TESTING_CHECK( icla_init() );
    icla_print_environment();

    icla_opts opts;
    opts.matrix = "rand_dominant";  // default
    opts.parse_opts( argc, argv );

    iclaInt_ptr success_array = NULL;
    icla_imalloc_cpu( &success_array, opts.ntest*opts.niter );

#else
int gtest_spotrf_gpu(iclaInt_ptr success_array, iclaInt_ptr len)
{
    int argc = 5;
    char** argv;
    argv = (char**)malloc(20*sizeof(char*));
    argv[0] = "helloWorld";
    argv[1] = "-l";
    argv[2] = "-c";
    argv[3] = "--niter";
    argv[4] = "1";

    TESTING_CHECK( icla_init() );
    icla_print_environment();

    icla_opts opts;
    opts.matrix = "rand_dominant";  // default
    opts.parse_opts( argc, argv );
#endif

    // constants
    const float c_neg_one = ICLA_S_NEG_ONE;
    const icla_int_t ione = 1;

    // locals
    real_Double_t   gflops, gpu_perf=0, gpu_time=0, cpu_perf=0, cpu_time=0;
    float *h_A, *h_R;
    iclaFloat_ptr d_A;
    icla_int_t N, n2, lda, ldda, info;
    float      Anorm, error, work[1], *sigma;
    int status = 0;


    opts.lapack |= opts.check;  // check (-c) implies lapack (-l)

    float tol = opts.tolerance * lapackf77_slamch("E");

    // for expert API testing
    icla_device_t cdev;
    icla_queue_t the_queue;
    icla_getdevice( &cdev );
    icla_queue_create( cdev, &the_queue );

    printf("%% uplo = %s\n", lapack_uplo_const(opts.uplo) );
    printf("%% N     CPU Gflop/s (sec)   GPU Gflop/s (sec)   ||R_icla - R_lapack||_F / ||R_lapack||_F\n");
    printf("%%=======================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N   = opts.nsize[itest];
            lda = max__(1, N);
            n2  = lda*N;
            ldda = max__(1, icla_roundup( N, opts.align ));  // multiple of 32 by default
            gflops = FLOPS_SPOTRF( N ) / 1e9;

            TESTING_CHECK( icla_smalloc_cpu( &h_A, n2 ));
            TESTING_CHECK( icla_smalloc_cpu( &sigma, N ));
            TESTING_CHECK( icla_smalloc_pinned( &h_R, n2 ));
            TESTING_CHECK( icla_smalloc( &d_A, ldda*N ));

            /* Initialize the matrix */
            icla_generate_matrix( opts, N, N, h_A, lda, sigma );
            lapackf77_slacpy( IclaFullStr, &N, &N, h_A, &lda, h_R, &lda );

            icla_ssetmatrix( N, N, h_A, lda, d_A, ldda, opts.queue );

            /* ====================================================================
               Performs operation using ICLA
               =================================================================== */
            if(opts.version == 1){
                //gpu_time = icla_wtime();
                //icla_spotrf_gpu( opts.uplo, N, d_A, ldda, &info );
                //icla_sgetrf_gpu( M, N, d_A, ldda, ipiv, &info, the_queue, &gpu_time);
                icla_spotrf_gpu( opts.uplo, N, d_A, ldda, &info, the_queue, &gpu_time );
                //gpu_time = icla_wtime() - gpu_time;
            }
#if 0
            else if(opts.version == 2){
                gpu_time = icla_wtime();
                //LL::  icla_spotrf_native(opts.uplo, N, d_A, ldda, &info );
                gpu_time = icla_wtime() - gpu_time;
            }
            else if(opts.version == 3 || opts.version == 4) {
                // expert interface
                icla_mode_t mode = (opts.version == 3) ? IclaHybrid : IclaNative;
                icla_int_t nb    = 32;//LL:: icla_get_spotrf_nb( N );
                icla_int_t recnb = 128;

                // query workspace
                void *hwork = NULL, *dwork=NULL;
                icla_int_t lhwork[1] = {-1}, ldwork[1] = {-1};
                /*LL::  icla_spotrf_expert_gpu_work(
                    opts.uplo, N, NULL, ldda, &info,
                    mode, nb, recnb,
                    NULL, lhwork, NULL, ldwork,
                    events, queues );*/

                // alloc workspace
                if( lhwork[0] > 0 ) {
                    icla_malloc_pinned( (void**)&hwork, lhwork[0] );
                }

                if( ldwork[0] > 0 ) {
                    icla_malloc( (void**)&dwork, ldwork[0] );
                }

                // time actual call only
                gpu_time = icla_wtime();
                /*LL::  icla_spotrf_expert_gpu_work(
                    opts.uplo, N, d_A, ldda, &info,
                    mode, nb, recnb,
                    hwork, lhwork, dwork, ldwork,
                    events, queues );*/
                icla_queue_sync( queues[0] );
                icla_queue_sync( queues[1] );
                gpu_time = icla_wtime() - gpu_time;

                // free workspace
                if( hwork != NULL) {
                    icla_free_pinned( hwork );
                }

                if( dwork != NULL ) {
                    icla_free( dwork );
                }
            }
#endif

            gpu_perf = gflops / gpu_time;
            if (info != 0) {
                printf("icla_spotrf_gpu returned error %lld: %s.\n",
                       (long long) info, icla_strerror( info ));
            }

            if ( opts.lapack ) {
                /* =====================================================================
                   Performs operation using LAPACK
                   =================================================================== */
                cpu_time = icla_wtime();
                lapackf77_spotrf( lapack_uplo_const(opts.uplo), &N, h_A, &lda, &info );
                cpu_time = icla_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0) {
                    printf("lapackf77_spotrf returned error %lld: %s.\n",
                           (long long) info, icla_strerror( info ));
                }

                /* =====================================================================
                   Check the result compared to LAPACK
                   =================================================================== */
                icla_sgetmatrix( N, N, d_A, ldda, h_R, lda, opts.queue );
                blasf77_saxpy(&n2, &c_neg_one, h_A, &ione, h_R, &ione);
                #ifndef ICLA_HAVE_HIP
                Anorm = lapackf77_slange("f", &N, &N, h_A, &lda, work);
                error = lapackf77_slange("f", &N, &N, h_R, &lda, work) / Anorm;
                #else
                // TODO: use slange when the herk/syrk implementations are standardized.
                // For HIP, the current herk/syrk routines overwrite the entire diagonal
                // blocks of the matrix, so using slange causes the error check to fail
                Anorm = safe_lapackf77_slansy( "f", lapack_uplo_const(opts.uplo), &N, h_A, &lda, work );
                error = safe_lapackf77_slansy( "f", lapack_uplo_const(opts.uplo), &N, h_R, &lda, work ) / Anorm;
                #endif

                if (N == 0)
                    error = 0.0;

                printf("%5lld   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %s\n",
                       (long long) N, cpu_perf, cpu_time, gpu_perf, gpu_time,
                       error, (error < tol ? "ok" : "failed") );
                status += ! (error < tol);
            }
            else {
                printf("%5lld     ---   (  ---  )   %7.2f (%7.2f)     ---  \n",
                       (long long) N, gpu_perf, gpu_time );
            }
            icla_free_cpu( h_A );
            icla_free_cpu( sigma );
            icla_free_pinned( h_R );
            icla_free( d_A );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

#ifdef BUILD_MAIN_
    icla_free_cpu(success_array);
#endif

    icla_queue_destroy( the_queue );

    opts.cleanup();
    TESTING_CHECK( icla_finalize() );
    return status;
}



//int gtest_spotrf_gpu(iclaInt_ptr success_array, iclaInt_ptr len)
#ifndef BUILD_MAIN_
#include "gtest/gtest.h"

//int main()
TEST(DnSpotrf, spotrfPerf_pricision)
{

//int gtest_sgetrf_gpu( iclaInt_ptr success_array )
	iclaInt_ptr sa;
    icla_int_t len = 0;
	sa = (iclaInt_ptr)malloc(20*sizeof(icla_int_t));
	gtest_spotrf_gpu(sa, &len);
    printf("lenlenlen=%d\n", len);
    printf("\n");
    for(int i=0; i<len; i++){
        printf("%d ", sa[i]);
        EXPECT_EQ(sa[i], 0);
    }
    printf("\n");

    free(sa);
    sa = NULL;


//	return 0;

}
#endif