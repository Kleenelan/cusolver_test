
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

#include "unistd.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing sgeqrf
*/

#ifdef BUILD_MAIN_
int main( int argc, char** argv)
{
    TESTING_CHECK( icla_init() );
    icla_print_environment();

    icla_opts opts;
    opts.parse_opts( argc, argv );

#else
int gtest_sgeqrf_gpu(iclaInt_ptr success_array, iclaInt_ptr len)
{
    int argc = 7;
    char** argv;
    argv = (char**)malloc(20*sizeof(char*));
    argv[0] = "helloWorld";
    argv[1] = "-l";
    argv[2] = "-c2";
    argv[3] = "--niter";
    argv[4] = "1";
    argv[5] = "--version";
    argv[6] = "2";

    TESTING_CHECK( icla_init() );
    icla_print_environment();

    icla_opts opts;
    opts.parse_opts( argc, argv );
#endif

    const float             d_neg_one = ICLA_D_NEG_ONE;
    const float             d_one     = ICLA_D_ONE;
    const float c_neg_one = ICLA_S_NEG_ONE;
    const float c_one     = ICLA_S_ONE;
    const float c_zero    = ICLA_S_ZERO;
    const icla_int_t        ione      = 1;

    real_Double_t    gflops, gpu_perf, gpu_time, cpu_perf=0, cpu_time=0;
    float           Anorm, error=0, error2=0;
    float *h_A, *h_R, *tau, *h_work, tmp[1], unused[1];
    iclaFloat_ptr d_A;
    icla_int_t M, N, n2, lda, ldda, lwork, info, min_mn, size;
    icla_int_t ISEED[4] = {0,0,0,1};



    int status = 0;
    float tol = opts.tolerance * lapackf77_slamch("E");

    // for expert API testing
    icla_device_t cdev;
    icla_queue_t the_queue;
    icla_getdevice( &cdev );
    icla_queue_create( cdev, &the_queue );

    if (opts.check == 2 && opts.version == 2) {
        opts.check = 1;
        printf( "%% version 2 requires check 1 (R - Q^H*A)\n" );
    }

    printf( "%% version %lld\n", (long long) opts.version );

    if ( opts.check == 1 ) {
        printf("%%   M     N   CPU Gflop/s (sec)   GPU Gflop/s (sec)   |R - Q^H*A|   |I - Q^H*Q|\n");
        printf("%%==============================================================================\n");
    }

    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            min_mn = min__( M, N );
            lda    = M;
            n2     = lda*N;
            ldda   = icla_roundup( M, opts.align );  // multiple of 32 by default
            gflops = FLOPS_SGEQRF( M, N ) / 1e9;

            // query for workspace size
            lwork = -1;
            lapackf77_sgeqrf( &M, &N, unused, &M, unused, tmp, &lwork, &info );
            lwork = (icla_int_t)ICLA_S_REAL( tmp[0] );

            TESTING_CHECK( icla_smalloc_cpu( &tau,    min_mn ));
            TESTING_CHECK( icla_smalloc_cpu( &h_A,    n2     ));
            TESTING_CHECK( icla_smalloc_cpu( &h_work, lwork  ));

            TESTING_CHECK( icla_smalloc_pinned( &h_R,    n2     ));

            TESTING_CHECK( icla_smalloc( &d_A,    ldda*N ));

            /* Initialize the matrix */
            icla_generate_matrix( opts, M, N, h_A, lda );
            lapackf77_slacpy( IclaFullStr, &M, &N, h_A, &lda, h_R, &lda );
            icla_ssetmatrix( M, N, h_R, lda, d_A, ldda, opts.queue );

            /* ====================================================================
               Performs operation using ICLA
               =================================================================== */
            if ( opts.version == 2 ) {
                // LAPACK complaint arguments
                icla_sgeqrf_gpu( M, N, d_A, ldda, tau, &info, the_queue, &gpu_time);
            }
            else {
                printf( "Unknown version %lld\n", (long long) opts.version );
                return -1;
            }
            gpu_perf = gflops / gpu_time;

            if (info != 0) {
                printf("icla_sgeqrf returned error %lld: %s.\n",
                       (long long) info, icla_strerror( info ));
            }

            if ( opts.check == 1 && (opts.version == 2 )) {
                /* =====================================================================
                   Check the result, following zqrt01 except using the reduced Q.
                   This works for any M,N (square, tall, wide).
                   Only for version 2, which has LAPACK complaint output.
                   Or   for version 3, after restoring diagonal blocks of A above.
                   =================================================================== */
                icla_sgetmatrix( M, N, d_A, ldda, h_R, lda, opts.queue );

                icla_int_t ldq = M;
                icla_int_t ldr = min_mn;
                float *Q, *R;
                float *work;
                TESTING_CHECK( icla_smalloc_cpu( &Q,    ldq*min_mn ));  // M by K
                TESTING_CHECK( icla_smalloc_cpu( &R,    ldr*N ));       // K by N
                TESTING_CHECK( icla_smalloc_cpu( &work, min_mn ));

                // generate M by K matrix Q, where K = min(M,N)
                lapackf77_slacpy( "Lower", &M, &min_mn, h_R, &lda, Q, &ldq );
                lapackf77_sorgqr( &M, &min_mn, &min_mn, Q, &ldq, tau, h_work, &lwork, &info );
                assert( info == 0 );

                // copy K by N matrix R
                lapackf77_slaset( "Lower", &min_mn, &N, &c_zero, &c_zero, R, &ldr );
                lapackf77_slacpy( "Upper", &min_mn, &N, h_R, &lda,        R, &ldr );

                // error = || R - Q^H*A || / (N * ||A||)
                blasf77_sgemm( "Conj", "NoTrans", &min_mn, &N, &M,
                               &c_neg_one, Q, &ldq, h_A, &lda, &c_one, R, &ldr );
                Anorm = lapackf77_slange( "1", &M,      &N, h_A, &lda, work );
                error = lapackf77_slange( "1", &min_mn, &N, R,   &ldr, work );
                if ( N > 0 && Anorm > 0 )
                    error /= (N*Anorm);

                // set R = I (K by K identity), then R = I - Q^H*Q
                // error = || I - Q^H*Q || / N
                lapackf77_slaset( "Upper", &min_mn, &min_mn, &c_zero, &c_one, R, &ldr );
                blasf77_ssyrk( "Upper", "Conj", &min_mn, &M, &d_neg_one, Q, &ldq, &d_one, R, &ldr );
                error2 = safe_lapackf77_slansy( "1", "Upper", &min_mn, R, &ldr, work );
                if ( N > 0 )
                    error2 /= N;

                icla_free_cpu( Q    );  Q    = NULL;
                icla_free_cpu( R    );  R    = NULL;
                icla_free_cpu( work );  work = NULL;
            }

            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = icla_wtime();
                lapackf77_sgeqrf( &M, &N, h_A, &lda, tau, h_work, &lwork, &info );
                cpu_time = icla_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0) {
                    printf("lapackf77_sgeqrf returned error %lld: %s.\n",
                           (long long) info, icla_strerror( info ));
                }
            }

            /* =====================================================================
               Print performance and error.
               =================================================================== */
            printf("%5lld %5lld   ", (long long) M, (long long) N );
            if ( opts.lapack ) {
                printf( "%7.2f (%7.2f)", cpu_perf, cpu_time );
            }
            else {
                printf("  ---   (  ---  )" );
            }
            printf( "   %7.2f (%7.2f)   ", gpu_perf, gpu_time );
            if ( opts.check == 1 ) {
                bool okay = (error < tol && error2 < tol);
                status += ! okay;
                printf( "%11.2e   %11.2e   %s\n", error, error2, (okay ?(success_array[itest*opts.niter + iter] = 0, "ok") : (success_array[itest*opts.niter + iter] = -1, "failed")) );
            }
            else if ( opts.check == 2 ) {
                if ( M >= N ) {
                    bool okay = (error < tol);
                    status += ! okay;
                    printf( "%10.2e   %s\n", error, (okay ? "ok" : "failed") );
                }
                else {
                    printf( "(error check only for M >= N)\n" );
                }
            }
            else {
                printf( "    ---\n" );
            }

            icla_free_cpu( tau    );
            icla_free_cpu( h_A    );
            icla_free_cpu( h_work );

            icla_free_pinned( h_R );

            icla_free( d_A );

            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    if ( opts.check == 1) {
        printf("\n\nTest status:\n");
        for( int itest = 0; itest < opts.ntest; ++itest ) {
            for( int iter = 0; iter < opts.niter; ++iter ) {
                printf("(%d, %s) ", (itest*opts.niter + iter), (success_array[itest*opts.niter + iter]==0? "OK" : "FAILED"));
            }
            printf(" ");
        }
        printf("\n\n");
    }
#ifdef BUILD_MAIN_
    icla_free_cpu(success_array);
#endif

    icla_queue_destroy( the_queue );

    opts.cleanup();
    TESTING_CHECK( icla_finalize() );
    return status;
}



#ifndef BUILD_MAIN_
#include "gtest/gtest.h"

TEST(DnSgeqrf, DnSgeqrf_perf_pricision)
{
	iclaInt_ptr sa;
    icla_int_t len = 0;
	sa = (iclaInt_ptr)malloc(20*sizeof(icla_int_t));
	gtest_sgeqrf_gpu(sa, &len);

    printf("\n");
    for(int i=0; i<len; i++){
        EXPECT_EQ(sa[i], 0);
    }
    printf("\n");

    free(sa);
    sa = NULL;
}
#endif



