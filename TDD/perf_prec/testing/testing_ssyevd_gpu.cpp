//testing_zheevd_gpu.cpp

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "icla_v2.h"
#include "icla_lapack.h"
#include "icla_operators.h"
#include "testings.h"

#define REAL

//#define PR(x) printf("_-_-_-_-_-_ %d\n", x)
#define PR(x)
/* ////////////////////////////////////////////////////////////////////////////
   -- Testing ssyevd_gpu
*/
#ifdef BUILD_MAIN_
int main( int argc, char** argv)
{
    TESTING_CHECK( icla_init() );
    icla_print_environment();

    icla_opts opts;
    opts.parse_opts( argc, argv );

    iclaInt_ptr success_array = NULL;
    icla_imalloc_cpu( &success_array, opts.ntest*opts.niter );

    iclaInt_ptr len;

#else
int gtest_ssyevd_gpu( iclaInt_ptr success_array, iclaInt_ptr len)
{
    int argc;
    char** argv;

    argc = 6;
    argv = (char**)malloc(argc*sizeof(char*));
    argv[0] = "helloWorld";
    argv[1] = "-l";
    argv[2] = "-c";
    argv[3] = "--niter";
    argv[4] = "1";
    argv[5] = "-JV";

    TESTING_CHECK( icla_init() );
    icla_print_environment();

    icla_opts opts;
    opts.parse_opts( argc, argv );

    if(argv != NULL){
        free(argv);
        argv = NULL;
    }

    *len = opts.ntest * opts.niter;
    printf("len = %d\n", *len);
#endif

    /* Constants */
    const float d_zero = 0;
    const icla_int_t izero = 0;
    const icla_int_t ione  = 1;

    /* Local variables */
    real_Double_t   gpu_time, cpu_time;
    float *h_A, *h_R, *h_Z, *h_work, aux_work[1], unused[1];
    iclaFloat_ptr d_R, d_Z;
    #ifdef COMPLEX
    float *rwork, aux_rwork[1];
    icla_int_t lrwork;
    #endif
    float *w1, *w2, result[4]={0, 0, 0, 0}, eps, abstol, runused[1];
    icla_int_t *iwork, *isuppz, *ifail, aux_iwork[1];
    icla_int_t N, Nfound, info, lwork, liwork, lda, ldda;
    eps = lapackf77_slamch( "E" );
    int status = 0;



    // checking NoVec requires LAPACK
    opts.lapack |= (opts.check && opts.jobz == IclaNoVec);

    float tol    = opts.tolerance * lapackf77_slamch("E");
    float tolulp = opts.tolerance * lapackf77_slamch("P");

    icla_device_t cdev;
    icla_queue_t the_queue;
    icla_getdevice( &cdev );
    icla_queue_create( cdev, &the_queue );

    #ifdef REAL
    if (opts.version == 3 || opts.version == 4) {
        printf("%% icla_ssyevr and icla_ssyevx not available for real precisions (single, float).\n");
        status = -1;
        return status;
    }
    #endif

    if (opts.version > 1) {
        fprintf( stderr, "%% error: no version %lld, only 1.\n", (long long) opts.version );
        status = -1;
        return status;
    }

    const char *versions[] = {
        "dummy",
        "ssyevd_gpu",
        "ssyevdx_gpu",
        "ssyevr_gpu (Complex only)",
        "ssyevx_gpu (Complex only)"
    };

    printf("%% jobz = %s, uplo = %s, version = %lld (%s)\n",
           lapack_vec_const(opts.jobz), lapack_uplo_const(opts.uplo),
           (long long)opts.version, versions[opts.version] );

    printf("%%   N   CPU Time (sec)   GPU Time (sec)   |S-S_icla|   |A-USU^H|   |I-U^H U|\n");
    printf("%%============================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            Nfound = N;
            lda  = N;
            ldda = icla_roundup( N, opts.align );  // multiple of 32 by default
            abstol = 0;  // auto, in ssyevr

            icla_range_t range;
            icla_int_t il, iu;
            float vl, vu;
            opts.get_range( N, &range, &vl, &vu, &il, &iu );

            /* Allocate host memory for the matrix */
            TESTING_CHECK( icla_smalloc_cpu( &h_A,    N*lda  ));
            TESTING_CHECK( icla_smalloc_cpu( &w1,     N      ));
            TESTING_CHECK( icla_smalloc_cpu( &w2,     N      ));

            // query for workspace sizes
            if ( opts.version == 1) {
                lwork = -1;
                liwork = -1;
//void lapackf77_ssyevd( const char* jobz, const char* uplo, int* n, float* a, int* lda, float* w, float* work, int* lwork, int* iwork, int* liwork, int* info )
                lapackf77_ssyevd( lapack_vec_const(opts.jobz), lapack_uplo_const(opts.uplo),
                                  &N, h_A, &ldda, w2,
                                  aux_work, &lwork,
                                  aux_iwork, &liwork,
                                  &info );
            }

            lwork  = (icla_int_t) ICLA_S_REAL( aux_work[0] );
            liwork = aux_iwork[0];
            TESTING_CHECK( icla_imalloc_cpu( &iwork,  liwork ));
            TESTING_CHECK( icla_smalloc_pinned( &h_R,    N*lda  ));
            TESTING_CHECK( icla_smalloc_pinned( &h_work, lwork  ));
            TESTING_CHECK( icla_smalloc( &d_R,    N*ldda ));

            /* Clear eigenvalues, for |S-S_icla| check when fraction < 1. */
            lapackf77_slaset( "Full", &N, &ione, &d_zero, &d_zero, w1, &N );
            lapackf77_slaset( "Full", &N, &ione, &d_zero, &d_zero, w2, &N );

            /* Initialize the matrix */
            icla_generate_matrix( opts, N, N, h_A, lda );
            //icla_ssetmatrix( N, N, h_A, lda, d_R, ldda, opts.queue );
            icla_ssetmatrix( N, N, h_A, lda, d_R, ldda, the_queue );

            /* ====================================================================
               Performs operation using ICLA
               =================================================================== */
            //gpu_time = icla_wtime();
            if ( opts.version == 1 ) {
                icla_ssyevd_gpu(opts.jobz, opts.uplo,
                                N, d_R,
                                ldda, w1 ,
                                &info,
                                the_queue, &gpu_time);
    //icla_queue_t the_queue,    real_Double_t *gpu_time_to
            }
            //gpu_time = icla_wtime() - gpu_time;
            if (info != 0) {
                printf("icla_ssyevd_gpu returned error %lld: %s.\n",
                       (long long) info, icla_strerror( info ));
            }

            bool okay = true;
            if ( opts.check && opts.jobz != IclaNoVec ) {
                /* =====================================================================
                   Check the results following the LAPACK's [zcds]drvst routine.
                   A is factored as A = U S U^H and the following 3 tests computed:
                   (1)    | A - U S U^H | / ( |A| N ) if all eigenvectors were computed
                          | U^H A U - S | / ( |A| Nfound ) otherwise
                   (2)    | I - U^H U   | / ( N )
                   (3)    | S(with U) - S(w/o U) | / | S |    // currently disabled, but compares to LAPACK
                   =================================================================== */
                //icla_sgetmatrix( N, N, d_R, ldda, h_R, lda, opts.queue );
                icla_sgetmatrix( N, N, d_R, ldda, h_R, lda, the_queue );

                float *work;
                TESTING_CHECK( icla_smalloc_cpu( &work, 2*N*N ));

                // e is unused since kband=0; tau is unused since itype=1
                if( Nfound == N ) {
                    lapackf77_ssyt21( &ione, lapack_uplo_const(opts.uplo), &N, &izero,
                                      h_A, &lda,
                                      w1, runused,
                                      h_R, &lda,
                                      h_R, &lda,
                                      unused, work,
                                      &result[0] );
                } else {
                    lapackf77_ssyt22( &ione, lapack_uplo_const(opts.uplo), &N, &Nfound, &izero,
                                      h_A, &lda,
                                      w1, runused,
                                      h_R, &lda,
                                      h_R, &lda,
                                      unused, work,
                                      &result[0] );
                }
                result[0] *= eps;
                result[1] *= eps;

                if(work != NULL)
                    icla_free_cpu( work );  //work=NULL;

                // Disable third eigenvalue check that calls routine again --
                // it obscures whether error occurs in first call above or in this call.
                // But see comparison to LAPACK below.
                //
                //icla_ssetmatrix( N, N, h_A, lda, d_R, ldda, opts.queue );
                //icla_ssyevd_gpu( IclaNoVec, opts.uplo,
                //                  N, d_R, ldda, w2,
                //                  h_R, lda,
                //                  h_work, lwork,
                //                  iwork, liwork,
                //                  &info);
                //if (info != 0) {
                //    printf("icla_ssyevd_gpu returned error %lld: %s.\n",
                //           (long long) info, icla_strerror( info ));
                //}
                //
                //float maxw=0, diff=0;
                //for( int j=0; j < N; j++ ) {
                //    maxw = max(maxw, fabs(w1[j]));
                //    maxw = max(maxw, fabs(w2[j]));
                //    diff = max(diff, fabs(w1[j] - w2[j]));
                //}
                //result[2] = diff / (N*maxw);
            }

            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = icla_wtime();
                if ( opts.version == 1 ) {
                    lapackf77_ssyevd( lapack_vec_const(opts.jobz), lapack_uplo_const(opts.uplo),
                                      &N, h_A, &lda, w2,
                                      h_work, &lwork,
                                      iwork, &liwork,
                                      &info );
//void lapackf77_ssyevd( const char* jobz, const char* uplo, int* n, float* a, int* lda, float* w, float* work, int* lwork, int* iwork, int* liwork, int* info );
                }

                cpu_time = icla_wtime() - cpu_time;
                if (info != 0) {
                    printf("lapackf77_ssyevd returned error %lld: %s.\n",
                           (long long) info, icla_strerror( info ));
                }

                // compare eigenvalues
                float maxw=0, diff=0;
                for( int j=0; j < Nfound; j++ ) {
                    maxw = max__(maxw, fabs(w1[j]));
                    maxw = max__(maxw, fabs(w2[j]));
                    diff = max__(diff, fabs(w1[j] - w2[j]));
                }
                result[3] = diff / (N*maxw);

                okay = okay && (result[3] < tolulp);
                printf("%5lld   %9.4f        %9.4f         %8.2e  ",
                       (long long) N, cpu_time, gpu_time, result[3] );
            }
            else {
                printf("%5lld      ---           %9.4f           ---     ",
                       (long long) N, gpu_time);
            }

            // print error checks
            if ( opts.check && opts.jobz != IclaNoVec ) {
                okay = okay && (result[0] < tol) && (result[1] < tol);
                printf("    %8.2e    %8.2e", result[0], result[1] );
            }
            else {
                printf("      ---         ---   ");
            }
            printf("   %s\n", (okay ? (success_array[itest*opts.niter + iter] = 0, "ok") : (success_array[itest*opts.niter + iter] = -1, "failed")));
//printf( "%11.2e   %11.2e   %s\n", error, error2, (okay ?(success_array[itest*opts.niter + iter] = 0, "ok") : (success_array[itest*opts.niter + iter] = -1, "failed")) );

            status += ! okay;

            icla_free_cpu( h_A    );
            icla_free_cpu( w1     );
            icla_free_cpu( w2     );
            icla_free_cpu( iwork  );

            icla_free_pinned( h_R    );
            icla_free_pinned( h_work );

            icla_free( d_R );

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

//int main()
TEST(DnSsyevd, DnSsyevd_perf_pricision)
{

	iclaInt_ptr sa;
    icla_int_t len = 0;
	sa = (iclaInt_ptr)malloc(20*sizeof(icla_int_t));
	gtest_ssyevd_gpu(sa, &len);
//printf("lenlenlen=%d\n", len);
//    printf("\n");
    for(int i=0; i<len; i++){
//        printf("%d ", sa[i]);
        EXPECT_EQ(sa[i], 0);
    }
    printf("\n");

    if(sa != NULL){
        free(sa);
        sa = NULL;
    }
}
#endif




