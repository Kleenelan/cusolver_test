
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "flops.h"
#include "icla_v2.h"
#include "icla_lapack.h"
#include "icla_operators.h"
#include "testings.h"
#if 0
int main(int argc, char **argv)
{
    #define  X(i_, j_)  ( X + (i_) + (j_)*lda)
    #define  Y(i_, j_)  ( Y + (i_) + (j_)*lda)

    #define dX(i_, j_)  (dX + (i_) + (j_)*ldda)
    #define dY(i_, j_)  (dY + (i_) + (j_)*ldda)

    TESTING_CHECK( icla_init() );
    icla_print_environment();

    real_Double_t   gflops, dev_perf, dev_time, cpu_perf, cpu_time;
    double          dev_error, work[1];
    icla_int_t ione     = 1;
    icla_int_t ISEED[4] = {0,0,0,1};
    icla_int_t M, N, lda, ldda, size;
    icla_int_t incx = 1;
    icla_int_t incy = 1;
    iclaDoubleComplex c_neg_one = ICLA_Z_NEG_ONE;
    iclaDoubleComplex alpha = ICLA_Z_MAKE(  1.5, -2.3 );
    iclaDoubleComplex *X, *Y, *Yresult;
    iclaDoubleComplex_ptr dX, dY;
    int status = 0;

    icla_opts opts;
    opts.parse_opts( argc, argv );

    double eps = lapackf77_dlamch("E");
    double tol = 3*eps;

    printf("%%   M   cnt     %s Gflop/s (ms)       CPU Gflop/s (ms)  %s error\n",
            g_platform_str, g_platform_str );
    printf("%%===========================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {

            M = opts.msize[itest];
            N = 100;
            lda    = icla_roundup( M, 8 );

            ldda   = icla_roundup( lda, opts.align );

            gflops = 2*M*N / 1e9;
            size   = ldda*N;

            TESTING_CHECK( icla_zmalloc_cpu( &X,       size ));
            TESTING_CHECK( icla_zmalloc_cpu( &Y,       size ));
            TESTING_CHECK( icla_zmalloc_cpu( &Yresult, size ));

            TESTING_CHECK( icla_zmalloc( &dX, size ));
            TESTING_CHECK( icla_zmalloc( &dY, size ));

            lapackf77_zlarnv( &ione, ISEED, &size, X );
            lapackf77_zlarnv( &ione, ISEED, &size, Y );

            double Xnorm = lapackf77_zlange( "F", &M, &N, X, &lda, work );
            double Ynorm = lapackf77_zlange( "F", &M, &N, Y, &lda, work );

            icla_zsetmatrix( M, N, X, lda, dX, ldda, opts.queue );
            icla_zsetmatrix( M, N, Y, lda, dY, ldda, opts.queue );

            icla_flush_cache( opts.cache );
            dev_time = icla_sync_wtime( opts.queue );
            for (int j = 0; j < N; ++j) {
                icla_zaxpy( M, alpha, dX(0,j), incx, dY(0,j), incy, opts.queue );
            }
            dev_time = icla_sync_wtime( opts.queue ) - dev_time;
            dev_perf = gflops / dev_time;

            icla_zgetmatrix( M, N, dY, ldda, Yresult, lda, opts.queue );

            icla_flush_cache( opts.cache );
            cpu_time = icla_wtime();
            for (int j = 0; j < N; ++j) {
                blasf77_zaxpy( &M, &alpha, X(0,j), &incx, Y(0,j), &incy );
            }
            cpu_time = icla_wtime() - cpu_time;
            cpu_perf = gflops / cpu_time;

            blasf77_zaxpy( &size, &c_neg_one, Y, &ione, Yresult, &ione );
            dev_error = lapackf77_zlange( "F", &M, &N, Yresult, &lda, work )
                            / (Xnorm + Ynorm);

            bool okay = (dev_error < tol);
            status += ! okay;
            printf("%5lld %5lld   %9.4f (%9.4f)   %9.4f (%9.4f)    %8.2e   %s\n",
                   (long long) M, (long long) N,
                   dev_perf,    1000.*dev_time,
                   cpu_perf,    1000.*cpu_time,
                   dev_error,
                   (okay ? "ok" : "failed"));

            icla_free_cpu( X );
            icla_free_cpu( Y );
            icla_free_cpu( Yresult );

            icla_free( dX );
            icla_free( dY );
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
#endif
