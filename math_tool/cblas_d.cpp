
#include "icla_internal.h"

#define REAL

extern "C"
double icla_cblas_dasum(
    icla_int_t n,
    const double *x, icla_int_t incx )
{
    if ( n <= 0 || incx <= 0 ) {
        return 0;
    }
    double result = 0;
    if ( incx == 1 ) {
        for( icla_int_t i=0; i < n; ++i ) {
            result += ICLA_D_ABS1( x[i] );
        }
    }
    else {
        icla_int_t nincx = n*incx;
        for( icla_int_t i=0; i < nincx; i += incx ) {
            result += ICLA_D_ABS1( x[i] );
        }
    }
    return result;
}

static inline double sqr( double x ) { return x*x; }

extern "C"
double icla_cblas_dnrm2(
    icla_int_t n,
    const double *x, icla_int_t incx )
{
    if (n <= 0 || incx <= 0) {
        return 0;
    }
    else {
        double scale = 0;
        double ssq   = 1;

        for( icla_int_t ix=0; ix < 1 + (n-1)*incx; ix += incx ) {
            if ( real( x[ix] ) != 0 ) {
                double temp = fabs( real( x[ix] ));
                if (scale < temp) {
                    ssq = 1 + ssq * sqr(scale/temp);
                    scale = temp;
                }
                else {
                    ssq += sqr(temp/scale);
                }
            }
            #ifdef COMPLEX
            if ( imag( x[ix] ) != 0 ) {
                double temp = fabs( imag( x[ix] ));
                if (scale < temp) {
                    ssq = 1 + ssq * sqr(scale/temp);
                    scale = temp;
                }
                else {
                    ssq += sqr(temp/scale);
                }
            }
            #endif
        }
        return scale*icla_dsqrt(ssq);
    }
}

#ifdef COMPLEX

extern "C"
double icla_cblas_ddot(
    icla_int_t n,
    const double *x, icla_int_t incx,
    const double *y, icla_int_t incy )
{

    double value = ICLA_D_ZERO;
    icla_int_t i;
    if ( incx == 1 && incy == 1 ) {
        for( i=0; i < n; ++i ) {
            value = value + conj( x[i] ) * y[i];
        }
    }
    else {
        icla_int_t ix=0, iy=0;
        if ( incx < 0 ) { ix = (-n + 1)*incx; }
        if ( incy < 0 ) { iy = (-n + 1)*incy; }
        for( i=0; i < n; ++i ) {
            value = value + conj( x[ix] ) * y[iy];
            ix += incx;
            iy += incy;
        }
    }
    return value;
}
#endif

extern "C"
double icla_cblas_ddot(
    icla_int_t n,
    const double *x, icla_int_t incx,
    const double *y, icla_int_t incy )
{

    double value = ICLA_D_ZERO;
    icla_int_t i;
    if ( incx == 1 && incy == 1 ) {
        for( i=0; i < n; ++i ) {
            value = value + x[i] * y[i];
        }
    }
    else {
        icla_int_t ix=0, iy=0;
        if ( incx < 0 ) { ix = (-n + 1)*incx; }
        if ( incy < 0 ) { iy = (-n + 1)*incy; }
        for( i=0; i < n; ++i ) {
            value = value + x[ix] * y[iy];
            ix += incx;
            iy += incy;
        }
    }
    return value;
}

#undef REAL
