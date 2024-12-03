
#ifndef LAPACK_HPP
#define LAPACK_HPP

#include <complex>

#include "icla_v2.h"
#include "icla_lapack.h"

namespace blas
{

template< typename T >
class traits
{
public:
    typedef T real_t;
    static T make( real_t r, real_t i )
        { return r; }
};

template< typename T >
class traits< std::complex<T> >
{
public:
    typedef T real_t;
    static std::complex<T> make( T r, T i )
        { return std::complex<T>( r, i ); }
};

template<>
class traits< iclaFloatComplex >
{
public:
    typedef float real_t;
    static iclaFloatComplex make( real_t r, real_t i )
        { return ICLA_C_MAKE( r, i ); }
};

template<>
class traits< iclaDoubleComplex >
{
public:
    typedef double real_t;
    static iclaDoubleComplex make( real_t r, real_t i )
        { return ICLA_Z_MAKE( r, i ); }
};

template< typename T1, typename T2 >
class traits2
{
public:
    typedef T1 scalar_t;
    typedef T1 real_t;
};

template<>
class traits2< float, double >
{
public:
    typedef double scalar_t;
    typedef double real_t;
};

template<>
class traits2< float, std::complex<float> >
{
public:
    typedef std::complex<float> scalar_t;
    typedef float real_t;
};

template<>
class traits2< float, iclaFloatComplex >
{
public:
    typedef iclaFloatComplex scalar_t;
    typedef float real_t;
};

template<>
class traits2< float, std::complex<double> >
{
public:
    typedef std::complex<double> scalar_t;
    typedef double real_t;
};

template<>
class traits2< float, iclaDoubleComplex >
{
public:
    typedef iclaDoubleComplex scalar_t;
    typedef double real_t;
};

template<>
class traits2< double, std::complex<float> >
{
public:
    typedef std::complex<double> scalar_t;
    typedef double real_t;
};

template<>
class traits2< double, iclaFloatComplex >
{
public:
    typedef iclaDoubleComplex scalar_t;
    typedef double real_t;
};

template<>
class traits2< double, std::complex<double> >
{
public:
    typedef std::complex<double> scalar_t;
    typedef double real_t;
};

template<>
class traits2< double, iclaDoubleComplex >
{
public:
    typedef iclaDoubleComplex scalar_t;
    typedef double real_t;
};

template<>
class traits2< std::complex<float>, std::complex<double> >
{
public:
    typedef std::complex<double> scalar_t;
    typedef double real_t;
};

template<>
class traits2< iclaFloatComplex, iclaDoubleComplex >
{
public:
    typedef iclaDoubleComplex scalar_t;
    typedef double real_t;
};

template< typename T1, typename T2, typename T3 >
class traits3
{
public:
    typedef typename
        traits2< typename traits2<T1,T2>::scalar_t, T3 >::scalar_t scalar_t;

    typedef typename
        traits2< typename traits2<T1,T2>::scalar_t, T3 >::real_t real_t;
};

inline float  asum( icla_int_t n, float* x, icla_int_t incx )
{
    return icla_cblas_sasum( n, x, incx );
}

inline double asum(
    icla_int_t n, double* x, icla_int_t incx )
{
    return icla_cblas_dasum( n, x, incx );
}

inline float  asum( icla_int_t n, iclaFloatComplex* x, icla_int_t incx )
{
    return icla_cblas_scasum( n, x, incx );
}

inline double asum(
    icla_int_t n, iclaDoubleComplex* x, icla_int_t incx )
{
    return icla_cblas_dzasum( n, x, incx );
}

inline float  dot(
    icla_int_t n,
    float* x, icla_int_t incx,
    float* y, icla_int_t incy )
{
    return icla_cblas_sdot( n, x, incx, y, incy );
}

inline double dot(
    icla_int_t n,
    double* x, icla_int_t incx,
    double* y, icla_int_t incy )
{
    return icla_cblas_ddot( n, x, incx, y, incy );
}

inline iclaFloatComplex  dot(
    icla_int_t n,
    iclaFloatComplex* x, icla_int_t incx,
    iclaFloatComplex* y, icla_int_t incy )
{
    return icla_cblas_cdotc( n, x, incx, y, incy );
}

inline iclaDoubleComplex dot(
    icla_int_t n,
    iclaDoubleComplex* x, icla_int_t incx,
    iclaDoubleComplex* y, icla_int_t incy )
{
    return icla_cblas_zdotc( n, x, incx, y, incy );
}

inline void copy(
    icla_int_t n,
    float* x, icla_int_t incx,
    float* y, icla_int_t incy )
{
    return blasf77_scopy( &n, x, &incx, y, &incy );
}

inline void copy(
    icla_int_t n,
    double* x, icla_int_t incx,
    double* y, icla_int_t incy )
{
    return blasf77_dcopy( &n, x, &incx, y, &incy );
}

inline void copy(
    icla_int_t n,
    iclaFloatComplex* x, icla_int_t incx,
    iclaFloatComplex* y, icla_int_t incy )
{
    return blasf77_ccopy( &n, x, &incx, y, &incy );
}

inline void copy(
    icla_int_t n,
    iclaDoubleComplex* x, icla_int_t incx,
    iclaDoubleComplex* y, icla_int_t incy )
{
    return blasf77_zcopy( &n, x, &incx, y, &incy );
}

inline void rot(
    icla_int_t n,
    float* x, icla_int_t incx,
    float* y, icla_int_t incy,
    float c, float s )
{
    blasf77_srot( &n, x, &incx, y, &incy, &c, &s );
}

inline void rot(
    icla_int_t n,
    double* x, icla_int_t incx,
    double* y, icla_int_t incy,
    double c, double s )
{
    blasf77_drot( &n, x, &incx, y, &incy, &c, &s );
}

inline void rot(
    icla_int_t n,
    iclaFloatComplex* x, icla_int_t incx,
    iclaFloatComplex* y, icla_int_t incy,
    float c, float s )
{
    blasf77_csrot( &n, x, &incx, y, &incy, &c, &s );
}

inline void rot(
    icla_int_t n,
    iclaDoubleComplex* x, icla_int_t incx,
    iclaDoubleComplex* y, icla_int_t incy,
    double c, double s )
{
    blasf77_zdrot( &n, x, &incx, y, &incy, &c, &s );
}

inline void scal(
    icla_int_t n, float alpha,
    float *x, icla_int_t incx )
{
    blasf77_sscal( &n, &alpha, x, &incx );
}

inline void scal(
    icla_int_t n, double alpha,
    double *x, icla_int_t incx )
{
    blasf77_dscal( &n, &alpha, x, &incx );
}

inline void scal(
    icla_int_t n, iclaFloatComplex alpha,
    iclaFloatComplex *x, icla_int_t incx )
{
    blasf77_cscal( &n, &alpha, x, &incx );
}

inline void scal(
    icla_int_t n, iclaDoubleComplex alpha,
    iclaDoubleComplex *x, icla_int_t incx )
{
    blasf77_zscal( &n, &alpha, x, &incx );
}

}

namespace lapack
{

inline void larnv(
    icla_int_t idist, icla_int_t iseed[4],
    icla_int_t n, float *x )
{
    lapackf77_slarnv( &idist, iseed, &n, x );
}

inline void larnv(
    icla_int_t idist, icla_int_t iseed[4],
    icla_int_t n, double *x )
{
    lapackf77_dlarnv( &idist, iseed, &n, x );
}

inline void larnv(
    icla_int_t idist, icla_int_t iseed[4],
    icla_int_t n, iclaFloatComplex *x )
{
    lapackf77_clarnv( &idist, iseed, &n, x );
}

inline void larnv(
    icla_int_t idist, icla_int_t iseed[4],
    icla_int_t n, iclaDoubleComplex *x )
{
    lapackf77_zlarnv( &idist, iseed, &n, x );
}

inline void larfg(
    icla_int_t n,
    float* alpha,
    float* x, icla_int_t incx,
    float* tau )
{
    lapackf77_slarfg( &n, alpha, x, &incx, tau );
}

inline void larfg(
    icla_int_t n,
    double* alpha,
    double* x, icla_int_t incx,
    double* tau )
{
    lapackf77_dlarfg( &n, alpha, x, &incx, tau );
}

inline void larfg(
    icla_int_t n,
    iclaFloatComplex* alpha,
    iclaFloatComplex* x, icla_int_t incx,
    iclaFloatComplex* tau )
{
    lapackf77_clarfg( &n, alpha, x, &incx, tau );
}

inline void larfg(
    icla_int_t n,
    iclaDoubleComplex* alpha,
    iclaDoubleComplex* x, icla_int_t incx,
    iclaDoubleComplex* tau )
{
    lapackf77_zlarfg( &n, alpha, x, &incx, tau );
}

inline void laset(
    const char* uplo, icla_int_t m, icla_int_t n,
    float diag, float offdiag,
    float* A, icla_int_t lda )
{
    lapackf77_slaset( uplo, &m, &n, &diag, &offdiag, A, &lda );
}

inline void laset(
    const char* uplo, icla_int_t m, icla_int_t n,
    double diag, double offdiag,
    double* A, icla_int_t lda )
{
    lapackf77_dlaset( uplo, &m, &n, &diag, &offdiag, A, &lda );
}

inline void laset(
    const char* uplo, icla_int_t m, icla_int_t n,
    iclaFloatComplex diag, iclaFloatComplex offdiag,
    iclaFloatComplex* A, icla_int_t lda )
{
    lapackf77_claset( uplo, &m, &n, &diag, &offdiag, A, &lda );
}

inline void laset(
    const char* uplo, icla_int_t m, icla_int_t n,
    iclaDoubleComplex diag, iclaDoubleComplex offdiag,
    iclaDoubleComplex* A, icla_int_t lda )
{
    lapackf77_zlaset( uplo, &m, &n, &diag, &offdiag, A, &lda );
}

inline void unmqr(
    const char* side, const char* trans,
    icla_int_t m, icla_int_t n, icla_int_t k,
    float* A,    icla_int_t lda,
    float* tau,
    float* C,    icla_int_t ldc,
    float* work, icla_int_t lwork,
    icla_int_t* info )
{
    if (*trans == 'c' || *trans == 'C') {
        trans = "T";
    }
    lapackf77_sormqr( side, trans, &m, &n, &k,
                      A, &lda, tau, C, &ldc, work, &lwork, info );
}

inline void unmqr(
    const char* side, const char* trans,
    icla_int_t m, icla_int_t n, icla_int_t k,
    double* A,    icla_int_t lda,
    double* tau,
    double* C,    icla_int_t ldc,
    double* work, icla_int_t lwork,
    icla_int_t* info )
{
    if (*trans == 'c' || *trans == 'C') {
        trans = "T";
    }
    lapackf77_dormqr( side, trans, &m, &n, &k,
                      A, &lda, tau, C, &ldc, work, &lwork, info );
}

inline void unmqr(
    const char* side, const char* trans,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaFloatComplex* A,    icla_int_t lda,
    iclaFloatComplex* tau,
    iclaFloatComplex* C,    icla_int_t ldc,
    iclaFloatComplex* work, icla_int_t lwork,
    icla_int_t* info )
{
    lapackf77_cunmqr( side, trans, &m, &n, &k,
                      A, &lda, tau, C, &ldc, work, &lwork, info );
}

inline void unmqr(
    const char* side, const char* trans,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDoubleComplex* A,    icla_int_t lda,
    iclaDoubleComplex* tau,
    iclaDoubleComplex* C,    icla_int_t ldc,
    iclaDoubleComplex* work, icla_int_t lwork,
    icla_int_t* info )
{
    lapackf77_zunmqr( side, trans, &m, &n, &k,
                      A, &lda, tau, C, &ldc, work, &lwork, info );
}

}

#endif

