
#include <exception>
#include <string>
#include <vector>
#include <limits>

#include "icla_v2.h"
#include "icla_lapack.hpp"

#include "icla_operators.h"

#include "icla_matrix.hpp"

#include "testings.h"
#undef max
#undef min
using std::max;
using std::min;

const icla_int_t idist_rand  = 1;
const icla_int_t idist_rands = 2;
const icla_int_t idist_randn = 3;

enum class MatrixType {
    rand      = 1,

    rands     = 2,

    randn     = 3,

    zero,
    ones,
    identity,
    jordan,
    kronecker,
    diag,
    svd,
    poev,
    heev,
    geev,
    geevx,
};

enum class Dist {
    rand      = 1,

    rands     = 2,

    randn     = 3,

    arith,
    geo,
    cluster0,
    cluster1,
    rarith,
    rgeo,
    rcluster0,
    rcluster1,
    logrand,
    specified,
};

const char *ansi_esc    = "\x1b[";
const char *ansi_red    = "\x1b[31m";
const char *ansi_bold   = "\x1b[1m";
const char *ansi_normal = "\x1b[0m";

template< typename FloatT >
inline FloatT rand( FloatT max_ )
{
    return max_ * rand() / FloatT(RAND_MAX);
}

inline bool begins( std::string const &str, std::string const &prefix )
{
    return (str.compare( 0, prefix.size(), prefix) == 0);
}

inline bool contains( std::string const &str, std::string const &pattern )
{
    return (str.find( pattern ) != std::string::npos);
}

template< typename FloatT >
void icla_generate_sigma(
    icla_opts& opts,
    Dist dist, bool rand_sign,
    typename blas::traits<FloatT>::real_t cond,
    typename blas::traits<FloatT>::real_t sigma_max,
    Matrix<FloatT>& A,
    Vector< typename blas::traits<FloatT>::real_t >& sigma )
{
    typedef typename blas::traits<FloatT>::real_t real_t;

    const FloatT c_zero = blas::traits<FloatT>::make( 0, 0 );

    icla_int_t minmn = min( A.m, A.n );
    assert( minmn == sigma.n );

    switch (dist) {
        case Dist::arith:
            for (icla_int_t i = 0; i < minmn; ++i) {
                sigma[i] = 1 - i / real_t(minmn - 1) * (1 - 1/cond);
            }
            break;

        case Dist::rarith:
            for (icla_int_t i = 0; i < minmn; ++i) {
                sigma[i] = 1 - (minmn - 1 - i) / real_t(minmn - 1) * (1 - 1/cond);
            }
            break;

        case Dist::geo:
            for (icla_int_t i = 0; i < minmn; ++i) {
                sigma[i] = pow( cond, -i / real_t(minmn - 1) );
            }
            break;

        case Dist::rgeo:
            for (icla_int_t i = 0; i < minmn; ++i) {
                sigma[i] = pow( cond, -(minmn - 1 - i) / real_t(minmn - 1) );
            }
            break;

        case Dist::cluster0:
            sigma[0] = 1;
            for (icla_int_t i = 1; i < minmn; ++i) {
                sigma[i] = 1/cond;
            }
            break;

        case Dist::rcluster0:
            for (icla_int_t i = 0; i < minmn-1; ++i) {
                sigma[i] = 1/cond;
            }
            sigma[minmn-1] = 1;
            break;

        case Dist::cluster1:
            for (icla_int_t i = 0; i < minmn-1; ++i) {
                sigma[i] = 1;
            }
            sigma[minmn-1] = 1/cond;
            break;

        case Dist::rcluster1:
            sigma[0] = 1/cond;
            for (icla_int_t i = 1; i < minmn; ++i) {
                sigma[i] = 1;
            }
            break;

        case Dist::logrand: {
            real_t range = log( 1/cond );
            lapack::larnv( idist_rand, opts.iseed, sigma.n, sigma(0) );
            for (icla_int_t i = 0; i < minmn; ++i) {
                sigma[i] = exp( sigma[i] * range );
            }

            if (minmn >= 2) {
                sigma[0] = 1;
                sigma[1] = 1/cond;
            }
            break;
        }

        case Dist::randn:
        case Dist::rands:
        case Dist::rand: {
            icla_int_t idist = (icla_int_t) dist;
            lapack::larnv( idist, opts.iseed, sigma.n, sigma(0) );
            break;
        }

        case Dist::specified:

            sigma_max = 1;
            rand_sign = false;
            break;
    }

    if (sigma_max != 1) {
        blas::scal( sigma.n, sigma_max, sigma(0), 1 );
    }

    if (rand_sign) {

        for (icla_int_t i = 0; i < minmn; ++i) {
            if (rand() > RAND_MAX/2) {
                sigma[i] = -sigma[i];
            }
        }
    }

    lapack::laset( "general", A.m, A.n, c_zero, c_zero, A(0,0), A.ld );
    for (icla_int_t i = 0; i < minmn; ++i) {
        *A(i,i) = blas::traits<FloatT>::make( sigma[i], 0 );
    }
}

template< typename FloatT >
void icla_generate_correlation_factor( Matrix<FloatT>& A )
{

    Vector<FloatT> x( A.n );
    for (icla_int_t j = 0; j < A.n; ++j) {
        x[j] = blas::dot( A.m, A(0,j), 1, A(0,j), 1 );
    }

    for (icla_int_t i = 0; i < A.n; ++i) {
        for (icla_int_t j = 0; j < A.n; ++j) {
            if ((x[i] < 1 && 1 < x[j]) || (x[i] > 1 && 1 > x[j])) {
                FloatT xij, d, t, c, s;
                xij = blas::dot( A.m, A(0,i), 1, A(0,j), 1 );
                d = sqrt( xij*xij - (x[i] - 1)*(x[j] - 1) );
                t = (xij + std::copysign( d, xij )) / (x[j] - 1);
                c = 1 / sqrt(1 + t*t);
                s = c*t;
                blas::rot( A.m, A(0,i), 1, A(0,j), 1, c, -s );
                x[i] = blas::dot( A.m, A(0,i), 1, A(0,i), 1 );

                x[i] = 1;
                x[j] = blas::dot( A.m, A(0,j), 1, A(0,j), 1 );
                break;
            }
        }
    }
}

template<>
void icla_generate_correlation_factor( Matrix<iclaFloatComplex>& A )
{
    assert( false );
}

template<>
void icla_generate_correlation_factor( Matrix<iclaDoubleComplex>& A )
{
    assert( false );
}

template< typename FloatT >
void icla_generate_svd(
    icla_opts& opts,
    Dist dist,
    typename blas::traits<FloatT>::real_t cond,
    typename blas::traits<FloatT>::real_t condD,
    typename blas::traits<FloatT>::real_t sigma_max,
    Matrix<FloatT>& A,
    Vector< typename blas::traits<FloatT>::real_t >& sigma )
{
    typedef typename blas::traits<FloatT>::real_t real_t;

    FloatT tmp;
    icla_int_t m = A.m;
    icla_int_t n = A.n;
    icla_int_t maxmn = max( m, n );
    icla_int_t minmn = min( m, n );
    icla_int_t sizeU;
    icla_int_t info = 0;
    Matrix<FloatT> U( maxmn, minmn );
    Vector<FloatT> tau( minmn );

    icla_int_t lwork = -1;
    lapack::unmqr( "Left", "NoTrans", A.m, A.n, minmn,
                   U(0,0), U.ld, tau(0), A(0,0), A.ld,
                   &tmp, lwork, &info );
    assert( info == 0 );
    lwork = icla_int_t( real( tmp ));
    icla_int_t lwork2 = -1;
    lapack::unmqr( "Right", "ConjTrans", A.m, A.n, minmn,
                   U(0,0), U.ld, tau(0), A(0,0), A.ld,
                   &tmp, lwork2, &info );
    assert( info == 0 );
    lwork2 = icla_int_t( real( tmp ));
    lwork = max( lwork, lwork2 );
    Vector<FloatT> work( lwork );

    icla_generate_sigma( opts, dist, false, cond, sigma_max, A, sigma );

    if (condD != 1) {
        real_t sum_sq = blas::dot( sigma.n, sigma(0), 1, sigma(0), 1 );
        real_t scale = sqrt( sigma.n / sum_sq );
        blas::scal( sigma.n, scale, sigma(0), 1 );

        for (icla_int_t i = 0; i < sigma.n; ++i) {
            *A(i,i) = blas::traits<FloatT>::make( *sigma(i), 0 );
        }
    }

    sizeU = U.size();
    lapack::larnv( idist_randn, opts.iseed, sizeU, U(0,0) );
    for (icla_int_t j = 0; j < minmn; ++j) {
        icla_int_t mj = m - j;
        lapack::larfg( mj, U(j,j), U(j+1,j), 1, tau(j) );
    }

    lapack::unmqr( "Left", "NoTrans", A.m, A.n, minmn,
                   U(0,0), U.ld, tau(0), A(0,0), A.ld,
                   work(0), lwork, &info );
    assert( info == 0 );

    lapack::larnv( idist_randn, opts.iseed, sizeU, U(0,0) );
    for (icla_int_t j = 0; j < minmn; ++j) {
        icla_int_t nj = n - j;
        lapack::larfg( nj, U(j,j), U(j+1,j), 1, tau(j) );
    }

    lapack::unmqr( "Right", "ConjTrans", A.m, A.n, minmn,
                   U(0,0), U.ld, tau(0), A(0,0), A.ld,
                   work(0), lwork, &info );
    assert( info == 0 );

    if (condD != 1) {

        icla_generate_correlation_factor( A );

        Vector<real_t> D( A.n );
        real_t range = log( condD );
        lapack::larnv( idist_rand, opts.iseed, D.n, D(0) );
        for (icla_int_t i = 0; i < D.n; ++i) {
            D[i] = exp( D[i] * range );
        }

        if (opts.verbose) {
            printf( "D = [" );
            for (icla_int_t i = 0; i < D.n; ++i) {
                printf( " %11.8g", D[i] );
            }
            printf( " ];\n" );
        }
        for (icla_int_t j = 0; j < A.n; ++j) {
            for (icla_int_t i = 0; i < A.m; ++i) {
                *A(i,j) = *A(i,j) * D[j];
            }
        }
    }
}

template< typename FloatT >
void icla_generate_heev(
    icla_opts& opts,
    Dist dist, bool rand_sign,
    typename blas::traits<FloatT>::real_t cond,
    typename blas::traits<FloatT>::real_t condD,
    typename blas::traits<FloatT>::real_t sigma_max,
    Matrix<FloatT>& A,
    Vector< typename blas::traits<FloatT>::real_t >& sigma )
{
    typedef typename blas::traits<FloatT>::real_t real_t;

    assert( A.m == A.n );

    FloatT tmp;
    icla_int_t n = A.n;
    icla_int_t sizeU;
    icla_int_t info = 0;
    Matrix<FloatT> U( n, n );
    Vector<FloatT> tau( n );

    icla_int_t lwork = -1;
    lapack::unmqr( "Left", "NoTrans", n, n, n,
                   U(0,0), U.ld, tau(0), A(0,0), A.ld,
                   &tmp, lwork, &info );
    assert( info == 0 );
    lwork = icla_int_t( real( tmp ));
    icla_int_t lwork2 = -1;
    lapack::unmqr( "Right", "ConjTrans", n, n, n,
                   U(0,0), U.ld, tau(0), A(0,0), A.ld,
                   &tmp, lwork2, &info );
    assert( info == 0 );
    lwork2 = icla_int_t( real( tmp ));
    lwork = max( lwork, lwork2 );
    Vector<FloatT> work( lwork );

    icla_generate_sigma( opts, dist, rand_sign, cond, sigma_max, A, sigma );

    sizeU = U.size();
    lapack::larnv( idist_randn, opts.iseed, sizeU, U(0,0) );
    for (icla_int_t j = 0; j < n; ++j) {
        icla_int_t nj = n - j;
        lapack::larfg( nj, U(j,j), U(j+1,j), 1, tau(j) );
    }

    lapack::unmqr( "Left", "NoTrans", n, n, n,
                   U(0,0), U.ld, tau(0), A(0,0), A.ld,
                   work(0), lwork, &info );
    assert( info == 0 );

    lapack::unmqr( "Right", "ConjTrans", n, n, n,
                   U(0,0), U.ld, tau(0), A(0,0), A.ld,
                   work(0), lwork, &info );
    assert( info == 0 );

    for (int i = 0; i < n; ++i) {
        *A(i,i) = blas::traits<FloatT>::make( real( *A(i,i) ), 0 );
    }

    if (condD != 1) {

        Vector<real_t> D( n );
        real_t range = log( condD );
        lapack::larnv( idist_rand, opts.iseed, n, D(0) );
        for (icla_int_t i = 0; i < n; ++i) {
            D[i] = exp( D[i] * range );
        }
        for (icla_int_t j = 0; j < n; ++j) {
            for (icla_int_t i = 0; i < n; ++i) {
                *A(i,j) = *A(i,j) * D[i] * D[j];
            }
        }
    }
}

template< typename FloatT >
void icla_generate_geev(
    icla_opts& opts,
    Dist dist,
    typename blas::traits<FloatT>::real_t cond,
    typename blas::traits<FloatT>::real_t condD,
    typename blas::traits<FloatT>::real_t sigma_max,
    Matrix<FloatT>& A,
    Vector< typename blas::traits<FloatT>::real_t >& sigma )
{
    throw std::exception();

}

template< typename FloatT >
void icla_generate_geevx(
    icla_opts& opts,
    Dist dist,
    typename blas::traits<FloatT>::real_t cond,
    typename blas::traits<FloatT>::real_t condD,
    typename blas::traits<FloatT>::real_t sigma_max,
    Matrix<FloatT>& A,
    Vector< typename blas::traits<FloatT>::real_t >& sigma )
{
    throw std::exception();

}

template< typename FloatT >
void icla_generate_matrix(
    icla_opts& opts,
    Matrix<FloatT>& A,
    Vector< typename blas::traits<FloatT>::real_t >& sigma )
{
    typedef typename blas::traits<FloatT>::real_t real_t;

    const real_t nan = std::numeric_limits<real_t>::quiet_NaN();
    const real_t d_zero = ICLA_D_ZERO;
    const real_t d_one  = ICLA_D_ONE;
    const real_t ufl = std::numeric_limits< real_t >::min();

    const real_t ofl = 1 / ufl;

    const real_t eps = std::numeric_limits< real_t >::epsilon();

    const FloatT c_zero = blas::traits<FloatT>::make( 0, 0 );
    const FloatT c_one  = blas::traits<FloatT>::make( 1, 0 );

    std::string name = opts.matrix;
    real_t cond = opts.cond;
    if (cond == 0) {
        cond = 1 / sqrt( eps );
    }
    real_t condD = opts.condD;
    real_t sigma_max = 1;
    icla_int_t minmn = min( A.m, A.n );

    lapack::laset( "general", sigma.n, 1, nan, nan, sigma(0), sigma.n );

    MatrixType type = MatrixType::identity;
    if      (name == "zero"
          || name == "zeros")         { type = MatrixType::zero;      }
    else if (name == "ones")          { type = MatrixType::ones;      }
    else if (name == "identity")      { type = MatrixType::identity;  }
    else if (name == "jordan")        { type = MatrixType::jordan;    }
    else if (name == "kronecker")     { type = MatrixType::kronecker; }
    else if (begins( name, "randn" )) { type = MatrixType::randn;     }
    else if (begins( name, "rands" )) { type = MatrixType::rands;     }
    else if (begins( name, "rand"  )) { type = MatrixType::rand;      }
    else if (begins( name, "diag"  )) { type = MatrixType::diag;      }
    else if (begins( name, "svd"   )) { type = MatrixType::svd;       }
    else if (begins( name, "poev"  )
          || begins( name, "spd"   )) { type = MatrixType::poev;      }
    else if (begins( name, "heev"  )
          || begins( name, "syev"  )) { type = MatrixType::heev;      }
    else if (begins( name, "geevx" )) { type = MatrixType::geevx;     }
    else if (begins( name, "geev"  )) { type = MatrixType::geev;      }
    else {
        fprintf( stderr, "Unrecognized matrix '%s'\n", name.c_str() );
        throw std::exception();
    }

    if (A.m != A.n
        && (type == MatrixType::jordan
            || type == MatrixType::poev
            || type == MatrixType::heev
            || type == MatrixType::geev
            || type == MatrixType::geevx))
    {
        fprintf( stderr, "Eigenvalue matrix requires m == n.\n" );
        throw std::exception();
    }

    if (opts.cond != 0
        && (type == MatrixType::zero
            || type == MatrixType::identity
            || type == MatrixType::jordan
            || type == MatrixType::randn
            || type == MatrixType::rands
            || type == MatrixType::rand))
    {
        fprintf( stderr, "%sWarning: --matrix %s ignores --cond %.2e.%s\n",
                 ansi_red, name.c_str(), opts.cond, ansi_normal );
    }

    Dist dist = Dist::rand;
    if      (contains( name, "_randn"     )) { dist = Dist::randn;     }
    else if (contains( name, "_rands"     )) { dist = Dist::rands;     }
    else if (contains( name, "_rand"      )) { dist = Dist::rand;      }

    else if (contains( name, "_logrand"   )) { dist = Dist::logrand;   }
    else if (contains( name, "_arith"     )) { dist = Dist::arith;     }
    else if (contains( name, "_geo"       )) { dist = Dist::geo;       }
    else if (contains( name, "_cluster1"  )) { dist = Dist::cluster1;  }
    else if (contains( name, "_cluster0"  )) { dist = Dist::cluster0;  }
    else if (contains( name, "_rarith"    )) { dist = Dist::rarith;    }
    else if (contains( name, "_rgeo"      )) { dist = Dist::rgeo;      }
    else if (contains( name, "_rcluster1" )) { dist = Dist::rcluster1; }
    else if (contains( name, "_rcluster0" )) { dist = Dist::rcluster0; }
    else if (contains( name, "_specified" )) { dist = Dist::specified; }

    if (opts.cond != 0
        && (dist == Dist::randn
            || dist == Dist::rands
            || dist == Dist::rand)
        && type != MatrixType::kronecker)
    {
        fprintf( stderr, "%sWarning: --matrix '%s' ignores --cond %.2e; use a different distribution.%s\n",
                 ansi_red, name.c_str(), opts.cond, ansi_normal );
    }

    if (type == MatrixType::poev
        && (dist == Dist::rands
            || dist == Dist::randn))
    {
        fprintf( stderr, "%sWarning: --matrix '%s' using rands or randn "
                 "will not generate SPD matrix; use rand instead.%s\n",
                 ansi_red, name.c_str(), ansi_normal );
    }

    if      (contains( name, "_small"  )) { sigma_max = sqrt( ufl ); }
    else if (contains( name, "_large"  )) { sigma_max = sqrt( ofl ); }
    else if (contains( name, "_ufl"    )) { sigma_max = ufl; }
    else if (contains( name, "_ofl"    )) { sigma_max = ofl; }

    switch (type) {
        case MatrixType::zero:
            lapack::laset( "general", A.m, A.n, c_zero, c_zero, A(0,0), A.ld );
            lapack::laset( "general", sigma.n, 1, d_zero, d_zero, sigma(0), sigma.n );
            break;

        case MatrixType::ones:
            lapack::laset( "general", A.m, A.n, c_one, c_one, A(0,0), A.ld );
            break;

        case MatrixType::identity:
            lapack::laset( "general", A.m, A.n, c_zero, c_one, A(0,0), A.ld );
            lapack::laset( "general", sigma.n, 1, d_one, d_one, sigma(0), sigma.n );
            break;

        case MatrixType::jordan: {
            icla_int_t n1 = A.n - 1;
            lapack::laset( "upper", A.n, A.n, c_zero, c_one, A(0,0), A.ld );

            lapack::laset( "lower", n1,  n1,  c_zero, c_one, A(1,0), A.ld );

            break;
        }

        case MatrixType::kronecker: {
            FloatT diag = blas::traits<FloatT>::make( 1 + A.m / cond, 0 );
            lapack::laset( "general", A.m, A.n, c_one, diag, A(0,0), A.ld );
            break;
        }

        case MatrixType::rand:
        case MatrixType::rands:
        case MatrixType::randn: {
            icla_int_t idist = (icla_int_t) type;
            icla_int_t sizeA = A.ld * A.n;
            lapack::larnv( idist, opts.iseed, sizeA, A(0,0) );
            if (sigma_max != 1) {
                FloatT scale = blas::traits<FloatT>::make( sigma_max, 0 );
                blas::scal( sizeA, scale, A(0,0), 1 );
            }
            break;
        }

        case MatrixType::diag:
            icla_generate_sigma( opts, dist, false, cond, sigma_max, A, sigma );
            break;

        case MatrixType::svd:
            icla_generate_svd( opts, dist, cond, condD, sigma_max, A, sigma );
            break;

        case MatrixType::poev:
            icla_generate_heev( opts, dist, false, cond, condD, sigma_max, A, sigma );
            break;

        case MatrixType::heev:
            icla_generate_heev( opts, dist, true, cond, condD, sigma_max, A, sigma );
            break;

        case MatrixType::geev:
            icla_generate_geev( opts, dist, cond, condD, sigma_max, A, sigma );
            break;

        case MatrixType::geevx:
            icla_generate_geevx( opts, dist, cond, condD, sigma_max, A, sigma );
            break;
    }

    if (contains( name, "_dominant" )) {

        for (int i = 0; i < minmn; ++i) {
            real_t sum = max( blas::asum( A.m, A(0,i), 1    ),

                              blas::asum( A.n, A(i,0), A.ld ) );

            *A(i,i) = blas::traits<FloatT>::make( sum, 0 );
        }

        lapack::laset( "general", sigma.n, 1, nan, nan, sigma(0), sigma.n );
    }
}

template< typename FloatT >
void icla_generate_matrix(
    icla_opts& opts,
    icla_int_t m, icla_int_t n,
    FloatT* A_ptr, icla_int_t lda,
    typename blas::traits<FloatT>::real_t* sigma_ptr
 )
{
    typedef typename blas::traits<FloatT>::real_t real_t;

    Vector<real_t> sigma( sigma_ptr, min(m,n) );
    if (sigma_ptr == nullptr) {
        sigma = Vector<real_t>( min(m,n) );
    }
    Matrix<FloatT> A( A_ptr, m, n, lda );
    icla_generate_matrix( opts, A, sigma );
}

template
void icla_generate_matrix(
    icla_opts& opts,
    icla_int_t m, icla_int_t n,
    float* A_ptr, icla_int_t lda,
    float* sigma_ptr );

template
void icla_generate_matrix(
    icla_opts& opts,
    icla_int_t m, icla_int_t n,
    double* A_ptr, icla_int_t lda,
    double* sigma_ptr );

template
void icla_generate_matrix(
    icla_opts& opts,
    icla_int_t m, icla_int_t n,
    iclaFloatComplex* A_ptr, icla_int_t lda,
    float* sigma_ptr );

template
void icla_generate_matrix(
    icla_opts& opts,
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex* A_ptr, icla_int_t lda,
    double* sigma_ptr );
