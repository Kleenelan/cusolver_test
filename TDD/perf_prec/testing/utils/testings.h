#ifndef TESTINGS_H
#define TESTINGS_H

#include <stdio.h>
#include <stdlib.h>

#if ! defined(ICLA_H) && ! defined(ICLA_V2_H)
#include "icla_v2.h"
#endif

// ignore/replace some cuBLAS calls
#ifdef ICLA_HAVE_HIP
#define cublasSetAtomicsMode(...)

#endif


#include <vector>
#include <string>
#include <cmath>


// special region to replace certain functions
#ifdef ICLA_HAVE_HIP
#define cublas_trans_const hipblas_trans_const
#define cublas_diag_const hipblas_diag_const
#define cublas_uplo_const hipblas_uplo_const
#define cublas_side_const hipblas_side_const
#endif

#include "icla_lapack.h"
#include "icla_lapack.hpp"  // C++ bindings; need traits
#include "icla_matrix.hpp"  // experimental Matrix and Vector classes
#include "testing_s.h"
#include "testing_d.h"
#include "testing_c.h"
#include "testing_z.h"

/***************************************************************************//**
 *  For portability to Windows
 */
#if defined( _WIN32 ) || defined( _WIN64 )
    // functions where Microsoft fails to provide C99 or POSIX standard
    // (only with Microsoft, not with nvcc on Windows)
    // in both icla_internal.h and testings.h
    #ifndef __NVCC__
        // note _snprintf has slightly different semantics than snprintf
        #define snprintf      _snprintf
        #define unlink        _unlink
    #endif
#endif


#ifdef __cplusplus
extern "C" {
#endif

void flops_init();


/***************************************************************************//**
    max that propogates nan consistently:
    max_nan( 1,   nan ) = nan
    max_nan( nan, 1   ) = nan

    isnan and isinf are hard to call portably from both C and C++.
    In Windows C,     include float.h, use _isnan as above (before VS 2015)
    In Unix C or C++, include math.h,  use isnan
    In std C++,       include cmath,   use std::isnan
    Sometimes in C++, include cmath,   use isnan is okay (on Linux but not MacOS)
    This makes writing a header inline function a nightmare. For now, do it
    here in testing to avoid potential issues in the ICLA library itself.
*******************************************************************************/
static inline double icla_max_nan( double x, double y )
{
    #ifdef isnan
        // with include <math.h> macro
        return (isnan(y) || x < y ? y : x);
    #else
        // with include <cmath> function
        return (std::isnan(y) || x < y ? y : x);
    #endif
}


/***************************************************************************//**
 *  Global utilities
 *  in both icla_internal.h and testings.h
 **/
#ifndef max__
#define max__(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef min__
#define min__(a, b) ((a) < (b) ? (a) : (b))
#endif

// suppress "warning: unused variable" in a portable fashion
#define ICLA_UNUSED(var)  ((void)var)

/***************************************************************************//**
 * Macros to handle error checking.
 */

#define TESTING_CHECK( err )                                                 \
    do {                                                                     \
        icla_int_t err_ = (err);                                            \
        if ( err_ != 0 ) {                                                   \
            fprintf( stderr, "Error: %s\nfailed at %s:%d: error %lld: %s\n", \
                     #err, __FILE__, __LINE__,                               \
                     (long long) err_, icla_strerror(err_) );               \
            exit(1);                                                         \
        }                                                                    \
    } while( 0 )


/***************************************************************************//**
 * Functions and data structures used for testing.
 */

void icla_assert( bool condition, const char* msg, ... );

void icla_assert_warn( bool condition, const char* msg, ... );

void icla_flush_cache( size_t cache_size );

#ifdef __cplusplus
}
#endif

/***************************************************************************//**
 * C++ Functions and data structures used for testing.
 */

// for sorting in descending order (e.g., singular values)
template< typename T >
bool greater( T a, T b )
{
    return (a > b);
}

#define MAX_NTEST 1050

typedef enum {
    IclaOptsDefault = 0,
    IclaOptsBatched = 1000
} icla_opts_t;

typedef enum {
    IclaSVD_all,
    IclaSVD_query,
    IclaSVD_doc,
    IclaSVD_doc_old,
    IclaSVD_min,
    IclaSVD_min_1,
    IclaSVD_min_old,
    IclaSVD_min_old_1,
    IclaSVD_min_fast,
    IclaSVD_min_fast_1,
    IclaSVD_opt,
    IclaSVD_opt_old,
    IclaSVD_opt_slow,
    IclaSVD_max
} icla_svd_work_t;

class icla_opts
{
public:
    // constructor
    icla_opts( icla_opts_t flag=IclaOptsDefault );

    // parse command line
    void parse_opts( int argc, char** argv );

    // set range, vl, vu, il, iu for eigen/singular value problems (gesvdx, syevdx, ...)
    void get_range( icla_int_t n, icla_range_t* range,
                    double* vl, double* vu,
                    icla_int_t* il, icla_int_t* iu );

    void get_range( icla_int_t n, icla_range_t* range,
                    float* vl, float* vu,
                    icla_int_t* il, icla_int_t* iu );

    // deallocate queues, etc.
    void cleanup();

    // matrix size
    icla_int_t ntest;
    icla_int_t msize[ MAX_NTEST ];
    icla_int_t nsize[ MAX_NTEST ];
    icla_int_t ksize[ MAX_NTEST ];
    icla_int_t batchcount;

    // lower/upper bandwidths
    icla_int_t kl;
    icla_int_t ku;

    icla_int_t default_nstart;
    icla_int_t default_nend;
    icla_int_t default_nstep;

    // scalars
    icla_int_t device;
    icla_int_t cache;
    icla_int_t align;
    icla_int_t nb;
    icla_int_t nrhs;
    icla_int_t nqueue;
    icla_int_t ngpu;
    icla_int_t nsub;
    icla_int_t niter;
    icla_int_t nthread;
    icla_int_t offset;
    icla_int_t itype;     // hegvd: problem type
    icla_int_t version;
    icla_int_t check;
    icla_int_t verbose;

    // ranges for eigen/singular values (gesvdx, heevdx, ...)
    double      fraction_lo;
    double      fraction_up;
    icla_int_t irange_lo;
    icla_int_t irange_up;
    double      vrange_lo;
    double      vrange_up;

    double      tolerance;

    // boolean arguments
    bool icla;
    bool lapack;
    bool warmup;

    // lapack options
    icla_uplo_t    uplo;
    icla_trans_t   transA;
    icla_trans_t   transB;
    icla_side_t    side;
    icla_diag_t    diag;
    icla_vec_t     jobz;    // heev:   no eigen vectors
    icla_vec_t     jobvr;   // geev:   no right eigen vectors
    icla_vec_t     jobvl;   // geev:   no left  eigen vectors

    // vectors of options
    std::vector< icla_svd_work_t > svd_work;
    std::vector< icla_vec_t > jobu;
    std::vector< icla_vec_t > jobv;

    // LAPACK test matrix generation
    std::string matrix;
    double      cond;
    double      condD;
    icla_int_t iseed[4];

    // queue for default device
    icla_queue_t   queue;
    icla_queue_t   queues2[3];  // 2 queues + 1 extra NULL entry to catch errors

    #ifdef ICLA_HAVE_CUDA
    // handle for directly calling cublas
    cublasHandle_t  handle;
    #elif defined(ICLA_HAVE_HIP)
    hipblasHandle_t handle;
    #endif
};

extern const char* g_platform_str;

// -----------------------------------------------------------------------------
template< typename FloatT >
void icla_generate_matrix(
    icla_opts& opts,
    Matrix< FloatT >& A,
    Vector< typename blas::traits<FloatT>::real_t >& sigma );

template< typename FloatT >
void icla_generate_matrix(
    icla_opts& opts,
    icla_int_t m, icla_int_t n,
    FloatT* A, icla_int_t lda,
    typename blas::traits<FloatT>::real_t* sigma=nullptr );

#endif /* TESTINGS_H */
