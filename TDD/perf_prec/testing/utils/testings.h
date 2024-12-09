#ifndef TESTINGS_H
#define TESTINGS_H

#include <stdio.h>
#include <stdlib.h>

#if ! defined(ICLA_H) && ! defined(ICLA_V2_H)
#include "icla_v2.h"
#endif

#ifdef ICLA_HAVE_HIP
#define cublasSetAtomicsMode(...)

#endif

#include <vector>
#include <string>
#include <cmath>

#ifdef ICLA_HAVE_HIP
#define cublas_trans_const hipblas_trans_const
#define cublas_diag_const hipblas_diag_const
#define cublas_uplo_const hipblas_uplo_const
#define cublas_side_const hipblas_side_const
#endif

#include "icla_lapack.h"
#include "icla_lapack.hpp"

#include "icla_matrix.hpp"

#include "testing_s.h"
#include "testing_d.h"
#include "testing_c.h"
#include "testing_z.h"

#if defined( _WIN32 ) || defined( _WIN64 )

    #ifndef __NVCC__

        #define snprintf      _snprintf
        #define unlink        _unlink
    #endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

void flops_init();

static inline double icla_max_nan( double x, double y )
{
    #ifdef isnan

        return (isnan(y) || x < y ? y : x);
    #else

        return (std::isnan(y) || x < y ? y : x);
    #endif
}

#ifndef max__
#define max__(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef min__
#define min__(a, b) ((a) < (b) ? (a) : (b))
#endif

#define ICLA_UNUSED(var)  ((void)var)

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

void icla_assert( bool condition, const char* msg, ... );

void icla_assert_warn( bool condition, const char* msg, ... );

void icla_flush_cache( size_t cache_size );

#ifdef __cplusplus
}
#endif

template< typename T >
bool greater( T a, T b )
{
    return (a > b);
}

#define MAX_NTEST 1050

typedef enum {
    iclaOptsDefault = 0,
    iclaOptsBatched = 1000
} icla_opts_t;

typedef enum {
    iclaSVD_all,
    iclaSVD_query,
    iclaSVD_doc,
    iclaSVD_doc_old,
    iclaSVD_min,
    iclaSVD_min_1,
    iclaSVD_min_old,
    iclaSVD_min_old_1,
    iclaSVD_min_fast,
    iclaSVD_min_fast_1,
    iclaSVD_opt,
    iclaSVD_opt_old,
    iclaSVD_opt_slow,
    iclaSVD_max
} icla_svd_work_t;

class icla_opts
{
public:

    icla_opts( icla_opts_t flag=iclaOptsDefault );

    void parse_opts( int argc, char** argv );

    void get_range( icla_int_t n, icla_range_t* range,
                    double* vl, double* vu,
                    icla_int_t* il, icla_int_t* iu );

    void get_range( icla_int_t n, icla_range_t* range,
                    float* vl, float* vu,
                    icla_int_t* il, icla_int_t* iu );

    void cleanup();

    icla_int_t ntest;
    icla_int_t msize[ MAX_NTEST ];
    icla_int_t nsize[ MAX_NTEST ];
    icla_int_t ksize[ MAX_NTEST ];
    icla_int_t batchcount;

    icla_int_t kl;
    icla_int_t ku;

    icla_int_t default_nstart;
    icla_int_t default_nend;
    icla_int_t default_nstep;

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
    icla_int_t itype;

    icla_int_t version;
    icla_int_t check;
    icla_int_t verbose;

    double      fraction_lo;
    double      fraction_up;
    icla_int_t irange_lo;
    icla_int_t irange_up;
    double      vrange_lo;
    double      vrange_up;

    double      tolerance;

    bool icla;
    bool lapack;
    bool warmup;

    icla_uplo_t    uplo;
    icla_trans_t   transA;
    icla_trans_t   transB;
    icla_side_t    side;
    icla_diag_t    diag;
    icla_vec_t     jobz;

    icla_vec_t     jobvr;

    icla_vec_t     jobvl;

    std::vector< icla_svd_work_t > svd_work;
    std::vector< icla_vec_t > jobu;
    std::vector< icla_vec_t > jobv;

    std::string matrix;
    double      cond;
    double      condD;
    icla_int_t iseed[4];

    icla_queue_t   queue;
    icla_queue_t   queues2[3];

    #ifdef ICLA_HAVE_CUDA

    cublasHandle_t  handle;
    #elif defined(ICLA_HAVE_HIP)
    hipblasHandle_t handle;
    #endif
};

extern const char* g_platform_str;

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

#endif

