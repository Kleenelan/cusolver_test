

#ifndef ICLA_TYPES_H
#define ICLA_TYPES_H


#include "icla_config.h"


#include <stdint.h>
#include <assert.h>



#ifdef HAVE_clAmdBlas
#define ICLA_HAVE_OPENCL
#endif



#if ! defined(ICLA_HAVE_CUDA) && ! defined(ICLA_HAVE_OPENCL) && ! defined(HAVE_MIC) && ! defined(ICLA_HAVE_HIP)


#define ICLA_HAVE_CUDA
#endif





#if __STDC_VERSION__ < 199901L
  #ifndef __func__
    #if __GNUC__ >= 2 || _MSC_VER >= 1300
      #define __func__ __FUNCTION__
    #else
      #define __func__ "<unknown>"
    #endif
  #endif
#endif





#if defined(ICLA_ILP64) || defined(MKL_ILP64)
typedef long long int icla_int_t;
#else
typedef int icla_int_t;
#endif

typedef int icla_index_t;
typedef unsigned int icla_uindex_t;


typedef double real_Double_t;






#if defined(ICLA_HAVE_CUDA)

    #ifndef CUBLAS_H_
    #include <cuda.h>
    #include <cublas_v2.h>
    #endif

    #include <cusparse_v2.h>
    #include <cusolverDn.h>

    #ifdef __cplusplus
    extern "C" {
    #endif


    struct icla_queue;
    typedef struct icla_queue* icla_queue_t;
    typedef cudaEvent_t    icla_event_t;
    typedef icla_int_t    icla_device_t;


    #if defined(__cplusplus) && CUDA_VERSION >= 7500
    #include <cuda_fp16.h>
    typedef __half           iclaHalf;
    #else


    typedef short            iclaHalf;
    #endif

    typedef cuDoubleComplex iclaDoubleComplex;
    typedef cuFloatComplex  iclaFloatComplex;

    cudaStream_t     icla_queue_get_cuda_stream    ( icla_queue_t queue );
    cublasHandle_t   icla_queue_get_cublas_handle  ( icla_queue_t queue );
    cusparseHandle_t icla_queue_get_cusparse_handle( icla_queue_t queue );




    #define ICLA_Z_MAKE(r,i)     make_cuDoubleComplex(r, i)
    #define ICLA_Z_REAL(a)       (a).x
    #define ICLA_Z_IMAG(a)       (a).y
    #define ICLA_Z_ADD(a, b)     cuCadd(a, b)
    #define ICLA_Z_SUB(a, b)     cuCsub(a, b)
    #define ICLA_Z_MUL(a, b)     cuCmul(a, b)
    #define ICLA_Z_DIV(a, b)     cuCdiv(a, b)
    #define ICLA_Z_ABS(a)        cuCabs(a)
    #define ICLA_Z_ABS1(a)       (fabs((a).x) + fabs((a).y))
    #define ICLA_Z_CONJ(a)       cuConj(a)

    #define ICLA_C_MAKE(r,i)     make_cuFloatComplex(r, i)
    #define ICLA_C_REAL(a)       (a).x
    #define ICLA_C_IMAG(a)       (a).y
    #define ICLA_C_ADD(a, b)     cuCaddf(a, b)
    #define ICLA_C_SUB(a, b)     cuCsubf(a, b)
    #define ICLA_C_MUL(a, b)     cuCmulf(a, b)
    #define ICLA_C_DIV(a, b)     cuCdivf(a, b)
    #define ICLA_C_ABS(a)        cuCabsf(a)
    #define ICLA_C_ABS1(a)       (fabsf((a).x) + fabsf((a).y))
    #define ICLA_C_CONJ(a)       cuConjf(a)

    #define iclaCfma cuCfma
    #define iclaCfmaf cuCfmaf




    #ifdef __cplusplus
    }
    #endif
#elif defined(ICLA_HAVE_HIP)


    #if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVCC)
      #define __HIP_PLATFORM_AMD__
    #endif

    #include <hip/hip_version.h>
    #include <hip/hip_runtime.h>


    #if HIP_VERSION >= 50200000
    #include <hipblas/hipblas.h>
    #include <hipsparse/hipsparse.h>
    #else
    #include <hipblas.h>
    #include <hipsparse.h>
    #endif



    #ifndef icla_unsupported
    #define icla_unsupported(fname) ((hipblasStatus_t)(fprintf(stderr, "ICLA: Unsupported function '" #fname "'\n"), HIPBLAS_STATUS_NOT_SUPPORTED))
    #endif

    #include <hip/hip_fp16.h>

    #ifdef __cplusplus
    extern "C" {
    #endif


    struct icla_queue;
    typedef struct icla_queue* icla_queue_t;
    typedef hipEvent_t  icla_event_t;
    typedef icla_int_t icla_device_t;

    #ifdef __cplusplus
    typedef __half           iclaHalf;
    #else

    typedef short            iclaHalf;
    #endif

    hipStream_t       icla_queue_get_hip_stream      ( icla_queue_t queue );
    hipblasHandle_t   icla_queue_get_hipblas_handle  ( icla_queue_t queue );
    hipsparseHandle_t icla_queue_get_hipsparse_handle( icla_queue_t queue );








    typedef struct {


        double x, y;

    } iclaDoubleComplex;



    #define ICLA_Z_MAKE(r, i)   ((iclaDoubleComplex){(double)(r), (double)(i)})
    #define ICLA_Z_REAL(a) (a).x
    #define ICLA_Z_IMAG(a) (a).y
    #define ICLA_Z_ADD(a, b) iclaCadd(a, b)
    #define ICLA_Z_SUB(a, b) iclaCsub(a, b)
    #define ICLA_Z_MUL(a, b) iclaCmul(a, b)
    #define ICLA_Z_DIV(a, b) iclaCdiv(a, b)
    #define ICLA_Z_ABS(a) (hypot(ICLA_Z_REAL(a), ICLA_Z_IMAG(a)))
    #define ICLA_Z_ABS1(a) (fabs(ICLA_Z_REAL(a)) + fabs(ICLA_Z_IMAG(a)))
    #define ICLA_Z_CONJ(a) iclaConj(a)



    __host__ __device__ static inline iclaDoubleComplex iclaCadd(iclaDoubleComplex a, iclaDoubleComplex b) {
        return ICLA_Z_MAKE(a.x+b.x, a.y+b.y);
    }
    __host__ __device__ static inline iclaDoubleComplex iclaCsub(iclaDoubleComplex a, iclaDoubleComplex b) {
        return ICLA_Z_MAKE(a.x-b.x, a.y-b.y);
    }
    __host__ __device__ static inline iclaDoubleComplex iclaCmul(iclaDoubleComplex a, iclaDoubleComplex b) {
        return ICLA_Z_MAKE(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
    }
    __host__ __device__ static inline iclaDoubleComplex iclaCdiv(iclaDoubleComplex a, iclaDoubleComplex b) {
        double sqabs = b.x*b.x + b.y*b.y;
        return ICLA_Z_MAKE(
            (a.x * b.x + a.y * b.y) / sqabs,
            (a.y * b.x - a.x * b.y) / sqabs
        );
    }
    __host__ __device__ static inline iclaDoubleComplex iclaConj(iclaDoubleComplex a) {
        return ICLA_Z_MAKE(a.x, -a.y);
    }
    __host__ __device__ static inline iclaDoubleComplex iclaCfma(iclaDoubleComplex a, iclaDoubleComplex b, iclaDoubleComplex c) {
        return iclaCadd(iclaCmul(a, b), c);
    }









    typedef struct {


        float x, y;

    } iclaFloatComplex;



    #define ICLA_C_MAKE(r, i)   ((iclaFloatComplex){(float)(r), (float)(i)})
    #define ICLA_C_REAL(a) (a).x
    #define ICLA_C_IMAG(a) (a).y
    #define ICLA_C_ADD(a, b) iclaCaddf(a, b)
    #define ICLA_C_SUB(a, b) iclaCsubf(a, b)
    #define ICLA_C_MUL(a, b) iclaCmulf(a, b)
    #define ICLA_C_DIV(a, b) iclaCdivf(a, b)
    #define ICLA_C_ABS(a) (hypotf(ICLA_C_REAL(a), ICLA_C_IMAG(a)))
    #define ICLA_C_ABS1(a) (fabsf(ICLA_C_REAL(a)) + fabs(ICLA_C_IMAG(a)))
    #define ICLA_C_CONJ(a) iclaConjf(a)



    __host__ __device__ static inline iclaFloatComplex iclaCaddf(iclaFloatComplex a, iclaFloatComplex b) {
        return ICLA_C_MAKE(a.x+b.x, a.y+b.y);
    }
    __host__ __device__ static inline iclaFloatComplex iclaCsubf(iclaFloatComplex a, iclaFloatComplex b) {
        return ICLA_C_MAKE(a.x-b.x, a.y-b.y);
    }
    __host__ __device__ static inline iclaFloatComplex iclaCmulf(iclaFloatComplex a, iclaFloatComplex b) {
        return ICLA_C_MAKE(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
    }
    __host__ __device__ static inline iclaFloatComplex iclaCdivf(iclaFloatComplex a, iclaFloatComplex b) {
        float sqabs = b.x*b.x + b.y*b.y;
        return ICLA_C_MAKE(
            (a.x * b.x + a.y * b.y) / sqabs,
            (a.y * b.x - a.x * b.y) / sqabs
        );
    }
    __host__ __device__ static inline iclaFloatComplex iclaConjf(iclaFloatComplex a) {
        return ICLA_C_MAKE(a.x, -a.y);
    }
    __host__ __device__ static inline iclaFloatComplex iclaCfmaf(iclaFloatComplex a, iclaFloatComplex b, iclaFloatComplex c) {
        return iclaCaddf(iclaCmulf(a, b), c);
    }


    #ifdef __cplusplus
    }
    #endif

#elif defined(ICLA_HAVE_OPENCL)
    #include <clBLAS.h>

    #ifdef __cplusplus
    extern "C" {
    #endif

    typedef cl_command_queue  icla_queue_t;
    typedef cl_event          icla_event_t;
    typedef cl_device_id      icla_device_t;

    typedef short         iclaHalf;
    typedef DoubleComplex iclaDoubleComplex;
    typedef FloatComplex  iclaFloatComplex;

    cl_command_queue icla_queue_get_cl_queue( icla_queue_t queue );

    #define ICLA_Z_MAKE(r,i)     doubleComplex(r,i)
    #define ICLA_Z_REAL(a)       (a).s[0]
    #define ICLA_Z_IMAG(a)       (a).s[1]
    #define ICLA_Z_ADD(a, b)     ICLA_Z_MAKE((a).s[0] + (b).s[0], (a).s[1] + (b).s[1])
    #define ICLA_Z_SUB(a, b)     ICLA_Z_MAKE((a).s[0] - (b).s[0], (a).s[1] - (b).s[1])
    #define ICLA_Z_MUL(a, b)     ((a) * (b))
    #define ICLA_Z_DIV(a, b)     ((a) / (b))
    #define ICLA_Z_ABS(a)        icla_cabs(a)
    #define ICLA_Z_ABS1(a)       (fabs((a).s[0]) + fabs((a).s[1]))
    #define ICLA_Z_CONJ(a)       ICLA_Z_MAKE((a).s[0], -(a).s[1])

    #define ICLA_C_MAKE(r,i)     floatComplex(r,i)
    #define ICLA_C_REAL(a)       (a).s[0]
    #define ICLA_C_IMAG(a)       (a).s[1]
    #define ICLA_C_ADD(a, b)     ICLA_C_MAKE((a).s[0] + (b).s[0], (a).s[1] + (b).s[1])
    #define ICLA_C_SUB(a, b)     ICLA_C_MAKE((a).s[0] - (b).s[0], (a).s[1] - (b).s[1])
    #define ICLA_C_MUL(a, b)     ((a) * (b))
    #define ICLA_C_DIV(a, b)     ((a) / (b))
    #define ICLA_C_ABS(a)        icla_cabsf(a)
    #define ICLA_C_ABS1(a)       (fabsf((a).s[0]) + fabsf((a).s[1]))
    #define ICLA_C_CONJ(a)       ICLA_C_MAKE((a).s[0], -(a).s[1])

    #ifdef __cplusplus
    }
    #endif
#elif defined(HAVE_MIC)
    #include <complex>

    #ifdef __cplusplus
    extern "C" {
    #endif

    typedef int   icla_queue_t;
    typedef int   icla_event_t;
    typedef int   icla_device_t;

    typedef short                 iclaHalf;
    typedef std::complex<float>   iclaFloatComplex;
    typedef std::complex<double>  iclaDoubleComplex;

    #define ICLA_Z_MAKE(r, i)    std::complex<double>(r,i)
    #define ICLA_Z_REAL(x)       (x).real()
    #define ICLA_Z_IMAG(x)       (x).imag()
    #define ICLA_Z_ADD(a, b)     ((a)+(b))
    #define ICLA_Z_SUB(a, b)     ((a)-(b))
    #define ICLA_Z_MUL(a, b)     ((a)*(b))
    #define ICLA_Z_DIV(a, b)     ((a)/(b))
    #define ICLA_Z_ABS(a)        abs(a)
    #define ICLA_Z_ABS1(a)       (fabs((a).real()) + fabs((a).imag()))
    #define ICLA_Z_CONJ(a)       conj(a)

    #define ICLA_C_MAKE(r, i)    std::complex<float> (r,i)
    #define ICLA_C_REAL(x)       (x).real()
    #define ICLA_C_IMAG(x)       (x).imag()
    #define ICLA_C_ADD(a, b)     ((a)+(b))
    #define ICLA_C_SUB(a, b)     ((a)-(b))
    #define ICLA_C_MUL(a, b)     ((a)*(b))
    #define ICLA_C_DIV(a, b)     ((a)/(b))
    #define ICLA_C_ABS(a)        abs(a)
    #define ICLA_C_ABS1(a)       (fabs((a).real()) + fabs((a).imag()))
    #define ICLA_C_CONJ(a)       conj(a)

    #ifdef __cplusplus
    }
    #endif
#else
    #error "One of ICLA_HAVE_CUDA, ICLA_HAVE_HIP, ICLA_HAVE_OPENCL, or HAVE_MIC must be defined. For example, add -DICLA_HAVE_CUDA to CFLAGS, or #define ICLA_HAVE_CUDA before #include <icla.h>. In ICLA, this happens in Makefile."
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define ICLA_Z_EQUAL(a,b)        (ICLA_Z_REAL(a)==ICLA_Z_REAL(b) && ICLA_Z_IMAG(a)==ICLA_Z_IMAG(b))
#define ICLA_Z_NEGATE(a)         ICLA_Z_MAKE( -ICLA_Z_REAL(a), -ICLA_Z_IMAG(a))

#define ICLA_C_EQUAL(a,b)        (ICLA_C_REAL(a)==ICLA_C_REAL(b) && ICLA_C_IMAG(a)==ICLA_C_IMAG(b))
#define ICLA_C_NEGATE(a)         ICLA_C_MAKE( -ICLA_C_REAL(a), -ICLA_C_IMAG(a))

#define ICLA_D_MAKE(r,i)         (r)
#define ICLA_D_REAL(x)           (x)
#define ICLA_D_IMAG(x)           (0.0)
#define ICLA_D_ADD(a, b)         ((a) + (b))
#define ICLA_D_SUB(a, b)         ((a) - (b))
#define ICLA_D_MUL(a, b)         ((a) * (b))
#define ICLA_D_DIV(a, b)         ((a) / (b))
#define ICLA_D_ABS(a)            ((a)>0 ? (a) : -(a))
#define ICLA_D_ABS1(a)           ((a)>0 ? (a) : -(a))
#define ICLA_D_CONJ(a)           (a)
#define ICLA_D_EQUAL(a,b)        ((a) == (b))
#define ICLA_D_NEGATE(a)         (-a)

#define ICLA_S_MAKE(r,i)         (r)
#define ICLA_S_REAL(x)           (x)
#define ICLA_S_IMAG(x)           (0.0)
#define ICLA_S_ADD(a, b)         ((a) + (b))
#define ICLA_S_SUB(a, b)         ((a) - (b))
#define ICLA_S_MUL(a, b)         ((a) * (b))
#define ICLA_S_DIV(a, b)         ((a) / (b))
#define ICLA_S_ABS(a)            ((a)>0 ? (a) : -(a))
#define ICLA_S_ABS1(a)           ((a)>0 ? (a) : -(a))
#define ICLA_S_CONJ(a)           (a)
#define ICLA_S_EQUAL(a,b)        ((a) == (b))
#define ICLA_S_NEGATE(a)         (-a)

#define ICLA_Z_ZERO              ICLA_Z_MAKE( 0.0, 0.0)
#define ICLA_Z_ONE               ICLA_Z_MAKE( 1.0, 0.0)
#define ICLA_Z_HALF              ICLA_Z_MAKE( 0.5, 0.0)
#define ICLA_Z_NEG_ONE           ICLA_Z_MAKE(-1.0, 0.0)
#define ICLA_Z_NEG_HALF          ICLA_Z_MAKE(-0.5, 0.0)

#define ICLA_C_ZERO              ICLA_C_MAKE( 0.0, 0.0)
#define ICLA_C_ONE               ICLA_C_MAKE( 1.0, 0.0)
#define ICLA_C_HALF              ICLA_C_MAKE( 0.5, 0.0)
#define ICLA_C_NEG_ONE           ICLA_C_MAKE(-1.0, 0.0)
#define ICLA_C_NEG_HALF          ICLA_C_MAKE(-0.5, 0.0)

#define ICLA_D_ZERO              ( 0.0)
#define ICLA_D_ONE               ( 1.0)
#define ICLA_D_HALF              ( 0.5)
#define ICLA_D_NEG_ONE           (-1.0)
#define ICLA_D_NEG_HALF          (-0.5)

#define ICLA_S_ZERO              ( 0.0)
#define ICLA_S_ONE               ( 1.0)
#define ICLA_S_HALF              ( 0.5)
#define ICLA_S_NEG_ONE           (-1.0)
#define ICLA_S_NEG_HALF          (-0.5)

#ifndef CBLAS_SADDR
#define CBLAS_SADDR(a)  &(a)
#endif


double icla_cabs ( iclaDoubleComplex x );
float  icla_cabsf( iclaFloatComplex  x );

#if defined(ICLA_HAVE_OPENCL)

    typedef cl_mem icla_ptr;
    typedef cl_mem iclaInt_ptr;
    typedef cl_mem iclaIndex_ptr;
    typedef cl_mem iclaFloat_ptr;
    typedef cl_mem iclaDouble_ptr;
    typedef cl_mem iclaFloatComplex_ptr;
    typedef cl_mem iclaDoubleComplex_ptr;

    typedef cl_mem icla_const_ptr;
    typedef cl_mem iclaInt_const_ptr;
    typedef cl_mem iclaIndex_const_ptr;
    typedef cl_mem iclaFloat_const_ptr;
    typedef cl_mem iclaDouble_const_ptr;
    typedef cl_mem iclaFloatComplex_const_ptr;
    typedef cl_mem iclaDoubleComplex_const_ptr;
#else

    typedef void               *icla_ptr;
    typedef icla_int_t        *iclaInt_ptr;
    typedef icla_index_t      *iclaIndex_ptr;
    typedef icla_uindex_t     *iclaUIndex_ptr;
    typedef float              *iclaFloat_ptr;
    typedef double             *iclaDouble_ptr;
    typedef iclaFloatComplex  *iclaFloatComplex_ptr;
    typedef iclaDoubleComplex *iclaDoubleComplex_ptr;
    typedef iclaHalf          *iclaHalf_ptr;

    typedef void               const *icla_const_ptr;
    typedef icla_int_t        const *iclaInt_const_ptr;
    typedef icla_index_t      const *iclaIndex_const_ptr;
    typedef icla_uindex_t     const *iclaUIndex_const_ptr;
    typedef float              const *iclaFloat_const_ptr;
    typedef double             const *iclaDouble_const_ptr;
    typedef iclaFloatComplex  const *iclaFloatComplex_const_ptr;
    typedef iclaDoubleComplex const *iclaDoubleComplex_const_ptr;
    typedef iclaHalf          const *iclaHalf_const_ptr;
#endif






#define ICLA_VERSION_MAJOR 2
#define ICLA_VERSION_MINOR 8
#define ICLA_VERSION_MICRO 0


#define ICLA_VERSION_STAGE "svn"

#define IclaMaxGPUs 8
#define IclaMaxAccelerators 8
#define IclaMaxSubs 16


#define IclaBigTileSize 1000000









#define ICLA_SUCCESS               0
#define ICLA_ERR                  -100
#define ICLA_ERR_NOT_INITIALIZED  -101
#define ICLA_ERR_REINITIALIZED    -102
#define ICLA_ERR_NOT_SUPPORTED    -103
#define ICLA_ERR_ILLEGAL_VALUE    -104
#define ICLA_ERR_NOT_FOUND        -105
#define ICLA_ERR_ALLOCATION       -106
#define ICLA_ERR_INTERNAL_LIMIT   -107
#define ICLA_ERR_UNALLOCATED      -108
#define ICLA_ERR_FILESYSTEM       -109
#define ICLA_ERR_UNEXPECTED       -110
#define ICLA_ERR_SEQUENCE_FLUSHED -111
#define ICLA_ERR_HOST_ALLOC       -112
#define ICLA_ERR_DEVICE_ALLOC     -113
#define ICLA_ERR_CUDASTREAM       -114
#define ICLA_ERR_INVALID_PTR      -115
#define ICLA_ERR_UNKNOWN          -116
#define ICLA_ERR_NOT_IMPLEMENTED  -117
#define ICLA_ERR_NAN              -118


#define ICLA_SLOW_CONVERGENCE     -201
#define ICLA_DIVERGENCE           -202
#define ICLA_NONSPD               -203
#define ICLA_ERR_BADPRECOND       -204
#define ICLA_NOTCONVERGED         -205




#define ICLA_ERR_CUSPARSE                            -3000
#define ICLA_ERR_CUSPARSE_NOT_INITIALIZED            -3001
#define ICLA_ERR_CUSPARSE_ALLOC_FAILED               -3002
#define ICLA_ERR_CUSPARSE_INVALID_VALUE              -3003
#define ICLA_ERR_CUSPARSE_ARCH_MISMATCH              -3004
#define ICLA_ERR_CUSPARSE_MAPPING_ERROR              -3005
#define ICLA_ERR_CUSPARSE_EXECUTION_FAILED           -3006
#define ICLA_ERR_CUSPARSE_INTERNAL_ERROR             -3007
#define ICLA_ERR_CUSPARSE_MATRIX_TYPE_NOT_SUPPORTED  -3008
#define ICLA_ERR_CUSPARSE_ZERO_PIVOT                 -3009










typedef enum {
    IclaFalse         = 0,
    IclaTrue          = 1
} icla_bool_t;

typedef enum {
    IclaRowMajor      = 101,
    IclaColMajor      = 102
} icla_order_t;



typedef enum {
    IclaNoTrans       = 111,
    IclaTrans         = 112,
    IclaConjTrans     = 113,
    Icla_ConjTrans    = IclaConjTrans
} icla_trans_t;

typedef enum {
    IclaUpper         = 121,
    IclaLower         = 122,
    IclaFull          = 123,

    IclaHessenberg    = 124

} icla_uplo_t;

typedef icla_uplo_t icla_type_t;


typedef enum {
    IclaNonUnit       = 131,
    IclaUnit          = 132
} icla_diag_t;

typedef enum {
    IclaLeft          = 141,
    IclaRight         = 142,
    IclaBothSides     = 143

} icla_side_t;

typedef enum {
    IclaOneNorm       = 171,

    IclaRealOneNorm   = 172,
    IclaTwoNorm       = 173,
    IclaFrobeniusNorm = 174,
    IclaInfNorm       = 175,
    IclaRealInfNorm   = 176,
    IclaMaxNorm       = 177,
    IclaRealMaxNorm   = 178
} icla_norm_t;

typedef enum {
    IclaDistUniform   = 201,

    IclaDistSymmetric = 202,
    IclaDistNormal    = 203
} icla_dist_t;

typedef enum {
    IclaHermGeev      = 241,

    IclaHermPoev      = 242,
    IclaNonsymPosv    = 243,
    IclaSymPosv       = 244
} icla_sym_t;

typedef enum {
    IclaNoPacking     = 291,

    IclaPackSubdiag   = 292,
    IclaPackSupdiag   = 293,
    IclaPackColumn    = 294,
    IclaPackRow       = 295,
    IclaPackLowerBand = 296,
    IclaPackUpeprBand = 297,
    IclaPackAll       = 298
} icla_pack_t;

typedef enum {
    IclaNoVec         = 301,

    IclaVec           = 302,

    IclaIVec          = 303,

    IclaAllVec        = 304,

    IclaSomeVec       = 305,

    IclaOverwriteVec  = 306,

    IclaBacktransVec  = 307

} icla_vec_t;

typedef enum {
    IclaRangeAll      = 311,

    IclaRangeV        = 312,
    IclaRangeI        = 313
} icla_range_t;

typedef enum {
    IclaQ             = 322,

    IclaP             = 323
} icla_vect_t;

typedef enum {
    IclaForward       = 391,

    IclaBackward      = 392
} icla_direct_t;

typedef enum {
    IclaColumnwise    = 401,

    IclaRowwise       = 402
} icla_storev_t;

typedef enum {
    IclaHybrid        = 701,
    IclaNative        = 702
} icla_mode_t;


typedef enum {
    Icla_CSR          = 611,
    Icla_ELLPACKT     = 612,
    Icla_ELL          = 613,
    Icla_DENSE        = 614,
    Icla_BCSR         = 615,
    Icla_CSC          = 616,
    Icla_HYB          = 617,
    Icla_COO          = 618,
    Icla_ELLRT        = 619,
    Icla_SPMVFUNCTION = 620,
    Icla_SELLP        = 621,
    Icla_ELLD         = 622,
    Icla_CSRLIST      = 623,
    Icla_CSRD         = 624,
    Icla_CSRL         = 627,
    Icla_CSRU         = 628,
    Icla_CSRCOO       = 629,
    Icla_CUCSR        = 630,
    Icla_COOLIST      = 631,
    Icla_CSR5         = 632
} icla_storage_t;


typedef enum {
    Icla_CG           = 431,
    Icla_CGMERGE      = 432,
    Icla_GMRES        = 433,
    Icla_BICGSTAB     = 434,
  Icla_BICGSTABMERGE  = 435,
  Icla_BICGSTABMERGE2 = 436,
    Icla_JACOBI       = 437,
    Icla_GS           = 438,
    Icla_ITERREF      = 439,
    Icla_BCSRLU       = 440,
    Icla_PCG          = 441,
    Icla_PGMRES       = 442,
    Icla_PBICGSTAB    = 443,
    Icla_PASTIX       = 444,
    Icla_ILU          = 445,
    Icla_ICC          = 446,
    Icla_PARILU       = 447,
    Icla_PARIC        = 448,
    Icla_BAITER       = 449,
    Icla_LOBPCG       = 450,
    Icla_NONE         = 451,
    Icla_FUNCTION     = 452,
    Icla_IDR          = 453,
    Icla_PIDR         = 454,
    Icla_CGS          = 455,
    Icla_PCGS         = 456,
    Icla_CGSMERGE     = 457,
    Icla_PCGSMERGE    = 458,
    Icla_TFQMR        = 459,
    Icla_PTFQMR       = 460,
    Icla_TFQMRMERGE   = 461,
    Icla_PTFQMRMERGE  = 462,
    Icla_QMR          = 463,
    Icla_PQMR         = 464,
    Icla_QMRMERGE     = 465,
    Icla_PQMRMERGE    = 466,
    Icla_BOMBARD      = 490,
    Icla_BOMBARDMERGE = 491,
    Icla_PCGMERGE     = 492,
    Icla_BAITERO      = 493,
    Icla_IDRMERGE     = 494,
  Icla_PBICGSTABMERGE = 495,
    Icla_PARICT       = 496,
    Icla_CUSTOMIC     = 497,
    Icla_CUSTOMILU    = 498,
    Icla_PIDRMERGE    = 499,
    Icla_BICG         = 500,
    Icla_BICGMERGE    = 501,
    Icla_PBICG        = 502,
    Icla_PBICGMERGE   = 503,
    Icla_LSQR         = 504,
    Icla_PARILUT      = 505,
    Icla_ISAI         = 506,
    Icla_CUSOLVE      = 507,
    Icla_VBJACOBI     = 508,
    Icla_PARDISO      = 509,
    Icla_SYNCFREESOLVE= 510,
    Icla_ILUT         = 511
} icla_solver_type;

typedef enum {
    Icla_CGSO         = 561,
    Icla_FUSED_CGSO   = 562,
    Icla_MGSO         = 563
} icla_ortho_t;

typedef enum {
    Icla_CPU          = 571,
    Icla_DEV          = 572
} icla_location_t;

typedef enum {
    Icla_GENERAL      = 581,
    Icla_SYMMETRIC    = 582
} icla_symmetry_t;

typedef enum {
    Icla_ORDERED      = 591,
    Icla_DIAGFIRST    = 592,
    Icla_UNITY        = 593,
    Icla_VALUE        = 594
} icla_diagorder_t;

typedef enum {
    Icla_DCOMPLEX     = 501,
    Icla_FCOMPLEX     = 502,
    Icla_DOUBLE       = 503,
    Icla_FLOAT        = 504
} icla_precision;

typedef enum {
    Icla_NOSCALE      = 511,
    Icla_UNITROW      = 512,
    Icla_UNITDIAG     = 513,
    Icla_UNITCOL      = 514,
    Icla_UNITROWCOL   = 515,
    Icla_UNITDIAGCOL  = 516,
} icla_scale_t;


typedef enum {
    Icla_SOLVE        = 801,
    Icla_SETUPSOLVE   = 802,
    Icla_APPLYSOLVE   = 803,
    Icla_DESTROYSOLVE = 804,
    Icla_INFOSOLVE    = 805,
    Icla_GENERATEPREC = 806,
    Icla_PRECONDLEFT  = 807,
    Icla_PRECONDRIGHT = 808,
    Icla_TRANSPOSE    = 809,
    Icla_SPMV         = 810
} icla_operation_t;

typedef enum {
    Icla_PREC_SS           = 900,
    Icla_PREC_SST          = 901,
    Icla_PREC_HS           = 902,
    Icla_PREC_HST          = 903,
    Icla_PREC_SH           = 904,
    Icla_PREC_SHT          = 905,

    Icla_PREC_XHS_H        = 910,
    Icla_PREC_XHS_HTC      = 911,
    Icla_PREC_XHS_161616   = 912,
    Icla_PREC_XHS_161616TC = 913,
    Icla_PREC_XHS_161632TC = 914,
    Icla_PREC_XSH_S        = 915,
    Icla_PREC_XSH_STC      = 916,
    Icla_PREC_XSH_163232TC = 917,
    Icla_PREC_XSH_323232TC = 918,

    Icla_REFINE_IRSTRS   = 920,
    Icla_REFINE_IRDTRS   = 921,
    Icla_REFINE_IRGMSTRS = 922,
    Icla_REFINE_IRGMDTRS = 923,
    Icla_REFINE_GMSTRS   = 924,
    Icla_REFINE_GMDTRS   = 925,
    Icla_REFINE_GMGMSTRS = 926,
    Icla_REFINE_GMGMDTRS = 927,

    Icla_PREC_HD         = 930,
} icla_refinement_t;

typedef enum {
    Icla_MP_BASE_SS              = 950,
    Icla_MP_BASE_DD              = 951,
    Icla_MP_BASE_XHS             = 952,
    Icla_MP_BASE_XSH             = 953,
    Icla_MP_BASE_XHD             = 954,
    Icla_MP_BASE_XDH             = 955,

    Icla_MP_ENABLE_DFLT_MATH     = 960,
    Icla_MP_ENABLE_TC_MATH       = 961,
    Icla_MP_SGEMM                = 962,
    Icla_MP_HGEMM                = 963,
    Icla_MP_GEMEX_I32_O32_C32    = 964,
    Icla_MP_GEMEX_I16_O32_C32    = 965,
    Icla_MP_GEMEX_I16_O16_C32    = 966,
    Icla_MP_GEMEX_I16_O16_C16    = 967,

    Icla_MP_TC_SGEMM             = 968,
    Icla_MP_TC_HGEMM             = 969,
    Icla_MP_TC_GEMEX_I32_O32_C32 = 970,
    Icla_MP_TC_GEMEX_I16_O32_C32 = 971,
    Icla_MP_TC_GEMEX_I16_O16_C32 = 972,
    Icla_MP_TC_GEMEX_I16_O16_C16 = 973,

} icla_mp_type_t;






#define Icla2lapack_Min  IclaFalse
#define Icla2lapack_Max  IclaRowwise





#define IclaRowMajorStr      "Row"
#define IclaColMajorStr      "Col"

#define IclaNoTransStr       "NoTrans"
#define IclaTransStr         "Trans"
#define IclaConjTransStr     "ConjTrans"
#define Icla_ConjTransStr    "ConjTrans"

#define IclaUpperStr         "Upper"
#define IclaLowerStr         "Lower"
#define IclaFullStr          "Full"

#define IclaNonUnitStr       "NonUnit"
#define IclaUnitStr          "Unit"

#define IclaLeftStr          "Left"
#define IclaRightStr         "Right"
#define IclaBothSidesStr     "Both"

#define IclaOneNormStr       "1"
#define IclaTwoNormStr       "2"
#define IclaFrobeniusNormStr "Fro"
#define IclaInfNormStr       "Inf"
#define IclaMaxNormStr       "Max"

#define IclaForwardStr       "Forward"
#define IclaBackwardStr      "Backward"

#define IclaColumnwiseStr    "Columnwise"
#define IclaRowwiseStr       "Rowwise"

#define IclaNoVecStr         "NoVec"
#define IclaVecStr           "Vec"
#define IclaIVecStr          "IVec"
#define IclaAllVecStr        "All"
#define IclaSomeVecStr       "Some"
#define IclaOverwriteVecStr  "Overwrite"






icla_bool_t   icla_bool_const  ( char lapack_char );
icla_order_t  icla_order_const ( char lapack_char );
icla_trans_t  icla_trans_const ( char lapack_char );
icla_uplo_t   icla_uplo_const  ( char lapack_char );
icla_diag_t   icla_diag_const  ( char lapack_char );
icla_side_t   icla_side_const  ( char lapack_char );
icla_norm_t   icla_norm_const  ( char lapack_char );
icla_dist_t   icla_dist_const  ( char lapack_char );
icla_sym_t    icla_sym_const   ( char lapack_char );
icla_pack_t   icla_pack_const  ( char lapack_char );
icla_vec_t    icla_vec_const   ( char lapack_char );
icla_range_t  icla_range_const ( char lapack_char );
icla_vect_t   icla_vect_const  ( char lapack_char );
icla_direct_t icla_direct_const( char lapack_char );
icla_storev_t icla_storev_const( char lapack_char );











const char* lapack_const_str   ( int            icla_const );
const char* lapack_bool_const  ( icla_bool_t   icla_const );
const char* lapack_order_const ( icla_order_t  icla_const );
const char* lapack_trans_const ( icla_trans_t  icla_const );
const char* lapack_uplo_const  ( icla_uplo_t   icla_const );
const char* lapack_diag_const  ( icla_diag_t   icla_const );
const char* lapack_side_const  ( icla_side_t   icla_const );
const char* lapack_norm_const  ( icla_norm_t   icla_const );
const char* lapack_dist_const  ( icla_dist_t   icla_const );
const char* lapack_sym_const   ( icla_sym_t    icla_const );
const char* lapack_pack_const  ( icla_pack_t   icla_const );
const char* lapack_vec_const   ( icla_vec_t    icla_const );
const char* lapack_range_const ( icla_range_t  icla_const );
const char* lapack_vect_const  ( icla_vect_t   icla_const );
const char* lapack_direct_const( icla_direct_t icla_const );
const char* lapack_storev_const( icla_storev_t icla_const );

static inline char lapacke_const       ( int icla_const            ) { return *lapack_const_str   ( icla_const ); }
static inline char lapacke_bool_const  ( icla_bool_t   icla_const ) { return *lapack_bool_const  ( icla_const ); }
static inline char lapacke_order_const ( icla_order_t  icla_const ) { return *lapack_order_const ( icla_const ); }
static inline char lapacke_trans_const ( icla_trans_t  icla_const ) { return *lapack_trans_const ( icla_const ); }
static inline char lapacke_uplo_const  ( icla_uplo_t   icla_const ) { return *lapack_uplo_const  ( icla_const ); }
static inline char lapacke_diag_const  ( icla_diag_t   icla_const ) { return *lapack_diag_const  ( icla_const ); }
static inline char lapacke_side_const  ( icla_side_t   icla_const ) { return *lapack_side_const  ( icla_const ); }
static inline char lapacke_norm_const  ( icla_norm_t   icla_const ) { return *lapack_norm_const  ( icla_const ); }
static inline char lapacke_dist_const  ( icla_dist_t   icla_const ) { return *lapack_dist_const  ( icla_const ); }
static inline char lapacke_sym_const   ( icla_sym_t    icla_const ) { return *lapack_sym_const   ( icla_const ); }
static inline char lapacke_pack_const  ( icla_pack_t   icla_const ) { return *lapack_pack_const  ( icla_const ); }
static inline char lapacke_vec_const   ( icla_vec_t    icla_const ) { return *lapack_vec_const   ( icla_const ); }
static inline char lapacke_range_const ( icla_range_t  icla_const ) { return *lapack_range_const ( icla_const ); }
static inline char lapacke_vect_const  ( icla_vect_t   icla_const ) { return *lapack_vect_const  ( icla_const ); }
static inline char lapacke_direct_const( icla_direct_t icla_const ) { return *lapack_direct_const( icla_const ); }
static inline char lapacke_storev_const( icla_storev_t icla_const ) { return *lapack_storev_const( icla_const ); }




#if defined(ICLA_HAVE_OPENCL)
clblasOrder          clblas_order_const( icla_order_t order );
clblasTranspose      clblas_trans_const( icla_trans_t trans );
clblasUplo           clblas_uplo_const ( icla_uplo_t  uplo  );
clblasDiag           clblas_diag_const ( icla_diag_t  diag  );
clblasSide           clblas_side_const ( icla_side_t  side  );
#endif




#if defined(CUBLAS_V2_H_)
cublasOperation_t    cublas_trans_const ( icla_trans_t trans );
cublasFillMode_t     cublas_uplo_const  ( icla_uplo_t  uplo  );
cublasDiagType_t     cublas_diag_const  ( icla_diag_t  diag  );
cublasSideMode_t     cublas_side_const  ( icla_side_t  side  );

#define icla_backend_trans_const cublas_trans_const
#define icla_backend_uplo_const cublas_uplo_const
#define icla_backend_diag_const cublas_diag_const
#define icla_backend_side_const cublas_side_const
#endif




#if defined(ICLA_HAVE_HIP)
hipblasOperation_t   hipblas_trans_const( icla_trans_t trans );
hipblasFillMode_t    hipblas_uplo_const (icla_uplo_t uplo    );
hipblasDiagType_t    hipblas_diag_const (icla_diag_t diag    );
hipblasSideMode_t    hipblas_side_const (icla_side_t side    );

#define icla_backend_trans_const hipblas_trans_const
#define icla_backend_uplo_const hipblas_uplo_const
#define icla_backend_diag_const hipblas_diag_const
#define icla_backend_side_const hipblas_side_const
#endif




#if defined(HAVE_CBLAS)
#include <cblas.h>
enum CBLAS_ORDER     cblas_order_const  ( icla_order_t order );
enum CBLAS_TRANSPOSE cblas_trans_const  ( icla_trans_t trans );
enum CBLAS_UPLO      cblas_uplo_const   ( icla_uplo_t  uplo  );
enum CBLAS_DIAG      cblas_diag_const   ( icla_diag_t  diag  );
enum CBLAS_SIDE      cblas_side_const   ( icla_side_t  side  );
#endif


#ifdef __cplusplus
}
#endif

#endif
