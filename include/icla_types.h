
#ifndef ICLA_TYPES_H
#define ICLA_TYPES_H

#include "icla_config.h"

#include <stdint.h>
#include <assert.h>


#if ! defined(ICLA_HAVE_CUDA) && defined(ICLA_HAVE_HIP)

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
#else
    #error "One of ICLA_HAVE_CUDA, ICLA_HAVE_HIP, must be defined. For example, add -DICLA_HAVE_CUDA to CFLAGS, or #define ICLA_HAVE_CUDA before #include <icla.h>. In ICLA, this happens in Makefile."
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

#define ICLA_VERSION_MAJOR 1
#define ICLA_VERSION_MINOR 0
#define ICLA_VERSION_MICRO 0

#define ICLA_VERSION_STAGE "svn"

#define iclaMaxGPUs 8
#define iclaMaxAccelerators 8
#define iclaMaxSubs 16

#define iclaBigTileSize 1000000

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
    iclaFalse         = 0,
    iclaTrue          = 1
} icla_bool_t;

typedef enum {
    iclaRowMajor      = 101,
    iclaColMajor      = 102
} icla_order_t;

typedef enum {
    iclaNoTrans       = 111,
    iclaTrans         = 112,
    iclaConjTrans     = 113,
    icla_ConjTrans    = iclaConjTrans
} icla_trans_t;

typedef enum {
    iclaUpper         = 121,
    iclaLower         = 122,
    iclaFull          = 123,

    iclaHessenberg    = 124

} icla_uplo_t;

typedef icla_uplo_t icla_type_t;

typedef enum {
    iclaNonUnit       = 131,
    iclaUnit          = 132
} icla_diag_t;

typedef enum {
    iclaLeft          = 141,
    iclaRight         = 142,
    iclaBothSides     = 143

} icla_side_t;

typedef enum {
    iclaOneNorm       = 171,

    iclaRealOneNorm   = 172,
    iclaTwoNorm       = 173,
    iclaFrobeniusNorm = 174,
    iclaInfNorm       = 175,
    iclaRealInfNorm   = 176,
    iclaMaxNorm       = 177,
    iclaRealMaxNorm   = 178
} icla_norm_t;

typedef enum {
    iclaDistUniform   = 201,

    iclaDistSymmetric = 202,
    iclaDistNormal    = 203
} icla_dist_t;

typedef enum {
    iclaHermGeev      = 241,

    iclaHermPoev      = 242,
    iclaNonsymPosv    = 243,
    iclaSymPosv       = 244
} icla_sym_t;

typedef enum {
    iclaNoPacking     = 291,

    iclaPackSubdiag   = 292,
    iclaPackSupdiag   = 293,
    iclaPackColumn    = 294,
    iclaPackRow       = 295,
    iclaPackLowerBand = 296,
    iclaPackUpeprBand = 297,
    iclaPackAll       = 298
} icla_pack_t;

typedef enum {
    iclaNoVec         = 301,

    iclaVec           = 302,

    iclaIVec          = 303,

    iclaAllVec        = 304,

    iclaSomeVec       = 305,

    iclaOverwriteVec  = 306,

    iclaBacktransVec  = 307

} icla_vec_t;

typedef enum {
    iclaRangeAll      = 311,

    iclaRangeV        = 312,
    iclaRangeI        = 313
} icla_range_t;

typedef enum {
    iclaQ             = 322,

    iclaP             = 323
} icla_vect_t;

typedef enum {
    iclaForward       = 391,

    iclaBackward      = 392
} icla_direct_t;

typedef enum {
    iclaColumnwise    = 401,

    iclaRowwise       = 402
} icla_storev_t;

typedef enum {
    iclaHybrid        = 701,
    iclaNative        = 702
} icla_mode_t;

typedef enum {
    icla_CSR          = 611,
    icla_ELLPACKT     = 612,
    icla_ELL          = 613,
    icla_DENSE        = 614,
    icla_BCSR         = 615,
    icla_CSC          = 616,
    icla_HYB          = 617,
    icla_COO          = 618,
    icla_ELLRT        = 619,
    icla_SPMVFUNCTION = 620,
    icla_SELLP        = 621,
    icla_ELLD         = 622,
    icla_CSRLIST      = 623,
    icla_CSRD         = 624,
    icla_CSRL         = 627,
    icla_CSRU         = 628,
    icla_CSRCOO       = 629,
    icla_CUCSR        = 630,
    icla_COOLIST      = 631,
    icla_CSR5         = 632
} icla_storage_t;

typedef enum {
    icla_CG           = 431,
    icla_CGMERGE      = 432,
    icla_GMRES        = 433,
    icla_BICGSTAB     = 434,
  icla_BICGSTABMERGE  = 435,
  icla_BICGSTABMERGE2 = 436,
    icla_JACOBI       = 437,
    icla_GS           = 438,
    icla_ITERREF      = 439,
    icla_BCSRLU       = 440,
    icla_PCG          = 441,
    icla_PGMRES       = 442,
    icla_PBICGSTAB    = 443,
    icla_PASTIX       = 444,
    icla_ILU          = 445,
    icla_ICC          = 446,
    icla_PARILU       = 447,
    icla_PARIC        = 448,
    icla_BAITER       = 449,
    icla_LOBPCG       = 450,
    icla_NONE         = 451,
    icla_FUNCTION     = 452,
    icla_IDR          = 453,
    icla_PIDR         = 454,
    icla_CGS          = 455,
    icla_PCGS         = 456,
    icla_CGSMERGE     = 457,
    icla_PCGSMERGE    = 458,
    icla_TFQMR        = 459,
    icla_PTFQMR       = 460,
    icla_TFQMRMERGE   = 461,
    icla_PTFQMRMERGE  = 462,
    icla_QMR          = 463,
    icla_PQMR         = 464,
    icla_QMRMERGE     = 465,
    icla_PQMRMERGE    = 466,
    icla_BOMBARD      = 490,
    icla_BOMBARDMERGE = 491,
    icla_PCGMERGE     = 492,
    icla_BAITERO      = 493,
    icla_IDRMERGE     = 494,
  icla_PBICGSTABMERGE = 495,
    icla_PARICT       = 496,
    icla_CUSTOMIC     = 497,
    icla_CUSTOMILU    = 498,
    icla_PIDRMERGE    = 499,
    icla_BICG         = 500,
    icla_BICGMERGE    = 501,
    icla_PBICG        = 502,
    icla_PBICGMERGE   = 503,
    icla_LSQR         = 504,
    icla_PARILUT      = 505,
    icla_ISAI         = 506,
    icla_CUSOLVE      = 507,
    icla_VBJACOBI     = 508,
    icla_PARDISO      = 509,
    icla_SYNCFREESOLVE= 510,
    icla_ILUT         = 511
} icla_solver_type;

typedef enum {
    icla_CGSO         = 561,
    icla_FUSED_CGSO   = 562,
    icla_MGSO         = 563
} icla_ortho_t;

typedef enum {
    icla_CPU          = 571,
    icla_DEV          = 572
} icla_location_t;

typedef enum {
    icla_GENERAL      = 581,
    icla_SYMMETRIC    = 582
} icla_symmetry_t;

typedef enum {
    icla_ORDERED      = 591,
    icla_DIAGFIRST    = 592,
    icla_UNITY        = 593,
    icla_VALUE        = 594
} icla_diagorder_t;

typedef enum {
    icla_DCOMPLEX     = 501,
    icla_FCOMPLEX     = 502,
    icla_DOUBLE       = 503,
    icla_FLOAT        = 504
} icla_precision;

typedef enum {
    icla_NOSCALE      = 511,
    icla_UNITROW      = 512,
    icla_UNITDIAG     = 513,
    icla_UNITCOL      = 514,
    icla_UNITROWCOL   = 515,

    icla_UNITDIAGCOL  = 516,

} icla_scale_t;

typedef enum {
    icla_SOLVE        = 801,
    icla_SETUPSOLVE   = 802,
    icla_APPLYSOLVE   = 803,
    icla_DESTROYSOLVE = 804,
    icla_INFOSOLVE    = 805,
    icla_GENERATEPREC = 806,
    icla_PRECONDLEFT  = 807,
    icla_PRECONDRIGHT = 808,
    icla_TRANSPOSE    = 809,
    icla_SPMV         = 810
} icla_operation_t;

typedef enum {
    icla_PREC_SS           = 900,
    icla_PREC_SST          = 901,
    icla_PREC_HS           = 902,
    icla_PREC_HST          = 903,
    icla_PREC_SH           = 904,
    icla_PREC_SHT          = 905,

    icla_PREC_XHS_H        = 910,
    icla_PREC_XHS_HTC      = 911,
    icla_PREC_XHS_161616   = 912,
    icla_PREC_XHS_161616TC = 913,
    icla_PREC_XHS_161632TC = 914,
    icla_PREC_XSH_S        = 915,
    icla_PREC_XSH_STC      = 916,
    icla_PREC_XSH_163232TC = 917,
    icla_PREC_XSH_323232TC = 918,

    icla_REFINE_IRSTRS   = 920,
    icla_REFINE_IRDTRS   = 921,
    icla_REFINE_IRGMSTRS = 922,
    icla_REFINE_IRGMDTRS = 923,
    icla_REFINE_GMSTRS   = 924,
    icla_REFINE_GMDTRS   = 925,
    icla_REFINE_GMGMSTRS = 926,
    icla_REFINE_GMGMDTRS = 927,

    icla_PREC_HD         = 930,
} icla_refinement_t;

typedef enum {
    icla_MP_BASE_SS              = 950,
    icla_MP_BASE_DD              = 951,
    icla_MP_BASE_XHS             = 952,
    icla_MP_BASE_XSH             = 953,
    icla_MP_BASE_XHD             = 954,
    icla_MP_BASE_XDH             = 955,

    icla_MP_ENABLE_DFLT_MATH     = 960,
    icla_MP_ENABLE_TC_MATH       = 961,
    icla_MP_SGEMM                = 962,
    icla_MP_HGEMM                = 963,
    icla_MP_GEMEX_I32_O32_C32    = 964,
    icla_MP_GEMEX_I16_O32_C32    = 965,
    icla_MP_GEMEX_I16_O16_C32    = 966,
    icla_MP_GEMEX_I16_O16_C16    = 967,

    icla_MP_TC_SGEMM             = 968,
    icla_MP_TC_HGEMM             = 969,
    icla_MP_TC_GEMEX_I32_O32_C32 = 970,
    icla_MP_TC_GEMEX_I16_O32_C32 = 971,
    icla_MP_TC_GEMEX_I16_O16_C32 = 972,
    icla_MP_TC_GEMEX_I16_O16_C16 = 973,

} icla_mp_type_t;

#define icla2lapack_Min  iclaFalse

#define icla2lapack_Max  iclaRowwise

#define iclaRowMajorStr      "Row"
#define iclaColMajorStr      "Col"

#define iclaNoTransStr       "NoTrans"
#define iclaTransStr         "Trans"
#define iclaConjTransStr     "ConjTrans"
#define icla_ConjTransStr    "ConjTrans"

#define iclaUpperStr         "Upper"
#define iclaLowerStr         "Lower"
#define iclaFullStr          "Full"

#define iclaNonUnitStr       "NonUnit"
#define iclaUnitStr          "Unit"

#define iclaLeftStr          "Left"
#define iclaRightStr         "Right"
#define iclaBothSidesStr     "Both"

#define iclaOneNormStr       "1"
#define iclaTwoNormStr       "2"
#define iclaFrobeniusNormStr "Fro"
#define iclaInfNormStr       "Inf"
#define iclaMaxNormStr       "Max"

#define iclaForwardStr       "Forward"
#define iclaBackwardStr      "Backward"

#define iclaColumnwiseStr    "Columnwise"
#define iclaRowwiseStr       "Rowwise"

#define iclaNoVecStr         "NoVec"
#define iclaVecStr           "Vec"
#define iclaIVecStr          "IVec"
#define iclaAllVecStr        "All"
#define iclaSomeVecStr       "Some"
#define iclaOverwriteVecStr  "Overwrite"

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

