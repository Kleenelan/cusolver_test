
#ifndef ICLA_COPY_V1_H
#define ICLA_COPY_V1_H

#ifdef ICLA_NO_V1
#error "Since ICLA_NO_V1 is defined, icla.h is invalid; use icla_v2.h"
#endif

#include "icla_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define icla_setvector_v1(           n, elemSize, hx_src, incx, dy_dst, incy ) \
        icla_setvector_v1_internal(  n, elemSize, hx_src, incx, dy_dst, incy, __func__, __FILE__, __LINE__ )

#define icla_getvector_v1(           n, elemSize, dx_src, incx, hy_dst, incy ) \
        icla_getvector_v1_internal(  n, elemSize, dx_src, incx, hy_dst, incy, __func__, __FILE__, __LINE__ )

#define icla_copyvector_v1(          n, elemSize, dx_src, incx, dy_dst, incy ) \
        icla_copyvector_v1_internal( n, elemSize, dx_src, incx, dy_dst, incy, __func__, __FILE__, __LINE__ )

void
icla_setvector_v1_internal(
    icla_int_t n, icla_int_t elemSize,
    const void *hx_src, icla_int_t incx,
    icla_ptr   dy_dst, icla_int_t incy,
    const char* func, const char* file, int line );

void
icla_getvector_v1_internal(
    icla_int_t n, icla_int_t elemSize,
    icla_const_ptr dx_src, icla_int_t incx,
    void           *hy_dst, icla_int_t incy,
    const char* func, const char* file, int line );

void
icla_copyvector_v1_internal(
    icla_int_t n, icla_int_t elemSize,
    icla_const_ptr dx_src, icla_int_t incx,
    icla_ptr       dy_dst, icla_int_t incy,
    const char* func, const char* file, int line );

#define icla_setmatrix_v1(           m, n, elemSize, hA_src, lda,  dB_dst, lddb ) \
        icla_setmatrix_v1_internal(  m, n, elemSize, hA_src, lda,  dB_dst, lddb, __func__, __FILE__, __LINE__ )

#define icla_getmatrix_v1(           m, n, elemSize, dA_src, ldda, hB_dst, ldb ) \
        icla_getmatrix_v1_internal(  m, n, elemSize, dA_src, ldda, hB_dst, ldb, __func__, __FILE__, __LINE__ )

#define icla_copymatrix_v1(          m, n, elemSize, dA_src, ldda, dB_dst, lddb ) \
        icla_copymatrix_v1_internal( m, n, elemSize, dA_src, ldda, dB_dst, lddb, __func__, __FILE__, __LINE__ )

void
icla_setmatrix_v1_internal(
    icla_int_t m, icla_int_t n, icla_int_t elemSize,
    const void *hA_src, icla_int_t lda,
    icla_ptr   dB_dst, icla_int_t lddb,
    const char* func, const char* file, int line );

void
icla_getmatrix_v1_internal(
    icla_int_t m, icla_int_t n, icla_int_t elemSize,
    icla_const_ptr dA_src, icla_int_t ldda,
    void           *hB_dst, icla_int_t ldb,
    const char* func, const char* file, int line );

void
icla_copymatrix_v1_internal(
    icla_int_t m, icla_int_t n, icla_int_t elemSize,
    icla_const_ptr dA_src, icla_int_t ldda,
    icla_ptr       dB_dst, icla_int_t lddb,
    const char* func, const char* file, int line );

#define icla_isetvector_v1(           n, hx_src, incx, dy_dst, incy ) \
        icla_isetvector_v1_internal(  n, hx_src, incx, dy_dst, incy, __func__, __FILE__, __LINE__ )

#define icla_igetvector_v1(           n, dx_src, incx, hy_dst, incy ) \
        icla_igetvector_v1_internal(  n, dx_src, incx, hy_dst, incy, __func__, __FILE__, __LINE__ )

#define icla_icopyvector_v1(          n, dx_src, incx, dy_dst, incy ) \
        icla_icopyvector_v1_internal( n, dx_src, incx, dy_dst, incy, __func__, __FILE__, __LINE__ )

static inline void
icla_isetvector_v1_internal(
    icla_int_t n,
    const icla_int_t *hx_src, icla_int_t incx,
    iclaInt_ptr       dy_dst, icla_int_t incy,
    const char* func, const char* file, int line )
{
    icla_setvector_v1_internal( n, sizeof(icla_int_t),
                                 hx_src, incx,
                                 dy_dst, incy,
                                 func, file, line );
}

static inline void
icla_igetvector_v1_internal(
    icla_int_t n,
    iclaInt_const_ptr dx_src, icla_int_t incx,
    icla_int_t       *hy_dst, icla_int_t incy,
    const char* func, const char* file, int line )
{
    icla_getvector_v1_internal( n, sizeof(icla_int_t),
                                 dx_src, incx,
                                 hy_dst, incy,
                                 func, file, line );
}

static inline void
icla_icopyvector_v1_internal(
    icla_int_t n,
    iclaInt_const_ptr dx_src, icla_int_t incx,
    iclaInt_ptr       dy_dst, icla_int_t incy,
    const char* func, const char* file, int line )
{
    icla_copyvector_v1_internal( n, sizeof(icla_int_t),
                                  dx_src, incx,
                                  dy_dst, incy,
                                  func, file, line );
}

#define icla_isetmatrix_v1(           m, n, hA_src, lda,  dB_dst, lddb ) \
        icla_isetmatrix_v1_internal(  m, n, hA_src, lda,  dB_dst, lddb, __func__, __FILE__, __LINE__ )

#define icla_igetmatrix_v1(           m, n, dA_src, ldda, hB_dst, ldb ) \
        icla_igetmatrix_v1_internal(  m, n, dA_src, ldda, hB_dst, ldb, __func__, __FILE__, __LINE__ )

#define icla_icopymatrix_v1(          m, n, dA_src, ldda, dB_dst, lddb ) \
        icla_icopymatrix_v1_internal( m, n, dA_src, ldda, dB_dst, lddb, __func__, __FILE__, __LINE__ )

static inline void
icla_isetmatrix_v1_internal(
    icla_int_t m, icla_int_t n,
    const icla_int_t *hA_src, icla_int_t lda,
    iclaInt_ptr       dB_dst, icla_int_t lddb,
    const char* func, const char* file, int line )
{
    icla_setmatrix_v1_internal( m, n, sizeof(icla_int_t),
                                 hA_src, lda,
                                 dB_dst, lddb,
                                 func, file, line );
}

static inline void
icla_igetmatrix_v1_internal(
    icla_int_t m, icla_int_t n,
    iclaInt_const_ptr dA_src, icla_int_t ldda,
    icla_int_t       *hB_dst, icla_int_t ldb,
    const char* func, const char* file, int line )
{
    icla_getmatrix_v1_internal( m, n, sizeof(icla_int_t),
                                 dA_src, ldda,
                                 hB_dst, ldb,
                                 func, file, line );
}

static inline void
icla_icopymatrix_v1_internal(
    icla_int_t m, icla_int_t n,
    iclaInt_const_ptr dA_src, icla_int_t ldda,
    iclaInt_ptr       dB_dst, icla_int_t lddb,
    const char* func, const char* file, int line )
{
    icla_copymatrix_v1_internal( m, n, sizeof(icla_int_t),
                                  dA_src, ldda,
                                  dB_dst, lddb,
                                  func, file, line );
}

#define icla_index_setvector_v1(           n, hx_src, incx, dy_dst, incy ) \
        icla_index_setvector_v1_internal(  n, hx_src, incx, dy_dst, incy, __func__, __FILE__, __LINE__ )

#define icla_index_getvector_v1(           n, dx_src, incx, hy_dst, incy ) \
        icla_index_getvector_v1_internal(  n, dx_src, incx, hy_dst, incy, __func__, __FILE__, __LINE__ )

#define icla_index_copyvector_v1(          n, dx_src, incx, dy_dst, incy ) \
        icla_index_copyvector_v1_internal( n, dx_src, incx, dy_dst, incy, __func__, __FILE__, __LINE__ )

static inline void
icla_index_setvector_v1_internal(
    icla_int_t n,
    const icla_index_t *hx_src, icla_int_t incx,
    iclaIndex_ptr       dy_dst, icla_int_t incy,
    const char* func, const char* file, int line )
{
    icla_setvector_v1_internal( n, sizeof(icla_index_t),
                                 hx_src, incx,
                                 dy_dst, incy,
                                 func, file, line );
}

static inline void
icla_index_getvector_v1_internal(
    icla_int_t n,
    iclaIndex_const_ptr dx_src, icla_int_t incx,
    icla_index_t       *hy_dst, icla_int_t incy,
    const char* func, const char* file, int line )
{
    icla_getvector_v1_internal( n, sizeof(icla_index_t),
                                 dx_src, incx,
                                 hy_dst, incy,
                                 func, file, line );
}

static inline void
icla_index_copyvector_v1_internal(
    icla_int_t n,
    iclaIndex_const_ptr dx_src, icla_int_t incx,
    iclaIndex_ptr       dy_dst, icla_int_t incy,
    const char* func, const char* file, int line )
{
    icla_copyvector_v1_internal( n, sizeof(icla_index_t),
                                  dx_src, incx,
                                  dy_dst, incy,
                                  func, file, line );
}

#define icla_index_setmatrix_v1(           m, n, hA_src, lda,  dB_dst, lddb ) \
        icla_index_setmatrix_v1_internal(  m, n, hA_src, lda,  dB_dst, lddb, __func__, __FILE__, __LINE__ )

#define icla_index_getmatrix_v1(           m, n, dA_src, ldda, hB_dst, ldb ) \
        icla_index_getmatrix_v1_internal(  m, n, dA_src, ldda, hB_dst, ldb, __func__, __FILE__, __LINE__ )

#define icla_index_copymatrix_v1(          m, n, dA_src, ldda, dB_dst, lddb ) \
        icla_index_copymatrix_v1_internal( m, n, dA_src, ldda, dB_dst, lddb, __func__, __FILE__, __LINE__ )

static inline void
icla_index_setmatrix_v1_internal(
    icla_int_t m, icla_int_t n,
    const icla_index_t *hA_src, icla_int_t lda,
    iclaIndex_ptr       dB_dst, icla_int_t lddb,
    const char* func, const char* file, int line )
{
    icla_setmatrix_v1_internal( m, n, sizeof(icla_index_t),
                                 hA_src, lda,
                                 dB_dst, lddb,
                                 func, file, line );
}

static inline void
icla_index_getmatrix_v1_internal(
    icla_int_t m, icla_int_t n,
    iclaIndex_const_ptr dA_src, icla_int_t ldda,
    icla_index_t       *hB_dst, icla_int_t ldb,
    const char* func, const char* file, int line )
{
    icla_getmatrix_v1_internal( m, n, sizeof(icla_index_t),
                                 dA_src, ldda,
                                 hB_dst, ldb,
                                 func, file, line );
}

static inline void
icla_index_copymatrix_v1_internal(
    icla_int_t m, icla_int_t n,
    iclaIndex_const_ptr dA_src, icla_int_t ldda,
    iclaIndex_ptr       dB_dst, icla_int_t lddb,
    const char* func, const char* file, int line )
{
    icla_copymatrix_v1_internal( m, n, sizeof(icla_index_t),
                                  dA_src, ldda,
                                  dB_dst, lddb,
                                  func, file, line );
}

#ifdef __cplusplus
}
#endif

#endif

