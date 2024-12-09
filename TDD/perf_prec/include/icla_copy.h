

#ifndef ICLA_COPY_H
#define ICLA_COPY_H

#include "icla_types.h"

#ifdef __cplusplus
extern "C" {
#endif


#define icla_setvector(                 n, elemSize, hx_src, incx, dy_dst, incy, queue ) \
        icla_setvector_internal(        n, elemSize, hx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

#define icla_getvector(                 n, elemSize, dx_src, incx, hy_dst, incy, queue ) \
        icla_getvector_internal(        n, elemSize, dx_src, incx, hy_dst, incy, queue, __func__, __FILE__, __LINE__ )

#define icla_copyvector(                n, elemSize, dx_src, incx, dy_dst, incy, queue ) \
        icla_copyvector_internal(       n, elemSize, dx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

#define icla_setvector_async(           n, elemSize, hx_src, incx, dy_dst, incy, queue ) \
        icla_setvector_async_internal(  n, elemSize, hx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

#define icla_getvector_async(           n, elemSize, dx_src, incx, hy_dst, incy, queue ) \
        icla_getvector_async_internal(  n, elemSize, dx_src, incx, hy_dst, incy, queue, __func__, __FILE__, __LINE__ )

#define icla_copyvector_async(          n, elemSize, dx_src, incx, dy_dst, incy, queue ) \
        icla_copyvector_async_internal( n, elemSize, dx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

void
icla_setvector_internal(
    icla_int_t n, icla_int_t elemSize,
    const void *hx_src, icla_int_t incx,
    icla_ptr   dy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line );

void
icla_getvector_internal(
    icla_int_t n, icla_int_t elemSize,
    icla_const_ptr dx_src, icla_int_t incx,
    void           *hy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line );

void
icla_copyvector_internal(
    icla_int_t n, icla_int_t elemSize,
    icla_const_ptr dx_src, icla_int_t incx,
    icla_ptr       dy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line );

void
icla_setvector_async_internal(
    icla_int_t n, icla_int_t elemSize,
    const void *hx_src, icla_int_t incx,
    icla_ptr   dy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line );

void
icla_getvector_async_internal(
    icla_int_t n, icla_int_t elemSize,
    icla_const_ptr dx_src, icla_int_t incx,
    void           *hy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line );

void
icla_copyvector_async_internal(
    icla_int_t n, icla_int_t elemSize,
    icla_const_ptr dx_src, icla_int_t incx,
    icla_ptr       dy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line );





#define icla_setmatrix(                 m, n, elemSize, hA_src, lda,  dB_dst, lddb, queue ) \
        icla_setmatrix_internal(        m, n, elemSize, hA_src, lda,  dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

#define icla_getmatrix(                 m, n, elemSize, dA_src, ldda, hB_dst, ldb,  queue ) \
        icla_getmatrix_internal(        m, n, elemSize, dA_src, ldda, hB_dst, ldb,  queue, __func__, __FILE__, __LINE__ )

#define icla_copymatrix(                m, n, elemSize, dA_src, ldda, dB_dst, lddb, queue ) \
        icla_copymatrix_internal(       m, n, elemSize, dA_src, ldda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

#define icla_setmatrix_async(           m, n, elemSize, hA_src, lda,  dB_dst, lddb, queue ) \
        icla_setmatrix_async_internal(  m, n, elemSize, hA_src, lda,  dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

#define icla_getmatrix_async(           m, n, elemSize, dA_src, ldda, hB_dst, ldb,  queue ) \
        icla_getmatrix_async_internal(  m, n, elemSize, dA_src, ldda, hB_dst, ldb,  queue, __func__, __FILE__, __LINE__ )

#define icla_copymatrix_async(          m, n, elemSize, dA_src, ldda, dB_dst, lddb, queue ) \
        icla_copymatrix_async_internal( m, n, elemSize, dA_src, ldda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

void
icla_setmatrix_internal(
    icla_int_t m, icla_int_t n, icla_int_t elemSize,
    const void *hA_src, icla_int_t lda,
    icla_ptr   dB_dst, icla_int_t lddb,
    icla_queue_t queue,
    const char* func, const char* file, int line );

void
icla_getmatrix_internal(
    icla_int_t m, icla_int_t n, icla_int_t elemSize,
    icla_const_ptr dA_src, icla_int_t ldda,
    void           *hB_dst, icla_int_t ldb,
    icla_queue_t queue,
    const char* func, const char* file, int line );

void
icla_copymatrix_internal(
    icla_int_t m, icla_int_t n, icla_int_t elemSize,
    icla_const_ptr dA_src, icla_int_t ldda,
    icla_ptr       dB_dst, icla_int_t lddb,
    icla_queue_t queue,
    const char* func, const char* file, int line );

void
icla_setmatrix_async_internal(
    icla_int_t m, icla_int_t n, icla_int_t elemSize,
    const void *hA_src, icla_int_t lda,
    icla_ptr   dB_dst, icla_int_t lddb,
    icla_queue_t queue,
    const char* func, const char* file, int line );

void
icla_getmatrix_async_internal(
    icla_int_t m, icla_int_t n, icla_int_t elemSize,
    icla_const_ptr dA_src, icla_int_t ldda,
    void           *hB_dst, icla_int_t ldb,
    icla_queue_t queue,
    const char* func, const char* file, int line );

void
icla_copymatrix_async_internal(
    icla_int_t m, icla_int_t n, icla_int_t elemSize,
    icla_const_ptr dA_src, icla_int_t ldda,
    icla_ptr       dB_dst, icla_int_t lddb,
    icla_queue_t queue,
    const char* func, const char* file, int line );







#define icla_isetvector(                 n, hx_src, incx, dy_dst, incy, queue ) \
        icla_isetvector_internal(        n, hx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )



#define icla_igetvector(                 n, dx_src, incx, hy_dst, incy, queue ) \
        icla_igetvector_internal(        n, dx_src, incx, hy_dst, incy, queue, __func__, __FILE__, __LINE__ )



#define icla_icopyvector(                n, dx_src, incx, dy_dst, incy, queue ) \
        icla_icopyvector_internal(       n, dx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )



#define icla_isetvector_async(           n, hx_src, incx, dy_dst, incy, queue ) \
        icla_isetvector_async_internal(  n, hx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )



#define icla_igetvector_async(           n, dx_src, incx, hy_dst, incy, queue ) \
        icla_igetvector_async_internal(  n, dx_src, incx, hy_dst, incy, queue, __func__, __FILE__, __LINE__ )



#define icla_icopyvector_async(          n, dx_src, incx, dy_dst, incy, queue ) \
        icla_icopyvector_async_internal( n, dx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

static inline void
icla_isetvector_internal(
    icla_int_t n,
    const icla_int_t *hx_src, icla_int_t incx,
    iclaInt_ptr       dy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_setvector_internal( n, sizeof(icla_int_t),
                              hx_src, incx,
                              dy_dst, incy,
                              queue, func, file, line );
}

static inline void
icla_igetvector_internal(
    icla_int_t n,
    iclaInt_const_ptr dx_src, icla_int_t incx,
    icla_int_t       *hy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_getvector_internal( n, sizeof(icla_int_t),
                              dx_src, incx,
                              hy_dst, incy,
                              queue, func, file, line );
}

static inline void
icla_icopyvector_internal(
    icla_int_t n,
    iclaInt_const_ptr dx_src, icla_int_t incx,
    iclaInt_ptr       dy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_copyvector_internal( n, sizeof(icla_int_t),
                               dx_src, incx,
                               dy_dst, incy,
                               queue, func, file, line );
}

static inline void
icla_isetvector_async_internal(
    icla_int_t n,
    const icla_int_t *hx_src, icla_int_t incx,
    iclaInt_ptr       dy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_setvector_async_internal( n, sizeof(icla_int_t),
                                    hx_src, incx,
                                    dy_dst, incy,
                                    queue, func, file, line );
}

static inline void
icla_igetvector_async_internal(
    icla_int_t n,
    iclaInt_const_ptr dx_src, icla_int_t incx,
    icla_int_t       *hy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_getvector_async_internal( n, sizeof(icla_int_t),
                                    dx_src, incx,
                                    hy_dst, incy,
                                    queue, func, file, line );
}

static inline void
icla_icopyvector_async_internal(
    icla_int_t n,
    iclaInt_const_ptr dx_src, icla_int_t incx,
    iclaInt_ptr       dy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_copyvector_async_internal( n, sizeof(icla_int_t),
                                     dx_src, incx,
                                     dy_dst, incy,
                                     queue, func, file, line );
}







#define icla_isetmatrix(                 m, n, hA_src, lda,  dB_dst, lddb, queue ) \
        icla_isetmatrix_internal(        m, n, hA_src, lda,  dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )



#define icla_igetmatrix(                 m, n, dA_src, ldda, hB_dst, ldb,  queue ) \
        icla_igetmatrix_internal(        m, n, dA_src, ldda, hB_dst, ldb,  queue, __func__, __FILE__, __LINE__ )



#define icla_icopymatrix(                m, n, dA_src, ldda, dB_dst, lddb, queue ) \
        icla_icopymatrix_internal(       m, n, dA_src, ldda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )



#define icla_isetmatrix_async(           m, n, hA_src, lda,  dB_dst, lddb, queue ) \
        icla_isetmatrix_async_internal(  m, n, hA_src, lda,  dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )



#define icla_igetmatrix_async(           m, n, dA_src, ldda, hB_dst, ldb,  queue ) \
        icla_igetmatrix_async_internal(  m, n, dA_src, ldda, hB_dst, ldb,  queue, __func__, __FILE__, __LINE__ )



#define icla_icopymatrix_async(          m, n, dA_src, ldda, dB_dst, lddb, queue ) \
        icla_icopymatrix_async_internal( m, n, dA_src, ldda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

static inline void
icla_isetmatrix_internal(
    icla_int_t m, icla_int_t n,
    const icla_int_t *hA_src, icla_int_t lda,
    iclaInt_ptr       dB_dst, icla_int_t lddb,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_setmatrix_internal( m, n, sizeof(icla_int_t),
                              hA_src, lda,
                              dB_dst, lddb,
                              queue, func, file, line );
}

static inline void
icla_igetmatrix_internal(
    icla_int_t m, icla_int_t n,
    iclaInt_const_ptr dA_src, icla_int_t ldda,
    icla_int_t       *hB_dst, icla_int_t ldb,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_getmatrix_internal( m, n, sizeof(icla_int_t),
                              dA_src, ldda,
                              hB_dst, ldb,
                              queue, func, file, line );
}

static inline void
icla_icopymatrix_internal(
    icla_int_t m, icla_int_t n,
    iclaInt_const_ptr dA_src, icla_int_t ldda,
    iclaInt_ptr       dB_dst, icla_int_t lddb,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_copymatrix_internal( m, n, sizeof(icla_int_t),
                               dA_src, ldda,
                               dB_dst, lddb,
                               queue, func, file, line );
}

static inline void
icla_isetmatrix_async_internal(
    icla_int_t m, icla_int_t n,
    const icla_int_t *hA_src, icla_int_t lda,
    iclaInt_ptr       dB_dst, icla_int_t lddb,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_setmatrix_async_internal( m, n, sizeof(icla_int_t),
                                    hA_src, lda,
                                    dB_dst, lddb,
                                    queue, func, file, line );
}

static inline void
icla_igetmatrix_async_internal(
    icla_int_t m, icla_int_t n,
    iclaInt_const_ptr dA_src, icla_int_t ldda,
    icla_int_t       *hB_dst, icla_int_t ldb,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_getmatrix_async_internal( m, n, sizeof(icla_int_t),
                                    dA_src, ldda,
                                    hB_dst, ldb,
                                    queue, func, file, line );
}

static inline void
icla_icopymatrix_async_internal(
    icla_int_t m, icla_int_t n,
    iclaInt_const_ptr dA_src, icla_int_t ldda,
    iclaInt_ptr       dB_dst, icla_int_t lddb,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_copymatrix_async_internal( m, n, sizeof(icla_int_t),
                                     dA_src, ldda,
                                     dB_dst, lddb,
                                     queue, func, file, line );
}







#define icla_index_setvector(                 n, hx_src, incx, dy_dst, incy, queue ) \
        icla_index_setvector_internal(        n, hx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )



#define icla_index_getvector(                 n, dx_src, incx, hy_dst, incy, queue ) \
        icla_index_getvector_internal(        n, dx_src, incx, hy_dst, incy, queue, __func__, __FILE__, __LINE__ )



#define icla_index_copyvector(                n, dx_src, incx, dy_dst, incy, queue ) \
        icla_index_copyvector_internal(       n, dx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )



#define icla_index_setvector_async(           n, hx_src, incx, dy_dst, incy, queue ) \
        icla_index_setvector_async_internal(  n, hx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )



#define icla_index_getvector_async(           n, dx_src, incx, hy_dst, incy, queue ) \
        icla_index_getvector_async_internal(  n, dx_src, incx, hy_dst, incy, queue, __func__, __FILE__, __LINE__ )



#define icla_index_copyvector_async(          n, dx_src, incx, dy_dst, incy, queue ) \
        icla_index_copyvector_async_internal( n, dx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

static inline void
icla_index_setvector_internal(
    icla_int_t n,
    const icla_index_t *hx_src, icla_int_t incx,
    iclaIndex_ptr       dy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_setvector_internal( n, sizeof(icla_index_t),
                              hx_src, incx,
                              dy_dst, incy,
                              queue, func, file, line );
}

static inline void
icla_index_getvector_internal(
    icla_int_t n,
    iclaIndex_const_ptr dx_src, icla_int_t incx,
    icla_index_t       *hy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_getvector_internal( n, sizeof(icla_index_t),
                              dx_src, incx,
                              hy_dst, incy,
                              queue, func, file, line );
}

static inline void
icla_index_copyvector_internal(
    icla_int_t n,
    iclaIndex_const_ptr dx_src, icla_int_t incx,
    iclaIndex_ptr       dy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_copyvector_internal( n, sizeof(icla_index_t),
                               dx_src, incx,
                               dy_dst, incy,
                               queue, func, file, line );
}

static inline void
icla_index_setvector_async_internal(
    icla_int_t n,
    const icla_index_t *hx_src, icla_int_t incx,
    iclaIndex_ptr       dy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_setvector_async_internal( n, sizeof(icla_index_t),
                                    hx_src, incx,
                                    dy_dst, incy,
                                    queue, func, file, line );
}

static inline void
icla_index_getvector_async_internal(
    icla_int_t n,
    iclaIndex_const_ptr dx_src, icla_int_t incx,
    icla_index_t       *hy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_getvector_async_internal( n, sizeof(icla_index_t),
                                    dx_src, incx,
                                    hy_dst, incy,
                                    queue, func, file, line );
}

static inline void
icla_index_copyvector_async_internal(
    icla_int_t n,
    iclaIndex_const_ptr dx_src, icla_int_t incx,
    iclaIndex_ptr       dy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_copyvector_async_internal( n, sizeof(icla_index_t),
                                     dx_src, incx,
                                     dy_dst, incy,
                                     queue, func, file, line );
}






#define icla_uindex_setvector(                 n, hx_src, incx, dy_dst, incy, queue ) \
        icla_uindex_setvector_internal(        n, hx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )



#define icla_uindex_getvector(                 n, dx_src, incx, hy_dst, incy, queue ) \
        icla_uindex_getvector_internal(        n, dx_src, incx, hy_dst, incy, queue, __func__, __FILE__, __LINE__ )



#define icla_uindex_copyvector(                n, dx_src, incx, dy_dst, incy, queue ) \
        icla_uindex_copyvector_internal(       n, dx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )



#define icla_uindex_setvector_async(           n, hx_src, incx, dy_dst, incy, queue ) \
        icla_uindex_setvector_async_internal(  n, hx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )



#define icla_uindex_getvector_async(           n, dx_src, incx, hy_dst, incy, queue ) \
        icla_uindex_getvector_async_internal(  n, dx_src, incx, hy_dst, incy, queue, __func__, __FILE__, __LINE__ )



#define icla_uindex_copyvector_async(          n, dx_src, incx, dy_dst, incy, queue ) \
        icla_uindex_copyvector_async_internal( n, dx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

static inline void
icla_uindex_setvector_internal(
    icla_int_t n,
    const icla_uindex_t *hx_src, icla_int_t incx,
    iclaUIndex_ptr       dy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_setvector_internal( n, sizeof(icla_uindex_t),
                              hx_src, incx,
                              dy_dst, incy,
                              queue, func, file, line );
}

static inline void
icla_uindex_getvector_internal(
    icla_int_t n,
    iclaUIndex_const_ptr dx_src, icla_int_t incx,
    icla_uindex_t       *hy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_getvector_internal( n, sizeof(icla_uindex_t),
                              dx_src, incx,
                              hy_dst, incy,
                              queue, func, file, line );
}

static inline void
icla_uindex_copyvector_internal(
    icla_int_t n,
    iclaUIndex_const_ptr dx_src, icla_int_t incx,
    iclaUIndex_ptr       dy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_copyvector_internal( n, sizeof(icla_uindex_t),
                               dx_src, incx,
                               dy_dst, incy,
                               queue, func, file, line );
}

static inline void
icla_uindex_setvector_async_internal(
    icla_int_t n,
    const icla_uindex_t *hx_src, icla_int_t incx,
    iclaUIndex_ptr       dy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_setvector_async_internal( n, sizeof(icla_uindex_t),
                                    hx_src, incx,
                                    dy_dst, incy,
                                    queue, func, file, line );
}

static inline void
icla_uindex_getvector_async_internal(
    icla_int_t n,
    iclaUIndex_const_ptr dx_src, icla_int_t incx,
    icla_uindex_t       *hy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_getvector_async_internal( n, sizeof(icla_uindex_t),
                                    dx_src, incx,
                                    hy_dst, incy,
                                    queue, func, file, line );
}

static inline void
icla_uindex_copyvector_async_internal(
    icla_int_t n,
    iclaUIndex_const_ptr dx_src, icla_int_t incx,
    iclaUIndex_ptr       dy_dst, icla_int_t incy,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_copyvector_async_internal( n, sizeof(icla_uindex_t),
                                     dx_src, incx,
                                     dy_dst, incy,
                                     queue, func, file, line );
}







#define icla_index_setmatrix(                 m, n, hA_src, lda,  dB_dst, lddb, queue ) \
        icla_index_setmatrix_internal(        m, n, hA_src, lda,  dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )



#define icla_index_getmatrix(                 m, n, dA_src, ldda, hB_dst, ldb,  queue ) \
        icla_index_getmatrix_internal(        m, n, dA_src, ldda, hB_dst, ldb,  queue, __func__, __FILE__, __LINE__ )



#define icla_index_copymatrix(                m, n, dA_src, ldda, dB_dst, lddb, queue ) \
        icla_index_copymatrix_internal(       m, n, dA_src, ldda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )



#define icla_index_setmatrix_async(           m, n, hA_src, lda,  dB_dst, lddb, queue ) \
        icla_index_setmatrix_async_internal(  m, n, hA_src, lda,  dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )



#define icla_index_getmatrix_async(           m, n, dA_src, ldda, hB_dst, ldb,  queue ) \
        icla_index_getmatrix_async_internal(  m, n, dA_src, ldda, hB_dst, ldb,  queue, __func__, __FILE__, __LINE__ )



#define icla_index_copymatrix_async(          m, n, dA_src, ldda, dB_dst, lddb, queue ) \
        icla_index_copymatrix_async_internal( m, n, dA_src, ldda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

static inline void
icla_index_setmatrix_internal(
    icla_int_t m, icla_int_t n,
    const icla_index_t *hA_src, icla_int_t lda,
    iclaIndex_ptr       dB_dst, icla_int_t lddb,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_setmatrix_internal( m, n, sizeof(icla_index_t),
                              hA_src, lda,
                              dB_dst, lddb,
                              queue, func, file, line );
}

static inline void
icla_index_getmatrix_internal(
    icla_int_t m, icla_int_t n,
    iclaIndex_const_ptr dA_src, icla_int_t ldda,
    icla_index_t       *hB_dst, icla_int_t ldb,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_getmatrix_internal( m, n, sizeof(icla_index_t),
                              dA_src, ldda,
                              hB_dst, ldb,
                              queue, func, file, line );
}

static inline void
icla_index_copymatrix_internal(
    icla_int_t m, icla_int_t n,
    iclaIndex_const_ptr dA_src, icla_int_t ldda,
    iclaIndex_ptr       dB_dst, icla_int_t lddb,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_copymatrix_internal( m, n, sizeof(icla_index_t),
                               dA_src, ldda,
                               dB_dst, lddb,
                               queue, func, file, line );
}

static inline void
icla_index_setmatrix_async_internal(
    icla_int_t m, icla_int_t n,
    const icla_index_t *hA_src, icla_int_t lda,
    iclaIndex_ptr       dB_dst, icla_int_t lddb,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_setmatrix_async_internal( m, n, sizeof(icla_index_t),
                                    hA_src, lda,
                                    dB_dst, lddb,
                                    queue, func, file, line );
}

static inline void
icla_index_getmatrix_async_internal(
    icla_int_t m, icla_int_t n,
    iclaIndex_const_ptr dA_src, icla_int_t ldda,
    icla_index_t       *hB_dst, icla_int_t ldb,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_getmatrix_async_internal( m, n, sizeof(icla_index_t),
                                    dA_src, ldda,
                                    hB_dst, ldb,
                                    queue, func, file, line );
}

static inline void
icla_index_copymatrix_async_internal(
    icla_int_t m, icla_int_t n,
    iclaIndex_const_ptr dA_src, icla_int_t ldda,
    iclaIndex_ptr       dB_dst, icla_int_t lddb,
    icla_queue_t queue,
    const char* func, const char* file, int line )
{
    icla_copymatrix_async_internal( m, n, sizeof(icla_index_t),
                                     dA_src, ldda,
                                     dB_dst, lddb,
                                     queue, func, file, line );
}

#ifdef __cplusplus
}
#endif

#endif
