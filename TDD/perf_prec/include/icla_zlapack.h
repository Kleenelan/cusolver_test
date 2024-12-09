

#ifndef ICLA_ZLAPACK_H
#define ICLA_ZLAPACK_H

#include "icla_types.h"
#include "icla_mangling.h"

#define ICLA_COMPLEX

#ifdef __cplusplus
extern "C" {
#endif


#define blasf77_izamax     FORTRAN_NAME( izamax, IZAMAX )
#define blasf77_zaxpy      FORTRAN_NAME( zaxpy,  ZAXPY  )
#define blasf77_zcopy      FORTRAN_NAME( zcopy,  ZCOPY  )
#define blasf77_zgbmv      FORTRAN_NAME( zgbmv,  ZGBMV  )
#define blasf77_zgemm      FORTRAN_NAME( zgemm,  ZGEMM  )
#define blasf77_zgemv      FORTRAN_NAME( zgemv,  ZGEMV  )
#define blasf77_zgerc      FORTRAN_NAME( zgerc,  ZGERC  )
#define blasf77_zgeru      FORTRAN_NAME( zgeru,  ZGERU  )
#define blasf77_zhemm      FORTRAN_NAME( zhemm,  ZHEMM  )
#define blasf77_zhemv      FORTRAN_NAME( zhemv,  ZHEMV  )
#define blasf77_zher       FORTRAN_NAME( zher,   ZHER   )
#define blasf77_zher2      FORTRAN_NAME( zher2,  ZHER2  )
#define blasf77_zher2k     FORTRAN_NAME( zher2k, ZHER2K )
#define blasf77_zherk      FORTRAN_NAME( zherk,  ZHERK  )
#define blasf77_zscal      FORTRAN_NAME( zscal,  ZSCAL  )
#define blasf77_zdscal     FORTRAN_NAME( zdscal, ZDSCAL )
#define blasf77_zswap      FORTRAN_NAME( zswap,  ZSWAP  )
#define blasf77_zsymm      FORTRAN_NAME( zsymm,  ZSYMM  )
#define blasf77_zsyr2k     FORTRAN_NAME( zsyr2k, ZSYR2K )
#define blasf77_zsyrk      FORTRAN_NAME( zsyrk,  ZSYRK  )
#define blasf77_zrotg      FORTRAN_NAME( zrotg,  ZROTG  )
#define blasf77_zrot       FORTRAN_NAME( zrot,   ZROT   )
#define blasf77_zdrot      FORTRAN_NAME( zdrot,  ZDROT  )
#define blasf77_ztrmm      FORTRAN_NAME( ztrmm,  ZTRMM  )
#define blasf77_ztrmv      FORTRAN_NAME( ztrmv,  ZTRMV  )
#define blasf77_ztrsm      FORTRAN_NAME( ztrsm,  ZTRSM  )
#define blasf77_ztrsv      FORTRAN_NAME( ztrsv,  ZTRSV  )

#define lapackf77_dlaed2   FORTRAN_NAME( dlaed2, DLAED2 )
#define lapackf77_dlaed4   FORTRAN_NAME( dlaed4, DLAED4 )
#define lapackf77_dlaln2   FORTRAN_NAME( dlaln2, DLALN2 )
#define lapackf77_dlamc3   FORTRAN_NAME( dlamc3, DLAMC3 )
#define lapackf77_dlamrg   FORTRAN_NAME( dlamrg, DLAMRG )
#define lapackf77_dlasrt   FORTRAN_NAME( dlasrt, DLASRT )
#define lapackf77_dstebz   FORTRAN_NAME( dstebz, DSTEBZ )

#define lapackf77_dbdsdc   FORTRAN_NAME( dbdsdc, DBDSDC )
#define lapackf77_zbdsqr   FORTRAN_NAME( zbdsqr, ZBDSQR )
#define lapackf77_zgbtrf   FORTRAN_NAME( zgbtrf, ZGBTRF )
#define lapackf77_zgebak   FORTRAN_NAME( zgebak, ZGEBAK )
#define lapackf77_zgebal   FORTRAN_NAME( zgebal, ZGEBAL )
#define lapackf77_zgebd2   FORTRAN_NAME( zgebd2, ZGEBD2 )
#define lapackf77_zgebrd   FORTRAN_NAME( zgebrd, ZGEBRD )
#define lapackf77_zgbbrd   FORTRAN_NAME( zgbbrd, ZGBBRD )
#define lapackf77_zgbsv    FORTRAN_NAME( zgbsv,  ZGBSV  )
#define lapackf77_zgbtrs   FORTRAN_NAME( zgbtrs, ZGBTRS )
#define lapackf77_zgeev    FORTRAN_NAME( zgeev,  ZGEEV  )
#define lapackf77_zgehd2   FORTRAN_NAME( zgehd2, ZGEHD2 )
#define lapackf77_zgehrd   FORTRAN_NAME( zgehrd, ZGEHRD )
#define lapackf77_zgelqf   FORTRAN_NAME( zgelqf, ZGELQF )
#define lapackf77_zgels    FORTRAN_NAME( zgels,  ZGELS  )
#define lapackf77_zgeqlf   FORTRAN_NAME( zgeqlf, ZGEQLF )
#define lapackf77_zgeqp3   FORTRAN_NAME( zgeqp3, ZGEQP3 )
#define lapackf77_zgeqrf   FORTRAN_NAME( zgeqrf, ZGEQRF )
#define lapackf77_zgerqf   FORTRAN_NAME( zgerqf, ZGERQF )
#define lapackf77_zgesdd   FORTRAN_NAME( zgesdd, ZGESDD )
#define lapackf77_zgesv    FORTRAN_NAME( zgesv,  ZGESV  )
#define lapackf77_zgesvd   FORTRAN_NAME( zgesvd, ZGESVD )
#define lapackf77_zgetrf   FORTRAN_NAME( zgetrf, ZGETRF )
#define lapackf77_zgetri   FORTRAN_NAME( zgetri, ZGETRI )
#define lapackf77_zgetrs   FORTRAN_NAME( zgetrs, ZGETRS )
#define lapackf77_zgglse   FORTRAN_NAME( zgglse, ZGGLSE )
#define lapackf77_zggrqf   FORTRAN_NAME( zggrqf, ZGGRQF )
#define lapackf77_zhetf2   FORTRAN_NAME( zhetf2, ZHETF2 )
#define lapackf77_zhetrs   FORTRAN_NAME( zhetrs, ZHETRS )
#define lapackf77_zhbtrd   FORTRAN_NAME( zhbtrd, ZHBTRD )
#define lapackf77_zheev    FORTRAN_NAME( zheev,  ZHEEV  )
#define lapackf77_zheevd   FORTRAN_NAME( zheevd, ZHEEVD )
#define lapackf77_zheevr   FORTRAN_NAME( zheevr, ZHEEVR )
#define lapackf77_zheevx   FORTRAN_NAME( zheevx, ZHEEVX )
#define lapackf77_zhegs2   FORTRAN_NAME( zhegs2, ZHEGS2 )
#define lapackf77_zhegst   FORTRAN_NAME( zhegst, ZHEGST )
#define lapackf77_zhegvd   FORTRAN_NAME( zhegvd, ZHEGVD )
#define lapackf77_zhetd2   FORTRAN_NAME( zhetd2, ZHETD2 )
#define lapackf77_zhetrd   FORTRAN_NAME( zhetrd, ZHETRD )
#define lapackf77_zhetrf   FORTRAN_NAME( zhetrf, ZHETRF )
#define lapackf77_zhesv    FORTRAN_NAME( zhesv,  ZHESV )
#define lapackf77_zhseqr   FORTRAN_NAME( zhseqr, ZHSEQR )
#define lapackf77_zlabrd   FORTRAN_NAME( zlabrd, ZLABRD )
#define lapackf77_zlacgv   FORTRAN_NAME( zlacgv, ZLACGV )
#define lapackf77_zlacp2   FORTRAN_NAME( zlacp2, ZLACP2 )
#define lapackf77_zlacpy   FORTRAN_NAME( zlacpy, ZLACPY )
#define lapackf77_zlacrm   FORTRAN_NAME( zlacrm, ZLACRM )
#define lapackf77_zladiv   FORTRAN_NAME( zladiv, ZLADIV )
#define lapackf77_zlahef   FORTRAN_NAME( zlahef, ZLAHEF )
#define lapackf77_zlangb   FORTRAN_NAME( zlangb, ZLANGB )
#define lapackf77_zlange   FORTRAN_NAME( zlange, ZLANGE )
#define lapackf77_zlanhe   FORTRAN_NAME( zlanhe, ZLANHE )
#define lapackf77_zlanht   FORTRAN_NAME( zlanht, ZLANHT )
#define lapackf77_zlansy   FORTRAN_NAME( zlansy, ZLANSY )
#define lapackf77_zlantr   FORTRAN_NAME( zlantr, ZLANTR )
#define lapackf77_dlapy3   FORTRAN_NAME( dlapy3, DLAPY3 )
#define lapackf77_zlaqp2   FORTRAN_NAME( zlaqp2, ZLAQP2 )
#define lapackf77_zlarcm   FORTRAN_NAME( zlarcm, ZLARCM )
#define lapackf77_zlarf    FORTRAN_NAME( zlarf,  ZLARF  )
#define lapackf77_zlarfb   FORTRAN_NAME( zlarfb, ZLARFB )
#define lapackf77_zlarfg   FORTRAN_NAME( zlarfg, ZLARFG )
#define lapackf77_zlarft   FORTRAN_NAME( zlarft, ZLARFT )
#define lapackf77_zlarfx   FORTRAN_NAME( zlarfx, ZLARFX )
#define lapackf77_zlarnv   FORTRAN_NAME( zlarnv, ZLARNV )
#define lapackf77_zlartg   FORTRAN_NAME( zlartg, ZLARTG )
#define lapackf77_zlascl   FORTRAN_NAME( zlascl, ZLASCL )
#define lapackf77_zlaset   FORTRAN_NAME( zlaset, ZLASET )
#define lapackf77_zlaswp   FORTRAN_NAME( zlaswp, ZLASWP )
#define lapackf77_zlatrd   FORTRAN_NAME( zlatrd, ZLATRD )
#define lapackf77_zlatrs   FORTRAN_NAME( zlatrs, ZLATRS )
#define lapackf77_zlauum   FORTRAN_NAME( zlauum, ZLAUUM )
#define lapackf77_zlavhe   FORTRAN_NAME( zlavhe, ZLAVHE )
#define lapackf77_zposv    FORTRAN_NAME( zposv,  ZPOSV  )
#define lapackf77_zpotrf   FORTRAN_NAME( zpotrf, ZPOTRF )
#define lapackf77_zpotri   FORTRAN_NAME( zpotri, ZPOTRI )
#define lapackf77_zpotrs   FORTRAN_NAME( zpotrs, ZPOTRS )
#define lapackf77_zstedc   FORTRAN_NAME( zstedc, ZSTEDC )
#define lapackf77_zstein   FORTRAN_NAME( zstein, ZSTEIN )
#define lapackf77_zstemr   FORTRAN_NAME( zstemr, ZSTEMR )
#define lapackf77_zsteqr   FORTRAN_NAME( zsteqr, ZSTEQR )
#define lapackf77_zsymv    FORTRAN_NAME( zsymv,  ZSYMV  )
#define lapackf77_zsyr     FORTRAN_NAME( zsyr,   ZSYR   )
#define lapackf77_zsysv    FORTRAN_NAME( zsysv,  ZSYSV  )
#define lapackf77_ztrevc   FORTRAN_NAME( ztrevc, ZTREVC )
#define lapackf77_ztrevc3  FORTRAN_NAME( ztrevc3, ZTREVC3 )
#define lapackf77_ztrtri   FORTRAN_NAME( ztrtri, ZTRTRI )
#define lapackf77_zung2r   FORTRAN_NAME( zung2r, ZUNG2R )
#define lapackf77_zungbr   FORTRAN_NAME( zungbr, ZUNGBR )
#define lapackf77_zunghr   FORTRAN_NAME( zunghr, ZUNGHR )
#define lapackf77_zunglq   FORTRAN_NAME( zunglq, ZUNGLQ )
#define lapackf77_zungql   FORTRAN_NAME( zungql, ZUNGQL )
#define lapackf77_zungqr   FORTRAN_NAME( zungqr, ZUNGQR )
#define lapackf77_zungtr   FORTRAN_NAME( zungtr, ZUNGTR )
#define lapackf77_zunm2r   FORTRAN_NAME( zunm2r, ZUNM2R )
#define lapackf77_zunmbr   FORTRAN_NAME( zunmbr, ZUNMBR )
#define lapackf77_zunmlq   FORTRAN_NAME( zunmlq, ZUNMLQ )
#define lapackf77_zunmql   FORTRAN_NAME( zunmql, ZUNMQL )
#define lapackf77_zunmqr   FORTRAN_NAME( zunmqr, ZUNMQR )
#define lapackf77_zunmrq   FORTRAN_NAME( zunmrq, ZUNMRQ )
#define lapackf77_zunmtr   FORTRAN_NAME( zunmtr, ZUNMTR )



#ifdef ICLA_WITH_MKL
#define lapackf77_zgetrf_batch   FORTRAN_NAME( zgetrf_batch, ZGETRF_BATCH )
#endif



#define lapackf77_zbdt01   FORTRAN_NAME( zbdt01, ZBDT01 )
#define lapackf77_zget22   FORTRAN_NAME( zget22, ZGET22 )
#define lapackf77_zhet21   FORTRAN_NAME( zhet21, ZHET21 )
#define lapackf77_zhet22   FORTRAN_NAME( zhet22, ZHET22 )
#define lapackf77_zhst01   FORTRAN_NAME( zhst01, ZHST01 )
#define lapackf77_zlarfy   FORTRAN_NAME( zlarfy, ZLARFY )
#define lapackf77_zlatms   FORTRAN_NAME( zlatms, ZLATMS )
#define lapackf77_zqpt01   FORTRAN_NAME( zqpt01, ZQPT01 )
#define lapackf77_zqrt02   FORTRAN_NAME( zqrt02, ZQRT02 )
#define lapackf77_zstt21   FORTRAN_NAME( zstt21, ZSTT21 )
#define lapackf77_zunt01   FORTRAN_NAME( zunt01, ZUNT01 )

icla_int_t blasf77_izamax(
                     const icla_int_t *n,
                     const iclaDoubleComplex *x, const icla_int_t *incx );

void blasf77_zaxpy(  const icla_int_t *n,
                     const iclaDoubleComplex *alpha,
                     const iclaDoubleComplex *x, const icla_int_t *incx,
                           iclaDoubleComplex *y, const icla_int_t *incy );

void blasf77_zcopy(  const icla_int_t *n,
                     const iclaDoubleComplex *x, const icla_int_t *incx,
                           iclaDoubleComplex *y, const icla_int_t *incy );

void blasf77_zgbmv(  const char *transa,
                     const icla_int_t *m,  const icla_int_t *n,
                     const icla_int_t *kl, const icla_int_t *ku,
                     const iclaDoubleComplex *alpha,
                     const iclaDoubleComplex *A, const icla_int_t *lda,
                     const iclaDoubleComplex *x, const icla_int_t *incx,
                     const iclaDoubleComplex *beta,
                           iclaDoubleComplex *y, const icla_int_t *incy );

void blasf77_zgemm(  const char *transa, const char *transb,
                     const icla_int_t *m, const icla_int_t *n, const icla_int_t *k,
                     const iclaDoubleComplex *alpha,
                     const iclaDoubleComplex *A, const icla_int_t *lda,
                     const iclaDoubleComplex *B, const icla_int_t *ldb,
                     const iclaDoubleComplex *beta,
                           iclaDoubleComplex *C, const icla_int_t *ldc );

void blasf77_zgemv(  const char *transa,
                     const icla_int_t *m, const icla_int_t *n,
                     const iclaDoubleComplex *alpha,
                     const iclaDoubleComplex *A, const icla_int_t *lda,
                     const iclaDoubleComplex *x, const icla_int_t *incx,
                     const iclaDoubleComplex *beta,
                           iclaDoubleComplex *y, const icla_int_t *incy );

void blasf77_zgerc(  const icla_int_t *m, const icla_int_t *n,
                     const iclaDoubleComplex *alpha,
                     const iclaDoubleComplex *x, const icla_int_t *incx,
                     const iclaDoubleComplex *y, const icla_int_t *incy,
                           iclaDoubleComplex *A, const icla_int_t *lda );

void blasf77_zgeru(  const icla_int_t *m, const icla_int_t *n,
                     const iclaDoubleComplex *alpha,
                     const iclaDoubleComplex *x, const icla_int_t *incx,
                     const iclaDoubleComplex *y, const icla_int_t *incy,
                           iclaDoubleComplex *A, const icla_int_t *lda );

void blasf77_zhemm(  const char *side, const char *uplo,
                     const icla_int_t *m, const icla_int_t *n,
                     const iclaDoubleComplex *alpha,
                     const iclaDoubleComplex *A, const icla_int_t *lda,
                     const iclaDoubleComplex *B, const icla_int_t *ldb,
                     const iclaDoubleComplex *beta,
                           iclaDoubleComplex *C, const icla_int_t *ldc );

void blasf77_zhemv(  const char *uplo,
                     const icla_int_t *n,
                     const iclaDoubleComplex *alpha,
                     const iclaDoubleComplex *A, const icla_int_t *lda,
                     const iclaDoubleComplex *x, const icla_int_t *incx,
                     const iclaDoubleComplex *beta,
                           iclaDoubleComplex *y, const icla_int_t *incy );

void blasf77_zher(   const char *uplo,
                     const icla_int_t *n,
                     const double *alpha,
                     const iclaDoubleComplex *x, const icla_int_t *incx,
                           iclaDoubleComplex *A, const icla_int_t *lda );

void blasf77_zher2(  const char *uplo,
                     const icla_int_t *n,
                     const iclaDoubleComplex *alpha,
                     const iclaDoubleComplex *x, const icla_int_t *incx,
                     const iclaDoubleComplex *y, const icla_int_t *incy,
                           iclaDoubleComplex *A, const icla_int_t *lda );

void blasf77_zher2k( const char *uplo, const char *trans,
                     const icla_int_t *n, const icla_int_t *k,
                     const iclaDoubleComplex *alpha,
                     const iclaDoubleComplex *A, const icla_int_t *lda,
                     const iclaDoubleComplex *B, const icla_int_t *ldb,
                     const double *beta,
                           iclaDoubleComplex *C, const icla_int_t *ldc );

void blasf77_zherk(  const char *uplo, const char *trans,
                     const icla_int_t *n, const icla_int_t *k,
                     const double *alpha,
                     const iclaDoubleComplex *A, const icla_int_t *lda,
                     const double *beta,
                           iclaDoubleComplex *C, const icla_int_t *ldc );

void blasf77_zscal(  const icla_int_t *n,
                     const iclaDoubleComplex *alpha,
                           iclaDoubleComplex *x, const icla_int_t *incx );

void blasf77_zdscal( const icla_int_t *n,
                     const double *alpha,
                           iclaDoubleComplex *x, const icla_int_t *incx );

void blasf77_zswap(  const icla_int_t *n,
                     iclaDoubleComplex *x, const icla_int_t *incx,
                     iclaDoubleComplex *y, const icla_int_t *incy );



void blasf77_zsymm(  const char *side, const char *uplo,
                     const icla_int_t *m, const icla_int_t *n,
                     const iclaDoubleComplex *alpha,
                     const iclaDoubleComplex *A, const icla_int_t *lda,
                     const iclaDoubleComplex *B, const icla_int_t *ldb,
                     const iclaDoubleComplex *beta,
                           iclaDoubleComplex *C, const icla_int_t *ldc );

void blasf77_zsyr2k( const char *uplo, const char *trans,
                     const icla_int_t *n, const icla_int_t *k,
                     const iclaDoubleComplex *alpha,
                     const iclaDoubleComplex *A, const icla_int_t *lda,
                     const iclaDoubleComplex *B, const icla_int_t *ldb,
                     const iclaDoubleComplex *beta,
                           iclaDoubleComplex *C, const icla_int_t *ldc );

void blasf77_zsyrk(  const char *uplo, const char *trans,
                     const icla_int_t *n, const icla_int_t *k,
                     const iclaDoubleComplex *alpha,
                     const iclaDoubleComplex *A, const icla_int_t *lda,
                     const iclaDoubleComplex *beta,
                           iclaDoubleComplex *C, const icla_int_t *ldc );

void blasf77_zrotg(  iclaDoubleComplex *ca, const iclaDoubleComplex *cb,
                     double *c, iclaDoubleComplex *s );

void blasf77_zrot(   const icla_int_t *n,
                     iclaDoubleComplex *x, const icla_int_t *incx,
                     iclaDoubleComplex *y, const icla_int_t *incy,
                     const double *c, const iclaDoubleComplex *s );

void blasf77_zdrot(  const icla_int_t *n,
                     iclaDoubleComplex *x, const icla_int_t *incx,
                     iclaDoubleComplex *y, const icla_int_t *incy,
                     const double *c, const double *s );

void blasf77_ztrmm(  const char *side, const char *uplo, const char *transa, const char *diag,
                     const icla_int_t *m, const icla_int_t *n,
                     const iclaDoubleComplex *alpha,
                     const iclaDoubleComplex *A, const icla_int_t *lda,
                           iclaDoubleComplex *B, const icla_int_t *ldb );

void blasf77_ztrmv(  const char *uplo, const char *transa, const char *diag,
                     const icla_int_t *n,
                     const iclaDoubleComplex *A, const icla_int_t *lda,
                           iclaDoubleComplex *x, const icla_int_t *incx );

void blasf77_ztrsm(  const char *side, const char *uplo, const char *transa, const char *diag,
                     const icla_int_t *m, const icla_int_t *n,
                     const iclaDoubleComplex *alpha,
                     const iclaDoubleComplex *A, const icla_int_t *lda,
                           iclaDoubleComplex *B, const icla_int_t *ldb );

void blasf77_ztrsv(  const char *uplo, const char *transa, const char *diag,
                     const icla_int_t *n,
                     const iclaDoubleComplex *A, const icla_int_t *lda,
                           iclaDoubleComplex *x, const icla_int_t *incx );


double icla_cblas_dzasum(
    icla_int_t n,
    const iclaDoubleComplex *x, icla_int_t incx );

double icla_cblas_dznrm2(
    icla_int_t n,
    const iclaDoubleComplex *x, icla_int_t incx );

iclaDoubleComplex icla_cblas_zdotc(
    icla_int_t n,
    const iclaDoubleComplex *x, icla_int_t incx,
    const iclaDoubleComplex *y, icla_int_t incy );

iclaDoubleComplex icla_cblas_zdotu(
    icla_int_t n,
    const iclaDoubleComplex *x, icla_int_t incx,
    const iclaDoubleComplex *y, icla_int_t incy );


#ifdef ICLA_REAL
void   lapackf77_dbdsdc( const char *uplo, const char *compq,
                         const icla_int_t *n,
                         double *d, double *e,
                         double *U,  const icla_int_t *ldu,
                         double *VT, const icla_int_t *ldvt,
                         double *Q, icla_int_t *IQ,
                         double *work, icla_int_t *iwork,
                         icla_int_t *info );
#endif

void   lapackf77_zbdsqr( const char *uplo,
                         const icla_int_t *n, const icla_int_t *ncvt, const icla_int_t *nru,  const icla_int_t *ncc,
                         double *d, double *e,
                         iclaDoubleComplex *Vt, const icla_int_t *ldvt,
                         iclaDoubleComplex *U, const icla_int_t *ldu,
                         iclaDoubleComplex *C, const icla_int_t *ldc,
                         double *work,
                         icla_int_t *info );

void   lapackf77_zgbtrf( const icla_int_t  *m,  const icla_int_t *n,
                         const icla_int_t  *kl, const icla_int_t *ku,
                         iclaDoubleComplex *AB, const icla_int_t *ldab,
                         icla_int_t *ipiv,
                         icla_int_t *info );

void   lapackf77_zgebak( const char *job, const char *side,
                         const icla_int_t *n,
                         const icla_int_t *ilo, const icla_int_t *ihi,
                         const double *scale, const icla_int_t *m,
                         iclaDoubleComplex *V, const icla_int_t *ldv,
                         icla_int_t *info );

void   lapackf77_zgebal( const char *job,
                         const icla_int_t *n,
                         iclaDoubleComplex *A, const icla_int_t *lda,
                         icla_int_t *ilo, icla_int_t *ihi,
                         double *scale,
                         icla_int_t *info );

void   lapackf77_zgebd2( const icla_int_t *m, const icla_int_t *n,
                         iclaDoubleComplex *A, const icla_int_t *lda,
                         double *d, double *e,
                         iclaDoubleComplex *tauq,
                         iclaDoubleComplex *taup,
                         iclaDoubleComplex *work,
                         icla_int_t *info );

void   lapackf77_zgebrd( const icla_int_t *m, const icla_int_t *n,
                         iclaDoubleComplex *A, const icla_int_t *lda,
                         double *d, double *e,
                         iclaDoubleComplex *tauq,
                         iclaDoubleComplex *taup,
                         iclaDoubleComplex *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_zgbbrd( const char *vect, const icla_int_t *m,
                         const icla_int_t *n, const icla_int_t *ncc,
                         const icla_int_t *kl, const icla_int_t *ku,
                         iclaDoubleComplex *Ab, const icla_int_t *ldab,
                         double *d, double *e,
                         iclaDoubleComplex *Q, const icla_int_t *ldq,
                         iclaDoubleComplex *PT, const icla_int_t *ldpt,
                         iclaDoubleComplex *C, const icla_int_t *ldc,
                         iclaDoubleComplex *work,
                         #ifdef ICLA_COMPLEX
                         double *rwork,
                         #endif
                         icla_int_t *info );

void   lapackf77_zgbsv( const icla_int_t *n,
                        const icla_int_t *kl, const icla_int_t *ku,
                        const icla_int_t *nrhs,
                        iclaDoubleComplex *ab, const icla_int_t *ldab,
                        icla_int_t *ipiv,
                        iclaDoubleComplex *B, const icla_int_t *ldb,
                        icla_int_t *info );

void   lapackf77_zgbtrs( const char *trans,
                         const icla_int_t *n,
                         const icla_int_t *kl, const icla_int_t *ku,
                         const icla_int_t *nrhs,
                         iclaDoubleComplex *ab, const icla_int_t *ldab,
                         icla_int_t *ipiv,
                         iclaDoubleComplex *B, const icla_int_t *ldb,
                         icla_int_t *info );

void   lapackf77_zgeev(  const char *jobvl, const char *jobvr,
                         const icla_int_t *n,
                         iclaDoubleComplex *A,    const icla_int_t *lda,
                         #ifdef ICLA_COMPLEX
                         iclaDoubleComplex *w,
                         #else
                         double *wr, double *wi,
                         #endif
                         iclaDoubleComplex *Vl,   const icla_int_t *ldvl,
                         iclaDoubleComplex *Vr,   const icla_int_t *ldvr,
                         iclaDoubleComplex *work, const icla_int_t *lwork,
                         #ifdef ICLA_COMPLEX
                         double *rwork,
                         #endif
                         icla_int_t *info );

void   lapackf77_zgehd2( const icla_int_t *n,
                         const icla_int_t *ilo, const icla_int_t *ihi,
                         iclaDoubleComplex *A, const icla_int_t *lda,
                         iclaDoubleComplex *tau,
                         iclaDoubleComplex *work,
                         icla_int_t *info );

void   lapackf77_zgehrd( const icla_int_t *n,
                         const icla_int_t *ilo, const icla_int_t *ihi,
                         iclaDoubleComplex *A, const icla_int_t *lda,
                         iclaDoubleComplex *tau,
                         iclaDoubleComplex *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_zgelqf( const icla_int_t *m, const icla_int_t *n,
                         iclaDoubleComplex *A, const icla_int_t *lda,
                         iclaDoubleComplex *tau,
                         iclaDoubleComplex *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_zgels(  const char *trans,
                         const icla_int_t *m, const icla_int_t *n, const icla_int_t *nrhs,
                         iclaDoubleComplex *A, const icla_int_t *lda,
                         iclaDoubleComplex *B, const icla_int_t *ldb,
                         iclaDoubleComplex *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_zgeqlf( const icla_int_t *m, const icla_int_t *n,
                         iclaDoubleComplex *A, const icla_int_t *lda,
                         iclaDoubleComplex *tau,
                         iclaDoubleComplex *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_zgeqp3( const icla_int_t *m, const icla_int_t *n,
                         iclaDoubleComplex *A, const icla_int_t *lda,
                         icla_int_t *jpvt,
                         iclaDoubleComplex *tau,
                         iclaDoubleComplex *work, const icla_int_t *lwork,
                         #ifdef ICLA_COMPLEX
                         double *rwork,
                         #endif
                         icla_int_t *info );

void   lapackf77_zgeqrf( const icla_int_t *m, const icla_int_t *n,
                         iclaDoubleComplex *A, const icla_int_t *lda,
                         iclaDoubleComplex *tau,
                         iclaDoubleComplex *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_zgerqf( const icla_int_t *m, const icla_int_t *n,
                         iclaDoubleComplex *A, const icla_int_t *lda,
                         iclaDoubleComplex *tau,
                         iclaDoubleComplex *work, const icla_int_t *lwork,
                         icla_int_t *info);

void   lapackf77_zgesdd( const char *jobz,
                         const icla_int_t *m, const icla_int_t *n,
                         iclaDoubleComplex *A, const icla_int_t *lda,
                         double *s,
                         iclaDoubleComplex *U,  const icla_int_t *ldu,
                         iclaDoubleComplex *Vt, const icla_int_t *ldvt,
                         iclaDoubleComplex *work, const icla_int_t *lwork,
                         #ifdef ICLA_COMPLEX
                         double *rwork,
                         #endif
                         icla_int_t *iwork,
                         icla_int_t *info );

void   lapackf77_zgesv(  const icla_int_t *n, const icla_int_t *nrhs,
                         iclaDoubleComplex *A, const icla_int_t *lda,
                         icla_int_t *ipiv,
                         iclaDoubleComplex *B,  const icla_int_t *ldb,
                         icla_int_t *info );

void   lapackf77_zgesvd( const char *jobu, const char *jobvt,
                         const icla_int_t *m, const icla_int_t *n,
                         iclaDoubleComplex *A, const icla_int_t *lda,
                         double *s,
                         iclaDoubleComplex *U,  const icla_int_t *ldu,
                         iclaDoubleComplex *Vt, const icla_int_t *ldvt,
                         iclaDoubleComplex *work, const icla_int_t *lwork,
                         #ifdef ICLA_COMPLEX
                         double *rwork,
                         #endif
                         icla_int_t *info );

void   lapackf77_zgetrf( const icla_int_t *m, const icla_int_t *n,
                         iclaDoubleComplex *A, const icla_int_t *lda,
                         icla_int_t *ipiv,
                         icla_int_t *info );

void   lapackf77_zgetri( const icla_int_t *n,
                         iclaDoubleComplex *A, const icla_int_t *lda,
                         const icla_int_t *ipiv,
                         iclaDoubleComplex *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_zgetrs( const char *trans,
                         const icla_int_t *n, const icla_int_t *nrhs,
                         const iclaDoubleComplex *A, const icla_int_t *lda,
                         const icla_int_t *ipiv,
                         iclaDoubleComplex *B, const icla_int_t *ldb,
                         icla_int_t *info );

void   lapackf77_zgglse( icla_int_t *m, icla_int_t *n, icla_int_t *p,
                         iclaDoubleComplex *A, icla_int_t *lda,
                         iclaDoubleComplex *B, icla_int_t *ldb,
                         iclaDoubleComplex *c, iclaDoubleComplex *d,
                         iclaDoubleComplex *x,
                         iclaDoubleComplex *work, icla_int_t *lwork,
                         icla_int_t *info);

void   lapackf77_zggrqf( icla_int_t *m, icla_int_t *p, icla_int_t *n,
                         iclaDoubleComplex *A, icla_int_t *lda,
                         iclaDoubleComplex *tauA, iclaDoubleComplex *B,
                         icla_int_t *ldb, iclaDoubleComplex *tauB,
                         iclaDoubleComplex *work, icla_int_t *lwork,
                         icla_int_t *info);

void   lapackf77_zhetf2( const char *uplo, const icla_int_t *n,
                         iclaDoubleComplex *A, const icla_int_t *lda,
                         icla_int_t *ipiv,
                         icla_int_t *info );

void   lapackf77_zhetrs( const char *uplo,
                         const icla_int_t *n, const icla_int_t *nrhs,
                         const iclaDoubleComplex *A, const icla_int_t *lda,
                         const icla_int_t *ipiv,
                         iclaDoubleComplex *B, const icla_int_t *ldb,
                         icla_int_t *info );

void   lapackf77_zhbtrd( const char *vect, const char *uplo,
                         const icla_int_t *n, const icla_int_t *kd,
                         iclaDoubleComplex *Ab, const icla_int_t *ldab,
                         double *d, double *e,
                         iclaDoubleComplex *Q, const icla_int_t *ldq,
                         iclaDoubleComplex *work,
                         icla_int_t *info );

void   lapackf77_zheev(  const char *jobz, const char *uplo,
                         const icla_int_t *n,
                         iclaDoubleComplex *A, const icla_int_t *lda,
                         double *w,
                         iclaDoubleComplex *work, const icla_int_t *lwork,
                         #ifdef ICLA_COMPLEX
                         double *rwork,
                         #endif
                         icla_int_t *info );

void   lapackf77_zheevd( const char *jobz, const char *uplo,
                         const icla_int_t *n,
                         iclaDoubleComplex *A, const icla_int_t *lda,
                         double *w,
                         iclaDoubleComplex *work, const icla_int_t *lwork,
                         #ifdef ICLA_COMPLEX
                         double *rwork, const icla_int_t *lrwork,
                         #endif
                         icla_int_t *iwork, const icla_int_t *liwork,
                         icla_int_t *info );

void   lapackf77_zheevr( const char *jobz, const char *range, const char *uplo,
                         const icla_int_t *n,
                         iclaDoubleComplex *A, const icla_int_t *lda,
                         const double *vl, const double *vu,
                         const icla_int_t *il, const icla_int_t *iu,
                         const double *abstol,
                         icla_int_t *m, double *w,
                         iclaDoubleComplex *Z, const icla_int_t *ldz,
                         icla_int_t *isuppz,
                         iclaDoubleComplex *work, const icla_int_t *lwork,
                         #ifdef ICLA_COMPLEX
                         double *rwork, const icla_int_t *lrwork,
                         #endif
                         icla_int_t *iwork, const icla_int_t *liwork,
                         icla_int_t *info);

void   lapackf77_zheevx( const char *jobz, const char *range, const char *uplo,
                         const icla_int_t *n,
                         iclaDoubleComplex *A, const icla_int_t *lda,
                         const double *vl, const double *vu,
                         const icla_int_t *il, const icla_int_t *iu,
                         const double *abstol,
                         icla_int_t *m, double *w,
                         iclaDoubleComplex *Z, const icla_int_t *ldz,
                         iclaDoubleComplex *work, const icla_int_t *lwork,
                         #ifdef ICLA_COMPLEX
                         double *rwork,
                         #endif
                         icla_int_t *iwork, icla_int_t *ifail,
                         icla_int_t *info);

void   lapackf77_zhegs2( const icla_int_t *itype, const char *uplo,
                         const icla_int_t *n,
                         iclaDoubleComplex *A, const icla_int_t *lda,
                         const iclaDoubleComplex *B, const icla_int_t *ldb,
                         icla_int_t *info );

void   lapackf77_zhegst( const icla_int_t *itype, const char *uplo,
                         const icla_int_t *n,
                         iclaDoubleComplex *A, const icla_int_t *lda,
                         const iclaDoubleComplex *B, const icla_int_t *ldb,
                         icla_int_t *info );

void   lapackf77_zhegvd( const icla_int_t *itype, const char *jobz, const char *uplo,
                         const icla_int_t *n,
                         iclaDoubleComplex *A, const icla_int_t *lda,
                         iclaDoubleComplex *B, const icla_int_t *ldb,
                         double *w,
                         iclaDoubleComplex *work, const icla_int_t *lwork,
                         #ifdef ICLA_COMPLEX
                         double *rwork, const icla_int_t *lrwork,
                         #endif
                         icla_int_t *iwork, const icla_int_t *liwork,
                         icla_int_t *info );

void   lapackf77_zhesv( const char *uplo,
                        const icla_int_t *n, const icla_int_t *nrhs,
                        iclaDoubleComplex *A, const icla_int_t *lda, icla_int_t *ipiv,
                        iclaDoubleComplex *B, const icla_int_t *ldb,
                        iclaDoubleComplex *work, const icla_int_t *lwork,
                        icla_int_t *info );

void   lapackf77_zhetd2( const char *uplo,
                         const icla_int_t *n,
                         iclaDoubleComplex *A, const icla_int_t *lda,
                         double *d, double *e,
                         iclaDoubleComplex *tau,
                         icla_int_t *info );

void   lapackf77_zhetrd( const char *uplo,
                         const icla_int_t *n,
                         iclaDoubleComplex *A, const icla_int_t *lda,
                         double *d, double *e,
                         iclaDoubleComplex *tau,
                         iclaDoubleComplex *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_zhetrf( const char *uplo,
                         const icla_int_t *n,
                         iclaDoubleComplex *A, const icla_int_t *lda,
                         icla_int_t *ipiv,
                         iclaDoubleComplex *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_zhseqr( const char *job, const char *compz,
                         const icla_int_t *n,
                         const icla_int_t *ilo, const icla_int_t *ihi,
                         iclaDoubleComplex *H, const icla_int_t *ldh,
                         #ifdef ICLA_COMPLEX
                         iclaDoubleComplex *w,
                         #else
                         double *wr, double *wi,
                         #endif
                         iclaDoubleComplex *Z, const icla_int_t *ldz,
                         iclaDoubleComplex *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_zlabrd( const icla_int_t *m, const icla_int_t *n, const icla_int_t *nb,
                         iclaDoubleComplex *A, const icla_int_t *lda,
                         double *d, double *e,
                         iclaDoubleComplex *tauq,
                         iclaDoubleComplex *taup,
                         iclaDoubleComplex *X, const icla_int_t *ldx,
                         iclaDoubleComplex *Y, const icla_int_t *ldy );

#ifdef ICLA_COMPLEX
void   lapackf77_zlacgv( const icla_int_t *n,
                         iclaDoubleComplex *x, const icla_int_t *incx );
#endif

#ifdef ICLA_COMPLEX
void   lapackf77_zlacp2( const char *uplo,
                         const icla_int_t *m, const icla_int_t *n,
                         const double *A, const icla_int_t *lda,
                         iclaDoubleComplex *B, const icla_int_t *ldb );
#endif

void   lapackf77_zlacpy( const char *uplo,
                         const icla_int_t *m, const icla_int_t *n,
                         const iclaDoubleComplex *A, const icla_int_t *lda,
                         iclaDoubleComplex *B, const icla_int_t *ldb );

#ifdef ICLA_COMPLEX
void   lapackf77_zlacrm( const icla_int_t *m, const icla_int_t *n,
                         const iclaDoubleComplex *A, const icla_int_t *lda,
                         const double             *B, const icla_int_t *ldb,
                         iclaDoubleComplex       *C, const icla_int_t *ldc,
                         double *rwork );
#endif

#ifdef ICLA_COMPLEX
void   lapackf77_zladiv( iclaDoubleComplex *ret_val,
                         const iclaDoubleComplex *x,
                         const iclaDoubleComplex *y );
#else
void   lapackf77_zladiv( const double *a, const double *b,
                         const double *c, const double *d,
                         double *p, double *q );
#endif

void   lapackf77_zlahef( const char *uplo,
                         const icla_int_t *n, const icla_int_t *nb,
                         icla_int_t *kb,
                         iclaDoubleComplex *A, const icla_int_t *lda,
                         icla_int_t *ipiv,
                         iclaDoubleComplex *work, const icla_int_t *ldwork,
                         icla_int_t *info );

double lapackf77_zlangb( const char *norm,
                         const icla_int_t *n, const icla_int_t *kl, const icla_int_t *ku,
                         const iclaDoubleComplex *AB, const icla_int_t *ldab,
                         double *work );

double lapackf77_zlange( const char *norm,
                         const icla_int_t *m, const icla_int_t *n,
                         const iclaDoubleComplex *A, const icla_int_t *lda,
                         double *work );

double lapackf77_zlanhe( const char *norm, const char *uplo,
                         const icla_int_t *n,
                         const iclaDoubleComplex *A, const icla_int_t *lda,
                         double *work );

double lapackf77_zlanht( const char *norm, const icla_int_t *n,
                         const double *d, const iclaDoubleComplex *e );

double lapackf77_zlansy( const char *norm, const char *uplo,
                         const icla_int_t *n,
                         const iclaDoubleComplex *A, const icla_int_t *lda,
                         double *work );

double lapackf77_zlantr( const char *norm, const char *uplo, const char *diag,
                         const icla_int_t *m, const icla_int_t *n,
                         const iclaDoubleComplex *A, const icla_int_t *lda,
                         double *work );

void   lapackf77_zlaqp2( const icla_int_t *m, const icla_int_t *n, const icla_int_t *offset,
                         iclaDoubleComplex *A, const icla_int_t *lda,
                         icla_int_t *jpvt,
                         iclaDoubleComplex *tau,
                         double *vn1, double *vn2,
                         iclaDoubleComplex *work );

#ifdef ICLA_COMPLEX
void   lapackf77_zlarcm( const icla_int_t *m, const icla_int_t *n,
                         const double             *A, const icla_int_t *lda,
                         const iclaDoubleComplex *B, const icla_int_t *ldb,
                         iclaDoubleComplex       *C, const icla_int_t *ldc,
                         double *rwork );
#endif

void   lapackf77_zlarf(  const char *side, const icla_int_t *m, const icla_int_t *n,
                         const iclaDoubleComplex *v, const icla_int_t *incv,
                         const iclaDoubleComplex *tau,
                         iclaDoubleComplex *C, const icla_int_t *ldc,
                         iclaDoubleComplex *work );

void   lapackf77_zlarfb( const char *side, const char *trans, const char *direct, const char *storev,
                         const icla_int_t *m, const icla_int_t *n, const icla_int_t *k,
                         const iclaDoubleComplex *V, const icla_int_t *ldv,
                         const iclaDoubleComplex *T, const icla_int_t *ldt,
                         iclaDoubleComplex *C, const icla_int_t *ldc,
                         iclaDoubleComplex *work, const icla_int_t *ldwork );

void   lapackf77_zlarfg( const icla_int_t *n,
                         iclaDoubleComplex *alpha,
                         iclaDoubleComplex *x, const icla_int_t *incx,
                         iclaDoubleComplex *tau );

void   lapackf77_zlarft( const char *direct, const char *storev,
                         const icla_int_t *n, const icla_int_t *k,
                         const iclaDoubleComplex *V, const icla_int_t *ldv,
                         const iclaDoubleComplex *tau,
                         iclaDoubleComplex *T, const icla_int_t *ldt );

void   lapackf77_zlarfx( const char *side, const icla_int_t *m, const icla_int_t *n,
                         const iclaDoubleComplex *V,
                         const iclaDoubleComplex *tau,
                         iclaDoubleComplex *C, const icla_int_t *ldc,
                         iclaDoubleComplex *work );

void   lapackf77_zlarnv( const icla_int_t *idist, icla_int_t *iseed, const icla_int_t *n,
                         iclaDoubleComplex *x );

void   lapackf77_zlartg( const iclaDoubleComplex *f,
                         const iclaDoubleComplex *g,
                         double *cs,
                         iclaDoubleComplex *sn,
                         iclaDoubleComplex *r );

void   lapackf77_zlascl( const char *type,
                         const icla_int_t *kl, const icla_int_t *ku,
                         const double *cfrom,
                         const double *cto,
                         const icla_int_t *m, const icla_int_t *n,
                         iclaDoubleComplex *A, const icla_int_t *lda,
                         icla_int_t *info );

void   lapackf77_zlaset( const char *uplo,
                         const icla_int_t *m, const icla_int_t *n,
                         const iclaDoubleComplex *alpha,
                         const iclaDoubleComplex *beta,
                         iclaDoubleComplex *A, const icla_int_t *lda );

void   lapackf77_zlaswp( const icla_int_t *n,
                         iclaDoubleComplex *A, const icla_int_t *lda,
                         const icla_int_t *k1, const icla_int_t *k2,
                         const icla_int_t *ipiv,
                         const icla_int_t *incx );

void   lapackf77_zlatrd( const char *uplo,
                         const icla_int_t *n, const icla_int_t *nb,
                         iclaDoubleComplex *A, const icla_int_t *lda,
                         double *e,
                         iclaDoubleComplex *tau,
                         iclaDoubleComplex *work, const icla_int_t *ldwork );

void   lapackf77_zlatrs( const char *uplo, const char *trans, const char *diag,
                         const char *normin,
                         const icla_int_t *n,
                         const iclaDoubleComplex *A, const icla_int_t *lda,
                         iclaDoubleComplex *x, double *scale,
                         double *cnorm,
                         icla_int_t *info );

void   lapackf77_zlauum( const char *uplo,
                         const icla_int_t *n,
                         iclaDoubleComplex *A, const icla_int_t *lda,
                         icla_int_t *info );

void   lapackf77_zlavhe( const char *uplo, const char *trans, const char *diag,
                         icla_int_t *n, icla_int_t *nrhs,
                         iclaDoubleComplex *A, icla_int_t *lda,
                         icla_int_t *ipiv,
                         iclaDoubleComplex *B, icla_int_t *ldb,
                         icla_int_t *info );

void   lapackf77_zposv(  const char *uplo,
                         const icla_int_t *n, const icla_int_t *nrhs,
                         iclaDoubleComplex *A, const icla_int_t *lda,
                         iclaDoubleComplex *B,  const icla_int_t *ldb,
                         icla_int_t *info );

void   lapackf77_zpotrf( const char *uplo,
                         const icla_int_t *n,
                         iclaDoubleComplex *A, const icla_int_t *lda,
                         icla_int_t *info );

void   lapackf77_zpotri( const char *uplo,
                         const icla_int_t *n,
                         iclaDoubleComplex *A, const icla_int_t *lda,
                         icla_int_t *info );

void   lapackf77_zpotrs( const char *uplo,
                         const icla_int_t *n, const icla_int_t *nrhs,
                         const iclaDoubleComplex *A, const icla_int_t *lda,
                         iclaDoubleComplex *B, const icla_int_t *ldb,
                         icla_int_t *info );

void   lapackf77_zstedc( const char *compz,
                         const icla_int_t *n,
                         double *d, double *e,
                         iclaDoubleComplex *Z, const icla_int_t *ldz,
                         iclaDoubleComplex *work, const icla_int_t *lwork,
                         #ifdef ICLA_COMPLEX
                         double *rwork, const icla_int_t *lrwork,
                         #endif
                         icla_int_t *iwork, const icla_int_t *liwork,
                         icla_int_t *info );

void   lapackf77_zstein( const icla_int_t *n,
                         const double *d, const double *e,
                         const icla_int_t *m,
                         const double *w,
                         const icla_int_t *iblock,
                         const icla_int_t *isplit,
                         iclaDoubleComplex *Z, const icla_int_t *ldz,
                         double *work, icla_int_t *iwork, icla_int_t *ifailv,
                         icla_int_t *info );

void   lapackf77_zstemr( const char *jobz, const char *range,
                         const icla_int_t *n,
                         double *d, double *e,
                         const double *vl, const double *vu,
                         const icla_int_t *il, const icla_int_t *iu,
                         icla_int_t *m,
                         double *w,
                         iclaDoubleComplex *Z, const icla_int_t *ldz,
                         const icla_int_t *nzc, icla_int_t *isuppz, icla_int_t *tryrac,
                         double *work, const icla_int_t *lwork,
                         icla_int_t *iwork, const icla_int_t *liwork,
                         icla_int_t *info );

void   lapackf77_zsteqr( const char *compz,
                         const icla_int_t *n,
                         double *d, double *e,
                         iclaDoubleComplex *Z, const icla_int_t *ldz,
                         double *work,
                         icla_int_t *info );

#ifdef ICLA_COMPLEX
void   lapackf77_zsymv(  const char *uplo,
                         const icla_int_t *n,
                         const iclaDoubleComplex *alpha,
                         const iclaDoubleComplex *A, const icla_int_t *lda,
                         const iclaDoubleComplex *x, const icla_int_t *incx,
                         const iclaDoubleComplex *beta,
                               iclaDoubleComplex *y, const icla_int_t *incy );

void   lapackf77_zsyr(   const char *uplo,
                         const icla_int_t *n,
                         const iclaDoubleComplex *alpha,
                         const iclaDoubleComplex *x, const icla_int_t *incx,
                               iclaDoubleComplex *A, const icla_int_t *lda );

void   lapackf77_zsysv(  const char *uplo,
                         const icla_int_t *n, const icla_int_t *nrhs,
                         iclaDoubleComplex *A, const icla_int_t *lda, icla_int_t *ipiv,
                         iclaDoubleComplex *B, const icla_int_t *ldb,
                         iclaDoubleComplex *work, const icla_int_t *lwork,
                         icla_int_t *info );

#endif

void   lapackf77_ztrevc( const char *side, const char *howmny,

                         #ifdef ICLA_COMPLEX
                         const
                         #endif
                         icla_int_t *select, const icla_int_t *n,

                         #ifdef ICLA_REAL
                         const
                         #endif
                         iclaDoubleComplex *T,  const icla_int_t *ldt,
                         iclaDoubleComplex *Vl, const icla_int_t *ldvl,
                         iclaDoubleComplex *Vr, const icla_int_t *ldvr,
                         const icla_int_t *mm, icla_int_t *m,
                         iclaDoubleComplex *work,
                         #ifdef ICLA_COMPLEX
                         double *rwork,
                         #endif
                         icla_int_t *info );

void   lapackf77_ztrevc3( const char *side, const char *howmny,
                          icla_int_t *select, const icla_int_t *n,
                          iclaDoubleComplex *T,  const icla_int_t *ldt,
                          iclaDoubleComplex *VL, const icla_int_t *ldvl,
                          iclaDoubleComplex *VR, const icla_int_t *ldvr,
                          const icla_int_t *mm,
                          const icla_int_t *mout,
                          iclaDoubleComplex *work, const icla_int_t *lwork,
                          #ifdef ICLA_COMPLEX
                          double *rwork,
                          #endif
                          icla_int_t *info );

void   lapackf77_ztrtri( const char *uplo, const char *diag,
                         const icla_int_t *n,
                         iclaDoubleComplex *A, const icla_int_t *lda,
                         icla_int_t *info );

void   lapackf77_zung2r( const icla_int_t *m, const icla_int_t *n, const icla_int_t *k,
                         iclaDoubleComplex *A, const icla_int_t *lda,
                         const iclaDoubleComplex *tau,
                         iclaDoubleComplex *work,
                         icla_int_t *info );

void   lapackf77_zungbr( const char *vect,
                         const icla_int_t *m, const icla_int_t *n, const icla_int_t *k,
                         iclaDoubleComplex *A, const icla_int_t *lda,
                         const iclaDoubleComplex *tau,
                         iclaDoubleComplex *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_zunghr( const icla_int_t *n,
                         const icla_int_t *ilo, const icla_int_t *ihi,
                         iclaDoubleComplex *A, const icla_int_t *lda,
                         const iclaDoubleComplex *tau,
                         iclaDoubleComplex *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_zunglq( const icla_int_t *m, const icla_int_t *n, const icla_int_t *k,
                         iclaDoubleComplex *A, const icla_int_t *lda,
                         const iclaDoubleComplex *tau,
                         iclaDoubleComplex *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_zungql( const icla_int_t *m, const icla_int_t *n, const icla_int_t *k,
                         iclaDoubleComplex *A, const icla_int_t *lda,
                         const iclaDoubleComplex *tau,
                         iclaDoubleComplex *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_zungqr( const icla_int_t *m, const icla_int_t *n, const icla_int_t *k,
                         iclaDoubleComplex *A, const icla_int_t *lda,
                         const iclaDoubleComplex *tau,
                         iclaDoubleComplex *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_zungtr( const char *uplo,
                         const icla_int_t *n,
                         iclaDoubleComplex *A, const icla_int_t *lda,
                         const iclaDoubleComplex *tau,
                         iclaDoubleComplex *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_zunm2r( const char *side, const char *trans,
                         const icla_int_t *m, const icla_int_t *n, const icla_int_t *k,
                         const iclaDoubleComplex *A, const icla_int_t *lda,
                         const iclaDoubleComplex *tau,
                         iclaDoubleComplex *C, const icla_int_t *ldc,
                         iclaDoubleComplex *work,
                         icla_int_t *info );

void   lapackf77_zunmbr( const char *vect, const char *side, const char *trans,
                         const icla_int_t *m, const icla_int_t *n, const icla_int_t *k,
                         const iclaDoubleComplex *A, const icla_int_t *lda,
                         const iclaDoubleComplex *tau,
                         iclaDoubleComplex *C, const icla_int_t *ldc,
                         iclaDoubleComplex *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_zunmlq( const char *side, const char *trans,
                         const icla_int_t *m, const icla_int_t *n, const icla_int_t *k,
                         const iclaDoubleComplex *A, const icla_int_t *lda,
                         const iclaDoubleComplex *tau,
                         iclaDoubleComplex *C, const icla_int_t *ldc,
                         iclaDoubleComplex *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_zunmql( const char *side, const char *trans,
                         const icla_int_t *m, const icla_int_t *n, const icla_int_t *k,
                         const iclaDoubleComplex *A, const icla_int_t *lda,
                         const iclaDoubleComplex *tau,
                         iclaDoubleComplex *C, const icla_int_t *ldc,
                         iclaDoubleComplex *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_zunmqr( const char *side, const char *trans,
                         const icla_int_t *m, const icla_int_t *n, const icla_int_t *k,
                         const iclaDoubleComplex *A, const icla_int_t *lda,
                         const iclaDoubleComplex *tau,
                         iclaDoubleComplex *C, const icla_int_t *ldc,
                         iclaDoubleComplex *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_zunmrq( const char *side, const char *trans,
                         icla_int_t *m, icla_int_t *n, icla_int_t *k,
                         iclaDoubleComplex *A, icla_int_t *lda,
                         iclaDoubleComplex *tau,
                         iclaDoubleComplex *C, icla_int_t *ldc,
                         iclaDoubleComplex *work, icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_zunmtr( const char *side, const char *uplo, const char *trans,
                         const icla_int_t *m, const icla_int_t *n,
                         const iclaDoubleComplex *A, const icla_int_t *lda,
                         const iclaDoubleComplex *tau,
                         iclaDoubleComplex *C, const icla_int_t *ldc,
                         iclaDoubleComplex *work, const icla_int_t *lwork,
                         icla_int_t *info );


void   lapackf77_dstebz( const char *range, const char *order,
                         const icla_int_t *n,
                         const double *vl, const double *vu,
                         const icla_int_t *il, const icla_int_t *iu,
                         const double *abstol,
                         const double *d, const double *e,
                         icla_int_t *m, icla_int_t *nsplit,
                         double *w,
                         icla_int_t *iblock, icla_int_t *isplit,
                         double *work,
                         icla_int_t *iwork,
                         icla_int_t *info );

void   lapackf77_dlaln2( const icla_int_t *ltrans,
                         const icla_int_t *na, const icla_int_t *nw,
                         const double *smin, const double *ca,
                         const double *a,  const icla_int_t *lda,
                         const double *d1, const double *d2,
                         const double *b,  const icla_int_t *ldb,
                         const double *wr, const double *wi,
                         double *x, const icla_int_t *ldx,
                         double *scale, double *xnorm,
                         icla_int_t *info );

double lapackf77_dlamc3( const double *a, const double *b );

void   lapackf77_dlamrg( const icla_int_t *n1, const icla_int_t *n2,
                         const double *a,
                         const icla_int_t *dtrd1, const icla_int_t *dtrd2,
                         icla_int_t *index );

double lapackf77_dlapy3( const double *x, const double *y, const double *z );

void   lapackf77_dlaed2( icla_int_t *k, const icla_int_t *n, const icla_int_t *n1,
                         double *d,
                         double *q, const icla_int_t *ldq,
                         icla_int_t *indxq,
                         double *rho, const double *z,
                         double *dlamda, double *w, double *q2,
                         icla_int_t *indx, icla_int_t *indxc, icla_int_t *indxp,
                         icla_int_t *coltyp,
                         icla_int_t *info);

void   lapackf77_dlaed4( const icla_int_t *n, const icla_int_t *i,
                         const double *d,
                         const double *z,
                         double *delta,
                         const double *rho,
                         double *dlam,
                         icla_int_t *info );

void   lapackf77_dlasrt( const char *id, const icla_int_t *n, double *d,
                         icla_int_t *info );

void   lapackf77_zbdt01( const icla_int_t *m, const icla_int_t *n, const icla_int_t *kd,
                         iclaDoubleComplex *A, const icla_int_t *lda,
                         iclaDoubleComplex *Q, const icla_int_t *ldq,
                         double *d, double *e,
                         iclaDoubleComplex *Pt, const icla_int_t *ldpt,
                         iclaDoubleComplex *work,
                         #ifdef ICLA_COMPLEX
                         double *rwork,
                         #endif
                         double *resid );

void   lapackf77_zget22( const char *transa, const char *transe, const char *transw, const icla_int_t *n,
                         iclaDoubleComplex *A, const icla_int_t *lda,
                         iclaDoubleComplex *E, const icla_int_t *lde,
                         #ifdef ICLA_COMPLEX
                         iclaDoubleComplex *w,
                         #else
                         iclaDoubleComplex *wr,
                         iclaDoubleComplex *wi,
                         #endif
                         iclaDoubleComplex *work,
                         #ifdef ICLA_COMPLEX
                         double *rwork,
                         #endif
                         double *result );

void   lapackf77_zhet21( const icla_int_t *itype, const char *uplo,
                         const icla_int_t *n, const icla_int_t *kband,
                         iclaDoubleComplex *A, const icla_int_t *lda,
                         double *d, double *e,
                         iclaDoubleComplex *U, const icla_int_t *ldu,
                         iclaDoubleComplex *V, const icla_int_t *ldv,
                         iclaDoubleComplex *tau,
                         iclaDoubleComplex *work,
                         #ifdef ICLA_COMPLEX
                         double *rwork,
                         #endif
                         double *result );

void   lapackf77_zhet22( const icla_int_t *itype, const char *uplo,
                         const icla_int_t *n, const icla_int_t *m, const icla_int_t *kband,
                         iclaDoubleComplex *A, const icla_int_t *lda,
                         double *d, double *e,
                         iclaDoubleComplex *U, const icla_int_t *ldu,
                         iclaDoubleComplex *V, const icla_int_t *ldv,
                         iclaDoubleComplex *tau,
                         iclaDoubleComplex *work,
                         #ifdef ICLA_COMPLEX
                         double *rwork,
                         #endif
                         double *result );

void   lapackf77_zhst01( const icla_int_t *n, const icla_int_t *ilo, const icla_int_t *ihi,
                         iclaDoubleComplex *A, const icla_int_t *lda,
                         iclaDoubleComplex *H, const icla_int_t *ldh,
                         iclaDoubleComplex *Q, const icla_int_t *ldq,
                         iclaDoubleComplex *work, const icla_int_t *lwork,
                         #ifdef ICLA_COMPLEX
                         double *rwork,
                         #endif
                         double *result );

void   lapackf77_zstt21( const icla_int_t *n, const icla_int_t *kband,
                         double *AD,
                         double *AE,
                         double *SD,
                         double *SE,
                         iclaDoubleComplex *U, const icla_int_t *ldu,
                         iclaDoubleComplex *work,
                         #ifdef ICLA_COMPLEX
                         double *rwork,
                         #endif
                         double *result );

void   lapackf77_zunt01( const char *rowcol, const icla_int_t *m, const icla_int_t *n,
                         iclaDoubleComplex *U, const icla_int_t *ldu,
                         iclaDoubleComplex *work, const icla_int_t *lwork,
                         #ifdef ICLA_COMPLEX
                         double *rwork,
                         #endif
                         double *resid );

void   lapackf77_zlarfy( const char *uplo, const icla_int_t *n,
                         iclaDoubleComplex *V, const icla_int_t *incv,
                         iclaDoubleComplex *tau,
                         iclaDoubleComplex *C, const icla_int_t *ldc,
                         iclaDoubleComplex *work );

double lapackf77_zqpt01( const icla_int_t *m, const icla_int_t *n, const icla_int_t *k,
                         iclaDoubleComplex *A,
                         iclaDoubleComplex *Af, const icla_int_t *lda,
                         iclaDoubleComplex *tau, icla_int_t *jpvt,
                         iclaDoubleComplex *work, const icla_int_t *lwork );

void   lapackf77_zqrt02( const icla_int_t *m, const icla_int_t *n, const icla_int_t *k,
                         iclaDoubleComplex *A,
                         iclaDoubleComplex *AF,
                         iclaDoubleComplex *Q,
                         iclaDoubleComplex *R, const icla_int_t *lda,
                         iclaDoubleComplex *tau,
                         iclaDoubleComplex *work, const icla_int_t *lwork,
                         double *rwork,
                         double *result );

void   lapackf77_zlatms( const icla_int_t *m, const icla_int_t *n,
                         const char *dist, icla_int_t *iseed, const char *sym,
                         double *d,
                         const icla_int_t *mode, const double *cond,
                         const double *dmax,
                         const icla_int_t *kl, const icla_int_t *ku, const char *pack,
                         iclaDoubleComplex *A, const icla_int_t *lda,
                         iclaDoubleComplex *work,
                         icla_int_t *info );

#ifdef ICLA_WITH_MKL
void   lapackf77_zgetrf_batch(
                         icla_int_t *m_array, icla_int_t *n_array,
                         iclaDoubleComplex **A_array, icla_int_t *lda_array,
                         icla_int_t **ipiv_array,
                         icla_int_t *group_count, icla_int_t *group_size,
                         icla_int_t *info_array );
#endif

#ifdef __cplusplus
}
#endif

#undef ICLA_COMPLEX

#endif

