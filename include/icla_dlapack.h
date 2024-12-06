

#ifndef ICLA_DLAPACK_H
#define ICLA_DLAPACK_H

#include "icla_types.h"
#include "icla_mangling.h"

#define ICLA_REAL

#ifdef __cplusplus
extern "C" {
#endif

#define blasf77_idamax     FORTRAN_NAME( idamax, IDAMAX )
#define blasf77_daxpy      FORTRAN_NAME( daxpy,  DAXPY  )
#define blasf77_dcopy      FORTRAN_NAME( dcopy,  DCOPY  )
#define blasf77_dgbmv      FORTRAN_NAME( dgbmv,  DGBMV  )
#define blasf77_dgemm      FORTRAN_NAME( dgemm,  DGEMM  )
#define blasf77_dgemv      FORTRAN_NAME( dgemv,  DGEMV  )
#define blasf77_dger      FORTRAN_NAME( dger,  DGER  )
#define blasf77_dger      FORTRAN_NAME( dger,  DGER  )
#define blasf77_dsymm      FORTRAN_NAME( dsymm,  DSYMM  )
#define blasf77_dsymv      FORTRAN_NAME( dsymv,  DSYMV  )
#define blasf77_dsyr       FORTRAN_NAME( dsyr,   DSYR   )
#define blasf77_dsyr2      FORTRAN_NAME( dsyr2,  DSYR2  )
#define blasf77_dsyr2k     FORTRAN_NAME( dsyr2k, DSYR2K )
#define blasf77_dsyrk      FORTRAN_NAME( dsyrk,  DSYRK  )
#define blasf77_dscal      FORTRAN_NAME( dscal,  DSCAL  )
#define blasf77_dscal     FORTRAN_NAME( dscal, DSCAL )
#define blasf77_dswap      FORTRAN_NAME( dswap,  DSWAP  )
#define blasf77_dsymm      FORTRAN_NAME( dsymm,  DSYMM  )
#define blasf77_dsyr2k     FORTRAN_NAME( dsyr2k, DSYR2K )
#define blasf77_dsyrk      FORTRAN_NAME( dsyrk,  DSYRK  )
#define blasf77_drotg      FORTRAN_NAME( drotg,  DROTG  )
#define blasf77_drot       FORTRAN_NAME( drot,   DROT   )
#define blasf77_drot      FORTRAN_NAME( drot,  DROT  )
#define blasf77_dtrmm      FORTRAN_NAME( dtrmm,  DTRMM  )
#define blasf77_dtrmv      FORTRAN_NAME( dtrmv,  DTRMV  )
#define blasf77_dtrsm      FORTRAN_NAME( dtrsm,  DTRSM  )
#define blasf77_dtrsv      FORTRAN_NAME( dtrsv,  DTRSV  )

#define lapackf77_dlaed2   FORTRAN_NAME( dlaed2, DLAED2 )
#define lapackf77_dlaed4   FORTRAN_NAME( dlaed4, DLAED4 )
#define lapackf77_dlaln2   FORTRAN_NAME( dlaln2, DLALN2 )
#define lapackf77_dlamc3   FORTRAN_NAME( dlamc3, DLAMC3 )
#define lapackf77_dlamrg   FORTRAN_NAME( dlamrg, DLAMRG )
#define lapackf77_dlasrt   FORTRAN_NAME( dlasrt, DLASRT )
#define lapackf77_dstebz   FORTRAN_NAME( dstebz, DSTEBZ )

#define lapackf77_dbdsdc   FORTRAN_NAME( dbdsdc, DBDSDC )
#define lapackf77_dbdsqr   FORTRAN_NAME( dbdsqr, DBDSQR )
#define lapackf77_dgbtrf   FORTRAN_NAME( dgbtrf, DGBTRF )
#define lapackf77_dgebak   FORTRAN_NAME( dgebak, DGEBAK )
#define lapackf77_dgebal   FORTRAN_NAME( dgebal, DGEBAL )
#define lapackf77_dgebd2   FORTRAN_NAME( dgebd2, DGEBD2 )
#define lapackf77_dgebrd   FORTRAN_NAME( dgebrd, DGEBRD )
#define lapackf77_dgbbrd   FORTRAN_NAME( dgbbrd, DGBBRD )
#define lapackf77_dgbsv    FORTRAN_NAME( dgbsv,  DGBSV  )
#define lapackf77_dgbtrs   FORTRAN_NAME( dgbtrs, DGBTRS )
#define lapackf77_dgeev    FORTRAN_NAME( dgeev,  DGEEV  )
#define lapackf77_dgehd2   FORTRAN_NAME( dgehd2, DGEHD2 )
#define lapackf77_dgehrd   FORTRAN_NAME( dgehrd, DGEHRD )
#define lapackf77_dgelqf   FORTRAN_NAME( dgelqf, DGELQF )
#define lapackf77_dgels    FORTRAN_NAME( dgels,  DGELS  )
#define lapackf77_dgeqlf   FORTRAN_NAME( dgeqlf, DGEQLF )
#define lapackf77_dgeqp3   FORTRAN_NAME( dgeqp3, DGEQP3 )
#define lapackf77_dgeqrf   FORTRAN_NAME( dgeqrf, DGEQRF )
#define lapackf77_dgerqf   FORTRAN_NAME( dgerqf, DGERQF )
#define lapackf77_dgesdd   FORTRAN_NAME( dgesdd, DGESDD )
#define lapackf77_dgesv    FORTRAN_NAME( dgesv,  DGESV  )
#define lapackf77_dgesvd   FORTRAN_NAME( dgesvd, DGESVD )
#define lapackf77_dgetrf   FORTRAN_NAME( dgetrf, DGETRF )
#define lapackf77_dgetri   FORTRAN_NAME( dgetri, DGETRI )
#define lapackf77_dgetrs   FORTRAN_NAME( dgetrs, DGETRS )
#define lapackf77_dgglse   FORTRAN_NAME( dgglse, DGGLSE )
#define lapackf77_dggrqf   FORTRAN_NAME( dggrqf, DGGRQF )
#define lapackf77_dsytf2   FORTRAN_NAME( dsytf2, DSYTF2 )
#define lapackf77_dsytrs   FORTRAN_NAME( dsytrs, DSYTRS )
#define lapackf77_dsbtrd   FORTRAN_NAME( dsbtrd, DSBTRD )
#define lapackf77_dsyev    FORTRAN_NAME( dsyev,  DSYEV  )
#define lapackf77_dsyevd   FORTRAN_NAME( dsyevd, DSYEVD )
#define lapackf77_dsyevr   FORTRAN_NAME( dsyevr, DSYEVR )
#define lapackf77_dsyevx   FORTRAN_NAME( dsyevx, DSYEVX )
#define lapackf77_dsygs2   FORTRAN_NAME( dsygs2, DSYGS2 )
#define lapackf77_dsygst   FORTRAN_NAME( dsygst, DSYGST )
#define lapackf77_dsygvd   FORTRAN_NAME( dsygvd, DSYGVD )
#define lapackf77_dsytd2   FORTRAN_NAME( dsytd2, DSYTD2 )
#define lapackf77_dsytrd   FORTRAN_NAME( dsytrd, DSYTRD )
#define lapackf77_dsytrf   FORTRAN_NAME( dsytrf, DSYTRF )
#define lapackf77_dsysv    FORTRAN_NAME( dsysv,  DSYSV )
#define lapackf77_dhseqr   FORTRAN_NAME( dhseqr, DHSEQR )
#define lapackf77_dlabrd   FORTRAN_NAME( dlabrd, DLABRD )
#define lapackf77_dlacgv   FORTRAN_NAME( dlacgv, DLACGV )
#define lapackf77_dlacp2   FORTRAN_NAME( dlacp2, DLACP2 )
#define lapackf77_dlacpy   FORTRAN_NAME( dlacpy, DLACPY )
#define lapackf77_dlacrm   FORTRAN_NAME( dlacrm, DLACRM )
#define lapackf77_dladiv   FORTRAN_NAME( dladiv, DLADIV )
#define lapackf77_dlasyf   FORTRAN_NAME( dlasyf, DLASYF )
#define lapackf77_dlangb   FORTRAN_NAME( dlangb, DLANGB )
#define lapackf77_dlange   FORTRAN_NAME( dlange, DLANGE )
#define lapackf77_dlansy   FORTRAN_NAME( dlansy, DLANSY )
#define lapackf77_dlanst   FORTRAN_NAME( dlanst, DLANST )
#define lapackf77_dlansy   FORTRAN_NAME( dlansy, DLANSY )
#define lapackf77_dlantr   FORTRAN_NAME( dlantr, DLANTR )
#define lapackf77_dlapy3   FORTRAN_NAME( dlapy3, DLAPY3 )
#define lapackf77_dlaqp2   FORTRAN_NAME( dlaqp2, DLAQP2 )
#define lapackf77_dlarcm   FORTRAN_NAME( dlarcm, DLARCM )
#define lapackf77_dlarf    FORTRAN_NAME( dlarf,  DLARF  )
#define lapackf77_dlarfb   FORTRAN_NAME( dlarfb, DLARFB )
#define lapackf77_dlarfg   FORTRAN_NAME( dlarfg, DLARFG )
#define lapackf77_dlarft   FORTRAN_NAME( dlarft, DLARFT )
#define lapackf77_dlarfx   FORTRAN_NAME( dlarfx, DLARFX )
#define lapackf77_dlarnv   FORTRAN_NAME( dlarnv, DLARNV )
#define lapackf77_dlartg   FORTRAN_NAME( dlartg, DLARTG )
#define lapackf77_dlascl   FORTRAN_NAME( dlascl, DLASCL )
#define lapackf77_dlaset   FORTRAN_NAME( dlaset, DLASET )
#define lapackf77_dlaswp   FORTRAN_NAME( dlaswp, DLASWP )
#define lapackf77_dlatrd   FORTRAN_NAME( dlatrd, DLATRD )
#define lapackf77_dlatrs   FORTRAN_NAME( dlatrs, DLATRS )
#define lapackf77_dlauum   FORTRAN_NAME( dlauum, DLAUUM )
#define lapackf77_dlavsy   FORTRAN_NAME( dlavsy, DLAVSY )
#define lapackf77_dposv    FORTRAN_NAME( dposv,  DPOSV  )
#define lapackf77_dpotrf   FORTRAN_NAME( dpotrf, DPOTRF )
#define lapackf77_dpotri   FORTRAN_NAME( dpotri, DPOTRI )
#define lapackf77_dpotrs   FORTRAN_NAME( dpotrs, DPOTRS )
#define lapackf77_dstedc   FORTRAN_NAME( dstedc, DSTEDC )
#define lapackf77_dstein   FORTRAN_NAME( dstein, DSTEIN )
#define lapackf77_dstemr   FORTRAN_NAME( dstemr, DSTEMR )
#define lapackf77_dsteqr   FORTRAN_NAME( dsteqr, DSTEQR )
#define lapackf77_dsymv    FORTRAN_NAME( dsymv,  DSYMV  )
#define lapackf77_dsyr     FORTRAN_NAME( dsyr,   DSYR   )
#define lapackf77_dsysv    FORTRAN_NAME( dsysv,  DSYSV  )
#define lapackf77_dtrevc   FORTRAN_NAME( dtrevc, DTREVC )
#define lapackf77_dtrevc3  FORTRAN_NAME( dtrevc3, DTREVC3 )
#define lapackf77_dtrtri   FORTRAN_NAME( dtrtri, DTRTRI )
#define lapackf77_dorg2r   FORTRAN_NAME( dorg2r, DORG2R )
#define lapackf77_dorgbr   FORTRAN_NAME( dorgbr, DORGBR )
#define lapackf77_dorghr   FORTRAN_NAME( dorghr, DORGHR )
#define lapackf77_dorglq   FORTRAN_NAME( dorglq, DORGLQ )
#define lapackf77_dorgql   FORTRAN_NAME( dorgql, DORGQL )
#define lapackf77_dorgqr   FORTRAN_NAME( dorgqr, DORGQR )
#define lapackf77_dorgtr   FORTRAN_NAME( dorgtr, DORGTR )
#define lapackf77_dorm2r   FORTRAN_NAME( dorm2r, DORM2R )
#define lapackf77_dormbr   FORTRAN_NAME( dormbr, DORMBR )
#define lapackf77_dormlq   FORTRAN_NAME( dormlq, DORMLQ )
#define lapackf77_dormql   FORTRAN_NAME( dormql, DORMQL )
#define lapackf77_dormqr   FORTRAN_NAME( dormqr, DORMQR )
#define lapackf77_dormrq   FORTRAN_NAME( dormrq, DORMRQ )
#define lapackf77_dormtr   FORTRAN_NAME( dormtr, DORMTR )



#ifdef ICLA_WITH_MKL
#define lapackf77_dgetrf_batch   FORTRAN_NAME( dgetrf_batch, DGETRF_BATCH )
#endif



#define lapackf77_dbdt01   FORTRAN_NAME( dbdt01, DBDT01 )
#define lapackf77_dget22   FORTRAN_NAME( dget22, DGET22 )
#define lapackf77_dsyt21   FORTRAN_NAME( dsyt21, DSYT21 )
#define lapackf77_dsyt22   FORTRAN_NAME( dsyt22, DSYT22 )
#define lapackf77_dhst01   FORTRAN_NAME( dhst01, DHST01 )
#define lapackf77_dlarfy   FORTRAN_NAME( dlarfy, DLARFY )
#define lapackf77_dlatms   FORTRAN_NAME( dlatms, DLATMS )
#define lapackf77_dqpt01   FORTRAN_NAME( dqpt01, DQPT01 )
#define lapackf77_dqrt02   FORTRAN_NAME( dqrt02, DQRT02 )
#define lapackf77_dstt21   FORTRAN_NAME( dstt21, DSTT21 )
#define lapackf77_dort01   FORTRAN_NAME( dort01, DORT01 )


icla_int_t blasf77_idamax(
                     const icla_int_t *n,
                     const double *x, const icla_int_t *incx );

void blasf77_daxpy(  const icla_int_t *n,
                     const double *alpha,
                     const double *x, const icla_int_t *incx,
                           double *y, const icla_int_t *incy );

void blasf77_dcopy(  const icla_int_t *n,
                     const double *x, const icla_int_t *incx,
                           double *y, const icla_int_t *incy );

void blasf77_dgbmv(  const char *transa,
                     const icla_int_t *m,  const icla_int_t *n,
                     const icla_int_t *kl, const icla_int_t *ku,
                     const double *alpha,
                     const double *A, const icla_int_t *lda,
                     const double *x, const icla_int_t *incx,
                     const double *beta,
                           double *y, const icla_int_t *incy );

void blasf77_dgemm(  const char *transa, const char *transb,
                     const icla_int_t *m, const icla_int_t *n, const icla_int_t *k,
                     const double *alpha,
                     const double *A, const icla_int_t *lda,
                     const double *B, const icla_int_t *ldb,
                     const double *beta,
                           double *C, const icla_int_t *ldc );

void blasf77_dgemv(  const char *transa,
                     const icla_int_t *m, const icla_int_t *n,
                     const double *alpha,
                     const double *A, const icla_int_t *lda,
                     const double *x, const icla_int_t *incx,
                     const double *beta,
                           double *y, const icla_int_t *incy );

void blasf77_dger(  const icla_int_t *m, const icla_int_t *n,
                     const double *alpha,
                     const double *x, const icla_int_t *incx,
                     const double *y, const icla_int_t *incy,
                           double *A, const icla_int_t *lda );

void blasf77_dger(  const icla_int_t *m, const icla_int_t *n,
                     const double *alpha,
                     const double *x, const icla_int_t *incx,
                     const double *y, const icla_int_t *incy,
                           double *A, const icla_int_t *lda );

void blasf77_dsymm(  const char *side, const char *uplo,
                     const icla_int_t *m, const icla_int_t *n,
                     const double *alpha,
                     const double *A, const icla_int_t *lda,
                     const double *B, const icla_int_t *ldb,
                     const double *beta,
                           double *C, const icla_int_t *ldc );

void blasf77_dsymv(  const char *uplo,
                     const icla_int_t *n,
                     const double *alpha,
                     const double *A, const icla_int_t *lda,
                     const double *x, const icla_int_t *incx,
                     const double *beta,
                           double *y, const icla_int_t *incy );

void blasf77_dsyr(   const char *uplo,
                     const icla_int_t *n,
                     const double *alpha,
                     const double *x, const icla_int_t *incx,
                           double *A, const icla_int_t *lda );

void blasf77_dsyr2(  const char *uplo,
                     const icla_int_t *n,
                     const double *alpha,
                     const double *x, const icla_int_t *incx,
                     const double *y, const icla_int_t *incy,
                           double *A, const icla_int_t *lda );

void blasf77_dsyr2k( const char *uplo, const char *trans,
                     const icla_int_t *n, const icla_int_t *k,
                     const double *alpha,
                     const double *A, const icla_int_t *lda,
                     const double *B, const icla_int_t *ldb,
                     const double *beta,
                           double *C, const icla_int_t *ldc );

void blasf77_dsyrk(  const char *uplo, const char *trans,
                     const icla_int_t *n, const icla_int_t *k,
                     const double *alpha,
                     const double *A, const icla_int_t *lda,
                     const double *beta,
                           double *C, const icla_int_t *ldc );

void blasf77_dscal(  const icla_int_t *n,
                     const double *alpha,
                           double *x, const icla_int_t *incx );

void blasf77_dscal( const icla_int_t *n,
                     const double *alpha,
                           double *x, const icla_int_t *incx );

void blasf77_dswap(  const icla_int_t *n,
                     double *x, const icla_int_t *incx,
                     double *y, const icla_int_t *incy );



void blasf77_dsymm(  const char *side, const char *uplo,
                     const icla_int_t *m, const icla_int_t *n,
                     const double *alpha,
                     const double *A, const icla_int_t *lda,
                     const double *B, const icla_int_t *ldb,
                     const double *beta,
                           double *C, const icla_int_t *ldc );

void blasf77_dsyr2k( const char *uplo, const char *trans,
                     const icla_int_t *n, const icla_int_t *k,
                     const double *alpha,
                     const double *A, const icla_int_t *lda,
                     const double *B, const icla_int_t *ldb,
                     const double *beta,
                           double *C, const icla_int_t *ldc );

void blasf77_dsyrk(  const char *uplo, const char *trans,
                     const icla_int_t *n, const icla_int_t *k,
                     const double *alpha,
                     const double *A, const icla_int_t *lda,
                     const double *beta,
                           double *C, const icla_int_t *ldc );

void blasf77_drotg(  double *ca, const double *cb,
                     double *c, double *s );

void blasf77_drot(   const icla_int_t *n,
                     double *x, const icla_int_t *incx,
                     double *y, const icla_int_t *incy,
                     const double *c, const double *s );

void blasf77_drot(  const icla_int_t *n,
                     double *x, const icla_int_t *incx,
                     double *y, const icla_int_t *incy,
                     const double *c, const double *s );

void blasf77_dtrmm(  const char *side, const char *uplo, const char *transa, const char *diag,
                     const icla_int_t *m, const icla_int_t *n,
                     const double *alpha,
                     const double *A, const icla_int_t *lda,
                           double *B, const icla_int_t *ldb );

void blasf77_dtrmv(  const char *uplo, const char *transa, const char *diag,
                     const icla_int_t *n,
                     const double *A, const icla_int_t *lda,
                           double *x, const icla_int_t *incx );

void blasf77_dtrsm(  const char *side, const char *uplo, const char *transa, const char *diag,
                     const icla_int_t *m, const icla_int_t *n,
                     const double *alpha,
                     const double *A, const icla_int_t *lda,
                           double *B, const icla_int_t *ldb );

void blasf77_dtrsv(  const char *uplo, const char *transa, const char *diag,
                     const icla_int_t *n,
                     const double *A, const icla_int_t *lda,
                           double *x, const icla_int_t *incx );


double icla_cblas_dasum(
    icla_int_t n,
    const double *x, icla_int_t incx );

double icla_cblas_dnrm2(
    icla_int_t n,
    const double *x, icla_int_t incx );

double icla_cblas_ddot(
    icla_int_t n,
    const double *x, icla_int_t incx,
    const double *y, icla_int_t incy );

double icla_cblas_ddot(
    icla_int_t n,
    const double *x, icla_int_t incx,
    const double *y, icla_int_t incy );

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

void   lapackf77_dbdsqr( const char *uplo,
                         const icla_int_t *n, const icla_int_t *ncvt, const icla_int_t *nru,  const icla_int_t *ncc,
                         double *d, double *e,
                         double *Vt, const icla_int_t *ldvt,
                         double *U, const icla_int_t *ldu,
                         double *C, const icla_int_t *ldc,
                         double *work,
                         icla_int_t *info );

void   lapackf77_dgbtrf( const icla_int_t  *m,  const icla_int_t *n,
                         const icla_int_t  *kl, const icla_int_t *ku,
                         double *AB, const icla_int_t *ldab,
                         icla_int_t *ipiv,
                         icla_int_t *info );

void   lapackf77_dgebak( const char *job, const char *side,
                         const icla_int_t *n,
                         const icla_int_t *ilo, const icla_int_t *ihi,
                         const double *scale, const icla_int_t *m,
                         double *V, const icla_int_t *ldv,
                         icla_int_t *info );

void   lapackf77_dgebal( const char *job,
                         const icla_int_t *n,
                         double *A, const icla_int_t *lda,
                         icla_int_t *ilo, icla_int_t *ihi,
                         double *scale,
                         icla_int_t *info );

void   lapackf77_dgebd2( const icla_int_t *m, const icla_int_t *n,
                         double *A, const icla_int_t *lda,
                         double *d, double *e,
                         double *tauq,
                         double *taup,
                         double *work,
                         icla_int_t *info );

void   lapackf77_dgebrd( const icla_int_t *m, const icla_int_t *n,
                         double *A, const icla_int_t *lda,
                         double *d, double *e,
                         double *tauq,
                         double *taup,
                         double *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_dgbbrd( const char *vect, const icla_int_t *m,
                         const icla_int_t *n, const icla_int_t *ncc,
                         const icla_int_t *kl, const icla_int_t *ku,
                         double *Ab, const icla_int_t *ldab,
                         double *d, double *e,
                         double *Q, const icla_int_t *ldq,
                         double *PT, const icla_int_t *ldpt,
                         double *C, const icla_int_t *ldc,
                         double *work,
                         #ifdef ICLA_COMPLEX
                         double *rwork,
                         #endif
                         icla_int_t *info );

void   lapackf77_dgbsv( const icla_int_t *n,
                        const icla_int_t *kl, const icla_int_t *ku,
                        const icla_int_t *nrhs,
                        double *ab, const icla_int_t *ldab,
                        icla_int_t *ipiv,
                        double *B, const icla_int_t *ldb,
                        icla_int_t *info );

void   lapackf77_dgbtrs( const char *trans,
                         const icla_int_t *n,
                         const icla_int_t *kl, const icla_int_t *ku,
                         const icla_int_t *nrhs,
                         double *ab, const icla_int_t *ldab,
                         icla_int_t *ipiv,
                         double *B, const icla_int_t *ldb,
                         icla_int_t *info );

void   lapackf77_dgeev(  const char *jobvl, const char *jobvr,
                         const icla_int_t *n,
                         double *A,    const icla_int_t *lda,
                         #ifdef ICLA_COMPLEX
                         double *w,
                         #else
                         double *wr, double *wi,
                         #endif
                         double *Vl,   const icla_int_t *ldvl,
                         double *Vr,   const icla_int_t *ldvr,
                         double *work, const icla_int_t *lwork,
                         #ifdef ICLA_COMPLEX
                         double *rwork,
                         #endif
                         icla_int_t *info );

void   lapackf77_dgehd2( const icla_int_t *n,
                         const icla_int_t *ilo, const icla_int_t *ihi,
                         double *A, const icla_int_t *lda,
                         double *tau,
                         double *work,
                         icla_int_t *info );

void   lapackf77_dgehrd( const icla_int_t *n,
                         const icla_int_t *ilo, const icla_int_t *ihi,
                         double *A, const icla_int_t *lda,
                         double *tau,
                         double *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_dgelqf( const icla_int_t *m, const icla_int_t *n,
                         double *A, const icla_int_t *lda,
                         double *tau,
                         double *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_dgels(  const char *trans,
                         const icla_int_t *m, const icla_int_t *n, const icla_int_t *nrhs,
                         double *A, const icla_int_t *lda,
                         double *B, const icla_int_t *ldb,
                         double *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_dgeqlf( const icla_int_t *m, const icla_int_t *n,
                         double *A, const icla_int_t *lda,
                         double *tau,
                         double *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_dgeqp3( const icla_int_t *m, const icla_int_t *n,
                         double *A, const icla_int_t *lda,
                         icla_int_t *jpvt,
                         double *tau,
                         double *work, const icla_int_t *lwork,
                         #ifdef ICLA_COMPLEX
                         double *rwork,
                         #endif
                         icla_int_t *info );

void   lapackf77_dgeqrf( const icla_int_t *m, const icla_int_t *n,
                         double *A, const icla_int_t *lda,
                         double *tau,
                         double *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_dgerqf( const icla_int_t *m, const icla_int_t *n,
                         double *A, const icla_int_t *lda,
                         double *tau,
                         double *work, const icla_int_t *lwork,
                         icla_int_t *info);

void   lapackf77_dgesdd( const char *jobz,
                         const icla_int_t *m, const icla_int_t *n,
                         double *A, const icla_int_t *lda,
                         double *s,
                         double *U,  const icla_int_t *ldu,
                         double *Vt, const icla_int_t *ldvt,
                         double *work, const icla_int_t *lwork,
                         #ifdef ICLA_COMPLEX
                         double *rwork,
                         #endif
                         icla_int_t *iwork,
                         icla_int_t *info );

void   lapackf77_dgesv(  const icla_int_t *n, const icla_int_t *nrhs,
                         double *A, const icla_int_t *lda,
                         icla_int_t *ipiv,
                         double *B,  const icla_int_t *ldb,
                         icla_int_t *info );

void   lapackf77_dgesvd( const char *jobu, const char *jobvt,
                         const icla_int_t *m, const icla_int_t *n,
                         double *A, const icla_int_t *lda,
                         double *s,
                         double *U,  const icla_int_t *ldu,
                         double *Vt, const icla_int_t *ldvt,
                         double *work, const icla_int_t *lwork,
                         #ifdef ICLA_COMPLEX
                         double *rwork,
                         #endif
                         icla_int_t *info );

void   lapackf77_dgetrf( const icla_int_t *m, const icla_int_t *n,
                         double *A, const icla_int_t *lda,
                         icla_int_t *ipiv,
                         icla_int_t *info );

void   lapackf77_dgetri( const icla_int_t *n,
                         double *A, const icla_int_t *lda,
                         const icla_int_t *ipiv,
                         double *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_dgetrs( const char *trans,
                         const icla_int_t *n, const icla_int_t *nrhs,
                         const double *A, const icla_int_t *lda,
                         const icla_int_t *ipiv,
                         double *B, const icla_int_t *ldb,
                         icla_int_t *info );

void   lapackf77_dgglse( icla_int_t *m, icla_int_t *n, icla_int_t *p,
                         double *A, icla_int_t *lda,
                         double *B, icla_int_t *ldb,
                         double *c, double *d,
                         double *x,
                         double *work, icla_int_t *lwork,
                         icla_int_t *info);

void   lapackf77_dggrqf( icla_int_t *m, icla_int_t *p, icla_int_t *n,
                         double *A, icla_int_t *lda,
                         double *tauA, double *B,
                         icla_int_t *ldb, double *tauB,
                         double *work, icla_int_t *lwork,
                         icla_int_t *info);

void   lapackf77_dsytf2( const char *uplo, const icla_int_t *n,
                         double *A, const icla_int_t *lda,
                         icla_int_t *ipiv,
                         icla_int_t *info );

void   lapackf77_dsytrs( const char *uplo,
                         const icla_int_t *n, const icla_int_t *nrhs,
                         const double *A, const icla_int_t *lda,
                         const icla_int_t *ipiv,
                         double *B, const icla_int_t *ldb,
                         icla_int_t *info );

void   lapackf77_dsbtrd( const char *vect, const char *uplo,
                         const icla_int_t *n, const icla_int_t *kd,
                         double *Ab, const icla_int_t *ldab,
                         double *d, double *e,
                         double *Q, const icla_int_t *ldq,
                         double *work,
                         icla_int_t *info );

void   lapackf77_dsyev(  const char *jobz, const char *uplo,
                         const icla_int_t *n,
                         double *A, const icla_int_t *lda,
                         double *w,
                         double *work, const icla_int_t *lwork,
                         #ifdef ICLA_COMPLEX
                         double *rwork,
                         #endif
                         icla_int_t *info );

void   lapackf77_dsyevd( const char *jobz, const char *uplo,
                         const icla_int_t *n,
                         double *A, const icla_int_t *lda,
                         double *w,
                         double *work, const icla_int_t *lwork,
                         #ifdef ICLA_COMPLEX
                         double *rwork, const icla_int_t *lrwork,
                         #endif
                         icla_int_t *iwork, const icla_int_t *liwork,
                         icla_int_t *info );

void   lapackf77_dsyevr( const char *jobz, const char *range, const char *uplo,
                         const icla_int_t *n,
                         double *A, const icla_int_t *lda,
                         const double *vl, const double *vu,
                         const icla_int_t *il, const icla_int_t *iu,
                         const double *abstol,
                         icla_int_t *m, double *w,
                         double *Z, const icla_int_t *ldz,
                         icla_int_t *isuppz,
                         double *work, const icla_int_t *lwork,
                         #ifdef ICLA_COMPLEX
                         double *rwork, const icla_int_t *lrwork,
                         #endif
                         icla_int_t *iwork, const icla_int_t *liwork,
                         icla_int_t *info);

void   lapackf77_dsyevx( const char *jobz, const char *range, const char *uplo,
                         const icla_int_t *n,
                         double *A, const icla_int_t *lda,
                         const double *vl, const double *vu,
                         const icla_int_t *il, const icla_int_t *iu,
                         const double *abstol,
                         icla_int_t *m, double *w,
                         double *Z, const icla_int_t *ldz,
                         double *work, const icla_int_t *lwork,
                         #ifdef ICLA_COMPLEX
                         double *rwork,
                         #endif
                         icla_int_t *iwork, icla_int_t *ifail,
                         icla_int_t *info);

void   lapackf77_dsygs2( const icla_int_t *itype, const char *uplo,
                         const icla_int_t *n,
                         double *A, const icla_int_t *lda,
                         const double *B, const icla_int_t *ldb,
                         icla_int_t *info );

void   lapackf77_dsygst( const icla_int_t *itype, const char *uplo,
                         const icla_int_t *n,
                         double *A, const icla_int_t *lda,
                         const double *B, const icla_int_t *ldb,
                         icla_int_t *info );

void   lapackf77_dsygvd( const icla_int_t *itype, const char *jobz, const char *uplo,
                         const icla_int_t *n,
                         double *A, const icla_int_t *lda,
                         double *B, const icla_int_t *ldb,
                         double *w,
                         double *work, const icla_int_t *lwork,
                         #ifdef ICLA_COMPLEX
                         double *rwork, const icla_int_t *lrwork,
                         #endif
                         icla_int_t *iwork, const icla_int_t *liwork,
                         icla_int_t *info );

void   lapackf77_dsysv( const char *uplo,
                        const icla_int_t *n, const icla_int_t *nrhs,
                        double *A, const icla_int_t *lda, icla_int_t *ipiv,
                        double *B, const icla_int_t *ldb,
                        double *work, const icla_int_t *lwork,
                        icla_int_t *info );

void   lapackf77_dsytd2( const char *uplo,
                         const icla_int_t *n,
                         double *A, const icla_int_t *lda,
                         double *d, double *e,
                         double *tau,
                         icla_int_t *info );

void   lapackf77_dsytrd( const char *uplo,
                         const icla_int_t *n,
                         double *A, const icla_int_t *lda,
                         double *d, double *e,
                         double *tau,
                         double *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_dsytrf( const char *uplo,
                         const icla_int_t *n,
                         double *A, const icla_int_t *lda,
                         icla_int_t *ipiv,
                         double *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_dhseqr( const char *job, const char *compz,
                         const icla_int_t *n,
                         const icla_int_t *ilo, const icla_int_t *ihi,
                         double *H, const icla_int_t *ldh,
                         #ifdef ICLA_COMPLEX
                         double *w,
                         #else
                         double *wr, double *wi,
                         #endif
                         double *Z, const icla_int_t *ldz,
                         double *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_dlabrd( const icla_int_t *m, const icla_int_t *n, const icla_int_t *nb,
                         double *A, const icla_int_t *lda,
                         double *d, double *e,
                         double *tauq,
                         double *taup,
                         double *X, const icla_int_t *ldx,
                         double *Y, const icla_int_t *ldy );

#ifdef ICLA_COMPLEX
void   lapackf77_dlacgv( const icla_int_t *n,
                         double *x, const icla_int_t *incx );
#endif

#ifdef ICLA_COMPLEX
void   lapackf77_dlacp2( const char *uplo,
                         const icla_int_t *m, const icla_int_t *n,
                         const double *A, const icla_int_t *lda,
                         double *B, const icla_int_t *ldb );
#endif

void   lapackf77_dlacpy( const char *uplo,
                         const icla_int_t *m, const icla_int_t *n,
                         const double *A, const icla_int_t *lda,
                         double *B, const icla_int_t *ldb );

#ifdef ICLA_COMPLEX
void   lapackf77_dlacrm( const icla_int_t *m, const icla_int_t *n,
                         const double *A, const icla_int_t *lda,
                         const double             *B, const icla_int_t *ldb,
                         double       *C, const icla_int_t *ldc,
                         double *rwork );
#endif

#ifdef ICLA_COMPLEX
void   lapackf77_dladiv( double *ret_val,
                         const double *x,
                         const double *y );
#else
void   lapackf77_dladiv( const double *a, const double *b,
                         const double *c, const double *d,
                         double *p, double *q );
#endif

void   lapackf77_dlasyf( const char *uplo,
                         const icla_int_t *n, const icla_int_t *nb,
                         icla_int_t *kb,
                         double *A, const icla_int_t *lda,
                         icla_int_t *ipiv,
                         double *work, const icla_int_t *ldwork,
                         icla_int_t *info );

double lapackf77_dlangb( const char *norm,
                         const icla_int_t *n, const icla_int_t *kl, const icla_int_t *ku,
                         const double *AB, const icla_int_t *ldab,
                         double *work );

double lapackf77_dlange( const char *norm,
                         const icla_int_t *m, const icla_int_t *n,
                         const double *A, const icla_int_t *lda,
                         double *work );

double lapackf77_dlansy( const char *norm, const char *uplo,
                         const icla_int_t *n,
                         const double *A, const icla_int_t *lda,
                         double *work );

double lapackf77_dlanst( const char *norm, const icla_int_t *n,
                         const double *d, const double *e );

double lapackf77_dlansy( const char *norm, const char *uplo,
                         const icla_int_t *n,
                         const double *A, const icla_int_t *lda,
                         double *work );

double lapackf77_dlantr( const char *norm, const char *uplo, const char *diag,
                         const icla_int_t *m, const icla_int_t *n,
                         const double *A, const icla_int_t *lda,
                         double *work );

void   lapackf77_dlaqp2( const icla_int_t *m, const icla_int_t *n, const icla_int_t *offset,
                         double *A, const icla_int_t *lda,
                         icla_int_t *jpvt,
                         double *tau,
                         double *vn1, double *vn2,
                         double *work );

#ifdef ICLA_COMPLEX
void   lapackf77_dlarcm( const icla_int_t *m, const icla_int_t *n,
                         const double             *A, const icla_int_t *lda,
                         const double *B, const icla_int_t *ldb,
                         double       *C, const icla_int_t *ldc,
                         double *rwork );
#endif

void   lapackf77_dlarf(  const char *side, const icla_int_t *m, const icla_int_t *n,
                         const double *v, const icla_int_t *incv,
                         const double *tau,
                         double *C, const icla_int_t *ldc,
                         double *work );

void   lapackf77_dlarfb( const char *side, const char *trans, const char *direct, const char *storev,
                         const icla_int_t *m, const icla_int_t *n, const icla_int_t *k,
                         const double *V, const icla_int_t *ldv,
                         const double *T, const icla_int_t *ldt,
                         double *C, const icla_int_t *ldc,
                         double *work, const icla_int_t *ldwork );

void   lapackf77_dlarfg( const icla_int_t *n,
                         double *alpha,
                         double *x, const icla_int_t *incx,
                         double *tau );

void   lapackf77_dlarft( const char *direct, const char *storev,
                         const icla_int_t *n, const icla_int_t *k,
                         const double *V, const icla_int_t *ldv,
                         const double *tau,
                         double *T, const icla_int_t *ldt );

void   lapackf77_dlarfx( const char *side, const icla_int_t *m, const icla_int_t *n,
                         const double *V,
                         const double *tau,
                         double *C, const icla_int_t *ldc,
                         double *work );

void   lapackf77_dlarnv( const icla_int_t *idist, icla_int_t *iseed, const icla_int_t *n,
                         double *x );

void   lapackf77_dlartg( const double *f,
                         const double *g,
                         double *cs,
                         double *sn,
                         double *r );

void   lapackf77_dlascl( const char *type,
                         const icla_int_t *kl, const icla_int_t *ku,
                         const double *cfrom,
                         const double *cto,
                         const icla_int_t *m, const icla_int_t *n,
                         double *A, const icla_int_t *lda,
                         icla_int_t *info );

void   lapackf77_dlaset( const char *uplo,
                         const icla_int_t *m, const icla_int_t *n,
                         const double *alpha,
                         const double *beta,
                         double *A, const icla_int_t *lda );

void   lapackf77_dlaswp( const icla_int_t *n,
                         double *A, const icla_int_t *lda,
                         const icla_int_t *k1, const icla_int_t *k2,
                         const icla_int_t *ipiv,
                         const icla_int_t *incx );

void   lapackf77_dlatrd( const char *uplo,
                         const icla_int_t *n, const icla_int_t *nb,
                         double *A, const icla_int_t *lda,
                         double *e,
                         double *tau,
                         double *work, const icla_int_t *ldwork );

void   lapackf77_dlatrs( const char *uplo, const char *trans, const char *diag,
                         const char *normin,
                         const icla_int_t *n,
                         const double *A, const icla_int_t *lda,
                         double *x, double *scale,
                         double *cnorm,
                         icla_int_t *info );

void   lapackf77_dlauum( const char *uplo,
                         const icla_int_t *n,
                         double *A, const icla_int_t *lda,
                         icla_int_t *info );

void   lapackf77_dlavsy( const char *uplo, const char *trans, const char *diag,
                         icla_int_t *n, icla_int_t *nrhs,
                         double *A, icla_int_t *lda,
                         icla_int_t *ipiv,
                         double *B, icla_int_t *ldb,
                         icla_int_t *info );

void   lapackf77_dposv(  const char *uplo,
                         const icla_int_t *n, const icla_int_t *nrhs,
                         double *A, const icla_int_t *lda,
                         double *B,  const icla_int_t *ldb,
                         icla_int_t *info );

void   lapackf77_dpotrf( const char *uplo,
                         const icla_int_t *n,
                         double *A, const icla_int_t *lda,
                         icla_int_t *info );

void   lapackf77_dpotri( const char *uplo,
                         const icla_int_t *n,
                         double *A, const icla_int_t *lda,
                         icla_int_t *info );

void   lapackf77_dpotrs( const char *uplo,
                         const icla_int_t *n, const icla_int_t *nrhs,
                         const double *A, const icla_int_t *lda,
                         double *B, const icla_int_t *ldb,
                         icla_int_t *info );

void   lapackf77_dstedc( const char *compz,
                         const icla_int_t *n,
                         double *d, double *e,
                         double *Z, const icla_int_t *ldz,
                         double *work, const icla_int_t *lwork,
                         #ifdef ICLA_COMPLEX
                         double *rwork, const icla_int_t *lrwork,
                         #endif
                         icla_int_t *iwork, const icla_int_t *liwork,
                         icla_int_t *info );

void   lapackf77_dstein( const icla_int_t *n,
                         const double *d, const double *e,
                         const icla_int_t *m,
                         const double *w,
                         const icla_int_t *iblock,
                         const icla_int_t *isplit,
                         double *Z, const icla_int_t *ldz,
                         double *work, icla_int_t *iwork, icla_int_t *ifailv,
                         icla_int_t *info );

void   lapackf77_dstemr( const char *jobz, const char *range,
                         const icla_int_t *n,
                         double *d, double *e,
                         const double *vl, const double *vu,
                         const icla_int_t *il, const icla_int_t *iu,
                         icla_int_t *m,
                         double *w,
                         double *Z, const icla_int_t *ldz,
                         const icla_int_t *nzc, icla_int_t *isuppz, icla_int_t *tryrac,
                         double *work, const icla_int_t *lwork,
                         icla_int_t *iwork, const icla_int_t *liwork,
                         icla_int_t *info );

void   lapackf77_dsteqr( const char *compz,
                         const icla_int_t *n,
                         double *d, double *e,
                         double *Z, const icla_int_t *ldz,
                         double *work,
                         icla_int_t *info );

#ifdef ICLA_COMPLEX
void   lapackf77_dsymv(  const char *uplo,
                         const icla_int_t *n,
                         const double *alpha,
                         const double *A, const icla_int_t *lda,
                         const double *x, const icla_int_t *incx,
                         const double *beta,
                               double *y, const icla_int_t *incy );

void   lapackf77_dsyr(   const char *uplo,
                         const icla_int_t *n,
                         const double *alpha,
                         const double *x, const icla_int_t *incx,
                               double *A, const icla_int_t *lda );

void   lapackf77_dsysv(  const char *uplo,
                         const icla_int_t *n, const icla_int_t *nrhs,
                         double *A, const icla_int_t *lda, icla_int_t *ipiv,
                         double *B, const icla_int_t *ldb,
                         double *work, const icla_int_t *lwork,
                         icla_int_t *info );

#endif

void   lapackf77_dtrevc( const char *side, const char *howmny,

                         #ifdef ICLA_COMPLEX
                         const
                         #endif
                         icla_int_t *select, const icla_int_t *n,

                         #ifdef ICLA_REAL
                         const
                         #endif
                         double *T,  const icla_int_t *ldt,
                         double *Vl, const icla_int_t *ldvl,
                         double *Vr, const icla_int_t *ldvr,
                         const icla_int_t *mm, icla_int_t *m,
                         double *work,
                         #ifdef ICLA_COMPLEX
                         double *rwork,
                         #endif
                         icla_int_t *info );

void   lapackf77_dtrevc3( const char *side, const char *howmny,
                          icla_int_t *select, const icla_int_t *n,
                          double *T,  const icla_int_t *ldt,
                          double *VL, const icla_int_t *ldvl,
                          double *VR, const icla_int_t *ldvr,
                          const icla_int_t *mm,
                          const icla_int_t *mout,
                          double *work, const icla_int_t *lwork,
                          #ifdef ICLA_COMPLEX
                          double *rwork,
                          #endif
                          icla_int_t *info );

void   lapackf77_dtrtri( const char *uplo, const char *diag,
                         const icla_int_t *n,
                         double *A, const icla_int_t *lda,
                         icla_int_t *info );

void   lapackf77_dorg2r( const icla_int_t *m, const icla_int_t *n, const icla_int_t *k,
                         double *A, const icla_int_t *lda,
                         const double *tau,
                         double *work,
                         icla_int_t *info );

void   lapackf77_dorgbr( const char *vect,
                         const icla_int_t *m, const icla_int_t *n, const icla_int_t *k,
                         double *A, const icla_int_t *lda,
                         const double *tau,
                         double *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_dorghr( const icla_int_t *n,
                         const icla_int_t *ilo, const icla_int_t *ihi,
                         double *A, const icla_int_t *lda,
                         const double *tau,
                         double *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_dorglq( const icla_int_t *m, const icla_int_t *n, const icla_int_t *k,
                         double *A, const icla_int_t *lda,
                         const double *tau,
                         double *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_dorgql( const icla_int_t *m, const icla_int_t *n, const icla_int_t *k,
                         double *A, const icla_int_t *lda,
                         const double *tau,
                         double *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_dorgqr( const icla_int_t *m, const icla_int_t *n, const icla_int_t *k,
                         double *A, const icla_int_t *lda,
                         const double *tau,
                         double *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_dorgtr( const char *uplo,
                         const icla_int_t *n,
                         double *A, const icla_int_t *lda,
                         const double *tau,
                         double *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_dorm2r( const char *side, const char *trans,
                         const icla_int_t *m, const icla_int_t *n, const icla_int_t *k,
                         const double *A, const icla_int_t *lda,
                         const double *tau,
                         double *C, const icla_int_t *ldc,
                         double *work,
                         icla_int_t *info );

void   lapackf77_dormbr( const char *vect, const char *side, const char *trans,
                         const icla_int_t *m, const icla_int_t *n, const icla_int_t *k,
                         const double *A, const icla_int_t *lda,
                         const double *tau,
                         double *C, const icla_int_t *ldc,
                         double *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_dormlq( const char *side, const char *trans,
                         const icla_int_t *m, const icla_int_t *n, const icla_int_t *k,
                         const double *A, const icla_int_t *lda,
                         const double *tau,
                         double *C, const icla_int_t *ldc,
                         double *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_dormql( const char *side, const char *trans,
                         const icla_int_t *m, const icla_int_t *n, const icla_int_t *k,
                         const double *A, const icla_int_t *lda,
                         const double *tau,
                         double *C, const icla_int_t *ldc,
                         double *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_dormqr( const char *side, const char *trans,
                         const icla_int_t *m, const icla_int_t *n, const icla_int_t *k,
                         const double *A, const icla_int_t *lda,
                         const double *tau,
                         double *C, const icla_int_t *ldc,
                         double *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_dormrq( const char *side, const char *trans,
                         icla_int_t *m, icla_int_t *n, icla_int_t *k,
                         double *A, icla_int_t *lda,
                         double *tau,
                         double *C, icla_int_t *ldc,
                         double *work, icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_dormtr( const char *side, const char *uplo, const char *trans,
                         const icla_int_t *m, const icla_int_t *n,
                         const double *A, const icla_int_t *lda,
                         const double *tau,
                         double *C, const icla_int_t *ldc,
                         double *work, const icla_int_t *lwork,
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


void   lapackf77_dbdt01( const icla_int_t *m, const icla_int_t *n, const icla_int_t *kd,
                         double *A, const icla_int_t *lda,
                         double *Q, const icla_int_t *ldq,
                         double *d, double *e,
                         double *Pt, const icla_int_t *ldpt,
                         double *work,
                         #ifdef ICLA_COMPLEX
                         double *rwork,
                         #endif
                         double *resid );

void   lapackf77_dget22( const char *transa, const char *transe, const char *transw, const icla_int_t *n,
                         double *A, const icla_int_t *lda,
                         double *E, const icla_int_t *lde,
                         #ifdef ICLA_COMPLEX
                         double *w,
                         #else
                         double *wr,
                         double *wi,
                         #endif
                         double *work,
                         #ifdef ICLA_COMPLEX
                         double *rwork,
                         #endif
                         double *result );

void   lapackf77_dsyt21( const icla_int_t *itype, const char *uplo,
                         const icla_int_t *n, const icla_int_t *kband,
                         double *A, const icla_int_t *lda,
                         double *d, double *e,
                         double *U, const icla_int_t *ldu,
                         double *V, const icla_int_t *ldv,
                         double *tau,
                         double *work,
                         #ifdef ICLA_COMPLEX
                         double *rwork,
                         #endif
                         double *result );

void   lapackf77_dsyt22( const icla_int_t *itype, const char *uplo,
                         const icla_int_t *n, const icla_int_t *m, const icla_int_t *kband,
                         double *A, const icla_int_t *lda,
                         double *d, double *e,
                         double *U, const icla_int_t *ldu,
                         double *V, const icla_int_t *ldv,
                         double *tau,
                         double *work,
                         #ifdef ICLA_COMPLEX
                         double *rwork,
                         #endif
                         double *result );

void   lapackf77_dhst01( const icla_int_t *n, const icla_int_t *ilo, const icla_int_t *ihi,
                         double *A, const icla_int_t *lda,
                         double *H, const icla_int_t *ldh,
                         double *Q, const icla_int_t *ldq,
                         double *work, const icla_int_t *lwork,
                         #ifdef ICLA_COMPLEX
                         double *rwork,
                         #endif
                         double *result );

void   lapackf77_dstt21( const icla_int_t *n, const icla_int_t *kband,
                         double *AD,
                         double *AE,
                         double *SD,
                         double *SE,
                         double *U, const icla_int_t *ldu,
                         double *work,
                         #ifdef ICLA_COMPLEX
                         double *rwork,
                         #endif
                         double *result );

void   lapackf77_dort01( const char *rowcol, const icla_int_t *m, const icla_int_t *n,
                         double *U, const icla_int_t *ldu,
                         double *work, const icla_int_t *lwork,
                         #ifdef ICLA_COMPLEX
                         double *rwork,
                         #endif
                         double *resid );

void   lapackf77_dlarfy( const char *uplo, const icla_int_t *n,
                         double *V, const icla_int_t *incv,
                         double *tau,
                         double *C, const icla_int_t *ldc,
                         double *work );

double lapackf77_dqpt01( const icla_int_t *m, const icla_int_t *n, const icla_int_t *k,
                         double *A,
                         double *Af, const icla_int_t *lda,
                         double *tau, icla_int_t *jpvt,
                         double *work, const icla_int_t *lwork );

void   lapackf77_dqrt02( const icla_int_t *m, const icla_int_t *n, const icla_int_t *k,
                         double *A,
                         double *AF,
                         double *Q,
                         double *R, const icla_int_t *lda,
                         double *tau,
                         double *work, const icla_int_t *lwork,
                         double *rwork,
                         double *result );

void   lapackf77_dlatms( const icla_int_t *m, const icla_int_t *n,
                         const char *dist, icla_int_t *iseed, const char *sym,
                         double *d,
                         const icla_int_t *mode, const double *cond,
                         const double *dmax,
                         const icla_int_t *kl, const icla_int_t *ku, const char *pack,
                         double *A, const icla_int_t *lda,
                         double *work,
                         icla_int_t *info );

#ifdef ICLA_WITH_MKL
void   lapackf77_dgetrf_batch(
                         icla_int_t *m_array, icla_int_t *n_array,
                         double **A_array, icla_int_t *lda_array,
                         icla_int_t **ipiv_array,
                         icla_int_t *group_count, icla_int_t *group_size,
                         icla_int_t *info_array );
#endif

#ifdef __cplusplus
}
#endif

#undef ICLA_REAL

#endif

