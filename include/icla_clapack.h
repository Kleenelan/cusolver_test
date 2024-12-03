
#ifndef ICLA_CLAPACK_H
#define ICLA_CLAPACK_H

#include "icla_types.h"
#include "icla_mangling.h"

#define ICLA_COMPLEX

#ifdef __cplusplus
extern "C" {
#endif

#define blasf77_icamax     FORTRAN_NAME( icamax, ICAMAX )
#define blasf77_caxpy      FORTRAN_NAME( caxpy,  CAXPY  )
#define blasf77_ccopy      FORTRAN_NAME( ccopy,  CCOPY  )
#define blasf77_cgbmv      FORTRAN_NAME( cgbmv,  CGBMV  )
#define blasf77_cgemm      FORTRAN_NAME( cgemm,  CGEMM  )
#define blasf77_cgemv      FORTRAN_NAME( cgemv,  CGEMV  )
#define blasf77_cgerc      FORTRAN_NAME( cgerc,  CGERC  )
#define blasf77_cgeru      FORTRAN_NAME( cgeru,  CGERU  )
#define blasf77_chemm      FORTRAN_NAME( chemm,  CHEMM  )
#define blasf77_chemv      FORTRAN_NAME( chemv,  CHEMV  )
#define blasf77_cher       FORTRAN_NAME( cher,   CHER   )
#define blasf77_cher2      FORTRAN_NAME( cher2,  CHER2  )
#define blasf77_cher2k     FORTRAN_NAME( cher2k, CHER2K )
#define blasf77_cherk      FORTRAN_NAME( cherk,  CHERK  )
#define blasf77_cscal      FORTRAN_NAME( cscal,  CSCAL  )
#define blasf77_csscal     FORTRAN_NAME( csscal, CSSCAL )
#define blasf77_cswap      FORTRAN_NAME( cswap,  CSWAP  )
#define blasf77_csymm      FORTRAN_NAME( csymm,  CSYMM  )
#define blasf77_csyr2k     FORTRAN_NAME( csyr2k, CSYR2K )
#define blasf77_csyrk      FORTRAN_NAME( csyrk,  CSYRK  )
#define blasf77_crotg      FORTRAN_NAME( crotg,  CROTG  )
#define blasf77_crot       FORTRAN_NAME( crot,   CROT   )
#define blasf77_csrot      FORTRAN_NAME( csrot,  CSROT  )
#define blasf77_ctrmm      FORTRAN_NAME( ctrmm,  CTRMM  )
#define blasf77_ctrmv      FORTRAN_NAME( ctrmv,  CTRMV  )
#define blasf77_ctrsm      FORTRAN_NAME( ctrsm,  CTRSM  )
#define blasf77_ctrsv      FORTRAN_NAME( ctrsv,  CTRSV  )

#define lapackf77_slaed2   FORTRAN_NAME( slaed2, SLAED2 )
#define lapackf77_slaed4   FORTRAN_NAME( slaed4, SLAED4 )
#define lapackf77_slaln2   FORTRAN_NAME( slaln2, SLALN2 )
#define lapackf77_slamc3   FORTRAN_NAME( slamc3, SLAMC3 )
#define lapackf77_slamrg   FORTRAN_NAME( slamrg, SLAMRG )
#define lapackf77_slasrt   FORTRAN_NAME( slasrt, SLASRT )
#define lapackf77_sstebz   FORTRAN_NAME( sstebz, SSTEBZ )

#define lapackf77_sbdsdc   FORTRAN_NAME( sbdsdc, SBDSDC )
#define lapackf77_cbdsqr   FORTRAN_NAME( cbdsqr, CBDSQR )
#define lapackf77_cgbtrf   FORTRAN_NAME( cgbtrf, CGBTRF )
#define lapackf77_cgebak   FORTRAN_NAME( cgebak, CGEBAK )
#define lapackf77_cgebal   FORTRAN_NAME( cgebal, CGEBAL )
#define lapackf77_cgebd2   FORTRAN_NAME( cgebd2, CGEBD2 )
#define lapackf77_cgebrd   FORTRAN_NAME( cgebrd, CGEBRD )
#define lapackf77_cgbbrd   FORTRAN_NAME( cgbbrd, CGBBRD )
#define lapackf77_cgbsv    FORTRAN_NAME( cgbsv,  CGBSV  )
#define lapackf77_cgbtrs   FORTRAN_NAME( cgbtrs, CGBTRS )
#define lapackf77_cgeev    FORTRAN_NAME( cgeev,  CGEEV  )
#define lapackf77_cgehd2   FORTRAN_NAME( cgehd2, CGEHD2 )
#define lapackf77_cgehrd   FORTRAN_NAME( cgehrd, CGEHRD )
#define lapackf77_cgelqf   FORTRAN_NAME( cgelqf, CGELQF )
#define lapackf77_cgels    FORTRAN_NAME( cgels,  CGELS  )
#define lapackf77_cgeqlf   FORTRAN_NAME( cgeqlf, CGEQLF )
#define lapackf77_cgeqp3   FORTRAN_NAME( cgeqp3, CGEQP3 )
#define lapackf77_cgeqrf   FORTRAN_NAME( cgeqrf, CGEQRF )
#define lapackf77_cgerqf   FORTRAN_NAME( cgerqf, CGERQF )
#define lapackf77_cgesdd   FORTRAN_NAME( cgesdd, CGESDD )
#define lapackf77_cgesv    FORTRAN_NAME( cgesv,  CGESV  )
#define lapackf77_cgesvd   FORTRAN_NAME( cgesvd, CGESVD )
#define lapackf77_cgetrf   FORTRAN_NAME( cgetrf, CGETRF )
#define lapackf77_cgetri   FORTRAN_NAME( cgetri, CGETRI )
#define lapackf77_cgetrs   FORTRAN_NAME( cgetrs, CGETRS )
#define lapackf77_cgglse   FORTRAN_NAME( cgglse, CGGLSE )
#define lapackf77_cggrqf   FORTRAN_NAME( cggrqf, CGGRQF )
#define lapackf77_chetf2   FORTRAN_NAME( chetf2, CHETF2 )
#define lapackf77_chetrs   FORTRAN_NAME( chetrs, CHETRS )
#define lapackf77_chbtrd   FORTRAN_NAME( chbtrd, CHBTRD )
#define lapackf77_cheev    FORTRAN_NAME( cheev,  CHEEV  )
#define lapackf77_cheevd   FORTRAN_NAME( cheevd, CHEEVD )
#define lapackf77_cheevr   FORTRAN_NAME( cheevr, CHEEVR )
#define lapackf77_cheevx   FORTRAN_NAME( cheevx, CHEEVX )
#define lapackf77_chegs2   FORTRAN_NAME( chegs2, CHEGS2 )
#define lapackf77_chegst   FORTRAN_NAME( chegst, CHEGST )
#define lapackf77_chegvd   FORTRAN_NAME( chegvd, CHEGVD )
#define lapackf77_chetd2   FORTRAN_NAME( chetd2, CHETD2 )
#define lapackf77_chetrd   FORTRAN_NAME( chetrd, CHETRD )
#define lapackf77_chetrf   FORTRAN_NAME( chetrf, CHETRF )
#define lapackf77_chesv    FORTRAN_NAME( chesv,  CHESV )
#define lapackf77_chseqr   FORTRAN_NAME( chseqr, CHSEQR )
#define lapackf77_clabrd   FORTRAN_NAME( clabrd, CLABRD )
#define lapackf77_clacgv   FORTRAN_NAME( clacgv, CLACGV )
#define lapackf77_clacp2   FORTRAN_NAME( clacp2, CLACP2 )
#define lapackf77_clacpy   FORTRAN_NAME( clacpy, CLACPY )
#define lapackf77_clacrm   FORTRAN_NAME( clacrm, CLACRM )
#define lapackf77_cladiv   FORTRAN_NAME( cladiv, CLADIV )
#define lapackf77_clahef   FORTRAN_NAME( clahef, CLAHEF )
#define lapackf77_clangb   FORTRAN_NAME( clangb, CLANGB )
#define lapackf77_clange   FORTRAN_NAME( clange, CLANGE )
#define lapackf77_clanhe   FORTRAN_NAME( clanhe, CLANHE )
#define lapackf77_clanht   FORTRAN_NAME( clanht, CLANHT )
#define lapackf77_clansy   FORTRAN_NAME( clansy, CLANSY )
#define lapackf77_clantr   FORTRAN_NAME( clantr, CLANTR )
#define lapackf77_slapy3   FORTRAN_NAME( slapy3, SLAPY3 )
#define lapackf77_claqp2   FORTRAN_NAME( claqp2, CLAQP2 )
#define lapackf77_clarcm   FORTRAN_NAME( clarcm, CLARCM )
#define lapackf77_clarf    FORTRAN_NAME( clarf,  CLARF  )
#define lapackf77_clarfb   FORTRAN_NAME( clarfb, CLARFB )
#define lapackf77_clarfg   FORTRAN_NAME( clarfg, CLARFG )
#define lapackf77_clarft   FORTRAN_NAME( clarft, CLARFT )
#define lapackf77_clarfx   FORTRAN_NAME( clarfx, CLARFX )
#define lapackf77_clarnv   FORTRAN_NAME( clarnv, CLARNV )
#define lapackf77_clartg   FORTRAN_NAME( clartg, CLARTG )
#define lapackf77_clascl   FORTRAN_NAME( clascl, CLASCL )
#define lapackf77_claset   FORTRAN_NAME( claset, CLASET )
#define lapackf77_claswp   FORTRAN_NAME( claswp, CLASWP )
#define lapackf77_clatrd   FORTRAN_NAME( clatrd, CLATRD )
#define lapackf77_clatrs   FORTRAN_NAME( clatrs, CLATRS )
#define lapackf77_clauum   FORTRAN_NAME( clauum, CLAUUM )
#define lapackf77_clavhe   FORTRAN_NAME( clavhe, CLAVHE )
#define lapackf77_cposv    FORTRAN_NAME( cposv,  CPOSV  )
#define lapackf77_cpotrf   FORTRAN_NAME( cpotrf, CPOTRF )
#define lapackf77_cpotri   FORTRAN_NAME( cpotri, CPOTRI )
#define lapackf77_cpotrs   FORTRAN_NAME( cpotrs, CPOTRS )
#define lapackf77_cstedc   FORTRAN_NAME( cstedc, CSTEDC )
#define lapackf77_cstein   FORTRAN_NAME( cstein, CSTEIN )
#define lapackf77_cstemr   FORTRAN_NAME( cstemr, CSTEMR )
#define lapackf77_csteqr   FORTRAN_NAME( csteqr, CSTEQR )
#define lapackf77_csymv    FORTRAN_NAME( csymv,  CSYMV  )
#define lapackf77_csyr     FORTRAN_NAME( csyr,   CSYR   )
#define lapackf77_csysv    FORTRAN_NAME( csysv,  CSYSV  )
#define lapackf77_ctrevc   FORTRAN_NAME( ctrevc, CTREVC )
#define lapackf77_ctrevc3  FORTRAN_NAME( ctrevc3, CTREVC3 )
#define lapackf77_ctrtri   FORTRAN_NAME( ctrtri, CTRTRI )
#define lapackf77_cung2r   FORTRAN_NAME( cung2r, CUNG2R )
#define lapackf77_cungbr   FORTRAN_NAME( cungbr, CUNGBR )
#define lapackf77_cunghr   FORTRAN_NAME( cunghr, CUNGHR )
#define lapackf77_cunglq   FORTRAN_NAME( cunglq, CUNGLQ )
#define lapackf77_cungql   FORTRAN_NAME( cungql, CUNGQL )
#define lapackf77_cungqr   FORTRAN_NAME( cungqr, CUNGQR )
#define lapackf77_cungtr   FORTRAN_NAME( cungtr, CUNGTR )
#define lapackf77_cunm2r   FORTRAN_NAME( cunm2r, CUNM2R )
#define lapackf77_cunmbr   FORTRAN_NAME( cunmbr, CUNMBR )
#define lapackf77_cunmlq   FORTRAN_NAME( cunmlq, CUNMLQ )
#define lapackf77_cunmql   FORTRAN_NAME( cunmql, CUNMQL )
#define lapackf77_cunmqr   FORTRAN_NAME( cunmqr, CUNMQR )
#define lapackf77_cunmrq   FORTRAN_NAME( cunmrq, CUNMRQ )
#define lapackf77_cunmtr   FORTRAN_NAME( cunmtr, CUNMTR )

#ifdef ICLA_WITH_MKL
#define lapackf77_cgetrf_batch   FORTRAN_NAME( cgetrf_batch, CGETRF_BATCH )
#endif

#define lapackf77_cbdt01   FORTRAN_NAME( cbdt01, CBDT01 )
#define lapackf77_cget22   FORTRAN_NAME( cget22, CGET22 )
#define lapackf77_chet21   FORTRAN_NAME( chet21, CHET21 )
#define lapackf77_chet22   FORTRAN_NAME( chet22, CHET22 )
#define lapackf77_chst01   FORTRAN_NAME( chst01, CHST01 )
#define lapackf77_clarfy   FORTRAN_NAME( clarfy, CLARFY )
#define lapackf77_clatms   FORTRAN_NAME( clatms, CLATMS )
#define lapackf77_cqpt01   FORTRAN_NAME( cqpt01, CQPT01 )
#define lapackf77_cqrt02   FORTRAN_NAME( cqrt02, CQRT02 )
#define lapackf77_cstt21   FORTRAN_NAME( cstt21, CSTT21 )
#define lapackf77_cunt01   FORTRAN_NAME( cunt01, CUNT01 )

icla_int_t blasf77_icamax(
                     const icla_int_t *n,
                     const iclaFloatComplex *x, const icla_int_t *incx );

void blasf77_caxpy(  const icla_int_t *n,
                     const iclaFloatComplex *alpha,
                     const iclaFloatComplex *x, const icla_int_t *incx,
                           iclaFloatComplex *y, const icla_int_t *incy );

void blasf77_ccopy(  const icla_int_t *n,
                     const iclaFloatComplex *x, const icla_int_t *incx,
                           iclaFloatComplex *y, const icla_int_t *incy );

void blasf77_cgbmv(  const char *transa,
                     const icla_int_t *m,  const icla_int_t *n,
                     const icla_int_t *kl, const icla_int_t *ku,
                     const iclaFloatComplex *alpha,
                     const iclaFloatComplex *A, const icla_int_t *lda,
                     const iclaFloatComplex *x, const icla_int_t *incx,
                     const iclaFloatComplex *beta,
                           iclaFloatComplex *y, const icla_int_t *incy );

void blasf77_cgemm(  const char *transa, const char *transb,
                     const icla_int_t *m, const icla_int_t *n, const icla_int_t *k,
                     const iclaFloatComplex *alpha,
                     const iclaFloatComplex *A, const icla_int_t *lda,
                     const iclaFloatComplex *B, const icla_int_t *ldb,
                     const iclaFloatComplex *beta,
                           iclaFloatComplex *C, const icla_int_t *ldc );

void blasf77_cgemv(  const char *transa,
                     const icla_int_t *m, const icla_int_t *n,
                     const iclaFloatComplex *alpha,
                     const iclaFloatComplex *A, const icla_int_t *lda,
                     const iclaFloatComplex *x, const icla_int_t *incx,
                     const iclaFloatComplex *beta,
                           iclaFloatComplex *y, const icla_int_t *incy );

void blasf77_cgerc(  const icla_int_t *m, const icla_int_t *n,
                     const iclaFloatComplex *alpha,
                     const iclaFloatComplex *x, const icla_int_t *incx,
                     const iclaFloatComplex *y, const icla_int_t *incy,
                           iclaFloatComplex *A, const icla_int_t *lda );

void blasf77_cgeru(  const icla_int_t *m, const icla_int_t *n,
                     const iclaFloatComplex *alpha,
                     const iclaFloatComplex *x, const icla_int_t *incx,
                     const iclaFloatComplex *y, const icla_int_t *incy,
                           iclaFloatComplex *A, const icla_int_t *lda );

void blasf77_chemm(  const char *side, const char *uplo,
                     const icla_int_t *m, const icla_int_t *n,
                     const iclaFloatComplex *alpha,
                     const iclaFloatComplex *A, const icla_int_t *lda,
                     const iclaFloatComplex *B, const icla_int_t *ldb,
                     const iclaFloatComplex *beta,
                           iclaFloatComplex *C, const icla_int_t *ldc );

void blasf77_chemv(  const char *uplo,
                     const icla_int_t *n,
                     const iclaFloatComplex *alpha,
                     const iclaFloatComplex *A, const icla_int_t *lda,
                     const iclaFloatComplex *x, const icla_int_t *incx,
                     const iclaFloatComplex *beta,
                           iclaFloatComplex *y, const icla_int_t *incy );

void blasf77_cher(   const char *uplo,
                     const icla_int_t *n,
                     const float *alpha,
                     const iclaFloatComplex *x, const icla_int_t *incx,
                           iclaFloatComplex *A, const icla_int_t *lda );

void blasf77_cher2(  const char *uplo,
                     const icla_int_t *n,
                     const iclaFloatComplex *alpha,
                     const iclaFloatComplex *x, const icla_int_t *incx,
                     const iclaFloatComplex *y, const icla_int_t *incy,
                           iclaFloatComplex *A, const icla_int_t *lda );

void blasf77_cher2k( const char *uplo, const char *trans,
                     const icla_int_t *n, const icla_int_t *k,
                     const iclaFloatComplex *alpha,
                     const iclaFloatComplex *A, const icla_int_t *lda,
                     const iclaFloatComplex *B, const icla_int_t *ldb,
                     const float *beta,
                           iclaFloatComplex *C, const icla_int_t *ldc );

void blasf77_cherk(  const char *uplo, const char *trans,
                     const icla_int_t *n, const icla_int_t *k,
                     const float *alpha,
                     const iclaFloatComplex *A, const icla_int_t *lda,
                     const float *beta,
                           iclaFloatComplex *C, const icla_int_t *ldc );

void blasf77_cscal(  const icla_int_t *n,
                     const iclaFloatComplex *alpha,
                           iclaFloatComplex *x, const icla_int_t *incx );

void blasf77_csscal( const icla_int_t *n,
                     const float *alpha,
                           iclaFloatComplex *x, const icla_int_t *incx );

void blasf77_cswap(  const icla_int_t *n,
                     iclaFloatComplex *x, const icla_int_t *incx,
                     iclaFloatComplex *y, const icla_int_t *incy );

void blasf77_csymm(  const char *side, const char *uplo,
                     const icla_int_t *m, const icla_int_t *n,
                     const iclaFloatComplex *alpha,
                     const iclaFloatComplex *A, const icla_int_t *lda,
                     const iclaFloatComplex *B, const icla_int_t *ldb,
                     const iclaFloatComplex *beta,
                           iclaFloatComplex *C, const icla_int_t *ldc );

void blasf77_csyr2k( const char *uplo, const char *trans,
                     const icla_int_t *n, const icla_int_t *k,
                     const iclaFloatComplex *alpha,
                     const iclaFloatComplex *A, const icla_int_t *lda,
                     const iclaFloatComplex *B, const icla_int_t *ldb,
                     const iclaFloatComplex *beta,
                           iclaFloatComplex *C, const icla_int_t *ldc );

void blasf77_csyrk(  const char *uplo, const char *trans,
                     const icla_int_t *n, const icla_int_t *k,
                     const iclaFloatComplex *alpha,
                     const iclaFloatComplex *A, const icla_int_t *lda,
                     const iclaFloatComplex *beta,
                           iclaFloatComplex *C, const icla_int_t *ldc );

void blasf77_crotg(  iclaFloatComplex *ca, const iclaFloatComplex *cb,
                     float *c, iclaFloatComplex *s );

void blasf77_crot(   const icla_int_t *n,
                     iclaFloatComplex *x, const icla_int_t *incx,
                     iclaFloatComplex *y, const icla_int_t *incy,
                     const float *c, const iclaFloatComplex *s );

void blasf77_csrot(  const icla_int_t *n,
                     iclaFloatComplex *x, const icla_int_t *incx,
                     iclaFloatComplex *y, const icla_int_t *incy,
                     const float *c, const float *s );

void blasf77_ctrmm(  const char *side, const char *uplo, const char *transa, const char *diag,
                     const icla_int_t *m, const icla_int_t *n,
                     const iclaFloatComplex *alpha,
                     const iclaFloatComplex *A, const icla_int_t *lda,
                           iclaFloatComplex *B, const icla_int_t *ldb );

void blasf77_ctrmv(  const char *uplo, const char *transa, const char *diag,
                     const icla_int_t *n,
                     const iclaFloatComplex *A, const icla_int_t *lda,
                           iclaFloatComplex *x, const icla_int_t *incx );

void blasf77_ctrsm(  const char *side, const char *uplo, const char *transa, const char *diag,
                     const icla_int_t *m, const icla_int_t *n,
                     const iclaFloatComplex *alpha,
                     const iclaFloatComplex *A, const icla_int_t *lda,
                           iclaFloatComplex *B, const icla_int_t *ldb );

void blasf77_ctrsv(  const char *uplo, const char *transa, const char *diag,
                     const icla_int_t *n,
                     const iclaFloatComplex *A, const icla_int_t *lda,
                           iclaFloatComplex *x, const icla_int_t *incx );

float icla_cblas_scasum(
    icla_int_t n,
    const iclaFloatComplex *x, icla_int_t incx );

float icla_cblas_scnrm2(
    icla_int_t n,
    const iclaFloatComplex *x, icla_int_t incx );

iclaFloatComplex icla_cblas_cdotc(
    icla_int_t n,
    const iclaFloatComplex *x, icla_int_t incx,
    const iclaFloatComplex *y, icla_int_t incy );

iclaFloatComplex icla_cblas_cdotu(
    icla_int_t n,
    const iclaFloatComplex *x, icla_int_t incx,
    const iclaFloatComplex *y, icla_int_t incy );

#ifdef ICLA_REAL
void   lapackf77_sbdsdc( const char *uplo, const char *compq,
                         const icla_int_t *n,
                         float *d, float *e,
                         float *U,  const icla_int_t *ldu,
                         float *VT, const icla_int_t *ldvt,
                         float *Q, icla_int_t *IQ,
                         float *work, icla_int_t *iwork,
                         icla_int_t *info );
#endif

void   lapackf77_cbdsqr( const char *uplo,
                         const icla_int_t *n, const icla_int_t *ncvt, const icla_int_t *nru,  const icla_int_t *ncc,
                         float *d, float *e,
                         iclaFloatComplex *Vt, const icla_int_t *ldvt,
                         iclaFloatComplex *U, const icla_int_t *ldu,
                         iclaFloatComplex *C, const icla_int_t *ldc,
                         float *work,
                         icla_int_t *info );

void   lapackf77_cgbtrf( const icla_int_t  *m,  const icla_int_t *n,
                         const icla_int_t  *kl, const icla_int_t *ku,
                         iclaFloatComplex *AB, const icla_int_t *ldab,
                         icla_int_t *ipiv,
                         icla_int_t *info );

void   lapackf77_cgebak( const char *job, const char *side,
                         const icla_int_t *n,
                         const icla_int_t *ilo, const icla_int_t *ihi,
                         const float *scale, const icla_int_t *m,
                         iclaFloatComplex *V, const icla_int_t *ldv,
                         icla_int_t *info );

void   lapackf77_cgebal( const char *job,
                         const icla_int_t *n,
                         iclaFloatComplex *A, const icla_int_t *lda,
                         icla_int_t *ilo, icla_int_t *ihi,
                         float *scale,
                         icla_int_t *info );

void   lapackf77_cgebd2( const icla_int_t *m, const icla_int_t *n,
                         iclaFloatComplex *A, const icla_int_t *lda,
                         float *d, float *e,
                         iclaFloatComplex *tauq,
                         iclaFloatComplex *taup,
                         iclaFloatComplex *work,
                         icla_int_t *info );

void   lapackf77_cgebrd( const icla_int_t *m, const icla_int_t *n,
                         iclaFloatComplex *A, const icla_int_t *lda,
                         float *d, float *e,
                         iclaFloatComplex *tauq,
                         iclaFloatComplex *taup,
                         iclaFloatComplex *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_cgbbrd( const char *vect, const icla_int_t *m,
                         const icla_int_t *n, const icla_int_t *ncc,
                         const icla_int_t *kl, const icla_int_t *ku,
                         iclaFloatComplex *Ab, const icla_int_t *ldab,
                         float *d, float *e,
                         iclaFloatComplex *Q, const icla_int_t *ldq,
                         iclaFloatComplex *PT, const icla_int_t *ldpt,
                         iclaFloatComplex *C, const icla_int_t *ldc,
                         iclaFloatComplex *work,
                         #ifdef ICLA_COMPLEX
                         float *rwork,
                         #endif
                         icla_int_t *info );

void   lapackf77_cgbsv( const icla_int_t *n,
                        const icla_int_t *kl, const icla_int_t *ku,
                        const icla_int_t *nrhs,
                        iclaFloatComplex *ab, const icla_int_t *ldab,
                        icla_int_t *ipiv,
                        iclaFloatComplex *B, const icla_int_t *ldb,
                        icla_int_t *info );

void   lapackf77_cgbtrs( const char *trans,
                         const icla_int_t *n,
                         const icla_int_t *kl, const icla_int_t *ku,
                         const icla_int_t *nrhs,
                         iclaFloatComplex *ab, const icla_int_t *ldab,
                         icla_int_t *ipiv,
                         iclaFloatComplex *B, const icla_int_t *ldb,
                         icla_int_t *info );

void   lapackf77_cgeev(  const char *jobvl, const char *jobvr,
                         const icla_int_t *n,
                         iclaFloatComplex *A,    const icla_int_t *lda,
                         #ifdef ICLA_COMPLEX
                         iclaFloatComplex *w,
                         #else
                         float *wr, float *wi,
                         #endif
                         iclaFloatComplex *Vl,   const icla_int_t *ldvl,
                         iclaFloatComplex *Vr,   const icla_int_t *ldvr,
                         iclaFloatComplex *work, const icla_int_t *lwork,
                         #ifdef ICLA_COMPLEX
                         float *rwork,
                         #endif
                         icla_int_t *info );

void   lapackf77_cgehd2( const icla_int_t *n,
                         const icla_int_t *ilo, const icla_int_t *ihi,
                         iclaFloatComplex *A, const icla_int_t *lda,
                         iclaFloatComplex *tau,
                         iclaFloatComplex *work,
                         icla_int_t *info );

void   lapackf77_cgehrd( const icla_int_t *n,
                         const icla_int_t *ilo, const icla_int_t *ihi,
                         iclaFloatComplex *A, const icla_int_t *lda,
                         iclaFloatComplex *tau,
                         iclaFloatComplex *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_cgelqf( const icla_int_t *m, const icla_int_t *n,
                         iclaFloatComplex *A, const icla_int_t *lda,
                         iclaFloatComplex *tau,
                         iclaFloatComplex *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_cgels(  const char *trans,
                         const icla_int_t *m, const icla_int_t *n, const icla_int_t *nrhs,
                         iclaFloatComplex *A, const icla_int_t *lda,
                         iclaFloatComplex *B, const icla_int_t *ldb,
                         iclaFloatComplex *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_cgeqlf( const icla_int_t *m, const icla_int_t *n,
                         iclaFloatComplex *A, const icla_int_t *lda,
                         iclaFloatComplex *tau,
                         iclaFloatComplex *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_cgeqp3( const icla_int_t *m, const icla_int_t *n,
                         iclaFloatComplex *A, const icla_int_t *lda,
                         icla_int_t *jpvt,
                         iclaFloatComplex *tau,
                         iclaFloatComplex *work, const icla_int_t *lwork,
                         #ifdef ICLA_COMPLEX
                         float *rwork,
                         #endif
                         icla_int_t *info );

void   lapackf77_cgeqrf( const icla_int_t *m, const icla_int_t *n,
                         iclaFloatComplex *A, const icla_int_t *lda,
                         iclaFloatComplex *tau,
                         iclaFloatComplex *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_cgerqf( const icla_int_t *m, const icla_int_t *n,
                         iclaFloatComplex *A, const icla_int_t *lda,
                         iclaFloatComplex *tau,
                         iclaFloatComplex *work, const icla_int_t *lwork,
                         icla_int_t *info);

void   lapackf77_cgesdd( const char *jobz,
                         const icla_int_t *m, const icla_int_t *n,
                         iclaFloatComplex *A, const icla_int_t *lda,
                         float *s,
                         iclaFloatComplex *U,  const icla_int_t *ldu,
                         iclaFloatComplex *Vt, const icla_int_t *ldvt,
                         iclaFloatComplex *work, const icla_int_t *lwork,
                         #ifdef ICLA_COMPLEX
                         float *rwork,
                         #endif
                         icla_int_t *iwork,
                         icla_int_t *info );

void   lapackf77_cgesv(  const icla_int_t *n, const icla_int_t *nrhs,
                         iclaFloatComplex *A, const icla_int_t *lda,
                         icla_int_t *ipiv,
                         iclaFloatComplex *B,  const icla_int_t *ldb,
                         icla_int_t *info );

void   lapackf77_cgesvd( const char *jobu, const char *jobvt,
                         const icla_int_t *m, const icla_int_t *n,
                         iclaFloatComplex *A, const icla_int_t *lda,
                         float *s,
                         iclaFloatComplex *U,  const icla_int_t *ldu,
                         iclaFloatComplex *Vt, const icla_int_t *ldvt,
                         iclaFloatComplex *work, const icla_int_t *lwork,
                         #ifdef ICLA_COMPLEX
                         float *rwork,
                         #endif
                         icla_int_t *info );

void   lapackf77_cgetrf( const icla_int_t *m, const icla_int_t *n,
                         iclaFloatComplex *A, const icla_int_t *lda,
                         icla_int_t *ipiv,
                         icla_int_t *info );

void   lapackf77_cgetri( const icla_int_t *n,
                         iclaFloatComplex *A, const icla_int_t *lda,
                         const icla_int_t *ipiv,
                         iclaFloatComplex *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_cgetrs( const char *trans,
                         const icla_int_t *n, const icla_int_t *nrhs,
                         const iclaFloatComplex *A, const icla_int_t *lda,
                         const icla_int_t *ipiv,
                         iclaFloatComplex *B, const icla_int_t *ldb,
                         icla_int_t *info );

void   lapackf77_cgglse( icla_int_t *m, icla_int_t *n, icla_int_t *p,
                         iclaFloatComplex *A, icla_int_t *lda,
                         iclaFloatComplex *B, icla_int_t *ldb,
                         iclaFloatComplex *c, iclaFloatComplex *d,
                         iclaFloatComplex *x,
                         iclaFloatComplex *work, icla_int_t *lwork,
                         icla_int_t *info);

void   lapackf77_cggrqf( icla_int_t *m, icla_int_t *p, icla_int_t *n,
                         iclaFloatComplex *A, icla_int_t *lda,
                         iclaFloatComplex *tauA, iclaFloatComplex *B,
                         icla_int_t *ldb, iclaFloatComplex *tauB,
                         iclaFloatComplex *work, icla_int_t *lwork,
                         icla_int_t *info);

void   lapackf77_chetf2( const char *uplo, const icla_int_t *n,
                         iclaFloatComplex *A, const icla_int_t *lda,
                         icla_int_t *ipiv,
                         icla_int_t *info );

void   lapackf77_chetrs( const char *uplo,
                         const icla_int_t *n, const icla_int_t *nrhs,
                         const iclaFloatComplex *A, const icla_int_t *lda,
                         const icla_int_t *ipiv,
                         iclaFloatComplex *B, const icla_int_t *ldb,
                         icla_int_t *info );

void   lapackf77_chbtrd( const char *vect, const char *uplo,
                         const icla_int_t *n, const icla_int_t *kd,
                         iclaFloatComplex *Ab, const icla_int_t *ldab,
                         float *d, float *e,
                         iclaFloatComplex *Q, const icla_int_t *ldq,
                         iclaFloatComplex *work,
                         icla_int_t *info );

void   lapackf77_cheev(  const char *jobz, const char *uplo,
                         const icla_int_t *n,
                         iclaFloatComplex *A, const icla_int_t *lda,
                         float *w,
                         iclaFloatComplex *work, const icla_int_t *lwork,
                         #ifdef ICLA_COMPLEX
                         float *rwork,
                         #endif
                         icla_int_t *info );

void   lapackf77_cheevd( const char *jobz, const char *uplo,
                         const icla_int_t *n,
                         iclaFloatComplex *A, const icla_int_t *lda,
                         float *w,
                         iclaFloatComplex *work, const icla_int_t *lwork,
                         #ifdef ICLA_COMPLEX
                         float *rwork, const icla_int_t *lrwork,
                         #endif
                         icla_int_t *iwork, const icla_int_t *liwork,
                         icla_int_t *info );

void   lapackf77_cheevr( const char *jobz, const char *range, const char *uplo,
                         const icla_int_t *n,
                         iclaFloatComplex *A, const icla_int_t *lda,
                         const float *vl, const float *vu,
                         const icla_int_t *il, const icla_int_t *iu,
                         const float *abstol,
                         icla_int_t *m, float *w,
                         iclaFloatComplex *Z, const icla_int_t *ldz,
                         icla_int_t *isuppz,
                         iclaFloatComplex *work, const icla_int_t *lwork,
                         #ifdef ICLA_COMPLEX
                         float *rwork, const icla_int_t *lrwork,
                         #endif
                         icla_int_t *iwork, const icla_int_t *liwork,
                         icla_int_t *info);

void   lapackf77_cheevx( const char *jobz, const char *range, const char *uplo,
                         const icla_int_t *n,
                         iclaFloatComplex *A, const icla_int_t *lda,
                         const float *vl, const float *vu,
                         const icla_int_t *il, const icla_int_t *iu,
                         const float *abstol,
                         icla_int_t *m, float *w,
                         iclaFloatComplex *Z, const icla_int_t *ldz,
                         iclaFloatComplex *work, const icla_int_t *lwork,
                         #ifdef ICLA_COMPLEX
                         float *rwork,
                         #endif
                         icla_int_t *iwork, icla_int_t *ifail,
                         icla_int_t *info);

void   lapackf77_chegs2( const icla_int_t *itype, const char *uplo,
                         const icla_int_t *n,
                         iclaFloatComplex *A, const icla_int_t *lda,
                         const iclaFloatComplex *B, const icla_int_t *ldb,
                         icla_int_t *info );

void   lapackf77_chegst( const icla_int_t *itype, const char *uplo,
                         const icla_int_t *n,
                         iclaFloatComplex *A, const icla_int_t *lda,
                         const iclaFloatComplex *B, const icla_int_t *ldb,
                         icla_int_t *info );

void   lapackf77_chegvd( const icla_int_t *itype, const char *jobz, const char *uplo,
                         const icla_int_t *n,
                         iclaFloatComplex *A, const icla_int_t *lda,
                         iclaFloatComplex *B, const icla_int_t *ldb,
                         float *w,
                         iclaFloatComplex *work, const icla_int_t *lwork,
                         #ifdef ICLA_COMPLEX
                         float *rwork, const icla_int_t *lrwork,
                         #endif
                         icla_int_t *iwork, const icla_int_t *liwork,
                         icla_int_t *info );

void   lapackf77_chesv( const char *uplo,
                        const icla_int_t *n, const icla_int_t *nrhs,
                        iclaFloatComplex *A, const icla_int_t *lda, icla_int_t *ipiv,
                        iclaFloatComplex *B, const icla_int_t *ldb,
                        iclaFloatComplex *work, const icla_int_t *lwork,
                        icla_int_t *info );

void   lapackf77_chetd2( const char *uplo,
                         const icla_int_t *n,
                         iclaFloatComplex *A, const icla_int_t *lda,
                         float *d, float *e,
                         iclaFloatComplex *tau,
                         icla_int_t *info );

void   lapackf77_chetrd( const char *uplo,
                         const icla_int_t *n,
                         iclaFloatComplex *A, const icla_int_t *lda,
                         float *d, float *e,
                         iclaFloatComplex *tau,
                         iclaFloatComplex *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_chetrf( const char *uplo,
                         const icla_int_t *n,
                         iclaFloatComplex *A, const icla_int_t *lda,
                         icla_int_t *ipiv,
                         iclaFloatComplex *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_chseqr( const char *job, const char *compz,
                         const icla_int_t *n,
                         const icla_int_t *ilo, const icla_int_t *ihi,
                         iclaFloatComplex *H, const icla_int_t *ldh,
                         #ifdef ICLA_COMPLEX
                         iclaFloatComplex *w,
                         #else
                         float *wr, float *wi,
                         #endif
                         iclaFloatComplex *Z, const icla_int_t *ldz,
                         iclaFloatComplex *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_clabrd( const icla_int_t *m, const icla_int_t *n, const icla_int_t *nb,
                         iclaFloatComplex *A, const icla_int_t *lda,
                         float *d, float *e,
                         iclaFloatComplex *tauq,
                         iclaFloatComplex *taup,
                         iclaFloatComplex *X, const icla_int_t *ldx,
                         iclaFloatComplex *Y, const icla_int_t *ldy );

#ifdef ICLA_COMPLEX
void   lapackf77_clacgv( const icla_int_t *n,
                         iclaFloatComplex *x, const icla_int_t *incx );
#endif

#ifdef ICLA_COMPLEX
void   lapackf77_clacp2( const char *uplo,
                         const icla_int_t *m, const icla_int_t *n,
                         const float *A, const icla_int_t *lda,
                         iclaFloatComplex *B, const icla_int_t *ldb );
#endif

void   lapackf77_clacpy( const char *uplo,
                         const icla_int_t *m, const icla_int_t *n,
                         const iclaFloatComplex *A, const icla_int_t *lda,
                         iclaFloatComplex *B, const icla_int_t *ldb );

#ifdef ICLA_COMPLEX
void   lapackf77_clacrm( const icla_int_t *m, const icla_int_t *n,
                         const iclaFloatComplex *A, const icla_int_t *lda,
                         const float             *B, const icla_int_t *ldb,
                         iclaFloatComplex       *C, const icla_int_t *ldc,
                         float *rwork );
#endif

#ifdef ICLA_COMPLEX
void   lapackf77_cladiv( iclaFloatComplex *ret_val,
                         const iclaFloatComplex *x,
                         const iclaFloatComplex *y );
#else

void   lapackf77_cladiv( const float *a, const float *b,
                         const float *c, const float *d,
                         float *p, float *q );
#endif

void   lapackf77_clahef( const char *uplo,
                         const icla_int_t *n, const icla_int_t *nb,
                         icla_int_t *kb,
                         iclaFloatComplex *A, const icla_int_t *lda,
                         icla_int_t *ipiv,
                         iclaFloatComplex *work, const icla_int_t *ldwork,
                         icla_int_t *info );

float lapackf77_clangb( const char *norm,
                         const icla_int_t *n, const icla_int_t *kl, const icla_int_t *ku,
                         const iclaFloatComplex *AB, const icla_int_t *ldab,
                         float *work );

float lapackf77_clange( const char *norm,
                         const icla_int_t *m, const icla_int_t *n,
                         const iclaFloatComplex *A, const icla_int_t *lda,
                         float *work );

float lapackf77_clanhe( const char *norm, const char *uplo,
                         const icla_int_t *n,
                         const iclaFloatComplex *A, const icla_int_t *lda,
                         float *work );

float lapackf77_clanht( const char *norm, const icla_int_t *n,
                         const float *d, const iclaFloatComplex *e );

float lapackf77_clansy( const char *norm, const char *uplo,
                         const icla_int_t *n,
                         const iclaFloatComplex *A, const icla_int_t *lda,
                         float *work );

float lapackf77_clantr( const char *norm, const char *uplo, const char *diag,
                         const icla_int_t *m, const icla_int_t *n,
                         const iclaFloatComplex *A, const icla_int_t *lda,
                         float *work );

void   lapackf77_claqp2( const icla_int_t *m, const icla_int_t *n, const icla_int_t *offset,
                         iclaFloatComplex *A, const icla_int_t *lda,
                         icla_int_t *jpvt,
                         iclaFloatComplex *tau,
                         float *vn1, float *vn2,
                         iclaFloatComplex *work );

#ifdef ICLA_COMPLEX
void   lapackf77_clarcm( const icla_int_t *m, const icla_int_t *n,
                         const float             *A, const icla_int_t *lda,
                         const iclaFloatComplex *B, const icla_int_t *ldb,
                         iclaFloatComplex       *C, const icla_int_t *ldc,
                         float *rwork );
#endif

void   lapackf77_clarf(  const char *side, const icla_int_t *m, const icla_int_t *n,
                         const iclaFloatComplex *v, const icla_int_t *incv,
                         const iclaFloatComplex *tau,
                         iclaFloatComplex *C, const icla_int_t *ldc,
                         iclaFloatComplex *work );

void   lapackf77_clarfb( const char *side, const char *trans, const char *direct, const char *storev,
                         const icla_int_t *m, const icla_int_t *n, const icla_int_t *k,
                         const iclaFloatComplex *V, const icla_int_t *ldv,
                         const iclaFloatComplex *T, const icla_int_t *ldt,
                         iclaFloatComplex *C, const icla_int_t *ldc,
                         iclaFloatComplex *work, const icla_int_t *ldwork );

void   lapackf77_clarfg( const icla_int_t *n,
                         iclaFloatComplex *alpha,
                         iclaFloatComplex *x, const icla_int_t *incx,
                         iclaFloatComplex *tau );

void   lapackf77_clarft( const char *direct, const char *storev,
                         const icla_int_t *n, const icla_int_t *k,
                         const iclaFloatComplex *V, const icla_int_t *ldv,
                         const iclaFloatComplex *tau,
                         iclaFloatComplex *T, const icla_int_t *ldt );

void   lapackf77_clarfx( const char *side, const icla_int_t *m, const icla_int_t *n,
                         const iclaFloatComplex *V,
                         const iclaFloatComplex *tau,
                         iclaFloatComplex *C, const icla_int_t *ldc,
                         iclaFloatComplex *work );

void   lapackf77_clarnv( const icla_int_t *idist, icla_int_t *iseed, const icla_int_t *n,
                         iclaFloatComplex *x );

void   lapackf77_clartg( const iclaFloatComplex *f,
                         const iclaFloatComplex *g,
                         float *cs,
                         iclaFloatComplex *sn,
                         iclaFloatComplex *r );

void   lapackf77_clascl( const char *type,
                         const icla_int_t *kl, const icla_int_t *ku,
                         const float *cfrom,
                         const float *cto,
                         const icla_int_t *m, const icla_int_t *n,
                         iclaFloatComplex *A, const icla_int_t *lda,
                         icla_int_t *info );

void   lapackf77_claset( const char *uplo,
                         const icla_int_t *m, const icla_int_t *n,
                         const iclaFloatComplex *alpha,
                         const iclaFloatComplex *beta,
                         iclaFloatComplex *A, const icla_int_t *lda );

void   lapackf77_claswp( const icla_int_t *n,
                         iclaFloatComplex *A, const icla_int_t *lda,
                         const icla_int_t *k1, const icla_int_t *k2,
                         const icla_int_t *ipiv,
                         const icla_int_t *incx );

void   lapackf77_clatrd( const char *uplo,
                         const icla_int_t *n, const icla_int_t *nb,
                         iclaFloatComplex *A, const icla_int_t *lda,
                         float *e,
                         iclaFloatComplex *tau,
                         iclaFloatComplex *work, const icla_int_t *ldwork );

void   lapackf77_clatrs( const char *uplo, const char *trans, const char *diag,
                         const char *normin,
                         const icla_int_t *n,
                         const iclaFloatComplex *A, const icla_int_t *lda,
                         iclaFloatComplex *x, float *scale,
                         float *cnorm,
                         icla_int_t *info );

void   lapackf77_clauum( const char *uplo,
                         const icla_int_t *n,
                         iclaFloatComplex *A, const icla_int_t *lda,
                         icla_int_t *info );

void   lapackf77_clavhe( const char *uplo, const char *trans, const char *diag,
                         icla_int_t *n, icla_int_t *nrhs,
                         iclaFloatComplex *A, icla_int_t *lda,
                         icla_int_t *ipiv,
                         iclaFloatComplex *B, icla_int_t *ldb,
                         icla_int_t *info );

void   lapackf77_cposv(  const char *uplo,
                         const icla_int_t *n, const icla_int_t *nrhs,
                         iclaFloatComplex *A, const icla_int_t *lda,
                         iclaFloatComplex *B,  const icla_int_t *ldb,
                         icla_int_t *info );

void   lapackf77_cpotrf( const char *uplo,
                         const icla_int_t *n,
                         iclaFloatComplex *A, const icla_int_t *lda,
                         icla_int_t *info );

void   lapackf77_cpotri( const char *uplo,
                         const icla_int_t *n,
                         iclaFloatComplex *A, const icla_int_t *lda,
                         icla_int_t *info );

void   lapackf77_cpotrs( const char *uplo,
                         const icla_int_t *n, const icla_int_t *nrhs,
                         const iclaFloatComplex *A, const icla_int_t *lda,
                         iclaFloatComplex *B, const icla_int_t *ldb,
                         icla_int_t *info );

void   lapackf77_cstedc( const char *compz,
                         const icla_int_t *n,
                         float *d, float *e,
                         iclaFloatComplex *Z, const icla_int_t *ldz,
                         iclaFloatComplex *work, const icla_int_t *lwork,
                         #ifdef ICLA_COMPLEX
                         float *rwork, const icla_int_t *lrwork,
                         #endif
                         icla_int_t *iwork, const icla_int_t *liwork,
                         icla_int_t *info );

void   lapackf77_cstein( const icla_int_t *n,
                         const float *d, const float *e,
                         const icla_int_t *m,
                         const float *w,
                         const icla_int_t *iblock,
                         const icla_int_t *isplit,
                         iclaFloatComplex *Z, const icla_int_t *ldz,
                         float *work, icla_int_t *iwork, icla_int_t *ifailv,
                         icla_int_t *info );

void   lapackf77_cstemr( const char *jobz, const char *range,
                         const icla_int_t *n,
                         float *d, float *e,
                         const float *vl, const float *vu,
                         const icla_int_t *il, const icla_int_t *iu,
                         icla_int_t *m,
                         float *w,
                         iclaFloatComplex *Z, const icla_int_t *ldz,
                         const icla_int_t *nzc, icla_int_t *isuppz, icla_int_t *tryrac,
                         float *work, const icla_int_t *lwork,
                         icla_int_t *iwork, const icla_int_t *liwork,
                         icla_int_t *info );

void   lapackf77_csteqr( const char *compz,
                         const icla_int_t *n,
                         float *d, float *e,
                         iclaFloatComplex *Z, const icla_int_t *ldz,
                         float *work,
                         icla_int_t *info );

#ifdef ICLA_COMPLEX
void   lapackf77_csymv(  const char *uplo,
                         const icla_int_t *n,
                         const iclaFloatComplex *alpha,
                         const iclaFloatComplex *A, const icla_int_t *lda,
                         const iclaFloatComplex *x, const icla_int_t *incx,
                         const iclaFloatComplex *beta,
                               iclaFloatComplex *y, const icla_int_t *incy );

void   lapackf77_csyr(   const char *uplo,
                         const icla_int_t *n,
                         const iclaFloatComplex *alpha,
                         const iclaFloatComplex *x, const icla_int_t *incx,
                               iclaFloatComplex *A, const icla_int_t *lda );

void   lapackf77_csysv(  const char *uplo,
                         const icla_int_t *n, const icla_int_t *nrhs,
                         iclaFloatComplex *A, const icla_int_t *lda, icla_int_t *ipiv,
                         iclaFloatComplex *B, const icla_int_t *ldb,
                         iclaFloatComplex *work, const icla_int_t *lwork,
                         icla_int_t *info );

#endif

void   lapackf77_ctrevc( const char *side, const char *howmny,

                         #ifdef ICLA_COMPLEX
                         const
                         #endif
                         icla_int_t *select, const icla_int_t *n,

                         #ifdef ICLA_REAL
                         const
                         #endif
                         iclaFloatComplex *T,  const icla_int_t *ldt,
                         iclaFloatComplex *Vl, const icla_int_t *ldvl,
                         iclaFloatComplex *Vr, const icla_int_t *ldvr,
                         const icla_int_t *mm, icla_int_t *m,
                         iclaFloatComplex *work,
                         #ifdef ICLA_COMPLEX
                         float *rwork,
                         #endif
                         icla_int_t *info );

void   lapackf77_ctrevc3( const char *side, const char *howmny,
                          icla_int_t *select, const icla_int_t *n,
                          iclaFloatComplex *T,  const icla_int_t *ldt,
                          iclaFloatComplex *VL, const icla_int_t *ldvl,
                          iclaFloatComplex *VR, const icla_int_t *ldvr,
                          const icla_int_t *mm,
                          const icla_int_t *mout,
                          iclaFloatComplex *work, const icla_int_t *lwork,
                          #ifdef ICLA_COMPLEX
                          float *rwork,
                          #endif
                          icla_int_t *info );

void   lapackf77_ctrtri( const char *uplo, const char *diag,
                         const icla_int_t *n,
                         iclaFloatComplex *A, const icla_int_t *lda,
                         icla_int_t *info );

void   lapackf77_cung2r( const icla_int_t *m, const icla_int_t *n, const icla_int_t *k,
                         iclaFloatComplex *A, const icla_int_t *lda,
                         const iclaFloatComplex *tau,
                         iclaFloatComplex *work,
                         icla_int_t *info );

void   lapackf77_cungbr( const char *vect,
                         const icla_int_t *m, const icla_int_t *n, const icla_int_t *k,
                         iclaFloatComplex *A, const icla_int_t *lda,
                         const iclaFloatComplex *tau,
                         iclaFloatComplex *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_cunghr( const icla_int_t *n,
                         const icla_int_t *ilo, const icla_int_t *ihi,
                         iclaFloatComplex *A, const icla_int_t *lda,
                         const iclaFloatComplex *tau,
                         iclaFloatComplex *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_cunglq( const icla_int_t *m, const icla_int_t *n, const icla_int_t *k,
                         iclaFloatComplex *A, const icla_int_t *lda,
                         const iclaFloatComplex *tau,
                         iclaFloatComplex *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_cungql( const icla_int_t *m, const icla_int_t *n, const icla_int_t *k,
                         iclaFloatComplex *A, const icla_int_t *lda,
                         const iclaFloatComplex *tau,
                         iclaFloatComplex *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_cungqr( const icla_int_t *m, const icla_int_t *n, const icla_int_t *k,
                         iclaFloatComplex *A, const icla_int_t *lda,
                         const iclaFloatComplex *tau,
                         iclaFloatComplex *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_cungtr( const char *uplo,
                         const icla_int_t *n,
                         iclaFloatComplex *A, const icla_int_t *lda,
                         const iclaFloatComplex *tau,
                         iclaFloatComplex *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_cunm2r( const char *side, const char *trans,
                         const icla_int_t *m, const icla_int_t *n, const icla_int_t *k,
                         const iclaFloatComplex *A, const icla_int_t *lda,
                         const iclaFloatComplex *tau,
                         iclaFloatComplex *C, const icla_int_t *ldc,
                         iclaFloatComplex *work,
                         icla_int_t *info );

void   lapackf77_cunmbr( const char *vect, const char *side, const char *trans,
                         const icla_int_t *m, const icla_int_t *n, const icla_int_t *k,
                         const iclaFloatComplex *A, const icla_int_t *lda,
                         const iclaFloatComplex *tau,
                         iclaFloatComplex *C, const icla_int_t *ldc,
                         iclaFloatComplex *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_cunmlq( const char *side, const char *trans,
                         const icla_int_t *m, const icla_int_t *n, const icla_int_t *k,
                         const iclaFloatComplex *A, const icla_int_t *lda,
                         const iclaFloatComplex *tau,
                         iclaFloatComplex *C, const icla_int_t *ldc,
                         iclaFloatComplex *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_cunmql( const char *side, const char *trans,
                         const icla_int_t *m, const icla_int_t *n, const icla_int_t *k,
                         const iclaFloatComplex *A, const icla_int_t *lda,
                         const iclaFloatComplex *tau,
                         iclaFloatComplex *C, const icla_int_t *ldc,
                         iclaFloatComplex *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_cunmqr( const char *side, const char *trans,
                         const icla_int_t *m, const icla_int_t *n, const icla_int_t *k,
                         const iclaFloatComplex *A, const icla_int_t *lda,
                         const iclaFloatComplex *tau,
                         iclaFloatComplex *C, const icla_int_t *ldc,
                         iclaFloatComplex *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_cunmrq( const char *side, const char *trans,
                         icla_int_t *m, icla_int_t *n, icla_int_t *k,
                         iclaFloatComplex *A, icla_int_t *lda,
                         iclaFloatComplex *tau,
                         iclaFloatComplex *C, icla_int_t *ldc,
                         iclaFloatComplex *work, icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_cunmtr( const char *side, const char *uplo, const char *trans,
                         const icla_int_t *m, const icla_int_t *n,
                         const iclaFloatComplex *A, const icla_int_t *lda,
                         const iclaFloatComplex *tau,
                         iclaFloatComplex *C, const icla_int_t *ldc,
                         iclaFloatComplex *work, const icla_int_t *lwork,
                         icla_int_t *info );

void   lapackf77_sstebz( const char *range, const char *order,
                         const icla_int_t *n,
                         const float *vl, const float *vu,
                         const icla_int_t *il, const icla_int_t *iu,
                         const float *abstol,
                         const float *d, const float *e,
                         icla_int_t *m, icla_int_t *nsplit,
                         float *w,
                         icla_int_t *iblock, icla_int_t *isplit,
                         float *work,
                         icla_int_t *iwork,
                         icla_int_t *info );

void   lapackf77_slaln2( const icla_int_t *ltrans,
                         const icla_int_t *na, const icla_int_t *nw,
                         const float *smin, const float *ca,
                         const float *a,  const icla_int_t *lda,
                         const float *d1, const float *d2,
                         const float *b,  const icla_int_t *ldb,
                         const float *wr, const float *wi,
                         float *x, const icla_int_t *ldx,
                         float *scale, float *xnorm,
                         icla_int_t *info );

float lapackf77_slamc3( const float *a, const float *b );

void   lapackf77_slamrg( const icla_int_t *n1, const icla_int_t *n2,
                         const float *a,
                         const icla_int_t *dtrd1, const icla_int_t *dtrd2,
                         icla_int_t *index );

float lapackf77_slapy3( const float *x, const float *y, const float *z );

void   lapackf77_slaed2( icla_int_t *k, const icla_int_t *n, const icla_int_t *n1,
                         float *d,
                         float *q, const icla_int_t *ldq,
                         icla_int_t *indxq,
                         float *rho, const float *z,
                         float *dlamda, float *w, float *q2,
                         icla_int_t *indx, icla_int_t *indxc, icla_int_t *indxp,
                         icla_int_t *coltyp,
                         icla_int_t *info);

void   lapackf77_slaed4( const icla_int_t *n, const icla_int_t *i,
                         const float *d,
                         const float *z,
                         float *delta,
                         const float *rho,
                         float *dlam,
                         icla_int_t *info );

void   lapackf77_slasrt( const char *id, const icla_int_t *n, float *d,
                         icla_int_t *info );

void   lapackf77_cbdt01( const icla_int_t *m, const icla_int_t *n, const icla_int_t *kd,
                         iclaFloatComplex *A, const icla_int_t *lda,
                         iclaFloatComplex *Q, const icla_int_t *ldq,
                         float *d, float *e,
                         iclaFloatComplex *Pt, const icla_int_t *ldpt,
                         iclaFloatComplex *work,
                         #ifdef ICLA_COMPLEX
                         float *rwork,
                         #endif
                         float *resid );

void   lapackf77_cget22( const char *transa, const char *transe, const char *transw, const icla_int_t *n,
                         iclaFloatComplex *A, const icla_int_t *lda,
                         iclaFloatComplex *E, const icla_int_t *lde,
                         #ifdef ICLA_COMPLEX
                         iclaFloatComplex *w,
                         #else
                         iclaFloatComplex *wr,
                         iclaFloatComplex *wi,
                         #endif
                         iclaFloatComplex *work,
                         #ifdef ICLA_COMPLEX
                         float *rwork,
                         #endif
                         float *result );

void   lapackf77_chet21( const icla_int_t *itype, const char *uplo,
                         const icla_int_t *n, const icla_int_t *kband,
                         iclaFloatComplex *A, const icla_int_t *lda,
                         float *d, float *e,
                         iclaFloatComplex *U, const icla_int_t *ldu,
                         iclaFloatComplex *V, const icla_int_t *ldv,
                         iclaFloatComplex *tau,
                         iclaFloatComplex *work,
                         #ifdef ICLA_COMPLEX
                         float *rwork,
                         #endif
                         float *result );

void   lapackf77_chet22( const icla_int_t *itype, const char *uplo,
                         const icla_int_t *n, const icla_int_t *m, const icla_int_t *kband,
                         iclaFloatComplex *A, const icla_int_t *lda,
                         float *d, float *e,
                         iclaFloatComplex *U, const icla_int_t *ldu,
                         iclaFloatComplex *V, const icla_int_t *ldv,
                         iclaFloatComplex *tau,
                         iclaFloatComplex *work,
                         #ifdef ICLA_COMPLEX
                         float *rwork,
                         #endif
                         float *result );

void   lapackf77_chst01( const icla_int_t *n, const icla_int_t *ilo, const icla_int_t *ihi,
                         iclaFloatComplex *A, const icla_int_t *lda,
                         iclaFloatComplex *H, const icla_int_t *ldh,
                         iclaFloatComplex *Q, const icla_int_t *ldq,
                         iclaFloatComplex *work, const icla_int_t *lwork,
                         #ifdef ICLA_COMPLEX
                         float *rwork,
                         #endif
                         float *result );

void   lapackf77_cstt21( const icla_int_t *n, const icla_int_t *kband,
                         float *AD,
                         float *AE,
                         float *SD,
                         float *SE,
                         iclaFloatComplex *U, const icla_int_t *ldu,
                         iclaFloatComplex *work,
                         #ifdef ICLA_COMPLEX
                         float *rwork,
                         #endif
                         float *result );

void   lapackf77_cunt01( const char *rowcol, const icla_int_t *m, const icla_int_t *n,
                         iclaFloatComplex *U, const icla_int_t *ldu,
                         iclaFloatComplex *work, const icla_int_t *lwork,
                         #ifdef ICLA_COMPLEX
                         float *rwork,
                         #endif
                         float *resid );

void   lapackf77_clarfy( const char *uplo, const icla_int_t *n,
                         iclaFloatComplex *V, const icla_int_t *incv,
                         iclaFloatComplex *tau,
                         iclaFloatComplex *C, const icla_int_t *ldc,
                         iclaFloatComplex *work );

float lapackf77_cqpt01( const icla_int_t *m, const icla_int_t *n, const icla_int_t *k,
                         iclaFloatComplex *A,
                         iclaFloatComplex *Af, const icla_int_t *lda,
                         iclaFloatComplex *tau, icla_int_t *jpvt,
                         iclaFloatComplex *work, const icla_int_t *lwork );

void   lapackf77_cqrt02( const icla_int_t *m, const icla_int_t *n, const icla_int_t *k,
                         iclaFloatComplex *A,
                         iclaFloatComplex *AF,
                         iclaFloatComplex *Q,
                         iclaFloatComplex *R, const icla_int_t *lda,
                         iclaFloatComplex *tau,
                         iclaFloatComplex *work, const icla_int_t *lwork,
                         float *rwork,
                         float *result );

void   lapackf77_clatms( const icla_int_t *m, const icla_int_t *n,
                         const char *dist, icla_int_t *iseed, const char *sym,
                         float *d,
                         const icla_int_t *mode, const float *cond,
                         const float *dmax,
                         const icla_int_t *kl, const icla_int_t *ku, const char *pack,
                         iclaFloatComplex *A, const icla_int_t *lda,
                         iclaFloatComplex *work,
                         icla_int_t *info );

#ifdef ICLA_WITH_MKL
void   lapackf77_cgetrf_batch(
                         icla_int_t *m_array, icla_int_t *n_array,
                         iclaFloatComplex **A_array, icla_int_t *lda_array,
                         icla_int_t **ipiv_array,
                         icla_int_t *group_count, icla_int_t *group_size,
                         icla_int_t *info_array );
#endif

#ifdef __cplusplus
}
#endif

#undef ICLA_COMPLEX

#endif

