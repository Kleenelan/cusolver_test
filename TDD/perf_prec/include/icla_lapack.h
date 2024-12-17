

#ifndef ICLA_LAPACK_H
#define ICLA_LAPACK_H

#include "icla_mangling.h"

#include "icla_zlapack.h"
#include "icla_clapack.h"
#include "icla_dlapack.h"
#include "icla_slapack.h"

#ifdef __cplusplus
extern "C" {
#endif

#define lapackf77_ieeeck FORTRAN_NAME( ieeeck, IEEECK )
#define lapackf77_lsame  FORTRAN_NAME( lsame,  LSAME  )

#define lapackf77_slamch FORTRAN_NAME( slamch, SLAMCH )
#define lapackf77_dlamch FORTRAN_NAME( dlamch, DLAMCH )
#define lapackf77_slabad FORTRAN_NAME( slabad, SLABAD )
#define lapackf77_dlabad FORTRAN_NAME( dlabad, DLABAD )
#define lapackf77_zcgesv FORTRAN_NAME( zcgesv, ZCGESV )
#define lapackf77_dsgesv FORTRAN_NAME( dsgesv, DSGESV )

#define lapackf77_dsterf FORTRAN_NAME( dsterf, DSTERF )
#define lapackf77_ssterf FORTRAN_NAME( ssterf, SSTERF )

#define lapackf77_zlag2c FORTRAN_NAME( zlag2c, ZLAG2C )
#define lapackf77_clag2z FORTRAN_NAME( clag2z, CLAG2Z )
#define lapackf77_dlag2s FORTRAN_NAME( dlag2s, DLAG2S )
#define lapackf77_slag2d FORTRAN_NAME( slag2d, SLAG2D )

#define lapackf77_zlat2c FORTRAN_NAME( zlat2c, ZLAT2C )
#define lapackf77_dlat2s FORTRAN_NAME( dlat2s, DLAT2S )


#define lapackf77_dlapy2 FORTRAN_NAME( dlapy2, DLAPY2 )
#define lapackf77_slapy2 FORTRAN_NAME( slapy2, SLAPY2 )

icla_int_t lapackf77_ieeeck( const icla_int_t *ispec, const float *zero, const float *one );

long   lapackf77_lsame(  const char *ca, const char *cb );

float  lapackf77_slamch( const char *cmach );
double lapackf77_dlamch( const char *cmach );


void   lapackf77_slabad( float  *Small, float  *large );
void   lapackf77_dlabad( double *Small, double *large );

void   lapackf77_zcgesv( const icla_int_t *n, const icla_int_t *nrhs,
                         iclaDoubleComplex *A, const icla_int_t *lda,
                         icla_int_t *ipiv,
                         const iclaDoubleComplex *B, const icla_int_t *ldb,
                               iclaDoubleComplex *X, const icla_int_t *ldx,
                         iclaDoubleComplex *work, iclaFloatComplex *swork, double *rwork,
                         icla_int_t *iter,
                         icla_int_t *info );

void   lapackf77_dsgesv( const icla_int_t *n, const icla_int_t *nrhs,
                         double *A, const icla_int_t *lda,
                         icla_int_t *ipiv,
                         const double *B, const icla_int_t *ldb,
                               double *X, const icla_int_t *ldx,
                         double *work, float *swork,
                         icla_int_t *iter,
                         icla_int_t *info );

void   lapackf77_dsterf( const icla_int_t *n,
                         double *d, double *e,
                         icla_int_t *info );

void   lapackf77_ssterf( const icla_int_t *n,
                         float *d, float *e,
                         icla_int_t *info );


void   lapackf77_zlag2c( const icla_int_t *m, const icla_int_t *n,
                         const iclaDoubleComplex *A,  const icla_int_t *lda,
                               iclaFloatComplex  *SA, const icla_int_t *ldsa,
                         icla_int_t *info );

void   lapackf77_clag2z( const icla_int_t *m, const icla_int_t *n,
                         const iclaFloatComplex  *SA, const icla_int_t *ldsa,
                               iclaDoubleComplex *A,  const icla_int_t *lda,
                         icla_int_t *info );

void   lapackf77_dlag2s( const icla_int_t *m, const icla_int_t *n,
                         const double *A,  const icla_int_t *lda,
                               float  *SA, const icla_int_t *ldsa,
                         icla_int_t *info );

void   lapackf77_slag2d( const icla_int_t *m, const icla_int_t *n,
                         const float  *SA, const icla_int_t *ldsa,
                               double *A,  const icla_int_t *lda,
                         icla_int_t *info );


void   lapackf77_zlat2c( const char *uplo, const icla_int_t *n,
                         const iclaDoubleComplex *A,  const icla_int_t *lda,
                               iclaFloatComplex  *SA, const icla_int_t *ldsa,
                         icla_int_t *info );

void   lapackf77_dlat2s( const char *uplo, const icla_int_t *n,
                         const double *A,  const icla_int_t *lda,
                               float  *SA, const icla_int_t *ldsa,
                         icla_int_t *info );

double lapackf77_dlapy2( const double *x, const double *y );
float  lapackf77_slapy2( const float  *x, const float  *y );

#ifdef __cplusplus
}
#endif

#endif

