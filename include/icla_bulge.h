/*
    -- ICLA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
*/

#ifndef ICLA_BULGE_H
#define ICLA_BULGE_H

#include "icla_types.h"
#include "icla_zbulge.h"
#include "icla_cbulge.h"
#include "icla_dbulge.h"
#include "icla_sbulge.h"

#ifdef __cplusplus
extern "C" {
#endif

    icla_int_t icla_yield();
    icla_int_t icla_bulge_getlwstg1(icla_int_t n, icla_int_t nb, icla_int_t *lda2);

    void cmp_vals(icla_int_t n, double *wr1, double *wr2, double *nrmI, double *nrm1, double *nrm2);

    void icla_bulge_findVTAUpos(icla_int_t n, icla_int_t nb, icla_int_t Vblksiz, icla_int_t sweep, icla_int_t st, icla_int_t ldv,
                                 icla_int_t *Vpos, icla_int_t *TAUpos);

    void icla_bulge_findVTpos(icla_int_t n, icla_int_t nb, icla_int_t Vblksiz, icla_int_t sweep, icla_int_t st, icla_int_t ldv, icla_int_t ldt,
                               icla_int_t *Vpos, icla_int_t *Tpos);

    void icla_bulge_findVTAUTpos(icla_int_t n, icla_int_t nb, icla_int_t Vblksiz, icla_int_t sweep, icla_int_t st, icla_int_t ldv, icla_int_t ldt,
                                  icla_int_t *Vpos, icla_int_t *TAUpos, icla_int_t *Tpos, icla_int_t *blkid);

    void icla_bulge_findpos(icla_int_t n, icla_int_t nb, icla_int_t Vblksiz, icla_int_t sweep, icla_int_t st, icla_int_t *myblkid);
    void icla_bulge_findpos113(icla_int_t n, icla_int_t nb, icla_int_t Vblksiz, icla_int_t sweep, icla_int_t st, icla_int_t *myblkid);

    icla_int_t icla_bulge_get_blkcnt(icla_int_t n, icla_int_t nb, icla_int_t Vblksiz);

    void findVTpos(icla_int_t n, icla_int_t nb, icla_int_t Vblksiz, icla_int_t sweep, icla_int_t st, icla_int_t *Vpos, icla_int_t *TAUpos, icla_int_t *Tpos, icla_int_t *myblkid);

#ifdef __cplusplus
}
#endif

#endif  // ICLA_BULGE_H
