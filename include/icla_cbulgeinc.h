
#ifndef ICLA_CBULGEINC_H
#define ICLA_CBULGEINC_H

#include "icla_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_THREADS_BLG         256

void findVTpos(
    icla_int_t N, icla_int_t NB, icla_int_t Vblksiz,
    icla_int_t sweep, icla_int_t st,
    icla_int_t *Vpos, icla_int_t *TAUpos, icla_int_t *Tpos,
    icla_int_t *myblkid);

void findVTsiz(
    icla_int_t N, icla_int_t NB, icla_int_t Vblksiz,
    icla_int_t *blkcnt, icla_int_t *LDV);

struct gbstrct_blg {
    iclaFloatComplex *dQ1;
    iclaFloatComplex *dT1;
    iclaFloatComplex *dT2;
    iclaFloatComplex *dV2;
    iclaFloatComplex *dE;
    iclaFloatComplex *T;
    iclaFloatComplex *A;
    iclaFloatComplex *V;
    iclaFloatComplex *TAU;
    iclaFloatComplex *E;
    iclaFloatComplex *E_CPU;
    int cores_num;
    int locores_num;
    int overlapQ1;
    int usemulticpu;
    int NB;
    int NBTILES;
    int N;
    int NE;
    int N_CPU;
    int N_GPU;
    int LDA;
    int LDE;
    int BAND;
    int grsiz;
    int Vblksiz;
    int WANTZ;
    icla_side_t SIDE;
    real_Double_t *timeblg;
    real_Double_t *timeaplQ;
    volatile int *ss_prog;
};

extern struct gbstrct_blg core_in_all;

#ifdef __cplusplus
}
#endif

#endif

