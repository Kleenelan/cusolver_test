
#ifndef ICLA_THREADSETTING_H
#define ICLA_THREADSETTING_H

#ifdef __cplusplus
extern "C" {
#endif

void icla_set_omp_numthreads(icla_int_t numthreads);
void icla_set_lapack_numthreads(icla_int_t numthreads);
icla_int_t icla_get_lapack_numthreads();
icla_int_t icla_get_parallel_numthreads();
icla_int_t icla_get_omp_numthreads();

#ifdef __cplusplus
}
#endif

#endif

