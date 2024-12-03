
#include "icla_internal.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(ICLA_WITH_MKL)
#include <mkl_service.h>
#endif

#if defined(HAVE_HWLOC)
#include <hwloc.h>
#endif

extern "C"
icla_int_t icla_get_parallel_numthreads()
{

    icla_int_t ncores = 0;

#ifdef HAVE_HWLOC

    hwloc_topology_t topology;
    hwloc_topology_init( &topology );
    hwloc_topology_load( topology );
    icla_int_t depth = hwloc_get_type_depth( topology, HWLOC_OBJ_CORE );
    if (depth != HWLOC_TYPE_DEPTH_UNKNOWN) {
        ncores = hwloc_get_nbobjs_by_depth( topology, depth );
    }
    hwloc_topology_destroy( topology );
#endif

    if ( ncores == 0 ) {
        #ifdef _MSC_VER

        SYSTEM_INFO sysinfo;
        GetSystemInfo( &sysinfo );
        ncores = sysinfo.dwNumberOfProcessors;
        #else
        ncores = sysconf( _SC_NPROCESSORS_ONLN );
        #endif
    }

    const char *threads_str = getenv("ICLA_NUM_THREADS");
    icla_int_t threads = 0;
    if ( threads_str != NULL ) {
        char* endptr;
        threads = strtol( threads_str, &endptr, 10 );
        if ( threads < 1 || *endptr != '\0' ) {
            threads = 1;
            fprintf( stderr, "$ICLA_NUM_THREADS='%s' is an invalid number; using %lld threads.\n",
                     threads_str, (long long) threads );
        }
    }
    else {
        #if defined(_OPENMP)
        #pragma omp parallel
        {
            threads = omp_get_num_threads();
        }
        #else
            threads = ncores;
        #endif
    }

    threads = max( 1, min( ncores, threads ));
    return threads;
}

extern "C"
icla_int_t icla_get_lapack_numthreads()
{
    icla_int_t threads = 1;

#if defined(ICLA_WITH_MKL)
    threads = mkl_get_max_threads();
#elif defined(_OPENMP)
    #pragma omp parallel
    {
        threads = omp_get_num_threads();
    }
#endif

    return threads;
}

extern "C"
void icla_set_lapack_numthreads(icla_int_t threads)
{
    if ( threads < 1 ) {
        return;
    }

#if defined(ICLA_WITH_MKL)
    mkl_set_num_threads( int(threads) );
#elif defined(_OPENMP)
    #pragma omp parallel
    {
        omp_set_num_threads( int(threads) );
    }
#endif
}

extern "C"
icla_int_t icla_get_omp_numthreads()
{
    icla_int_t threads = 1;

#if defined(_OPENMP)
    #pragma omp parallel
    {
        threads = omp_get_num_threads();
    }
#endif

    return threads;
}

extern "C"
void icla_set_omp_numthreads(icla_int_t threads)
{
    if ( threads < 1 ) {
        return;
    }

#if defined(_OPENMP)
    omp_set_num_threads( threads );
#endif
}
