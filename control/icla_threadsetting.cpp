/*
    -- ICLA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Azzam Haidar
       @author Mark Gates
*/
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


/***************************************************************************//**
    Purpose
    -------
    @return Number of threads to use for parallel sections of ICLA.
    Typically, it is initially set by the environment variables
    OMP_NUM_THREADS or ICLA_NUM_THREADS.

    If ICLA_NUM_THREADS is set, this returns
        min( num_cores, ICLA_NUM_THREADS );
    else if ICLA is compiled with OpenMP, this queries OpenMP and returns
        min( num_cores, OMP_NUM_THREADS );
    else this returns num_cores.

    For the number of cores, if ICLA is compiled with hwloc, this queries hwloc;
    else it queries sysconf (on Unix) or GetSystemInfo (on Windows).

    @sa icla_get_lapack_numthreads
    @sa icla_set_lapack_numthreads
    @ingroup icla_thread
*******************************************************************************/
extern "C"
icla_int_t icla_get_parallel_numthreads()
{
    // query number of cores
    icla_int_t ncores = 0;

#ifdef HAVE_HWLOC
    // hwloc gives physical cores, not hyperthreads
    // from http://stackoverflow.com/questions/12483399/getting-number-of-cores-not-ht-threads
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
        #ifdef _MSC_VER  // Windows
        SYSTEM_INFO sysinfo;
        GetSystemInfo( &sysinfo );
        ncores = sysinfo.dwNumberOfProcessors;
        #else
        ncores = sysconf( _SC_NPROCESSORS_ONLN );
        #endif
    }

    // query ICLA_NUM_THREADS or OpenMP
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

    // limit to range [1, number of cores]
    threads = max( 1, min( ncores, threads ));
    return threads;
}


/***************************************************************************//**
    Purpose
    -------
    @return Number of threads currently used for LAPACK and BLAS.

    Typically, the number of threads is initially set by the environment
    variables OMP_NUM_THREADS or MKL_NUM_THREADS.

    If ICLA is compiled with ICLA_WITH_MKL, this queries MKL;
    else if ICLA is compiled with OpenMP, this queries OpenMP;
    else this returns 1.

    @sa icla_get_parallel_numthreads
    @sa icla_set_lapack_numthreads
    @ingroup icla_thread
*******************************************************************************/
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


/***************************************************************************//**
    Purpose
    -------
    Sets the number of threads to use for LAPACK and BLAS.
    This is often used to set BLAS to be single-threaded during sections
    where ICLA uses explicit pthread parallelism. Example use:

        nthread_save = icla_get_lapack_numthreads();
        icla_set_lapack_numthreads( 1 );

        ... launch pthreads, do work, terminate pthreads ...

        icla_set_lapack_numthreads( nthread_save );

    If ICLA is compiled with ICLA_WITH_MKL, this sets MKL threads;
    else if ICLA is compiled with OpenMP, this sets OpenMP threads;
    else this does nothing.

    Arguments
    ---------
    @param[in]
    threads INTEGER
            Number of threads to use. threads >= 1.
            If threads < 1, this silently does nothing.

    @sa icla_get_parallel_numthreads
    @sa icla_get_lapack_numthreads
    @ingroup icla_thread
*******************************************************************************/
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


/***************************************************************************//**
    Purpose
    -------
    @return Number of threads currently used for OMP sections.
    Typically, the number of threads is initially set by the environment
    variable OMP_NUM_THREADS.

    @sa icla_get_parallel_numthreads
    @sa icla_set_lapack_numthreads
    @ingroup icla_thread
*******************************************************************************/
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


/***************************************************************************//**
    Purpose
    -------
    Sets the number of threads to use for parallel section.
    This is often used to set BLAS to be single-threaded during sections
    where ICLA uses explicit pthread parallelism. Example use:

        nthread_save = icla_get_omp_numthreads();
        icla_set_omp_numthreads( 1 );

        ... launch pthreads, do work, terminate pthreads ...

        icla_set_omp_numthreads( nthread_save );

    Arguments
    ---------
    @param[in]
    threads INTEGER
            Number of threads to use. threads >= 1.
            If threads < 1, this silently does nothing.

    @sa icla_get_parallel_numthreads
    @sa icla_get_lapack_numthreads
    @ingroup icla_thread
*******************************************************************************/
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
