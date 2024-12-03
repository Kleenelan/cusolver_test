/*
    -- ICLA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
*/

#ifndef ICLA_MANGLING_H
#define ICLA_MANGLING_H

#include "icla_mangling_cmake.h"

/* Define how to name mangle Fortran names.
 * If using CMake, it defines ICLA_GLOBAL in icla_mangling_cmake.h
 * Otherwise, the make.inc file should have one of -DADD_, -DNOCHANGE, or -DUPCASE.
 * If using outside of ICLA, put one of those in your compiler flags (e.g., CFLAGS).
 * These macros are used in:
 *   include/icla_*lapack.h
 *   control/icla_*f77.cpp
 */
#ifndef ICLA_FORTRAN_NAME
    #if defined(ICLA_GLOBAL)
        #define FORTRAN_NAME(lcname, UCNAME)  ICLA_GLOBAL( lcname, UCNAME )
    #elif defined(ADD_)
        #define FORTRAN_NAME(lcname, UCNAME)  lcname##_
    #elif defined(NOCHANGE)
        #define FORTRAN_NAME(lcname, UCNAME)  lcname
    #elif defined(UPCASE)
        #define FORTRAN_NAME(lcname, UCNAME)  UCNAME
    #else
        #error "One of ADD_, NOCHANGE, or UPCASE must be defined to set how Fortran functions are name mangled. For example, in ICLA, add -DADD_ to CFLAGS, FFLAGS, etc. in make.inc. If using CMake, it defines ICLA_GLOBAL instead."
        #define FORTRAN_NAME(lcname, UCNAME)  lcname##_error
    #endif
#endif

#endif  // ICLA_MANGLING_H
