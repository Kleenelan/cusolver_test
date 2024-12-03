
#ifndef ICLA_MANGLING_H
#define ICLA_MANGLING_H

#include "icla_mangling_cmake.h"

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

#endif

