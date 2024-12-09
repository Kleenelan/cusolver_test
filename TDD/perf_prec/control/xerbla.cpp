
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "icla_internal.h"

extern "C"
void icla_xerbla(const char *srname, icla_int_t neg_info)
{

    if ( neg_info < 0 ) {
        fprintf( stderr, "Error in %s, function-specific error (info = %lld)\n",
                 srname, (long long) -neg_info );
    }
    else if ( neg_info == 0 ) {
        fprintf( stderr, "No error, why is %s calling xerbla? (info = %lld)\n",
                 srname, (long long) -neg_info );
    }
    else if ( neg_info >= -ICLA_ERR ) {
        fprintf( stderr, "Error in %s, %s (info = %lld)\n",
                 srname, icla_strerror(-neg_info), (long long) -neg_info );
    }
    else {

        fprintf( stderr, "On entry to %s, parameter %lld had an illegal value (info = %lld)\n",
                 srname, (long long) neg_info, (long long) -neg_info );
    }
}
