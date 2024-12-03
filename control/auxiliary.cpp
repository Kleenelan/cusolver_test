
#include "icla_internal.h"

extern "C"
void icla_version( icla_int_t* major, icla_int_t* minor, icla_int_t* micro )
{
    if ( major != NULL && minor != NULL && micro != NULL ) {
        *major = ICLA_VERSION_MAJOR;
        *minor = ICLA_VERSION_MINOR;
        *micro = ICLA_VERSION_MICRO;
    }
}

extern "C"
icla_int_t icla_num_gpus( void )
{
    const char *ngpu_str = getenv("ICLA_NUM_GPUS");
    icla_int_t ngpu = 1;
    if ( ngpu_str != NULL ) {
        char* endptr;
        ngpu = strtol( ngpu_str, &endptr, 10 );
        icla_int_t ndevices;
        icla_device_t devices[ iclaMaxGPUs ];
        icla_getdevices( devices, iclaMaxGPUs, &ndevices );

        if ( ngpu < 1 || *endptr != '\0' ) {
            ngpu = 1;
            fprintf( stderr, "$ICLA_NUM_GPUS='%s' is an invalid number; using %lld GPU.\n",
                     ngpu_str, (long long) ngpu );
        }
        else if ( ngpu > iclaMaxGPUs || ngpu > ndevices ) {
            ngpu = min( ndevices, iclaMaxGPUs );
            fprintf( stderr, "$ICLA_NUM_GPUS='%s' exceeds iclaMaxGPUs=%d or available GPUs=%lld; using %lld GPUs.\n",
                     ngpu_str, iclaMaxGPUs, (long long) ndevices, (long long) ngpu );
        }
        assert( 1 <= ngpu && ngpu <= ndevices );
    }
    return ngpu;
}

extern "C"
void icla_swp2pswp( icla_trans_t trans, icla_int_t n, icla_int_t *ipiv, icla_int_t *newipiv)
{
    icla_int_t i, newind, ind;

    for (i=0; i < n; i++)
        newipiv[i] = -1;

    if (trans == iclaNoTrans) {
        for (i=0; i < n; i++) {
            newind = ipiv[i] - 1;
            if (newipiv[newind] == -1) {
                if (newipiv[i] == -1) {
                    newipiv[i] = newind;
                    if (newind > i)
                        newipiv[newind]= i;
                }
                else {
                    ind = newipiv[i];
                    newipiv[i] = newind;
                    if (newind > i)
                        newipiv[newind]= ind;
                }
            }
            else {
                if (newipiv[i] == -1) {
                    if (newind > i) {
                        ind = newipiv[newind];
                        newipiv[newind] = i;
                        newipiv[i] = ind;
                    }
                    else
                        newipiv[i] = newipiv[newind];
                }
                else {
                    ind = newipiv[i];
                    newipiv[i] = newipiv[newind];
                    if (newind > i)
                        newipiv[newind] = ind;
                }
            }
        }
    }
    else {

        for (i=n-1; i >= 0; i--) {
            newind = ipiv[i] - 1;
            if (newipiv[newind] == -1) {
                if (newipiv[i] == -1) {
                    newipiv[i] = newind;
                    if (newind > i)
                        newipiv[newind]= i;
                }
                else {
                    ind = newipiv[i];
                    newipiv[i] = newind;
                    if (newind > i)
                        newipiv[newind]= ind;
                }
            }
            else {
                if (newipiv[i] == -1) {
                    if (newind > i) {
                        ind = newipiv[newind];
                        newipiv[newind] = i;
                        newipiv[i] = ind;
                    }
                    else
                        newipiv[i] = newipiv[newind];
                }
                else {
                    ind = newipiv[i];
                    newipiv[i] = newipiv[newind];
                    if (newind > i)
                        newipiv[newind] = ind;
                }
            }
        }
    }
}

extern "C"
void icla_indices_1D_bcyclic( icla_int_t nb, icla_int_t ngpu, icla_int_t dev,
                               icla_int_t j0, icla_int_t j1,
                               icla_int_t* dj0, icla_int_t* dj1 )
{

    icla_int_t jblock = (j0 / nb) / ngpu;
    icla_int_t jdev   = (j0 / nb) % ngpu;
    if ( dev < jdev ) {
        jblock += 1;
    }
    *dj0 = jblock*nb;
    if ( dev == jdev ) {
        *dj0 += (j0 % nb);
    }

    j1 -= 1;
    jblock = (j1 / nb) / ngpu;
    jdev   = (j1 / nb) % ngpu;
    if ( dev > jdev ) {
        jblock -= 1;
    }
    if ( dev == jdev ) {
        *dj1 = jblock*nb + (j1 % nb) + 1;
    }
    else {
        *dj1 = jblock*nb + nb;
    }
}
