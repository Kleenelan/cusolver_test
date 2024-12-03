/*
    -- ICLA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates
*/
#include "icla_internal.h"


/***************************************************************************//**
    Returns version of ICLA, as defined by
    ICLA_VERSION_MAJOR, ICLA_VERSION_MINOR, ICLA_VERSION_MICRO constants.

    @param[out] major   Set to major version number.
    @param[out] minor   Set to minor version number.
    @param[out] micro   Set to micro version number.

    @ingroup icla_util
*******************************************************************************/
extern "C"
void icla_version( icla_int_t* major, icla_int_t* minor, icla_int_t* micro )
{
    if ( major != NULL && minor != NULL && micro != NULL ) {
        *major = ICLA_VERSION_MAJOR;
        *minor = ICLA_VERSION_MINOR;
        *micro = ICLA_VERSION_MICRO;
    }
}


/***************************************************************************//**
    Determines the number of GPUs to use, based on $ICLA_NUM_GPUS environment
    variable, and limited to actual number of GPUs available.
    If $ICLA_NUM_GPUS is not set, uses 1.

    @return Number of GPUs to use.

    @ingroup icla_util
*******************************************************************************/
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
        // if *endptr == '\0' then entire string was valid number (or empty)
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


/***************************************************************************//**
    Auxiliary function: ipiv(i) indicates that row i has been swapped with
    ipiv(i) from top to bottom. This function rearranges ipiv into newipiv
    where row i has to be moved to newipiv(i). The new pivoting allows for
    parallel processing vs the original one assumes a specific ordering and
    has to be done sequentially.

    @ingroup icla_internal
*******************************************************************************/
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
        // Transpose
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


/***************************************************************************//**
    Convert global indices [j0, j1) to local indices [dj0, dj1) on GPU dev,
    according to 1D block cyclic distribution.
    Note j0 and dj0 are inclusive, while j1 and dj1 are exclusive.
    This is consistent with the C++ container notion of first and last.

    Example with n = 75, nb = 10, ngpu = 3.
    Distribution of columns (ranges are inclusive):

                          local dj:  0- 9, 10-19, 20-29
        -----------------------------------------------
        dev 0:  3 blocks, global j:  0- 9, 30-39, 60-69
        dev 1:  3 blocks, global j: 10-19, 40-49, 70-74 (partial)
        dev 2:  2 block,  global j: 20-29, 50-59

    Calls return:

        input global j=13-68 inclusive      =>  output
        nb=10, ngpu=3, dev=0, j0=13, j1=69  =>  dj0=10, dj1=29 (i.e., global j=       30-39, 60-68)
        nb=10, ngpu=3, dev=1, j0=13, j1=69  =>  dj0= 3, dj1=20 (i.e., global j=13-19, 40-49)
        nb=10, ngpu=3, dev=2, j0=13, j1=69  =>  dj0= 0, dj1=20 (i.e., global j=20-29, 50-59)

        input global j=13-69 inclusive      =>  output
        nb=10, ngpu=3, dev=0, j0=13, j1=70  =>  dj0=10, dj1=30 (i.e., global j=       30-39, 60-69)
        nb=10, ngpu=3, dev=1, j0=13, j1=70  =>  dj0= 3, dj1=20 (i.e., global j=13-19, 40-49)
        nb=10, ngpu=3, dev=2, j0=13, j1=70  =>  dj0= 0, dj1=20 (i.e., global j=20-29, 50-59)

        input global j=13-70 inclusive      =>  output
        nb=10, ngpu=3, dev=0, j0=13, j1=71  =>  dj0=10, dj1=30 (i.e., global j=       30-39, 60-69)
        nb=10, ngpu=3, dev=1, j0=13, j1=71  =>  dj0= 3, dj1=21 (i.e., global j=13-19, 40-49, 70)
        nb=10, ngpu=3, dev=2, j0=13, j1=71  =>  dj0= 0, dj1=20 (i.e., global j=20-29, 50-59)

        input global j=13-71 inclusive      =>  output
        nb=10, ngpu=3, dev=0, j0=13, j1=72  =>  dj0=10, dj1=30 (i.e., global j=       30-39, 60-69)
        nb=10, ngpu=3, dev=1, j0=13, j1=72  =>  dj0= 3, dj1=22 (i.e., global j=13-19, 40-49, 70-71)
        nb=10, ngpu=3, dev=2, j0=13, j1=72  =>  dj0= 0, dj1=20 (i.e., global j=20-29, 50-59)

    @ingroup icla_internal
*******************************************************************************/
extern "C"
void icla_indices_1D_bcyclic( icla_int_t nb, icla_int_t ngpu, icla_int_t dev,
                               icla_int_t j0, icla_int_t j1,
                               icla_int_t* dj0, icla_int_t* dj1 )
{
    // on GPU jdev, which contains j0, dj0 maps to j0.
    // on other GPUs, dj0 is start of the block on that GPU after j0's block.
    icla_int_t jblock = (j0 / nb) / ngpu;
    icla_int_t jdev   = (j0 / nb) % ngpu;
    if ( dev < jdev ) {
        jblock += 1;
    }
    *dj0 = jblock*nb;
    if ( dev == jdev ) {
        *dj0 += (j0 % nb);
    }

    // on GPU jdev, which contains j1-1, dj1 maps to j1.
    // on other GPUs, dj1 is end of the block on that GPU before j1's block.
    // j1 points to element after end (e.g., n), so subtract 1 to get last
    // element, compute index, then add 1 to get just after that index again.
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
