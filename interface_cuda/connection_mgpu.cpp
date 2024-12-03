/*
    -- ICLA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Azzam Haidar
*/
#include <cuda_runtime.h>

#include "icla_internal.h"

extern "C" {

#if defined(ICLA_HAVE_CUDA) || defined(ICLA_HAVE_HIP)
icla_int_t icla_buildconnection_mgpu(
    icla_int_t gnode[iclaMaxGPUs+2][iclaMaxGPUs+2],
    icla_int_t *ncmplx, icla_int_t ngpu)
{
    icla_int_t *deviceid = NULL;
    icla_imalloc_cpu( &deviceid, ngpu );
    memset( deviceid, 0, ngpu*sizeof(icla_int_t) );

    ncmplx[0] = 0;

    int samecomplex = -1;
    cudaError_t err;
    cudaDeviceProp prop;

    icla_int_t cmplxnb = 0;
    icla_int_t cmplxid = 0;
    icla_int_t lcgpunb = 0;
    for( icla_int_t d = 0; d < ngpu; ++d ) {
        // check for unified memory & enable peer memory access between all GPUs.
        icla_setdevice( d );
        cudaGetDeviceProperties( &prop, int(d) );

        #ifdef ICLA_HAVE_CUDA
        if ( ! prop.unifiedAddressing ) {
        #elif defined(ICLA_HAVE_HIP)
        // assume it does, HIP does not have support for checking this
        if ( ! true ) {
        #endif
            printf( "device %lld doesn't support unified addressing\n", (long long) d );
            icla_free_cpu( deviceid );
            return -1;
        }

        // add this device to the list if not added yet.
        // not added yet meaning belong to a new complex
        if (deviceid[d] == 0) {
            cmplxnb = cmplxnb + 1;
            cmplxid = cmplxnb - 1;
            gnode[cmplxid][iclaMaxGPUs] = 1;
            lcgpunb = gnode[cmplxid][iclaMaxGPUs]-1;
            gnode[cmplxid][lcgpunb] = d;
            deviceid[d] = -1;
        }
        //printf("device %lld:\n", (long long) d );

        for( icla_int_t d2 = d+1; d2 < ngpu; ++d2 ) {
            // check for unified memory & enable peer memory access between all GPUs.
            icla_setdevice( d2 );
            cudaGetDeviceProperties( &prop, int(d2) );
            #ifdef ICLA_HAVE_CUDA
            if ( ! prop.unifiedAddressing ) {
            #elif defined(ICLA_HAVE_HIP)
            // assume it does, HIP does not have support for checking this
            if ( ! true ) {
            #endif
                printf( "device %lld doesn't support unified addressing\n", (long long) d2 );
                icla_free_cpu( deviceid );
                return -1;
            }

            /* TODO err = */ cudaDeviceCanAccessPeer( &samecomplex, int(d), int(d2) );

            //printf(" device %lld and device %lld have samecomplex = %lld\n",
            //       (long long) d, (long long) d2, (long long) samecomplex );
            if (samecomplex == 1) {
                // d and d2 are on the same complex so add them, note that d is already added
                // so just enable the peer Access for d and enable+add d2.
                // FOR d:
                icla_setdevice( d );
                err = cudaDeviceEnablePeerAccess( int(d2), 0 );
                //printf("enabling devide %lld ==> %lld  error %lld\n",
                //       (long long) d, (long long) d2, (long long) err );
                if ( err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled ) {
                    printf( "device %lld cudaDeviceEnablePeerAccess error %lld\n",
                            (long long) d2, (long long) err );
                    icla_free_cpu( deviceid );
                    return -2;
                }

                // FOR d2:
                icla_setdevice( d2 );
                err = cudaDeviceEnablePeerAccess( int(d), 0 );
                //printf("enabling devide %lld ==> %lld  error %lld\n",
                //       (long long) d2, (long long) d, (long long) err );
                if ((err == cudaSuccess) || (err == cudaErrorPeerAccessAlreadyEnabled)) {
                    if (deviceid[d2] == 0) {
                        //printf("adding device %lld\n", (long long) d2 );
                        gnode[cmplxid][iclaMaxGPUs] = gnode[cmplxid][iclaMaxGPUs]+1;
                        lcgpunb                      = gnode[cmplxid][iclaMaxGPUs]-1;
                        gnode[cmplxid][lcgpunb] = d2;
                        deviceid[d2] = -1;
                    }
                } else {
                    printf( "device %lld cudaDeviceEnablePeerAccess error %lld\n",
                            (long long) d, (long long) err );
                    icla_free_cpu( deviceid );
                    return -2;
                }
            }
        }
    }

    ncmplx[0] = cmplxnb;
    icla_free_cpu( deviceid );
    return cmplxnb;
#else
    // Err: CUDA only
    return -1;
#endif

}

} /* extern "C" */


