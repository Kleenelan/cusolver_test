
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

        icla_setdevice( d );
        cudaGetDeviceProperties( &prop, int(d) );

        #ifdef ICLA_HAVE_CUDA
        if ( ! prop.unifiedAddressing ) {
        #elif defined(ICLA_HAVE_HIP)

        if ( ! true ) {
        #endif
            printf( "device %lld doesn't support unified addressing\n", (long long) d );
            icla_free_cpu( deviceid );
            return -1;
        }

        if (deviceid[d] == 0) {
            cmplxnb = cmplxnb + 1;
            cmplxid = cmplxnb - 1;
            gnode[cmplxid][iclaMaxGPUs] = 1;
            lcgpunb = gnode[cmplxid][iclaMaxGPUs]-1;
            gnode[cmplxid][lcgpunb] = d;
            deviceid[d] = -1;
        }

        for( icla_int_t d2 = d+1; d2 < ngpu; ++d2 ) {

            icla_setdevice( d2 );
            cudaGetDeviceProperties( &prop, int(d2) );
            #ifdef ICLA_HAVE_CUDA
            if ( ! prop.unifiedAddressing ) {
            #elif defined(ICLA_HAVE_HIP)

            if ( ! true ) {
            #endif
                printf( "device %lld doesn't support unified addressing\n", (long long) d2 );
                icla_free_cpu( deviceid );
                return -1;
            }

 cudaDeviceCanAccessPeer( &samecomplex, int(d), int(d2) );

            if (samecomplex == 1) {

                icla_setdevice( d );
                err = cudaDeviceEnablePeerAccess( int(d2), 0 );

                if ( err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled ) {
                    printf( "device %lld cudaDeviceEnablePeerAccess error %lld\n",
                            (long long) d2, (long long) err );
                    icla_free_cpu( deviceid );
                    return -2;
                }

                icla_setdevice( d2 );
                err = cudaDeviceEnablePeerAccess( int(d), 0 );

                if ((err == cudaSuccess) || (err == cudaErrorPeerAccessAlreadyEnabled)) {
                    if (deviceid[d2] == 0) {

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

    return -1;
#endif

}

}

