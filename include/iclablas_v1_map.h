/*
    -- ICLA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
*/

#ifndef ICLABLAS_V1_MAP_H
#define ICLABLAS_V1_MAP_H

#ifdef ICLA_NO_V1
#error "Since ICLA_NO_V1 is defined, icla.h is invalid; use icla_v2.h"
#endif

// =============================================================================
// map function names to old v1 routines

#include "iclablas_s_v1_map.h"
#include "iclablas_d_v1_map.h"
#include "iclablas_c_v1_map.h"
#include "iclablas_z_v1_map.h"
#include "iclablas_ds_v1_map.h"
#include "iclablas_zc_v1_map.h"

#undef icla_queue_create

#undef icla_setvector
#undef icla_getvector
#undef icla_copyvector
#undef icla_setmatrix
#undef icla_getmatrix
#undef icla_copymatrix

#undef icla_isetvector
#undef icla_igetvector
#undef icla_icopyvector
#undef icla_isetmatrix
#undef icla_igetmatrix
#undef icla_icopymatrix

#undef icla_index_setvector
#undef icla_index_getvector
#undef icla_index_copyvector
#undef icla_index_setmatrix
#undef icla_index_getmatrix
#undef icla_index_copymatrix

#define icla_queue_create                  icla_queue_create_v1

#define icla_setvector                     icla_setvector_v1
#define icla_getvector                     icla_getvector_v1
#define icla_copyvector                    icla_copyvector_v1
#define icla_setmatrix                     icla_setmatrix_v1
#define icla_getmatrix                     icla_getmatrix_v1
#define icla_copymatrix                    icla_copymatrix_v1

#define icla_isetvector                    icla_isetvector_v1
#define icla_igetvector                    icla_igetvector_v1
#define icla_icopyvector                   icla_icopyvector_v1
#define icla_isetmatrix                    icla_isetmatrix_v1
#define icla_igetmatrix                    icla_igetmatrix_v1
#define icla_icopymatrix                   icla_icopymatrix_v1

#define icla_index_setvector               icla_index_setvector_v1
#define icla_index_getvector               icla_index_getvector_v1
#define icla_index_copyvector              icla_index_copyvector_v1
#define icla_index_setmatrix               icla_index_setmatrix_v1
#define icla_index_getmatrix               icla_index_getmatrix_v1
#define icla_index_copymatrix              icla_index_copymatrix_v1

#endif // ICLABLAS_V1_MAP_H
