/*
    -- ICLA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
*/

#ifndef ICLA_H
#define ICLA_H

#ifdef ICLA_NO_V1
#error "Since ICLA_NO_V1 is defined, icla.h is invalid; use icla_v2.h"
#endif

// =============================================================================
// ICLA configuration
#include "icla_config.h"


// icla v1 includes cublas.h by default, unless cublas_v2.h has already been included
#ifndef CUBLAS_V2_H_
#if defined(ICLA_HAVE_CUDA)
#include <cublas.h>
#endif
#endif

// Include the ICLA v2 and v1 APIs,
// then map names to the v1 API (e.g., icla_zgemm => icla_zgemm_v1).
// Some functions (like setmatrix_async) are the same in v1 and v2,
// so are provided by the v2 API.
#include "icla_v2.h"
#include "iclablas_v1.h"
#include "iclablas_v1_map.h"

#undef  ICLA_API
#define ICLA_API 1

#endif // ICLA_H
