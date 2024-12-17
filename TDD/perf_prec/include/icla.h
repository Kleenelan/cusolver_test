

#ifndef ICLA_H
#define ICLA_H

#ifdef ICLA_NO_V1
#error "Since ICLA_NO_V1 is defined, icla.h is invalid; use icla_v2.h"
#endif

#include "icla_config.h"

#ifndef CUBLAS_V2_H_
#if defined(ICLA_HAVE_CUDA)
#include <cublas.h>
#endif
#endif

#include "icla_v2.h"
#include "iclablas_v1.h"
#include "iclablas_v1_map.h"

#undef  ICLA_API
#define ICLA_API 1

#endif
