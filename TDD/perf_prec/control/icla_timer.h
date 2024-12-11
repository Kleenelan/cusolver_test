
#ifndef ICLA_TIMER_H
#define ICLA_TIMER_H

#include <stdio.h>

#include "icla_v2.h"

typedef double    icla_timer_t;
typedef long long icla_flops_t;

#if defined(ENABLE_TIMER)
    #include <stdio.h>
    #include <stdarg.h>

    #if defined(HAVE_PAPI)
        #include <papi.h>
        extern int gPAPI_flops_set;

    #endif
#endif

#ifndef __GNUC__
  #define  __attribute__(x)

#endif

static inline void timer_start( icla_timer_t &t )
{
    #if defined(ENABLE_TIMER)
    t = icla_wtime();
    #endif
}

static inline void timer_sync_start( icla_timer_t &t, icla_queue_t queue )
{
    #if defined(ENABLE_TIMER)
    icla_queue_sync( queue );
    t = icla_wtime();
    #endif
}

static inline icla_timer_t timer_stop( icla_timer_t &t )
{
    #if defined(ENABLE_TIMER)
    t = icla_wtime() - t;
    return t;
    #else
    return 0;
    #endif
}

static inline icla_timer_t timer_sync_stop( icla_timer_t &t, icla_queue_t queue )
{
    #if defined(ENABLE_TIMER)
    icla_queue_sync( queue );
    t = icla_wtime() - t;
    return t;
    #else
    return 0;
    #endif
}

static inline void flops_start( icla_flops_t &flops )
{
    #if defined(ENABLE_TIMER) && defined(HAVE_PAPI)
    PAPI_read( gPAPI_flops_set, &flops );
    #endif
}

static inline icla_flops_t flops_stop( icla_flops_t &flops )
{
    #if defined(ENABLE_TIMER) && defined(HAVE_PAPI)
    icla_flops_t end;
    PAPI_read( gPAPI_flops_set, &end );
    flops = end - flops;
    return flops;
    #else
    return 0;
    #endif
}

static inline int timer_printf( const char* format, ... )
    __attribute__((format(printf,1,2)));

static inline int timer_printf( const char* format, ... )
{
    int len = 0;
    #if defined(ENABLE_TIMER)
    va_list ap;
    va_start( ap, format );
    len = vprintf( format, ap );
    va_end( ap );
    #endif
    return len;
}

static inline int timer_fprintf( FILE* stream, const char* format, ... )
    __attribute__((format(printf,2,3)));

static inline int timer_fprintf( FILE* stream, const char* format, ... )
{
    int len = 0;
    #if defined(ENABLE_TIMER)
    va_list ap;
    va_start( ap, format );
    len = vfprintf( stream, format, ap );
    va_end( ap );
    #endif
    return len;
}

static inline int timer_snprintf( char* str, size_t size, const char* format, ... )
    __attribute__((format(printf,3,4)));

static inline int timer_snprintf( char* str, size_t size, const char* format, ... )
{
    int len = 0;
    #if defined(ENABLE_TIMER)
    va_list ap;
    va_start( ap, format );
    len = vsnprintf( str, size, format, ap );
    va_end( ap );
    #endif
    return len;
}

#endif

