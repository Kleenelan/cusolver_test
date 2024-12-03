/*
    -- ICLA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mathieu Faverge
       @author Mark Gates
*/

#ifndef ICLA_OPERATORS_H
#define ICLA_OPERATORS_H

#ifdef __cplusplus

// __host__ and __device__ are defined in CUDA headers.
#include "icla_types.h"

#ifdef ICLA_HAVE_OPENCL
#define __host__
#define __device__
#endif

/// @addtogroup icla_complex
/// In C++, including icla_operators.h defines the usual unary and binary
/// operators for complex numbers: +, +=, -, -=, *, *=, /, /=, ==, !=.
/// Additionally, real(), imag(), conj(), fabs(), and abs1() are defined
/// to apply to both complex and real numbers.
///
/// In C, there are equivalent macros:
/// ICLA_Z_{MAKE, REAL, IMAG, ADD, SUB, MUL, DIV, ABS, ABS1, CONJ} for double-complex,
/// ICLA_C_{...} for float-complex,
/// ICLA_D_{...} for double,
/// ICLA_S_{...} for float.
///
/// Just the double-complex versions are documented here.


// =============================================================================
// names to match C++ std complex functions

/// @return real component of complex number x; x for real number.
/// @ingroup icla_complex


// hip_complex.h manually uses the function name 'real' and 'imag' in the GLOBAL namespace (why they claim the name 'real' is beyond me...), but it should work the same as ours
__host__ __device__ static inline double real(const iclaDoubleComplex &x) { return ICLA_Z_REAL(x); }
__host__ __device__ static inline float  real(const iclaFloatComplex  &x) { return ICLA_C_REAL(x); }

__host__ __device__ static inline double imag(const iclaDoubleComplex &x) { return ICLA_Z_IMAG(x); }
__host__ __device__ static inline float  imag(const iclaFloatComplex  &x) { return ICLA_C_IMAG(x); }

__host__ __device__ static inline iclaDoubleComplex conj(const iclaDoubleComplex &x) { return ICLA_Z_CONJ(x); }
__host__ __device__ static inline iclaFloatComplex  conj(const iclaFloatComplex  &x) { return ICLA_C_CONJ(x); }
//#endif

__host__ __device__ static inline double real(const double             &x) { return x; }
__host__ __device__ static inline float  real(const float              &x) { return x; }

/// @return imaginary component of complex number x; 0 for real number.
/// @ingroup icla_complex
__host__ __device__ static inline double imag(const double             &x) { return 0.; }
__host__ __device__ static inline float  imag(const float              &x) { return 0.f; }

/// @return conjugate of complex number x; x for real number.
/// @ingroup icla_complex
__host__ __device__ static inline double             conj(const double             &x) { return x; }
__host__ __device__ static inline float              conj(const float              &x) { return x; }

/// @return 2-norm absolute value of complex number x: sqrt( real(x)^2 + imag(x)^2 ).
///         math.h or cmath provide fabs for real numbers.
/// @ingroup icla_complex
__host__ __device__ static inline double fabs(const iclaDoubleComplex &x) { return ICLA_Z_ABS(x); }
__host__ __device__ static inline float  fabs(const iclaFloatComplex  &x) { return ICLA_C_ABS(x); }
//__host__ __device__ static inline float  fabs(const float              &x) { return ICLA_S_ABS(x); }  // conflicts with std::fabs in .cu files
// already have fabs( double ) in math.h

/// @return 1-norm absolute value of complex nmuber x: | real(x) | + | imag(x) |.
/// @ingroup icla_complex
__host__ __device__ static inline double abs1(const iclaDoubleComplex &x) { return ICLA_Z_ABS1(x); }
__host__ __device__ static inline float  abs1(const iclaFloatComplex  &x) { return ICLA_C_ABS1(x); }
__host__ __device__ static inline double abs1(const double             &x) { return ICLA_D_ABS1(x); }
__host__ __device__ static inline float  abs1(const float              &x) { return ICLA_S_ABS1(x); }


// =============================================================================
// iclaDoubleComplex

// hip_complex.h also defines oeprators

// ---------- negate
__host__ __device__ static inline iclaDoubleComplex
operator - (const iclaDoubleComplex &a)
{
    return ICLA_Z_MAKE( -real(a),
                         -imag(a) );
}


// ---------- add
__host__ __device__ static inline iclaDoubleComplex
operator + (const iclaDoubleComplex a, const iclaDoubleComplex b)
{
    return ICLA_Z_MAKE( real(a) + real(b),
                         imag(a) + imag(b) );
}

__host__ __device__ static inline iclaDoubleComplex
operator + (const iclaDoubleComplex a, const double s)
{
    return ICLA_Z_MAKE( real(a) + s,
                         imag(a) );
}

__host__ __device__ static inline iclaDoubleComplex
operator + (const double s, const iclaDoubleComplex b)
{
    return ICLA_Z_MAKE( s + real(b),
                             imag(b) );
}

__host__ __device__ static inline iclaDoubleComplex&
operator += (iclaDoubleComplex &a, const iclaDoubleComplex b)
{
    a = ICLA_Z_MAKE( real(a) + real(b),
                      imag(a) + imag(b) );
    return a;
}

__host__ __device__ static inline iclaDoubleComplex&
operator += (iclaDoubleComplex &a, const double s)
{
    a = ICLA_Z_MAKE( real(a) + s,
                      imag(a) );
    return a;
}


// ---------- subtract
__host__ __device__ static inline iclaDoubleComplex
operator - (const iclaDoubleComplex a, const iclaDoubleComplex b)
{
    return ICLA_Z_MAKE( real(a) - real(b),
                         imag(a) - imag(b) );
}

__host__ __device__ static inline iclaDoubleComplex
operator - (const iclaDoubleComplex a, const double s)
{
    return ICLA_Z_MAKE( real(a) - s,
                         imag(a) );
}

__host__ __device__ static inline iclaDoubleComplex
operator - (const double s, const iclaDoubleComplex b)
{
    return ICLA_Z_MAKE( s - real(b),
                           - imag(b) );
}

__host__ __device__ static inline iclaDoubleComplex&
operator -= (iclaDoubleComplex &a, const iclaDoubleComplex b)
{
    a = ICLA_Z_MAKE( real(a) - real(b),
                      imag(a) - imag(b) );
    return a;
}

__host__ __device__ static inline iclaDoubleComplex&
operator -= (iclaDoubleComplex &a, const double s)
{
    a = ICLA_Z_MAKE( real(a) - s,
                      imag(a) );
    return a;
}


// ---------- multiply
__host__ __device__ static inline iclaDoubleComplex
operator * (const iclaDoubleComplex a, const iclaDoubleComplex b)
{
    return ICLA_Z_MAKE( real(a)*real(b) - imag(a)*imag(b),
                         imag(a)*real(b) + real(a)*imag(b) );
}

__host__ __device__ static inline iclaDoubleComplex
operator * (const iclaDoubleComplex a, const double s)
{
    return ICLA_Z_MAKE( real(a)*s,
                         imag(a)*s );
}

__host__ __device__ static inline iclaDoubleComplex
operator * (const iclaDoubleComplex a, const float s)
{
    return ICLA_Z_MAKE( real(a)*s,
                         imag(a)*s );
}



__host__ __device__ static inline iclaDoubleComplex
operator * (const double s, const iclaDoubleComplex a)
{
    return ICLA_Z_MAKE( real(a)*s,
                         imag(a)*s );
}

__host__ __device__ static inline iclaDoubleComplex&
operator *= (iclaDoubleComplex &a, const iclaDoubleComplex b)
{
    a = ICLA_Z_MAKE( real(a)*real(b) - imag(a)*imag(b),
                      imag(a)*real(b) + real(a)*imag(b) );
    return a;
}

__host__ __device__ static inline iclaDoubleComplex&
operator *= (iclaDoubleComplex &a, const double s)
{
    a = ICLA_Z_MAKE( real(a)*s,
                      imag(a)*s );
    return a;
}


// ---------- divide
/* From LAPACK DLADIV
 * Performs complex division in real arithmetic, avoiding unnecessary overflow.
 *
 *             a + i*b
 *  p + i*q = ---------
 *             c + i*d
 */
__host__ __device__ static inline iclaDoubleComplex
operator / (const iclaDoubleComplex x, const iclaDoubleComplex y)
{
    double a = real(x);
    double b = imag(x);
    double c = real(y);
    double d = imag(y);
    double e, f, p, q;
    if ( fabs( d ) < fabs( c ) ) {
        e = d / c;
        f = c + d*e;
        p = ( a + b*e ) / f;
        q = ( b - a*e ) / f;
    }
    else {
        e = c / d;
        f = d + c*e;
        p = (  b + a*e ) / f;
        q = ( -a + b*e ) / f;
    }
    return ICLA_Z_MAKE( p, q );
}

__host__ __device__ static inline iclaDoubleComplex
operator / (const iclaDoubleComplex a, const double s)
{
    return ICLA_Z_MAKE( real(a)/s,
                         imag(a)/s );
}

__host__ __device__ static inline iclaDoubleComplex
operator / (const double a, const iclaDoubleComplex y)
{
    double c = real(y);
    double d = imag(y);
    double e, f, p, q;
    if ( fabs( d ) < fabs( c ) ) {
        e = d / c;
        f = c + d*e;
        p =  a   / f;
        q = -a*e / f;
    }
    else {
        e = c / d;
        f = d + c*e;
        p =  a*e / f;
        q = -a   / f;
    }
    return ICLA_Z_MAKE( p, q );
}

__host__ __device__ static inline iclaDoubleComplex&
operator /= (iclaDoubleComplex &a, const iclaDoubleComplex b)
{
    a = a/b;
    return a;
}

__host__ __device__ static inline iclaDoubleComplex&
operator /= (iclaDoubleComplex &a, const double s)
{
    a = ICLA_Z_MAKE( real(a)/s,
                      imag(a)/s );
    return a;
}


// ---------- equality
__host__ __device__ static inline bool
operator == (const iclaDoubleComplex a, const iclaDoubleComplex b)
{
    return ( real(a) == real(b) &&
             imag(a) == imag(b) );
}

__host__ __device__ static inline bool
operator == (const iclaDoubleComplex a, const double s)
{
    return ( real(a) == s &&
             imag(a) == 0. );
}

__host__ __device__ static inline bool
operator == (const double s, const iclaDoubleComplex a)
{
    return ( real(a) == s &&
             imag(a) == 0. );
}


// ---------- not equality
__host__ __device__ static inline bool
operator != (const iclaDoubleComplex a, const iclaDoubleComplex b)
{
    return ! (a == b);
}
__host__ __device__ static inline bool
operator != (const iclaDoubleComplex a, const double s)
{
    return ! (a == s);
}

__host__ __device__ static inline bool
operator != (const double s, const iclaDoubleComplex a)
{
    return ! (a == s);
}



// =============================================================================
// iclaFloatComplex

// ---------- negate
__host__ __device__ static inline iclaFloatComplex
operator - (const iclaFloatComplex &a)
{
    return ICLA_C_MAKE( -real(a),
                         -imag(a) );
}


// ---------- add
__host__ __device__ static inline iclaFloatComplex
operator + (const iclaFloatComplex a, const iclaFloatComplex b)
{
    return ICLA_C_MAKE( real(a) + real(b),
                         imag(a) + imag(b) );
}

__host__ __device__ static inline iclaFloatComplex
operator + (const iclaFloatComplex a, const float s)
{
    return ICLA_C_MAKE( real(a) + s,
                         imag(a) );
}

__host__ __device__ static inline iclaFloatComplex
operator + (const float s, const iclaFloatComplex b)
{
    return ICLA_C_MAKE( s + real(b),
                             imag(b) );
}

__host__ __device__ static inline iclaFloatComplex&
operator += (iclaFloatComplex &a, const iclaFloatComplex b)
{
    a = ICLA_C_MAKE( real(a) + real(b),
                      imag(a) + imag(b) );
    return a;
}

__host__ __device__ static inline iclaFloatComplex&
operator += (iclaFloatComplex &a, const float s)
{
    a = ICLA_C_MAKE( real(a) + s,
                      imag(a) );
    return a;
}


// ---------- subtract
__host__ __device__ static inline iclaFloatComplex
operator - (const iclaFloatComplex a, const iclaFloatComplex b)
{
    return ICLA_C_MAKE( real(a) - real(b),
                         imag(a) - imag(b) );
}

__host__ __device__ static inline iclaFloatComplex
operator - (const iclaFloatComplex a, const float s)
{
    return ICLA_C_MAKE( real(a) - s,
                         imag(a) );
}

__host__ __device__ static inline iclaFloatComplex
operator - (const float s, const iclaFloatComplex b)
{
    return ICLA_C_MAKE( s - real(b),
                           - imag(b) );
}

__host__ __device__ static inline iclaFloatComplex&
operator -= (iclaFloatComplex &a, const iclaFloatComplex b)
{
    a = ICLA_C_MAKE( real(a) - real(b),
                      imag(a) - imag(b) );
    return a;
}

__host__ __device__ static inline iclaFloatComplex&
operator -= (iclaFloatComplex &a, const float s)
{
    a = ICLA_C_MAKE( real(a) - s,
                      imag(a) );
    return a;
}


// ---------- multiply
__host__ __device__ static inline iclaFloatComplex
operator * (const iclaFloatComplex a, const iclaFloatComplex b)
{
    return ICLA_C_MAKE( real(a)*real(b) - imag(a)*imag(b),
                         imag(a)*real(b) + real(a)*imag(b) );
}

__host__ __device__ static inline iclaFloatComplex
operator * (const iclaFloatComplex a, const float s)
{
    return ICLA_C_MAKE( real(a)*s,
                         imag(a)*s );
}

__host__ __device__ static inline iclaFloatComplex
operator * (const float s, const iclaFloatComplex a)
{
    return ICLA_C_MAKE( real(a)*s,
                         imag(a)*s );
}

__host__ __device__ static inline iclaFloatComplex&
operator *= (iclaFloatComplex &a, const iclaFloatComplex b)
{
    a = ICLA_C_MAKE( real(a)*real(b) - imag(a)*imag(b),
                      imag(a)*real(b) + real(a)*imag(b) );
    return a;
}

__host__ __device__ static inline iclaFloatComplex&
operator *= (iclaFloatComplex &a, const float s)
{
    a = ICLA_C_MAKE( real(a)*s,
                      imag(a)*s );
    return a;
}


// ---------- divide
/* From LAPACK DLADIV
 * Performs complex division in real arithmetic, avoiding unnecessary overflow.
 *
 *             a + i*b
 *  p + i*q = ---------
 *             c + i*d
 */
__host__ __device__ static inline iclaFloatComplex
operator / (const iclaFloatComplex x, const iclaFloatComplex y)
{
    float a = real(x);
    float b = imag(x);
    float c = real(y);
    float d = imag(y);
    float e, f, p, q;
    if ( fabs( d ) < fabs( c ) ) {
        e = d / c;
        f = c + d*e;
        p = ( a + b*e ) / f;
        q = ( b - a*e ) / f;
    }
    else {
        e = c / d;
        f = d + c*e;
        p = (  b + a*e ) / f;
        q = ( -a + b*e ) / f;
    }
    return ICLA_C_MAKE( p, q );
}

__host__ __device__ static inline iclaFloatComplex
operator / (const iclaFloatComplex a, const float s)
{
    return ICLA_C_MAKE( real(a)/s,
                         imag(a)/s );
}

__host__ __device__ static inline iclaFloatComplex
operator / (const float a, const iclaFloatComplex y)
{
    float c = real(y);
    float d = imag(y);
    float e, f, p, q;
    if ( fabs( d ) < fabs( c ) ) {
        e = d / c;
        f = c + d*e;
        p =  a   / f;
        q = -a*e / f;
    }
    else {
        e = c / d;
        f = d + c*e;
        p =  a*e / f;
        q = -a   / f;
    }
    return ICLA_C_MAKE( p, q );
}

__host__ __device__ static inline iclaFloatComplex&
operator /= (iclaFloatComplex &a, const iclaFloatComplex b)
{
    a = a/b;
    return a;
}

__host__ __device__ static inline iclaFloatComplex&
operator /= (iclaFloatComplex &a, const float s)
{
    a = ICLA_C_MAKE( real(a)/s,
                      imag(a)/s );
    return a;
}


// ---------- equality
__host__ __device__ static inline bool
operator == (const iclaFloatComplex a, const iclaFloatComplex b)
{
    return ( real(a) == real(b) &&
             imag(a) == imag(b) );
}

__host__ __device__ static inline bool
operator == (const iclaFloatComplex a, const float s)
{
    return ( real(a) == s &&
             imag(a) == 0. );
}

__host__ __device__ static inline bool
operator == (const float s, const iclaFloatComplex a)
{
    return ( real(a) == s &&
             imag(a) == 0. );
}


// ---------- not equality
__host__ __device__ static inline bool
operator != (const iclaFloatComplex a, const iclaFloatComplex b)
{
    return ! (a == b);
}

__host__ __device__ static inline bool
operator != (const iclaFloatComplex a, const float s)
{
    return ! (a == s);
}

__host__ __device__ static inline bool
operator != (const float s, const iclaFloatComplex a)
{
    return ! (a == s);
}


#ifdef ICLA_HAVE_OPENCL
#undef __host__
#undef __device__
#endif

#endif /* __cplusplus */

#endif /* ICLA_OPERATORS_H */
