/*
    -- ICLA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates
       @precisions normal z -> s d c
*/
#include "icla_internal.h"
#include "error.h"

#define COMPLEX

#define PRECISION_z

/* on some platforms (i.e. hipICLA on ROCm stack), we define custom types
 * So, to keep the C++ compiler from giving errors, we cast arguments to internal
 * BLAS routines. The hipify script should replace `cu*Complex` with appropriate HIP types
 *
 * FUTURE READERS: If hipBLAS changes numbers to `hipblas*Complex` rather than `hip*Complex`,
 *   these will need more complicated macro if/else blocks
 */
/*#ifdef PRECISION_z
  #ifdef ICLA_HAVE_HIP
    typedef hipDoubleComplex cuDoubleComplex;
  #else
    typedef cuDoubleComplex cuDoubleComplex;
  #endif
#elif defined(PRECISION_c)
  #ifdef ICLA_HAVE_HIP
    typedef hipComplex cuDoubleComplex;
  #else
    typedef cuFloatComplex cuDoubleComplex;
  #endif
#elif defined(PRECISION_d)
  typedef double cuDoubleComplex;
#else
  typedef float cuDoubleComplex;
#endif
*/
//#ifdef ICLA_HAVE_CUDA

// =============================================================================
// Level 1 BLAS

/***************************************************************************//**
    @return Index of element of vector x having max. absolute value;
            \f$ \text{argmax}_i\; | real(x_i) | + | imag(x_i) | \f$.

    @param[in]
    n       Number of elements in vector x. n >= 0.

    @param[in]
    dx      COMPLEX_16 array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx > 0.

    @param[in]
    queue   icla_queue_t
            Queue to execute in.

    @ingroup icla_iamax
*******************************************************************************/
extern "C" icla_int_t
icla_izamax(
    icla_int_t n,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    icla_queue_t queue )
{
    int result; /* not icla_int_t */
    cublasIzamax( queue->cublas_handle(), int(n), (cuDoubleComplex*)dx, int(incx), &result );
    return result;
}


/***************************************************************************//**
    @return Index of element of vector x having min. absolute value;
            \f$ \text{argmin}_i\; | real(x_i) | + | imag(x_i) | \f$.

    @param[in]
    n       Number of elements in vector x. n >= 0.

    @param[in]
    dx      COMPLEX_16 array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx > 0.

    @param[in]
    queue   icla_queue_t
            Queue to execute in.

    @ingroup icla_iamin
*******************************************************************************/
extern "C" icla_int_t
icla_izamin(
    icla_int_t n,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    icla_queue_t queue )
{
    int result; /* not icla_int_t */
    cublasIzamin( queue->cublas_handle(), int(n), (cuDoubleComplex*)dx, int(incx), &result );
    return result;
}


/***************************************************************************//**
    @return Sum of absolute values of vector x;
            \f$ \sum_i | real(x_i) | + | imag(x_i) | \f$.

    @param[in]
    n       Number of elements in vector x. n >= 0.

    @param[in]
    dx      COMPLEX_16 array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx > 0.

    @param[in]
    queue   icla_queue_t
            Queue to execute in.

    @ingroup icla_asum
*******************************************************************************/
extern "C" double
icla_dzasum(
    icla_int_t n,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    icla_queue_t queue )
{
    double result;
    cublasDzasum( queue->cublas_handle(), int(n), (cuDoubleComplex*)dx, int(incx), &result );
    return result;
}


/***************************************************************************//**
    Constant times a vector plus a vector; \f$ y = \alpha x + y \f$.

    @param[in]
    n       Number of elements in vectors x and y. n >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dx      COMPLEX_16 array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in,out]
    dy      COMPLEX_16 array on GPU device.
            The n element vector y of dimension (1 + (n-1)*incy).

    @param[in]
    incy    Stride between consecutive elements of dy. incy != 0.

    @param[in]
    queue   icla_queue_t
            Queue to execute in.

    @ingroup icla_axpy
*******************************************************************************/
extern "C" void
icla_zaxpy(
    icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex_ptr       dy, icla_int_t incy,
    icla_queue_t queue )
{
    cublasZaxpy( queue->cublas_handle(), int(n), (cuDoubleComplex*)&alpha, (cuDoubleComplex*)dx, int(incx), (cuDoubleComplex*)dy, int(incy) );
}


/***************************************************************************//**
    Copy vector x to vector y; \f$ y = x \f$.

    @param[in]
    n       Number of elements in vectors x and y. n >= 0.

    @param[in]
    dx      COMPLEX_16 array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[out]
    dy      COMPLEX_16 array on GPU device.
            The n element vector y of dimension (1 + (n-1)*incy).

    @param[in]
    incy    Stride between consecutive elements of dy. incy != 0.

    @param[in]
    queue   icla_queue_t
            Queue to execute in.

    @ingroup icla_copy
*******************************************************************************/
extern "C" void
icla_zcopy(
    icla_int_t n,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex_ptr       dy, icla_int_t incy,
    icla_queue_t queue )
{
    cublasZcopy( queue->cublas_handle(), int(n), (cuDoubleComplex*)dx, int(incx), (cuDoubleComplex*)dy, int(incy) );
}


#ifdef COMPLEX
/***************************************************************************//**
    @return Dot product of vectors x and y; \f$ x^H y \f$.

    @param[in]
    n       Number of elements in vector x and y. n >= 0.

    @param[in]
    dx      COMPLEX_16 array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in]
    dy      COMPLEX_16 array on GPU device.
            The n element vector y of dimension (1 + (n-1)*incy).

    @param[in]
    incy    Stride between consecutive elements of dy. incy != 0.

    @param[in]
    queue   icla_queue_t
            Queue to execute in.

    @ingroup icla__dot
*******************************************************************************/
extern "C"
iclaDoubleComplex icla_zdotc(
    icla_int_t n,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex_const_ptr dy, icla_int_t incy,
    icla_queue_t queue )
{
    iclaDoubleComplex result;
    cublasZdotc( queue->cublas_handle(), int(n), (cuDoubleComplex*)dx, int(incx), (cuDoubleComplex*)dy, int(incy), (cuDoubleComplex*)&result );
    return result;
}
#endif // COMPLEX


/***************************************************************************//**
    @return Dot product (unconjugated) of vectors x and y; \f$ x^T y \f$.

    @param[in]
    n       Number of elements in vector x and y. n >= 0.

    @param[in]
    dx      COMPLEX_16 array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in]
    dy      COMPLEX_16 array on GPU device.
            The n element vector y of dimension (1 + (n-1)*incy).

    @param[in]
    incy    Stride between consecutive elements of dy. incy != 0.

    @param[in]
    queue   icla_queue_t
            Queue to execute in.

    @ingroup icla__dot
*******************************************************************************/
extern "C"
iclaDoubleComplex icla_zdotu(
    icla_int_t n,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex_const_ptr dy, icla_int_t incy,
    icla_queue_t queue )
{
    iclaDoubleComplex result;
    cublasZdotu( queue->cublas_handle(), int(n), (cuDoubleComplex*)dx, int(incx), (cuDoubleComplex*)dy, int(incy), (cuDoubleComplex*)&result );
    return result;
}


/***************************************************************************//**
    @return 2-norm of vector x; \f$ \text{sqrt}( x^H x ) \f$.
            Avoids unnecesary over/underflow.

    @param[in]
    n       Number of elements in vector x and y. n >= 0.

    @param[in]
    dx      COMPLEX_16 array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx > 0.

    @param[in]
    queue   icla_queue_t
            Queue to execute in.

    @ingroup icla_nrm2
*******************************************************************************/
extern "C" double
icla_dznrm2(
    icla_int_t n,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    icla_queue_t queue )
{
    double result;
    cublasDznrm2( queue->cublas_handle(), int(n), (cuDoubleComplex*)dx, int(incx), &result );
    return result;
}


/***************************************************************************//**
    Apply Givens plane rotation, where cos (c) is real and sin (s) is complex.

    @param[in]
    n       Number of elements in vector x and y. n >= 0.

    @param[in,out]
    dx      COMPLEX_16 array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).
            On output, overwritten with c*x + s*y.

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in,out]
    dy      COMPLEX_16 array on GPU device.
            The n element vector y of dimension (1 + (n-1)*incy).
            On output, overwritten with -conj(s)*x + c*y.

    @param[in]
    incy    Stride between consecutive elements of dy. incy != 0.

    @param[in]
    c       double. cosine.

    @param[in]
    s       COMPLEX_16. sine. c and s define a rotation
            [ c         s ]  where c*c + s*conj(s) = 1.
            [ -conj(s)  c ]

    @param[in]
    queue   icla_queue_t
            Queue to execute in.

    @ingroup icla_rot
*******************************************************************************/
extern "C" void
icla_zrot(
    icla_int_t n,
    iclaDoubleComplex_ptr dx, icla_int_t incx,
    iclaDoubleComplex_ptr dy, icla_int_t incy,
    double c, iclaDoubleComplex s,
    icla_queue_t queue )
{
    cublasZrot( queue->cublas_handle(), int(n), (cuDoubleComplex*)dx, int(incx), (cuDoubleComplex*)dy, int(incy), &c, (cuDoubleComplex*)&s );
}


#ifdef COMPLEX
/***************************************************************************//**
    Apply Givens plane rotation, where cos (c) and sin (s) are real.

    @param[in]
    n       Number of elements in vector x and y. n >= 0.

    @param[in,out]
    dx      COMPLEX_16 array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).
            On output, overwritten with c*x + s*y.

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in,out]
    dy      COMPLEX_16 array on GPU device.
            The n element vector y of dimension (1 + (n-1)*incy).
            On output, overwritten with -conj(s)*x + c*y.

    @param[in]
    incy    Stride between consecutive elements of dy. incy != 0.

    @param[in]
    c       double. cosine.

    @param[in]
    s       double. sine. c and s define a rotation
            [  c  s ]  where c*c + s*s = 1.
            [ -s  c ]

    @param[in]
    queue   icla_queue_t
            Queue to execute in.

    @ingroup icla_rot
*******************************************************************************/
extern "C" void
icla_zdrot(
    icla_int_t n,
    iclaDoubleComplex_ptr dx, icla_int_t incx,
    iclaDoubleComplex_ptr dy, icla_int_t incy,
    double c, double s,
    icla_queue_t queue )
{
    cublasZdrot( queue->cublas_handle(), int(n), (cuDoubleComplex*)dx, int(incx), (cuDoubleComplex*)dy, int(incy), &c, &s );
}
#endif // COMPLEX


/***************************************************************************//**
    Generate a Givens plane rotation.
    The rotation annihilates the second entry of the vector, such that:

        (  c  s ) * ( a ) = ( r )
        ( -s  c )   ( b )   ( 0 )

    where \f$ c^2 + s^2 = 1 \f$ and \f$ r = a^2 + b^2 \f$.
    Further, this computes z such that

                { (sqrt(1 - z^2), z),    if |z| < 1,
        (c,s) = { (0, 1),                if |z| = 1,
                { (1/z, sqrt(1 - z^2)),  if |z| > 1.

    @param[in]
    a       On input, entry to be modified.
            On output, updated to r by applying the rotation.

    @param[in,out]
    b       On input, entry to be annihilated.
            On output, set to z.

    @param[in]
    c       On output, cosine of rotation.

    @param[in,out]
    s       On output, sine of rotation.

    @param[in]
    queue   icla_queue_t
            Queue to execute in.

    @ingroup icla_rotg
*******************************************************************************/
extern "C" void
icla_zrotg(
    iclaDoubleComplex *a, iclaDoubleComplex *b,
    double             *c, iclaDoubleComplex *s,
    icla_queue_t queue )
{
    cublasZrotg( queue->cublas_handle(), (cuDoubleComplex*)a, (cuDoubleComplex*)b, c, (cuDoubleComplex*)s );
}


#ifdef REAL
/***************************************************************************//**
    Apply modified plane rotation.

    @ingroup icla_rotm
*******************************************************************************/
extern "C" void
icla_zrotm(
    icla_int_t n,
    double *dx, icla_int_t incx,
    double *dy, icla_int_t incy,
    const double *param,
    icla_queue_t queue )
{
    cublasZrotm( queue->cublas_handle(), int(n), dx, int(incx), dy, int(incy), param );
}
#endif // REAL


#ifdef REAL
/***************************************************************************//**
    Generate modified plane rotation.

    @ingroup icla_rotmg
*******************************************************************************/
extern "C" void
icla_zrotmg(
    double *d1, double       *d2,
    double *x1, const double *y1,
    double *param,
    icla_queue_t queue )
{
    cublasZrotmg( queue->cublas_handle(), d1, d2, x1, y1, param );
}
#endif // REAL


/***************************************************************************//**
    Scales a vector by a constant; \f$ x = \alpha x \f$.

    @param[in]
    n       Number of elements in vector x. n >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in,out]
    dx      COMPLEX_16 array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx > 0.

    @param[in]
    queue   icla_queue_t
            Queue to execute in.

    @ingroup icla_scal
*******************************************************************************/
extern "C" void
icla_zscal(
    icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_ptr dx, icla_int_t incx,
    icla_queue_t queue )
{
    cublasZscal( queue->cublas_handle(), int(n), (cuDoubleComplex*)&alpha, (cuDoubleComplex*)dx, int(incx) );
}


#ifdef COMPLEX
/***************************************************************************//**
    Scales a vector by a real constant; \f$ x = \alpha x \f$.

    @param[in]
    n       Number of elements in vector x. n >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$ (real)

    @param[in,out]
    dx      COMPLEX_16 array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx > 0.

    @param[in]
    queue   icla_queue_t
            Queue to execute in.

    @ingroup icla_scal
*******************************************************************************/
extern "C" void
icla_zdscal(
    icla_int_t n,
    double alpha,
    iclaDoubleComplex_ptr dx, icla_int_t incx,
    icla_queue_t queue )
{
    cublasZdscal( queue->cublas_handle(), int(n), &alpha, (cuDoubleComplex*)dx, int(incx) );
}
#endif // COMPLEX


/***************************************************************************//**
    Swap vector x and y; \f$ x <-> y \f$.

    @param[in]
    n       Number of elements in vector x and y. n >= 0.

    @param[in,out]
    dx      COMPLEX_16 array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in,out]
    dy      COMPLEX_16 array on GPU device.
            The n element vector y of dimension (1 + (n-1)*incy).

    @param[in]
    incy    Stride between consecutive elements of dy. incy != 0.

    @param[in]
    queue   icla_queue_t
            Queue to execute in.

    @ingroup icla_swap
*******************************************************************************/
extern "C" void
icla_zswap(
    icla_int_t n,
    iclaDoubleComplex_ptr dx, icla_int_t incx,
    iclaDoubleComplex_ptr dy, icla_int_t incy,
    icla_queue_t queue )
{
    cublasZswap( queue->cublas_handle(), int(n), (cuDoubleComplex*)dx, int(incx), (cuDoubleComplex*)dy, int(incy) );
}


// =============================================================================
// Level 2 BLAS

/***************************************************************************//**
    Perform matrix-vector product.
        \f$ y = \alpha A   x + \beta y \f$  (transA == iclaNoTrans), or \n
        \f$ y = \alpha A^T x + \beta y \f$  (transA == iclaTrans),   or \n
        \f$ y = \alpha A^H x + \beta y \f$  (transA == iclaConjTrans).

    @param[in]
    transA  Operation to perform on A.

    @param[in]
    m       Number of rows of A. m >= 0.

    @param[in]
    n       Number of columns of A. n >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dA      COMPLEX_16 array of dimension (ldda,n), ldda >= max(1,m).
            The m-by-n matrix A, on GPU device.

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    dx      COMPLEX_16 array on GPU device.
            If transA == iclaNoTrans, the n element vector x of dimension (1 + (n-1)*incx); \n
            otherwise,                 the m element vector x of dimension (1 + (m-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in]
    beta    Scalar \f$ \beta \f$

    @param[in,out]
    dy      COMPLEX_16 array on GPU device.
            If transA == iclaNoTrans, the m element vector y of dimension (1 + (m-1)*incy); \n
            otherwise,                 the n element vector y of dimension (1 + (n-1)*incy).

    @param[in]
    incy    Stride between consecutive elements of dy. incy != 0.

    @param[in]
    queue   icla_queue_t
            Queue to execute in.

    @ingroup icla_gemv
*******************************************************************************/
extern "C" void
icla_zgemv(
    icla_trans_t transA,
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dy, icla_int_t incy,
    icla_queue_t queue )
{
    cublasZgemv(
        queue->cublas_handle(),
        cublas_trans_const( transA ),
        int(m), int(n),
        (cuDoubleComplex*)&alpha, (cuDoubleComplex*)dA, int(ldda),
                (cuDoubleComplex*)dx, int(incx),
        (cuDoubleComplex*)&beta,  (cuDoubleComplex*)dy, int(incy) );
}


#ifdef COMPLEX
/***************************************************************************//**
    Perform rank-1 update, \f$ A = \alpha x y^H + A \f$.

    @param[in]
    m       Number of rows of A. m >= 0.

    @param[in]
    n       Number of columns of A. n >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dx      COMPLEX_16 array on GPU device.
            The m element vector x of dimension (1 + (m-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in]
    dy      COMPLEX_16 array on GPU device.
            The n element vector y of dimension (1 + (n-1)*incy).

    @param[in]
    incy    Stride between consecutive elements of dy. incy != 0.

    @param[in,out]
    dA      COMPLEX_16 array on GPU device.
            The m-by-n matrix A of dimension (ldda,n), ldda >= max(1,m).

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    queue   icla_queue_t
            Queue to execute in.

    @ingroup icla_ger
*******************************************************************************/
extern "C" void
icla_zgerc(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex_const_ptr dy, icla_int_t incy,
    iclaDoubleComplex_ptr       dA, icla_int_t ldda,
    icla_queue_t queue )
{
    cublasZgerc(
        queue->cublas_handle(),
        int(m), int(n),
        (cuDoubleComplex*)&alpha, (cuDoubleComplex*)dx, int(incx),
                (cuDoubleComplex*)dy, int(incy),
                (cuDoubleComplex*)dA, int(ldda) );
}
#endif // COMPLEX


/***************************************************************************//**
    Perform rank-1 update (unconjugated), \f$ A = \alpha x y^T + A \f$.

    @param[in]
    m       Number of rows of A. m >= 0.

    @param[in]
    n       Number of columns of A. n >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dx      COMPLEX_16 array on GPU device.
            The m element vector x of dimension (1 + (m-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in]
    dy      COMPLEX_16 array on GPU device.
            The n element vector y of dimension (1 + (n-1)*incy).

    @param[in]
    incy    Stride between consecutive elements of dy. incy != 0.

    @param[in,out]
    dA      COMPLEX_16 array of dimension (ldda,n), ldda >= max(1,m).
            The m-by-n matrix A, on GPU device.

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    queue   icla_queue_t
            Queue to execute in.

    @ingroup icla_ger
*******************************************************************************/
extern "C" void
icla_zgeru(
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex_const_ptr dy, icla_int_t incy,
    iclaDoubleComplex_ptr       dA, icla_int_t ldda,
    icla_queue_t queue )
{
    cublasZgeru(
        queue->cublas_handle(),
        int(m), int(n),
        (cuDoubleComplex*)&alpha, (cuDoubleComplex*)dx, int(incx),
                (cuDoubleComplex*)dy, int(incy),
                (cuDoubleComplex*)dA, int(ldda) );
}


#ifdef COMPLEX
/***************************************************************************//**
    Perform Hermitian matrix-vector product, \f$ y = \alpha A x + \beta y, \f$
    where \f$ A \f$ is Hermitian.

    @param[in]
    uplo    Whether the upper or lower triangle of A is referenced.

    @param[in]
    n       Number of rows and columns of A. n >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dA      COMPLEX_16 array of dimension (ldda,n), ldda >= max(1,n).
            The n-by-n matrix A, on GPU device.

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    dx      COMPLEX_16 array on GPU device.
            The m element vector x of dimension (1 + (m-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in]
    beta    Scalar \f$ \beta \f$

    @param[in,out]
    dy      COMPLEX_16 array on GPU device.
            The n element vector y of dimension (1 + (n-1)*incy).

    @param[in]
    incy    Stride between consecutive elements of dy. incy != 0.

    @param[in]
    queue   icla_queue_t
            Queue to execute in.

    @ingroup icla_hemv
*******************************************************************************/
extern "C" void
icla_zhemv(
    icla_uplo_t uplo,
    icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dy, icla_int_t incy,
    icla_queue_t queue )
{
    cublasZhemv(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        int(n),
        (cuDoubleComplex*)&alpha, (cuDoubleComplex*)dA, int(ldda),
                (cuDoubleComplex*)dx, int(incx),
        (cuDoubleComplex*)&beta,  (cuDoubleComplex*)dy, int(incy) );
}
#endif // COMPLEX


#ifdef COMPLEX
/***************************************************************************//**
    Perform Hermitian rank-1 update, \f$ A = \alpha x x^H + A, \f$
    where \f$ A \f$ is Hermitian.

    @param[in]
    uplo    Whether the upper or lower triangle of A is referenced.

    @param[in]
    n       Number of rows and columns of A. n >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dx      COMPLEX_16 array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in,out]
    dA      COMPLEX_16 array of dimension (ldda,n), ldda >= max(1,n).
            The n-by-n matrix A, on GPU device.

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    queue   icla_queue_t
            Queue to execute in.

    @ingroup icla_her
*******************************************************************************/
extern "C" void
icla_zher(
    icla_uplo_t uplo,
    icla_int_t n,
    double alpha,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex_ptr       dA, icla_int_t ldda,
    icla_queue_t queue )
{
    cublasZher(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        int(n),
        (const double*)&alpha, (cuDoubleComplex*)dx, int(incx),
                (cuDoubleComplex*)dA, int(ldda) );
}
#endif // COMPLEX


#ifdef COMPLEX
/***************************************************************************//**
    Perform Hermitian rank-2 update, \f$ A = \alpha x y^H + conj(\alpha) y x^H + A, \f$
    where \f$ A \f$ is Hermitian.

    @param[in]
    uplo    Whether the upper or lower triangle of A is referenced.

    @param[in]
    n       Number of rows and columns of A. n >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dx      COMPLEX_16 array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in]
    dy      COMPLEX_16 array on GPU device.
            The n element vector y of dimension (1 + (n-1)*incy).

    @param[in]
    incy    Stride between consecutive elements of dy. incy != 0.

    @param[in,out]
    dA      COMPLEX_16 array of dimension (ldda,n), ldda >= max(1,n).
            The n-by-n matrix A, on GPU device.

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    queue   icla_queue_t
            Queue to execute in.

    @ingroup icla_her2
*******************************************************************************/
extern "C" void
icla_zher2(
    icla_uplo_t uplo,
    icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex_const_ptr dy, icla_int_t incy,
    iclaDoubleComplex_ptr       dA, icla_int_t ldda,
    icla_queue_t queue )
{
    cublasZher2(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        int(n),
        (cuDoubleComplex*)&alpha, (cuDoubleComplex*)dx, int(incx),
                (cuDoubleComplex*)dy, int(incy),
                (cuDoubleComplex*)dA, int(ldda) );
}
#endif // COMPLEX


/***************************************************************************//**
    Perform symmetric matrix-vector product, \f$ y = \alpha A x + \beta y, \f$
    where \f$ A \f$ is symmetric.

    @param[in]
    uplo    Whether the upper or lower triangle of A is referenced.

    @param[in]
    n       Number of rows and columns of A. n >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dA      COMPLEX_16 array of dimension (ldda,n), ldda >= max(1,n).
            The n-by-n matrix A, on GPU device.

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    dx      COMPLEX_16 array on GPU device.
            The m element vector x of dimension (1 + (m-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in]
    beta    Scalar \f$ \beta \f$

    @param[in,out]
    dy      COMPLEX_16 array on GPU device.
            The n element vector y of dimension (1 + (n-1)*incy).

    @param[in]
    incy    Stride between consecutive elements of dy. incy != 0.

    @param[in]
    queue   icla_queue_t
            Queue to execute in.

    @ingroup icla_symv
*******************************************************************************/
extern "C" void
icla_zsymv(
    icla_uplo_t uplo,
    icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dy, icla_int_t incy,
    icla_queue_t queue )
{
    cublasZsymv(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        int(n),
        (cuDoubleComplex*)&alpha, (cuDoubleComplex*)dA, int(ldda),
                (cuDoubleComplex*)dx, int(incx),
        (cuDoubleComplex*)&beta,  (cuDoubleComplex*)dy, int(incy) );
}


/***************************************************************************//**
    Perform symmetric rank-1 update, \f$ A = \alpha x x^T + A, \f$
    where \f$ A \f$ is symmetric.

    @param[in]
    uplo    Whether the upper or lower triangle of A is referenced.

    @param[in]
    n       Number of rows and columns of A. n >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dx      COMPLEX_16 array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in,out]
    dA      COMPLEX_16 array of dimension (ldda,n), ldda >= max(1,n).
            The n-by-n matrix A, on GPU device.

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    queue   icla_queue_t
            Queue to execute in.

    @ingroup icla_syr
*******************************************************************************/
extern "C" void
icla_zsyr(
    icla_uplo_t uplo,
    icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex_ptr       dA, icla_int_t ldda,
    icla_queue_t queue )
{
    cublasZsyr(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        int(n),
        (cuDoubleComplex*)&alpha, (cuDoubleComplex*)dx, int(incx),
                (cuDoubleComplex*)dA, int(ldda) );
}


/***************************************************************************//**
    Perform symmetric rank-2 update, \f$ A = \alpha x y^T + \alpha y x^T + A, \f$
    where \f$ A \f$ is symmetric.

    @param[in]
    uplo    Whether the upper or lower triangle of A is referenced.

    @param[in]
    n       Number of rows and columns of A. n >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dx      COMPLEX_16 array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in]
    dy      COMPLEX_16 array on GPU device.
            The n element vector y of dimension (1 + (n-1)*incy).

    @param[in]
    incy    Stride between consecutive elements of dy. incy != 0.

    @param[in,out]
    dA      COMPLEX_16 array of dimension (ldda,n), ldda >= max(1,n).
            The n-by-n matrix A, on GPU device.

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    queue   icla_queue_t
            Queue to execute in.

    @ingroup icla_syr2
*******************************************************************************/
extern "C" void
icla_zsyr2(
    icla_uplo_t uplo,
    icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dx, icla_int_t incx,
    iclaDoubleComplex_const_ptr dy, icla_int_t incy,
    iclaDoubleComplex_ptr       dA, icla_int_t ldda,
    icla_queue_t queue )
{
    cublasZsyr2(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        int(n),
        (cuDoubleComplex*)&alpha, (cuDoubleComplex*)dx, int(incx),
                (cuDoubleComplex*)dy, int(incy),
                (cuDoubleComplex*)dA, int(ldda) );
}


/***************************************************************************//**
    Perform triangular matrix-vector product.
        \f$ x = A   x \f$  (trans == iclaNoTrans), or \n
        \f$ x = A^T x \f$  (trans == iclaTrans),   or \n
        \f$ x = A^H x \f$  (trans == iclaConjTrans).

    @param[in]
    uplo    Whether the upper or lower triangle of A is referenced.

    @param[in]
    trans   Operation to perform on A.

    @param[in]
    diag    Whether the diagonal of A is assumed to be unit or non-unit.

    @param[in]
    n       Number of rows and columns of A. n >= 0.

    @param[in]
    dA      COMPLEX_16 array of dimension (ldda,n), ldda >= max(1,n).
            The n-by-n matrix A, on GPU device.

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    dx      COMPLEX_16 array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in]
    queue   icla_queue_t
            Queue to execute in.

    @ingroup icla_trmv
*******************************************************************************/
extern "C" void
icla_ztrmv(
    icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t n,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr       dx, icla_int_t incx,
    icla_queue_t queue )
{
    cublasZtrmv(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        cublas_diag_const( diag ),
        int(n),
        (cuDoubleComplex*)dA, int(ldda),
        (cuDoubleComplex*)dx, int(incx) );
}


/***************************************************************************//**
    Solve triangular matrix-vector system (one right-hand side).
        \f$ A   x = b \f$  (trans == iclaNoTrans), or \n
        \f$ A^T x = b \f$  (trans == iclaTrans),   or \n
        \f$ A^H x = b \f$  (trans == iclaConjTrans).

    @param[in]
    uplo    Whether the upper or lower triangle of A is referenced.

    @param[in]
    trans   Operation to perform on A.

    @param[in]
    diag    Whether the diagonal of A is assumed to be unit or non-unit.

    @param[in]
    n       Number of rows and columns of A. n >= 0.

    @param[in]
    dA      COMPLEX_16 array of dimension (ldda,n), ldda >= max(1,n).
            The n-by-n matrix A, on GPU device.

    @param[in]
    ldda    Leading dimension of dA.

    @param[in,out]
    dx      COMPLEX_16 array on GPU device.
            On entry, the n element RHS vector b of dimension (1 + (n-1)*incx).
            On exit, overwritten with the solution vector x.

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in]
    queue   icla_queue_t
            Queue to execute in.

    @ingroup icla_trsv
*******************************************************************************/
extern "C" void
icla_ztrsv(
    icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t n,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr       dx, icla_int_t incx,
    icla_queue_t queue )
{
    cublasZtrsv(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        cublas_diag_const( diag ),
        int(n),
        (cuDoubleComplex*)dA, int(ldda),
        (cuDoubleComplex*)dx, int(incx) );
}


// =============================================================================
// Level 3 BLAS

/***************************************************************************//**
    Perform matrix-matrix product, \f$ C = \alpha op(A) op(B) + \beta C \f$.

    @param[in]
    transA  Operation op(A) to perform on matrix A.

    @param[in]
    transB  Operation op(B) to perform on matrix B.

    @param[in]
    m       Number of rows of C and op(A). m >= 0.

    @param[in]
    n       Number of columns of C and op(B). n >= 0.

    @param[in]
    k       Number of columns of op(A) and rows of op(B). k >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dA      COMPLEX_16 array on GPU device.
            If transA == iclaNoTrans, the m-by-k matrix A of dimension (ldda,k), ldda >= max(1,m); \n
            otherwise,                 the k-by-m matrix A of dimension (ldda,m), ldda >= max(1,k).

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    dB      COMPLEX_16 array on GPU device.
            If transB == iclaNoTrans, the k-by-n matrix B of dimension (lddb,n), lddb >= max(1,k); \n
            otherwise,                 the n-by-k matrix B of dimension (lddb,k), lddb >= max(1,n).

    @param[in]
    lddb    Leading dimension of dB.

    @param[in]
    beta    Scalar \f$ \beta \f$

    @param[in,out]
    dC      COMPLEX_16 array on GPU device.
            The m-by-n matrix C of dimension (lddc,n), lddc >= max(1,m).

    @param[in]
    lddc    Leading dimension of dC.

    @param[in]
    queue   icla_queue_t
            Queue to execute in.

    @ingroup icla_gemm
*******************************************************************************/
extern "C" void
icla_zgemm(
    icla_trans_t transA, icla_trans_t transB,
    icla_int_t m, icla_int_t n, icla_int_t k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dB, icla_int_t lddb,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dC, icla_int_t lddc,
    icla_queue_t queue )
{
    cublasZgemm(
        queue->cublas_handle(),
        cublas_trans_const( transA ),
        cublas_trans_const( transB ),
        int(m), int(n), int(k),
        (cuDoubleComplex*)&alpha, (cuDoubleComplex*)dA, int(ldda),
                (cuDoubleComplex*)dB, int(lddb),
        (cuDoubleComplex*)&beta,  (cuDoubleComplex*)dC, int(lddc) );
}

#ifdef COMPLEX
/***************************************************************************//**
    Perform Hermitian matrix-matrix product.
        \f$ C = \alpha A B + \beta C \f$ (side == iclaLeft), or \n
        \f$ C = \alpha B A + \beta C \f$ (side == iclaRight),   \n
    where \f$ A \f$ is Hermitian.

    @param[in]
    side    Whether A is on the left or right.

    @param[in]
    uplo    Whether the upper or lower triangle of A is referenced.

    @param[in]
    m       Number of rows of C. m >= 0.

    @param[in]
    n       Number of columns of C. n >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dA      COMPLEX_16 array on GPU device.
            If side == iclaLeft, the m-by-m Hermitian matrix A of dimension (ldda,m), ldda >= max(1,m); \n
            otherwise,            the n-by-n Hermitian matrix A of dimension (ldda,n), ldda >= max(1,n).

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    dB      COMPLEX_16 array on GPU device.
            The m-by-n matrix B of dimension (lddb,n), lddb >= max(1,m).

    @param[in]
    lddb    Leading dimension of dB.

    @param[in]
    beta    Scalar \f$ \beta \f$

    @param[in,out]
    dC      COMPLEX_16 array on GPU device.
            The m-by-n matrix C of dimension (lddc,n), lddc >= max(1,m).

    @param[in]
    lddc    Leading dimension of dC.

    @param[in]
    queue   icla_queue_t
            Queue to execute in.

    @ingroup icla_hemm
*******************************************************************************/
extern "C" void
icla_zhemm(
    icla_side_t side, icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dB, icla_int_t lddb,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dC, icla_int_t lddc,
    icla_queue_t queue )
{
    cublasZhemm(
        queue->cublas_handle(),
        cublas_side_const( side ),
        cublas_uplo_const( uplo ),
        int(m), int(n),
        (cuDoubleComplex*)&alpha, (cuDoubleComplex*)dA, int(ldda),
                (cuDoubleComplex*)dB, int(lddb),
        (cuDoubleComplex*)&beta,  (cuDoubleComplex*)dC, int(lddc) );
}
#endif // COMPLEX


#ifdef COMPLEX
/***************************************************************************//**
    Perform Hermitian rank-k update.
        \f$ C = \alpha A A^H + \beta C \f$ (trans == iclaNoTrans), or \n
        \f$ C = \alpha A^H A + \beta C \f$ (trans == iclaConjTrans), \n
    where \f$ C \f$ is Hermitian.

    @param[in]
    uplo    Whether the upper or lower triangle of C is referenced.

    @param[in]
    trans   Operation to perform on A.

    @param[in]
    n       Number of rows and columns of C. n >= 0.

    @param[in]
    k       Number of columns of A (for iclaNoTrans)
            or rows of A (for iclaConjTrans). k >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dA      COMPLEX_16 array on GPU device.
            If trans == iclaNoTrans, the n-by-k matrix A of dimension (ldda,k), ldda >= max(1,n); \n
            otherwise,                the k-by-n matrix A of dimension (ldda,n), ldda >= max(1,k).

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    beta    Scalar \f$ \beta \f$

    @param[in,out]
    dC      COMPLEX_16 array on GPU device.
            The n-by-n Hermitian matrix C of dimension (lddc,n), lddc >= max(1,n).

    @param[in]
    lddc    Leading dimension of dC.

    @param[in]
    queue   icla_queue_t
            Queue to execute in.

    @ingroup icla_herk
*******************************************************************************/
extern "C" void
icla_zherk(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    double alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    double beta,
    iclaDoubleComplex_ptr       dC, icla_int_t lddc,
    icla_queue_t queue )
{
    cublasZherk(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        int(n), int(k),
        &alpha, (cuDoubleComplex*)dA, int(ldda),
        &beta,  (cuDoubleComplex*)dC, int(lddc) );
}
#endif // COMPLEX


#ifdef COMPLEX
/***************************************************************************//**
    Perform Hermitian rank-2k update.
        \f$ C = \alpha A B^H + \alpha B A^H \beta C \f$ (trans == iclaNoTrans), or \n
        \f$ C = \alpha A^H B + \alpha B^H A \beta C \f$ (trans == iclaConjTrans), \n
    where \f$ C \f$ is Hermitian.

    @param[in]
    uplo    Whether the upper or lower triangle of C is referenced.

    @param[in]
    trans   Operation to perform on A and B.

    @param[in]
    n       Number of rows and columns of C. n >= 0.

    @param[in]
    k       Number of columns of A and B (for iclaNoTrans)
            or rows of A and B (for iclaConjTrans). k >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dA      COMPLEX_16 array on GPU device.
            If trans == iclaNoTrans, the n-by-k matrix A of dimension (ldda,k), ldda >= max(1,n); \n
            otherwise,                the k-by-n matrix A of dimension (ldda,n), ldda >= max(1,k).

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    dB      COMPLEX_16 array on GPU device.
            If trans == iclaNoTrans, the n-by-k matrix B of dimension (lddb,k), lddb >= max(1,n); \n
            otherwise,                the k-by-n matrix B of dimension (lddb,n), lddb >= max(1,k).

    @param[in]
    lddb    Leading dimension of dB.

    @param[in]
    beta    Scalar \f$ \beta \f$

    @param[in,out]
    dC      COMPLEX_16 array on GPU device.
            The n-by-n Hermitian matrix C of dimension (lddc,n), lddc >= max(1,n).

    @param[in]
    lddc    Leading dimension of dC.

    @param[in]
    queue   icla_queue_t
            Queue to execute in.

    @ingroup icla_her2k
*******************************************************************************/
extern "C" void
icla_zher2k(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dB, icla_int_t lddb,
    double beta,
    iclaDoubleComplex_ptr       dC, icla_int_t lddc,
    icla_queue_t queue )
{
    cublasZher2k(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        int(n), int(k),
        (cuDoubleComplex*)&alpha, (cuDoubleComplex*)dA, int(ldda),
                (cuDoubleComplex*)dB, int(lddb),
        &beta,  (cuDoubleComplex*)dC, int(lddc) );
}
#endif // COMPLEX


/***************************************************************************//**
    Perform symmetric matrix-matrix product.
        \f$ C = \alpha A B + \beta C \f$ (side == iclaLeft), or \n
        \f$ C = \alpha B A + \beta C \f$ (side == iclaRight),   \n
    where \f$ A \f$ is symmetric.

    @param[in]
    side    Whether A is on the left or right.

    @param[in]
    uplo    Whether the upper or lower triangle of A is referenced.

    @param[in]
    m       Number of rows of C. m >= 0.

    @param[in]
    n       Number of columns of C. n >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dA      COMPLEX_16 array on GPU device.
            If side == iclaLeft, the m-by-m symmetric matrix A of dimension (ldda,m), ldda >= max(1,m); \n
            otherwise,            the n-by-n symmetric matrix A of dimension (ldda,n), ldda >= max(1,n).

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    dB      COMPLEX_16 array on GPU device.
            The m-by-n matrix B of dimension (lddb,n), lddb >= max(1,m).

    @param[in]
    lddb    Leading dimension of dB.

    @param[in]
    beta    Scalar \f$ \beta \f$

    @param[in,out]
    dC      COMPLEX_16 array on GPU device.
            The m-by-n matrix C of dimension (lddc,n), lddc >= max(1,m).

    @param[in]
    lddc    Leading dimension of dC.

    @param[in]
    queue   icla_queue_t
            Queue to execute in.

    @ingroup icla_symm
*******************************************************************************/
extern "C" void
icla_zsymm(
    icla_side_t side, icla_uplo_t uplo,
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dB, icla_int_t lddb,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dC, icla_int_t lddc,
    icla_queue_t queue )
{
    cublasZsymm(
        queue->cublas_handle(),
        cublas_side_const( side ),
        cublas_uplo_const( uplo ),
        int(m), int(n),
        (cuDoubleComplex*)&alpha, (cuDoubleComplex*)dA, int(ldda),
                (cuDoubleComplex*)dB, int(lddb),
        (cuDoubleComplex*)&beta,  (cuDoubleComplex*)dC, int(lddc) );
}


/***************************************************************************//**
    Perform symmetric rank-k update.
        \f$ C = \alpha A A^T + \beta C \f$ (trans == iclaNoTrans), or \n
        \f$ C = \alpha A^T A + \beta C \f$ (trans == iclaTrans),      \n
    where \f$ C \f$ is symmetric.

    @param[in]
    uplo    Whether the upper or lower triangle of C is referenced.

    @param[in]
    trans   Operation to perform on A.

    @param[in]
    n       Number of rows and columns of C. n >= 0.

    @param[in]
    k       Number of columns of A (for iclaNoTrans)
            or rows of A (for iclaTrans). k >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dA      COMPLEX_16 array on GPU device.
            If trans == iclaNoTrans, the n-by-k matrix A of dimension (ldda,k), ldda >= max(1,n); \n
            otherwise,                the k-by-n matrix A of dimension (ldda,n), ldda >= max(1,k).

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    beta    Scalar \f$ \beta \f$

    @param[in,out]
    dC      COMPLEX_16 array on GPU device.
            The n-by-n symmetric matrix C of dimension (lddc,n), lddc >= max(1,n).

    @param[in]
    lddc    Leading dimension of dC.

    @param[in]
    queue   icla_queue_t
            Queue to execute in.

    @ingroup icla_syrk
*******************************************************************************/
extern "C" void
icla_zsyrk(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dC, icla_int_t lddc,
    icla_queue_t queue )
{
    cublasZsyrk(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        int(n), int(k),
        (cuDoubleComplex*)&alpha, (cuDoubleComplex*)dA, int(ldda),
        (cuDoubleComplex*)&beta,  (cuDoubleComplex*)dC, int(lddc) );
}


/***************************************************************************//**
    Perform symmetric rank-2k update.
        \f$ C = \alpha A B^T + \alpha B A^T \beta C \f$ (trans == iclaNoTrans), or \n
        \f$ C = \alpha A^T B + \alpha B^T A \beta C \f$ (trans == iclaTrans),      \n
    where \f$ C \f$ is symmetric.

    @param[in]
    uplo    Whether the upper or lower triangle of C is referenced.

    @param[in]
    trans   Operation to perform on A and B.

    @param[in]
    n       Number of rows and columns of C. n >= 0.

    @param[in]
    k       Number of columns of A and B (for iclaNoTrans)
            or rows of A and B (for iclaTrans). k >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dA      COMPLEX_16 array on GPU device.
            If trans == iclaNoTrans, the n-by-k matrix A of dimension (ldda,k), ldda >= max(1,n); \n
            otherwise,                the k-by-n matrix A of dimension (ldda,n), ldda >= max(1,k).

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    dB      COMPLEX_16 array on GPU device.
            If trans == iclaNoTrans, the n-by-k matrix B of dimension (lddb,k), lddb >= max(1,n); \n
            otherwise,                the k-by-n matrix B of dimension (lddb,n), lddb >= max(1,k).

    @param[in]
    lddb    Leading dimension of dB.

    @param[in]
    beta    Scalar \f$ \beta \f$

    @param[in,out]
    dC      COMPLEX_16 array on GPU device.
            The n-by-n symmetric matrix C of dimension (lddc,n), lddc >= max(1,n).

    @param[in]
    lddc    Leading dimension of dC.

    @param[in]
    queue   icla_queue_t
            Queue to execute in.

    @ingroup icla_syr2k
*******************************************************************************/
extern "C" void
icla_zsyr2k(
    icla_uplo_t uplo, icla_trans_t trans,
    icla_int_t n, icla_int_t k,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_const_ptr dB, icla_int_t lddb,
    iclaDoubleComplex beta,
    iclaDoubleComplex_ptr       dC, icla_int_t lddc,
    icla_queue_t queue )
{
    cublasZsyr2k(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        int(n), int(k),
        (cuDoubleComplex*)&alpha, (cuDoubleComplex*)dA, int(ldda),
                (cuDoubleComplex*)dB, int(lddb),
        (cuDoubleComplex*)&beta,  (cuDoubleComplex*)dC, int(lddc) );
}


/***************************************************************************//**
    Perform triangular matrix-matrix product.
        \f$ B = \alpha op(A) B \f$ (side == iclaLeft), or \n
        \f$ B = \alpha B op(A) \f$ (side == iclaRight),   \n
    where \f$ A \f$ is triangular.

    @param[in]
    side    Whether A is on the left or right.

    @param[in]
    uplo    Whether A is upper or lower triangular.

    @param[in]
    trans   Operation to perform on A.

    @param[in]
    diag    Whether the diagonal of A is assumed to be unit or non-unit.

    @param[in]
    m       Number of rows of B. m >= 0.

    @param[in]
    n       Number of columns of B. n >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dA      COMPLEX_16 array on GPU device.
            If side == iclaLeft, the n-by-n triangular matrix A of dimension (ldda,n), ldda >= max(1,n); \n
            otherwise,            the m-by-m triangular matrix A of dimension (ldda,m), ldda >= max(1,m).

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    dB      COMPLEX_16 array on GPU device.
            The m-by-n matrix B of dimension (lddb,n), lddb >= max(1,m).

    @param[in]
    lddb    Leading dimension of dB.

    @param[in]
    queue   icla_queue_t
            Queue to execute in.

    @ingroup icla_trmm
*******************************************************************************/
extern "C" void
icla_ztrmm(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr       dB, icla_int_t lddb,
    icla_queue_t queue )
{
    #ifdef ICLA_HAVE_HIP
        // TODO: remove fallback when hipblas provides this routine
        cublasZtrmm(
		    queue->cublas_handle(),
		    cublas_side_const( side ),
                    cublas_uplo_const( uplo ),
                    cublas_trans_const( trans ),
                    cublas_diag_const( diag ),
		    int(m), int(n),
		    (cuDoubleComplex*)&alpha, (const cuDoubleComplex*)dA, int(ldda),
		    (cuDoubleComplex*)dB, int(lddb)
    #if (ROCM_VERSION >= 60000)
		    , (cuDoubleComplex*)dB, int(lddb)
    #endif
		    );
    #else
        cublasZtrmm(
                    queue->cublas_handle(),
                    cublas_side_const( side ),
                    cublas_uplo_const( uplo ),
                    cublas_trans_const( trans ),
                    cublas_diag_const( diag ),
                    int(m), int(n),
                    &alpha, dA, int(ldda),
                    dB, int(lddb),
                    dB, int(lddb) );  /* C same as B; less efficient */
    #endif
}


/***************************************************************************//**
    Solve triangular matrix-matrix system (multiple right-hand sides).
        \f$ op(A) X = \alpha B \f$ (side == iclaLeft), or \n
        \f$ X op(A) = \alpha B \f$ (side == iclaRight),   \n
    where \f$ A \f$ is triangular.

    @param[in]
    side    Whether A is on the left or right.

    @param[in]
    uplo    Whether A is upper or lower triangular.

    @param[in]
    trans   Operation to perform on A.

    @param[in]
    diag    Whether the diagonal of A is assumed to be unit or non-unit.

    @param[in]
    m       Number of rows of B. m >= 0.

    @param[in]
    n       Number of columns of B. n >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dA      COMPLEX_16 array on GPU device.
            If side == iclaLeft, the m-by-m triangular matrix A of dimension (ldda,m), ldda >= max(1,m); \n
            otherwise,            the n-by-n triangular matrix A of dimension (ldda,n), ldda >= max(1,n).

    @param[in]
    ldda    Leading dimension of dA.

    @param[in,out]
    dB      COMPLEX_16 array on GPU device.
            On entry, m-by-n matrix B of dimension (lddb,n), lddb >= max(1,m).
            On exit, overwritten with the solution matrix X.

    @param[in]
    lddb    Leading dimension of dB.

    @param[in]
    queue   icla_queue_t
            Queue to execute in.

    @ingroup icla_trsm
*******************************************************************************/
extern "C" void
icla_ztrsm(
    icla_side_t side, icla_uplo_t uplo, icla_trans_t trans, icla_diag_t diag,
    icla_int_t m, icla_int_t n,
    iclaDoubleComplex alpha,
    iclaDoubleComplex_const_ptr dA, icla_int_t ldda,
    iclaDoubleComplex_ptr       dB, icla_int_t lddb,
    icla_queue_t queue )
{
    cublasZtrsm(
        queue->cublas_handle(),
        cublas_side_const( side ),
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        cublas_diag_const( diag ),
        int(m), int(n),
        (cuDoubleComplex*)&alpha, (cuDoubleComplex*)dA, int(ldda),
                (cuDoubleComplex*)dB, int(lddb) );
}

//#endif // ICLA_HAVE_CUDA

#undef COMPLEX
