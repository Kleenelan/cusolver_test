
#ifndef ICLA_MATRIX_HPP
#define ICLA_MATRIX_HPP

#include <algorithm>  // copy, swap

#include <icla_types.h>

/******************************************************************************/
// Uses copy-and-swap idiom.
// https://stackoverflow.com/questions/3279543/what-is-the-copy-and-swap-idiom

template< typename FloatType >
class Vector
{
public:
    // constructor allocates new memory (unless n == 0)
    Vector( icla_int_t in_n=0 ):
        n    ( in_n ),
        data_( n > 0 ? new FloatType[n] : nullptr ),
        own_ ( true )
    {
        if (n < 0) { throw std::exception(); }
    }

    // constructor wraps existing memory; caller maintains ownership
    Vector( FloatType* data, icla_int_t in_n ):
        n    ( in_n ),
        data_( data ),
        own_ ( false )
    {
        if (n < 0) { throw std::exception(); }
    }

    // copy constructor
    Vector( Vector const &other ):
        n    ( other.n ),
        data_( nullptr ),
        own_ ( other.own_ )
    {
        if (other.own_) {
            if (n > 0) {
                data_ = new FloatType[n];
                std::copy( other.data_, other.data_ + n, data_ );
            }
        }
        else {
            data_ = other.data_;
        }
    }

    // move constructor, using copy & swap idiom
    Vector( Vector&& other )
        : Vector()
    {
        swap( *this, other );
    }

    // assignment operator, using copy & swap idiom
    Vector& operator= (Vector other)
    {
        swap( *this, other );
        return *this;
    }

    // destructor deletes memory if constructor allocated it
    // (i.e., not if wrapping existing memory)
    ~Vector()
    {
        if (own_) {
            delete[] data_;
            data_ = nullptr;
        }
    }

    friend void swap( Vector& first, Vector& second )
    {
        using std::swap;
        swap( first.n,     second.n     );
        swap( first.data_, second.data_ );
        swap( first.own_,  second.own_  );
    }

    // returns pointer to element i, because that's what we normally need to
    // call BLAS / LAPACK, which avoids littering the code with &.
    FloatType*       operator () ( icla_int_t i )       { return &data_[ i ]; }
    FloatType const* operator () ( icla_int_t i ) const { return &data_[ i ]; }

    // return element i itself, as usual in C/C++.
    // unfortunately, this won't work for matrices.
    FloatType&       operator [] ( icla_int_t i )       { return data_[ i ]; }
    FloatType const& operator [] ( icla_int_t i ) const { return data_[ i ]; }

    icla_int_t size() const { return n; }
    bool        own()  const { return own_; }

public:
    icla_int_t n;

private:
    FloatType *data_;
    bool own_;
};

/******************************************************************************/
template< typename FloatType >
class Matrix
{
public:
    // constructor allocates new memory
    // ld = m by default
    Matrix( icla_int_t in_m, icla_int_t in_n, icla_int_t in_ld=0 ):
        m( in_m ),
        n( in_n ),
        ld( in_ld == 0 ? m : in_ld ),
        data_( ld*n )
    {
        if (m  < 0) { throw std::exception(); }
        if (n  < 0) { throw std::exception(); }
        if (ld < m) { throw std::exception(); }
    }

    // constructor wraps existing memory; caller maintains ownership
    // ld = m by default
    Matrix( FloatType* data, icla_int_t in_m, icla_int_t in_n, icla_int_t in_ld=0 ):
        m( in_m ),
        n( in_n ),
        ld( in_ld == 0 ? m : in_ld ),
        data_( data, ld*n )
    {
        if (m  < 0) { throw std::exception(); }
        if (n  < 0) { throw std::exception(); }
        if (ld < m) { throw std::exception(); }
    }

    icla_int_t size() const { return data_.size(); }
    bool        own()  const { return data_.own(); }

    // returns pointer to element (i,j), because that's what we normally need to
    // call BLAS / LAPACK, which avoids littering the code with &.
    FloatType* operator () ( int i, int j )
        { return &data_[ i + j*ld ]; }

    FloatType const* operator () ( int i, int j ) const
        { return &data_[ i + j*ld ]; }

public:
    icla_int_t m, n, ld;

protected:
    Vector<FloatType> data_;
};

#endif        //  #ifndef ICLA_MATRIX_HPP
