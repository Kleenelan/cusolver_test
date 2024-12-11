
#ifndef ICLA_MATRIX_HPP
#define ICLA_MATRIX_HPP

#include <algorithm>

#include <icla_types.h>

template< typename FloatType >
class Vector
{
public:

    Vector( icla_int_t in_n=0 ):
        n    ( in_n ),
        data_( n > 0 ? new FloatType[n] : nullptr ),
        own_ ( true )
    {
        if (n < 0) { throw std::exception(); }
    }

    Vector( FloatType* data, icla_int_t in_n ):
        n    ( in_n ),
        data_( data ),
        own_ ( false )
    {
        if (n < 0) { throw std::exception(); }
    }

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

    Vector( Vector&& other )
        : Vector()
    {
        swap( *this, other );
    }

    Vector& operator= (Vector other)
    {
        swap( *this, other );
        return *this;
    }

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

    FloatType*       operator () ( icla_int_t i )       { return &data_[ i ]; }
    FloatType const* operator () ( icla_int_t i ) const { return &data_[ i ]; }

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

template< typename FloatType >
class Matrix
{
public:

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

    FloatType* operator () ( int i, int j )
        { return &data_[ i + j*ld ]; }

    FloatType const* operator () ( int i, int j ) const
        { return &data_[ i + j*ld ]; }

public:
    icla_int_t m, n, ld;

protected:
    Vector<FloatType> data_;
};

#endif

