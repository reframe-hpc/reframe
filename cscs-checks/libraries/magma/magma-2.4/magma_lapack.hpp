// Experimental !!
// Start of blas and lapack C++ bindings.
// Eventually will use blaspp and lapackpp from SLATE.

#ifndef LAPACK_HPP
#define LAPACK_HPP

#include <complex>

#include "magma_v2.h"
#include "magma_lapack.h"

// =============================================================================
namespace blas
{

// -----------------------------------------------------------------------------
// traits: given a type, defines its associated real type

// ----------------------------------------
template< typename T >
class traits
{
public:
    typedef T real_t;
    static T make( real_t r, real_t i )
        { return r; }
};

// ----------------------------------------
template< typename T >
class traits< std::complex<T> >
{
public:
    typedef T real_t;
    static std::complex<T> make( T r, T i )
        { return std::complex<T>( r, i ); }
};

// ----------------------------------------
template<>
class traits< magmaFloatComplex >
{
public:
    typedef float real_t;
    static magmaFloatComplex make( real_t r, real_t i )
        { return MAGMA_C_MAKE( r, i ); }
};

// ----------------------------------------
template<>
class traits< magmaDoubleComplex >
{
public:
    typedef double real_t;
    static magmaDoubleComplex make( real_t r, real_t i )
        { return MAGMA_Z_MAKE( r, i ); }
};


// -----------------------------------------------------------------------------
// traits2: given 2 types, defines their scalar and associated real types.
// Default is type T1, then overrides are given for cases where it should be T2
// or something different.

// ----------------------------------------
template< typename T1, typename T2 >
class traits2
{
public:
    typedef T1 scalar_t;
    typedef T1 real_t;
};

// ----------------------------------------
// float
template<>
class traits2< float, double >
{
public:
    typedef double scalar_t;
    typedef double real_t;
};

// ---------------
template<>
class traits2< float, std::complex<float> >
{
public:
    typedef std::complex<float> scalar_t;
    typedef float real_t;
};

template<>
class traits2< float, magmaFloatComplex >
{
public:
    typedef magmaFloatComplex scalar_t;
    typedef float real_t;
};

// ---------------
template<>
class traits2< float, std::complex<double> >
{
public:
    typedef std::complex<double> scalar_t;
    typedef double real_t;
};

template<>
class traits2< float, magmaDoubleComplex >
{
public:
    typedef magmaDoubleComplex scalar_t;
    typedef double real_t;
};

// ----------------------------------------
// double
template<>
class traits2< double, std::complex<float> >
{
public:
    typedef std::complex<double> scalar_t;
    typedef double real_t;
};

template<>
class traits2< double, magmaFloatComplex >
{
public:
    typedef magmaDoubleComplex scalar_t;
    typedef double real_t;
};

// ---------------
template<>
class traits2< double, std::complex<double> >
{
public:
    typedef std::complex<double> scalar_t;
    typedef double real_t;
};

template<>
class traits2< double, magmaDoubleComplex >
{
public:
    typedef magmaDoubleComplex scalar_t;
    typedef double real_t;
};

// ----------------------------------------
// complex<float>
template<>
class traits2< std::complex<float>, std::complex<double> >
{
public:
    typedef std::complex<double> scalar_t;
    typedef double real_t;
};

template<>
class traits2< magmaFloatComplex, magmaDoubleComplex >
{
public:
    typedef magmaDoubleComplex scalar_t;
    typedef double real_t;
};


// -----------------------------------------------------------------------------
// traits2: given 3 types, defines their scalar and associated real types.

// ----------------------------------------
template< typename T1, typename T2, typename T3 >
class traits3
{
public:
    typedef typename
        traits2< typename traits2<T1,T2>::scalar_t, T3 >::scalar_t scalar_t;

    typedef typename
        traits2< typename traits2<T1,T2>::scalar_t, T3 >::real_t real_t;
};


// -----------------------------------------------------------------------------
inline float  asum( magma_int_t n, float* x, magma_int_t incx )
{
    return magma_cblas_sasum( n, x, incx );
}

inline double asum(
    magma_int_t n, double* x, magma_int_t incx )
{
    return magma_cblas_dasum( n, x, incx );
}

inline float  asum( magma_int_t n, magmaFloatComplex* x, magma_int_t incx )
{
    return magma_cblas_scasum( n, x, incx );
}

inline double asum(
    magma_int_t n, magmaDoubleComplex* x, magma_int_t incx )
{
    return magma_cblas_dzasum( n, x, incx );
}


// -----------------------------------------------------------------------------
inline float  dot(
    magma_int_t n,
    float* x, magma_int_t incx,
    float* y, magma_int_t incy )
{
    return magma_cblas_sdot( n, x, incx, y, incy );
}

inline double dot(
    magma_int_t n,
    double* x, magma_int_t incx,
    double* y, magma_int_t incy )
{
    return magma_cblas_ddot( n, x, incx, y, incy );
}

inline magmaFloatComplex  dot(
    magma_int_t n,
    magmaFloatComplex* x, magma_int_t incx,
    magmaFloatComplex* y, magma_int_t incy )
{
    return magma_cblas_cdotc( n, x, incx, y, incy );
}

inline magmaDoubleComplex dot(
    magma_int_t n,
    magmaDoubleComplex* x, magma_int_t incx,
    magmaDoubleComplex* y, magma_int_t incy )
{
    return magma_cblas_zdotc( n, x, incx, y, incy );
}


// -----------------------------------------------------------------------------
inline void copy(
    magma_int_t n,
    float* x, magma_int_t incx,
    float* y, magma_int_t incy )
{
    return blasf77_scopy( &n, x, &incx, y, &incy );
}

inline void copy(
    magma_int_t n,
    double* x, magma_int_t incx,
    double* y, magma_int_t incy )
{
    return blasf77_dcopy( &n, x, &incx, y, &incy );
}

inline void copy(
    magma_int_t n,
    magmaFloatComplex* x, magma_int_t incx,
    magmaFloatComplex* y, magma_int_t incy )
{
    return blasf77_ccopy( &n, x, &incx, y, &incy );
}

inline void copy(
    magma_int_t n,
    magmaDoubleComplex* x, magma_int_t incx,
    magmaDoubleComplex* y, magma_int_t incy )
{
    return blasf77_zcopy( &n, x, &incx, y, &incy );
}


// -----------------------------------------------------------------------------
inline void rot(
    magma_int_t n,
    float* x, magma_int_t incx,
    float* y, magma_int_t incy,
    float c, float s )
{
    blasf77_srot( &n, x, &incx, y, &incy, &c, &s );
}

inline void rot(
    magma_int_t n,
    double* x, magma_int_t incx,
    double* y, magma_int_t incy,
    double c, double s )
{
    blasf77_drot( &n, x, &incx, y, &incy, &c, &s );
}

inline void rot(
    magma_int_t n,
    magmaFloatComplex* x, magma_int_t incx,
    magmaFloatComplex* y, magma_int_t incy,
    float c, float s )
{
    blasf77_csrot( &n, x, &incx, y, &incy, &c, &s );
}

inline void rot(
    magma_int_t n,
    magmaDoubleComplex* x, magma_int_t incx,
    magmaDoubleComplex* y, magma_int_t incy,
    double c, double s )
{
    blasf77_zdrot( &n, x, &incx, y, &incy, &c, &s );
}


// -----------------------------------------------------------------------------
inline void scal(
    magma_int_t n, float alpha,
    float *x, magma_int_t incx )
{
    blasf77_sscal( &n, &alpha, x, &incx );
}

inline void scal(
    magma_int_t n, double alpha,
    double *x, magma_int_t incx )
{
    blasf77_dscal( &n, &alpha, x, &incx );
}

inline void scal(
    magma_int_t n, magmaFloatComplex alpha,
    magmaFloatComplex *x, magma_int_t incx )
{
    blasf77_cscal( &n, &alpha, x, &incx );
}

inline void scal(
    magma_int_t n, magmaDoubleComplex alpha,
    magmaDoubleComplex *x, magma_int_t incx )
{
    blasf77_zscal( &n, &alpha, x, &incx );
}

}  // end namespace blas


// =============================================================================
namespace lapack
{

// -----------------------------------------------------------------------------
inline void larnv(
    magma_int_t idist, magma_int_t iseed[4],
    magma_int_t n, float *x )
{
    lapackf77_slarnv( &idist, iseed, &n, x );
}

inline void larnv(
    magma_int_t idist, magma_int_t iseed[4],
    magma_int_t n, double *x )
{
    lapackf77_dlarnv( &idist, iseed, &n, x );
}

inline void larnv(
    magma_int_t idist, magma_int_t iseed[4],
    magma_int_t n, magmaFloatComplex *x )
{
    lapackf77_clarnv( &idist, iseed, &n, x );
}

inline void larnv(
    magma_int_t idist, magma_int_t iseed[4],
    magma_int_t n, magmaDoubleComplex *x )
{
    lapackf77_zlarnv( &idist, iseed, &n, x );
}


// -----------------------------------------------------------------------------
inline void larfg(
    magma_int_t n,
    float* alpha,
    float* x, magma_int_t incx,
    float* tau )
{
    lapackf77_slarfg( &n, alpha, x, &incx, tau );
}

inline void larfg(
    magma_int_t n,
    double* alpha,
    double* x, magma_int_t incx,
    double* tau )
{
    lapackf77_dlarfg( &n, alpha, x, &incx, tau );
}

inline void larfg(
    magma_int_t n,
    magmaFloatComplex* alpha,
    magmaFloatComplex* x, magma_int_t incx,
    magmaFloatComplex* tau )
{
    lapackf77_clarfg( &n, alpha, x, &incx, tau );
}

inline void larfg(
    magma_int_t n,
    magmaDoubleComplex* alpha,
    magmaDoubleComplex* x, magma_int_t incx,
    magmaDoubleComplex* tau )
{
    lapackf77_zlarfg( &n, alpha, x, &incx, tau );
}


// -----------------------------------------------------------------------------
inline void laset(
    const char* uplo, magma_int_t m, magma_int_t n,
    float diag, float offdiag,
    float* A, magma_int_t lda )
{
    lapackf77_slaset( uplo, &m, &n, &diag, &offdiag, A, &lda );
}

inline void laset(
    const char* uplo, magma_int_t m, magma_int_t n,
    double diag, double offdiag,
    double* A, magma_int_t lda )
{
    lapackf77_dlaset( uplo, &m, &n, &diag, &offdiag, A, &lda );
}

inline void laset(
    const char* uplo, magma_int_t m, magma_int_t n,
    magmaFloatComplex diag, magmaFloatComplex offdiag,
    magmaFloatComplex* A, magma_int_t lda )
{
    lapackf77_claset( uplo, &m, &n, &diag, &offdiag, A, &lda );
}

inline void laset(
    const char* uplo, magma_int_t m, magma_int_t n,
    magmaDoubleComplex diag, magmaDoubleComplex offdiag,
    magmaDoubleComplex* A, magma_int_t lda )
{
    lapackf77_zlaset( uplo, &m, &n, &diag, &offdiag, A, &lda );
}


// -----------------------------------------------------------------------------
inline void unmqr(
    const char* side, const char* trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    float* A,    magma_int_t lda,
    float* tau,
    float* C,    magma_int_t ldc,
    float* work, magma_int_t lwork,
    magma_int_t* info )
{
    if (*trans == 'c' || *trans == 'C') {
        trans = "T";
    }
    lapackf77_sormqr( side, trans, &m, &n, &k,
                      A, &lda, tau, C, &ldc, work, &lwork, info );
}

inline void unmqr(
    const char* side, const char* trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    double* A,    magma_int_t lda,
    double* tau,
    double* C,    magma_int_t ldc,
    double* work, magma_int_t lwork,
    magma_int_t* info )
{
    if (*trans == 'c' || *trans == 'C') {
        trans = "T";
    }
    lapackf77_dormqr( side, trans, &m, &n, &k,
                      A, &lda, tau, C, &ldc, work, &lwork, info );
}

inline void unmqr(
    const char* side, const char* trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex* A,    magma_int_t lda,
    magmaFloatComplex* tau,
    magmaFloatComplex* C,    magma_int_t ldc,
    magmaFloatComplex* work, magma_int_t lwork,
    magma_int_t* info )
{
    lapackf77_cunmqr( side, trans, &m, &n, &k,
                      A, &lda, tau, C, &ldc, work, &lwork, info );
}

inline void unmqr(
    const char* side, const char* trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex* A,    magma_int_t lda,
    magmaDoubleComplex* tau,
    magmaDoubleComplex* C,    magma_int_t ldc,
    magmaDoubleComplex* work, magma_int_t lwork,
    magma_int_t* info )
{
    lapackf77_zunmqr( side, trans, &m, &n, &k,
                      A, &lda, tau, C, &ldc, work, &lwork, info );
}

}  // end namespace lapack

#endif        //  #ifndef LAPACK_HPP
