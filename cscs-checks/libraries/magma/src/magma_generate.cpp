/*
    -- MAGMA (version 2.4.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date June 2018

       @author Mark Gates

       Test matrix generation.
*/

#include <exception>
#include <string>
#include <vector>
#include <limits>

#include "magma_v2.h"
#include "magma_lapack.hpp"  // experimental C++ bindings
#include "magma_operators.h"

#include "magma_matrix.hpp"

// last (defines macros that conflict with std headers)
#include "testings.h"
#undef max
#undef min
using std::max;
using std::min;

/******************************************************************************/
// constants

const magma_int_t idist_rand  = 1;
const magma_int_t idist_randu = 2;
const magma_int_t idist_randn = 3;

enum class MatrixType {
    rand      = 1,  // maps to larnv idist
    randu     = 2,  // maps to larnv idist
    randn     = 3,  // maps to larnv idist
    zero,
    identity,
    jordan,
    diag,
    svd,
    poev,
    heev,
    geev,
    geevx,
};

enum class Dist {
    rand      = 1,  // maps to larnv idist
    randu     = 2,  // maps to larnv idist
    randn     = 3,  // maps to larnv idist
    arith,
    geo,
    cluster,
    cluster2,
    rarith,
    rgeo,
    rcluster,
    rcluster2,
    logrand,
    specified,
};

/******************************************************************************/
// ANSI color codes
const char *ansi_esc    = "\x1b[";
const char *ansi_red    = "\x1b[31m";
const char *ansi_bold   = "\x1b[1m";
const char *ansi_normal = "\x1b[0m";

/******************************************************************************/
// random number in (0, max_]
template< typename FloatT >
inline FloatT rand( FloatT max_ )
{
    return max_ * rand() / FloatT(RAND_MAX);
}

/******************************************************************************/
// true if str begins with prefix
inline bool begins( std::string const &str, std::string const &prefix )
{
    return (str.compare( 0, prefix.size(), prefix) == 0);
}

/******************************************************************************/
// true if str contains pattern
inline bool contains( std::string const &str, std::string const &pattern )
{
    return (str.find( pattern ) != std::string::npos);
}


/******************************************************************************/
template< typename FloatT >
void magma_generate_sigma(
    magma_opts& opts,
    Dist dist, bool rand_sign,
    typename blas::traits<FloatT>::real_t cond,
    typename blas::traits<FloatT>::real_t sigma_max,
    Matrix<FloatT>& A,
    Vector< typename blas::traits<FloatT>::real_t >& sigma )
{
    typedef typename blas::traits<FloatT>::real_t real_t;

    // constants
    const FloatT c_zero = blas::traits<FloatT>::make( 0, 0 );

    // locals
    magma_int_t minmn = min( A.m, A.n );
    assert( minmn == sigma.n );

    switch (dist) {
        case Dist::arith:
            for (magma_int_t i = 0; i < minmn; ++i) {
                sigma[i] = 1 - i / real_t(minmn - 1) * (1 - 1/cond);
            }
            break;

        case Dist::rarith:
            for (magma_int_t i = 0; i < minmn; ++i) {
                sigma[i] = 1 - (minmn - 1 - i) / real_t(minmn - 1) * (1 - 1/cond);
            }
            break;

        case Dist::geo:
            for (magma_int_t i = 0; i < minmn; ++i) {
                sigma[i] = pow( cond, -i / real_t(minmn - 1) );
            }
            break;

        case Dist::rgeo:
            for (magma_int_t i = 0; i < minmn; ++i) {
                sigma[i] = pow( cond, -(minmn - 1 - i) / real_t(minmn - 1) );
            }
            break;

        case Dist::cluster:
            sigma[0] = 1;
            for (magma_int_t i = 1; i < minmn; ++i) {
                sigma[i] = 1/cond;
            }
            break;

        case Dist::rcluster:
            for (magma_int_t i = 0; i < minmn-1; ++i) {
                sigma[i] = 1/cond;
            }
            sigma[minmn-1] = 1;
            break;

        case Dist::cluster2:
            for (magma_int_t i = 0; i < minmn-1; ++i) {
                sigma[i] = 1;
            }
            sigma[minmn-1] = 1/cond;
            break;

        case Dist::rcluster2:
            sigma[0] = 1/cond;
            for (magma_int_t i = 1; i < minmn; ++i) {
                sigma[i] = 1;
            }
            break;

        case Dist::logrand: {
            real_t range = log( 1/cond );
            lapack::larnv( idist_rand, opts.iseed, sigma.n, sigma(0) );
            for (magma_int_t i = 0; i < minmn; ++i) {
                sigma[i] = exp( sigma[i] * range );
            }
            // make cond exact
            if (minmn >= 2) {
                sigma[0] = 1;
                sigma[1] = 1/cond;
            }
            break;
        }

        case Dist::randn:
        case Dist::randu:
        case Dist::rand: {
            magma_int_t idist = (magma_int_t) dist;
            lapack::larnv( idist, opts.iseed, sigma.n, sigma(0) );
            break;
        }

        case Dist::specified:
            // user-specified sigma values; don't modify
            sigma_max = 1;
            rand_sign = false;
            break;
    }

    if (sigma_max != 1) {
        blas::scal( sigma.n, sigma_max, sigma(0), 1 );
    }

    if (rand_sign) {
        // apply random signs
        for (magma_int_t i = 0; i < minmn; ++i) {
            if (rand() > RAND_MAX/2) {
                sigma[i] = -sigma[i];
            }
        }
    }

    // copy sigma => A
    lapack::laset( "general", A.m, A.n, c_zero, c_zero, A(0,0), A.ld );
    for (magma_int_t i = 0; i < minmn; ++i) {
        *A(i,i) = blas::traits<FloatT>::make( sigma[i], 0 );
    }
}


/***************************************************************************//**
    Given A with singular values such that sum(sigma_i^2) = n,
    returns A with columns of unit norm, with the same condition number.
    see: Davies and Higham, 2000, Numerically stable generation of correlation
    matrices and their factors.
*******************************************************************************/
template< typename FloatT >
void magma_generate_correlation_factor( Matrix<FloatT>& A )
{
    //const FloatT eps = std::numeric_limits<FloatT>::epsilon();

    Vector<FloatT> x( A.n );
    for (magma_int_t j = 0; j < A.n; ++j) {
        x[j] = blas::dot( A.m, A(0,j), 1, A(0,j), 1 );
    }

    for (magma_int_t i = 0; i < A.n; ++i) {
        for (magma_int_t j = 0; j < A.n; ++j) {
            if ((x[i] < 1 && 1 < x[j]) || (x[i] > 1 && 1 > x[j])) {
                FloatT xij, d, t, c, s;
                xij = blas::dot( A.m, A(0,i), 1, A(0,j), 1 );
                d = sqrt( xij*xij - (x[i] - 1)*(x[j] - 1) );
                t = (xij + std::copysign( d, xij )) / (x[j] - 1);
                c = 1 / sqrt(1 + t*t);
                s = c*t;
                blas::rot( A.m, A(0,i), 1, A(0,j), 1, c, -s );
                x[i] = blas::dot( A.m, A(0,i), 1, A(0,i), 1 );
                //if (x[i] - 1 > 30*eps) {
                //    printf( "i %d, x[i] %.6f, x[i] - 1 %.6e, 30*eps %.6e\n",
                //            i, x[i], x[i] - 1, 30*eps );
                //}
                //assert( x[i] - 1 < 30*eps );
                x[i] = 1;
                x[j] = blas::dot( A.m, A(0,j), 1, A(0,j), 1 );
                break;
            }
        }
    }
}


/******************************************************************************/
// specialization to complex
// can't use Higham's algorithm in complex
template<>
void magma_generate_correlation_factor( Matrix<magmaFloatComplex>& A )
{
    assert( false );
}

template<>
void magma_generate_correlation_factor( Matrix<magmaDoubleComplex>& A )
{
    assert( false );
}


/******************************************************************************/
template< typename FloatT >
void magma_generate_svd(
    magma_opts& opts,
    Dist dist,
    typename blas::traits<FloatT>::real_t cond,
    typename blas::traits<FloatT>::real_t condD,
    typename blas::traits<FloatT>::real_t sigma_max,
    Matrix<FloatT>& A,
    Vector< typename blas::traits<FloatT>::real_t >& sigma )
{
    typedef typename blas::traits<FloatT>::real_t real_t;

    // locals
    FloatT tmp;
    magma_int_t m = A.m;
    magma_int_t n = A.n;
    magma_int_t maxmn = max( m, n );
    magma_int_t minmn = min( m, n );
    magma_int_t sizeU;
    magma_int_t info = 0;
    Matrix<FloatT> U( maxmn, minmn );
    Vector<FloatT> tau( minmn );

    // query for workspace
    magma_int_t lwork = -1;
    lapack::unmqr( "Left", "NoTrans", A.m, A.n, minmn,
                   U(0,0), U.ld, tau(0), A(0,0), A.ld,
                   &tmp, lwork, &info );
    assert( info == 0 );
    lwork = magma_int_t( real( tmp ));
    magma_int_t lwork2 = -1;
    lapack::unmqr( "Right", "ConjTrans", A.m, A.n, minmn,
                   U(0,0), U.ld, tau(0), A(0,0), A.ld,
                   &tmp, lwork2, &info );
    assert( info == 0 );
    lwork2 = magma_int_t( real( tmp ));
    lwork = max( lwork, lwork2 );
    Vector<FloatT> work( lwork );

    // ----------
    magma_generate_sigma( opts, dist, false, cond, sigma_max, A, sigma );

    // for generate correlation factor, need sum sigma_i^2 = n
    // scaling doesn't change cond
    if (condD != 1) {
        real_t sum_sq = blas::dot( sigma.n, sigma(0), 1, sigma(0), 1 );
        real_t scale = sqrt( sigma.n / sum_sq );
        blas::scal( sigma.n, scale, sigma(0), 1 );
        // copy sigma to diag(A)
        for (magma_int_t i = 0; i < sigma.n; ++i) {
            *A(i,i) = blas::traits<FloatT>::make( *sigma(i), 0 );
        }
    }

    // random U, m-by-minmn
    // just make each random column into a Householder vector;
    // no need to update subsequent columns (as in geqrf).
    sizeU = U.size();
    lapack::larnv( idist_randn, opts.iseed, sizeU, U(0,0) );
    for (magma_int_t j = 0; j < minmn; ++j) {
        magma_int_t mj = m - j;
        lapack::larfg( mj, U(j,j), U(j+1,j), 1, tau(j) );
    }

    // A = U*A
    lapack::unmqr( "Left", "NoTrans", A.m, A.n, minmn,
                   U(0,0), U.ld, tau(0), A(0,0), A.ld,
                   work(0), lwork, &info );
    assert( info == 0 );

    // random V, n-by-minmn (stored column-wise in U)
    lapack::larnv( idist_randn, opts.iseed, sizeU, U(0,0) );
    for (magma_int_t j = 0; j < minmn; ++j) {
        magma_int_t nj = n - j;
        lapack::larfg( nj, U(j,j), U(j+1,j), 1, tau(j) );
    }

    // A = A*V^H
    lapack::unmqr( "Right", "ConjTrans", A.m, A.n, minmn,
                   U(0,0), U.ld, tau(0), A(0,0), A.ld,
                   work(0), lwork, &info );
    assert( info == 0 );

    if (condD != 1) {
        // A = A*W, W orthogonal, such that A has unit column norms
        // i.e., A'*A is a correlation matrix with unit diagonal
        magma_generate_correlation_factor( A );

        // A = A*D col scaling
        Vector<real_t> D( A.n );
        real_t range = log( condD );
        lapack::larnv( idist_rand, opts.iseed, D.n, D(0) );
        for (magma_int_t i = 0; i < D.n; ++i) {
            D[i] = exp( D[i] * range );
        }
        // TODO: add argument to return D to caller?
        if (opts.verbose) {
            printf( "D = [" );
            for (magma_int_t i = 0; i < D.n; ++i) {
                printf( " %11.8g", D[i] );
            }
            printf( " ];\n" );
        }
        for (magma_int_t j = 0; j < A.n; ++j) {
            for (magma_int_t i = 0; i < A.m; ++i) {
                *A(i,j) *= D[j];
            }
        }
    }
}

/******************************************************************************/
template< typename FloatT >
void magma_generate_heev(
    magma_opts& opts,
    Dist dist, bool rand_sign,
    typename blas::traits<FloatT>::real_t cond,
    typename blas::traits<FloatT>::real_t condD,
    typename blas::traits<FloatT>::real_t sigma_max,
    Matrix<FloatT>& A,
    Vector< typename blas::traits<FloatT>::real_t >& sigma )
{
    typedef typename blas::traits<FloatT>::real_t real_t;

    // check inputs
    assert( A.m == A.n );

    // locals
    FloatT tmp;
    magma_int_t n = A.n;
    magma_int_t sizeU;
    magma_int_t info = 0;
    Matrix<FloatT> U( n, n );
    Vector<FloatT> tau( n );

    // query for workspace
    magma_int_t lwork = -1;
    lapack::unmqr( "Left", "NoTrans", n, n, n,
                   U(0,0), U.ld, tau(0), A(0,0), A.ld,
                   &tmp, lwork, &info );
    assert( info == 0 );
    lwork = magma_int_t( real( tmp ));
    magma_int_t lwork2 = -1;
    lapack::unmqr( "Right", "ConjTrans", n, n, n,
                   U(0,0), U.ld, tau(0), A(0,0), A.ld,
                   &tmp, lwork2, &info );
    assert( info == 0 );
    lwork2 = magma_int_t( real( tmp ));
    lwork = max( lwork, lwork2 );
    Vector<FloatT> work( lwork );

    // ----------
    magma_generate_sigma( opts, dist, rand_sign, cond, sigma_max, A, sigma );

    // random U, n-by-n
    // just make each random column into a Householder vector;
    // no need to update subsequent columns (as in geqrf).
    sizeU = U.size();
    lapack::larnv( idist_randn, opts.iseed, sizeU, U(0,0) );
    for (magma_int_t j = 0; j < n; ++j) {
        magma_int_t nj = n - j;
        lapack::larfg( nj, U(j,j), U(j+1,j), 1, tau(j) );
    }

    // A = U*A
    lapack::unmqr( "Left", "NoTrans", n, n, n,
                   U(0,0), U.ld, tau(0), A(0,0), A.ld,
                   work(0), lwork, &info );
    assert( info == 0 );

    // A = A*U^H
    lapack::unmqr( "Right", "ConjTrans", n, n, n,
                   U(0,0), U.ld, tau(0), A(0,0), A.ld,
                   work(0), lwork, &info );
    assert( info == 0 );

    // make diagonal real
    // usually LAPACK ignores imaginary part anyway, but Matlab doesn't
    for (int i = 0; i < n; ++i) {
        *A(i,i) = blas::traits<FloatT>::make( real( *A(i,i) ), 0 );
    }

    if (condD != 1) {
        // A = D*A*D row & column scaling
        Vector<real_t> D( n );
        real_t range = log( condD );
        lapack::larnv( idist_rand, opts.iseed, n, D(0) );
        for (magma_int_t i = 0; i < n; ++i) {
            D[i] = exp( D[i] * range );
        }
        for (magma_int_t j = 0; j < n; ++j) {
            for (magma_int_t i = 0; i < n; ++i) {
                *A(i,j) *= D[i] * D[j];
            }
        }
    }
}

/******************************************************************************/
template< typename FloatT >
void magma_generate_geev(
    magma_opts& opts,
    Dist dist,
    typename blas::traits<FloatT>::real_t cond,
    typename blas::traits<FloatT>::real_t condD,
    typename blas::traits<FloatT>::real_t sigma_max,
    Matrix<FloatT>& A,
    Vector< typename blas::traits<FloatT>::real_t >& sigma )
{
    throw std::exception();  // not implemented
}

/******************************************************************************/
template< typename FloatT >
void magma_generate_geevx(
    magma_opts& opts,
    Dist dist,
    typename blas::traits<FloatT>::real_t cond,
    typename blas::traits<FloatT>::real_t condD,
    typename blas::traits<FloatT>::real_t sigma_max,
    Matrix<FloatT>& A,
    Vector< typename blas::traits<FloatT>::real_t >& sigma )
{
    throw std::exception();  // not implemented
}

/***************************************************************************//**
    Purpose
    -------
    Generate an m-by-n test matrix A.
    Similar to but does not use LAPACK's libtmg.

    Arguments
    ---------
    @param[in]
    opts    MAGMA options. Uses matrix, cond, condD; see further details.

    @param[out]
    A       Complex array, dimension (lda, n).
            On output, the m-by-n test matrix A in an lda-by-n array.

    @param[in,out]
    sigma   Real array, dimension (min(m,n))
            For matrix with "_specified", on input contains user-specified
            singular or eigenvalues.
            On output, contains singular or eigenvalues, if known,
            else set to NaN. sigma is not necesarily sorted.

    Further Details
    ---------------
    The --matrix command line option specifies the matrix name according to the
    table below. Where indicated, names take an optional distribution suffix (#)
    and an optional scaling suffix (*). The default distribution is rand.
    Examples: rand, rand_small, svd_arith, heev_geo_small.

    The --cond and --condD command line options specify condition numbers as
    described below. Default cond = sqrt( 1/eps ) = 6.7e7 for double, condD = 1.

    Sigma is a diagonal matrix with entries sigma_i for i = 1, ..., n;
    Lambda is a diagonal matrix with entries lambda_i = sigma_i with random sign;
    U and V are random orthogonal matrices from the Haar distribution
    (See: Stewart, The efficient generation of random orthogonal matrices
     with an application to condition estimators, 1980);
    X is a random matrix.

    See LAPACK Working Note (LAWN) 41:
    Table  5 (Test matrices for the nonsymmetric eigenvalue problem)
    Table 10 (Test matrices for the symmetric eigenvalue problem)
    Table 11 (Test matrices for the singular value decomposition)

    Matrix      Description
    zero
    identity
    jordan      ones on diagonal and first subdiagonal

    rand*       matrix entries random uniform on (0, 1)
    randu*      matrix entries random uniform on (-1, 1)
    randn*      matrix entries random normal with mean 0, sigma 1

    diag#*      A = Sigma
    svd#*       A = U Sigma V^H
    poev#*      A = V Sigma V^H  (eigenvalues positive [1], i.e., matrix SPD)
    spd#*       alias for poev
    heev#*      A = V Lambda V^H (eigenvalues mixed signs)
    syev#*      alias for heev
    geev#*      A = V T V^H, Schur-form T                       [not yet implemented]
    geevx#*     A = X T X^{-1}, Schur-form T, X ill-conditioned [not yet implemented]

    # optional distribution suffix
    _rand       sigma_i random uniform on (0, 1) [default]
    _randu      sigma_i random uniform on (-1, 1)
    _randn      sigma_i random normal with mean 0, std 1
                [1] Note for _randu and _randn, Sigma contains negative values.
                _rand* do not use cond, so the condition number is arbitrary.

    _logrand    log(sigma_i) uniform on (log(1/cond), log(1))
    _arith      sigma_i = 1 - (i - 1)/(n - 1)*(1 - 1/cond); sigma_{i+1} - sigma_i is constant
    _geo        sigma_i = (cond)^{ -(i-1)/(n-1) };          sigma_{i+1} / sigma_i is constant
    _cluster    sigma = [ 1, 1/cond, ..., 1/cond ]; 1 unit value, n-1 small values
    _cluster2   sigma = [ 1, ..., 1, 1/cond ];      n-1 unit values, 1 small value
    _rarith     _arith,    reversed order
    _rgeo       _geo,      reversed order
    _rcluster   _cluster,  reversed order
    _rcluster2  _cluster2, reversed order
    _specified  user specified sigma on input

    * optional scaling & modifier suffix
    _ufl        scale near underflow         = 1e-308 for double
    _ofl        scale near overflow          = 2e+308 for double
    _small      scale near sqrt( underflow ) = 1e-154 for double
    _large      scale near sqrt( overflow  ) = 6e+153 for double
    _dominant   diagonally dominant: set A_ii = ± max( sum_j |A_ij|, sum_j |A_ji| )
                Note _dominant changes the singular or eigenvalues.

    [below scaling by D implemented, scaling by K not yet implemented]
    If condD != 1, then:
    For SVD, A = (U Sigma V^H) K D, where
    K is diagonal such that columns of (U Sigma V^H K) have unit norm,
    hence (A^T A) has unit diagonal,
    and D has log-random entries in ( log(1/condD), log(1) ).

    For heev, A0 = U Lambda U^H, A = D K A0 K D, where
    K is diagonal such that (K A0 K) has unit diagonal, and D is as above.

    Note using condD changes the singular or eigenvalues; on output, sigma
    contains the singular or eigenvalues of A0, not of A.
    See: Demmel and Veselic, Jacobi's method is more accurate than QR, 1992.

    @ingroup testing
*******************************************************************************/
template< typename FloatT >
void magma_generate_matrix(
    magma_opts& opts,
    Matrix<FloatT>& A,
    Vector< typename blas::traits<FloatT>::real_t >& sigma )
{
    typedef typename blas::traits<FloatT>::real_t real_t;

    // constants
    const real_t nan = std::numeric_limits<real_t>::quiet_NaN();
    const real_t d_zero = MAGMA_D_ZERO;
    const real_t d_one  = MAGMA_D_ONE;
    const real_t ufl = std::numeric_limits< real_t >::min();      // == lamch("safe min")  ==  1e-38 or  2e-308
    const real_t ofl = 1 / ufl;                                   //                            8e37 or   4e307
    const real_t eps = std::numeric_limits< real_t >::epsilon();  // == lamch("precision") == 1.2e-7 or 2.2e-16
    const FloatT c_zero = blas::traits<FloatT>::make( 0, 0 );
    const FloatT c_one  = blas::traits<FloatT>::make( 1, 0 );

    // locals
    std::string name = opts.matrix;
    real_t cond = opts.cond;
    if (cond == 0) {
        cond = 1 / sqrt( eps );
    }
    real_t condD = opts.condD;
    real_t sigma_max = 1;
    magma_int_t minmn = min( A.m, A.n );

    // ----------
    // set sigma to unknown (nan)
    lapack::laset( "general", sigma.n, 1, nan, nan, sigma(0), sigma.n );

    // ----- decode matrix type
    MatrixType type = MatrixType::identity;
    if      (name == "zero")          { type = MatrixType::zero;      }
    else if (name == "identity")      { type = MatrixType::identity;  }
    else if (name == "jordan")        { type = MatrixType::jordan;    }
    else if (begins( name, "randn" )) { type = MatrixType::randn;     }
    else if (begins( name, "randu" )) { type = MatrixType::randu;     }
    else if (begins( name, "rand"  )) { type = MatrixType::rand;      }
    else if (begins( name, "diag"  )) { type = MatrixType::diag;      }
    else if (begins( name, "svd"   )) { type = MatrixType::svd;       }
    else if (begins( name, "poev"  ) ||
             begins( name, "spd"   )) { type = MatrixType::poev;      }
    else if (begins( name, "heev"  ) ||
             begins( name, "syev"  )) { type = MatrixType::heev;      }
    else if (begins( name, "geevx" )) { type = MatrixType::geevx;     }
    else if (begins( name, "geev"  )) { type = MatrixType::geev;      }
    else {
        fprintf( stderr, "Unrecognized matrix '%s'\n", name.c_str() );
        throw std::exception();
    }

    if (A.m != A.n &&
        (type == MatrixType::jordan ||
         type == MatrixType::poev   ||
         type == MatrixType::heev   ||
         type == MatrixType::geev   ||
         type == MatrixType::geevx))
    {
        fprintf( stderr, "Eigenvalue matrix requires m == n.\n" );
        throw std::exception();
    }

    if (opts.cond != 0 &&
        (type == MatrixType::zero      ||
         type == MatrixType::identity  ||
         type == MatrixType::jordan    ||
         type == MatrixType::randn     ||
         type == MatrixType::randu     ||
         type == MatrixType::rand))
    {
        fprintf( stderr, "%sWarning: --matrix %s ignores --cond %.2e.%s\n",
                 ansi_red, name.c_str(), opts.cond, ansi_normal );
    }

    // ----- decode distribution
    Dist dist = Dist::rand;
    if      (contains( name, "_randn"     )) { dist = Dist::randn;     }
    else if (contains( name, "_randu"     )) { dist = Dist::randu;     }
    else if (contains( name, "_rand"      )) { dist = Dist::rand;      } // after randn, randu
    else if (contains( name, "_logrand"   )) { dist = Dist::logrand;   }
    else if (contains( name, "_arith"     )) { dist = Dist::arith;     }
    else if (contains( name, "_geo"       )) { dist = Dist::geo;       }
    else if (contains( name, "_cluster2"  )) { dist = Dist::cluster2;  }
    else if (contains( name, "_cluster"   )) { dist = Dist::cluster;   } // after cluster2
    else if (contains( name, "_rarith"    )) { dist = Dist::rarith;    }
    else if (contains( name, "_rgeo"      )) { dist = Dist::rgeo;      }
    else if (contains( name, "_rcluster2" )) { dist = Dist::rcluster2; }
    else if (contains( name, "_rcluster"  )) { dist = Dist::rcluster;  } // after rcluster2
    else if (contains( name, "_specified" )) { dist = Dist::specified; }

    if (opts.cond != 0 &&
        (dist == Dist::randn ||
         dist == Dist::randu ||
         dist == Dist::rand))
    {
        fprintf( stderr, "%sWarning: --matrix '%s' ignores --cond %.2e; use a different distribution.%s\n",
                 ansi_red, name.c_str(), opts.cond, ansi_normal );
    }

    if (type == MatrixType::poev &&
        (dist == Dist::randu ||
         dist == Dist::randn))
    {
        fprintf( stderr, "%sWarning: --matrix '%s' using randu or randn "
                 "will not generate SPD matrix; use rand instead.%s\n",
                 ansi_red, name.c_str(), ansi_normal );
    }

    // ----- decode scaling
    if      (contains( name, "_small"  )) { sigma_max = sqrt( ufl ); }
    else if (contains( name, "_large"  )) { sigma_max = sqrt( ofl ); }
    else if (contains( name, "_ufl"    )) { sigma_max = ufl; }
    else if (contains( name, "_ofl"    )) { sigma_max = ofl; }

    // ----- generate matrix
    switch (type) {
        case MatrixType::zero:
            lapack::laset( "general", A.m, A.n, c_zero, c_zero, A(0,0), A.ld );
            lapack::laset( "general", sigma.n, 1, d_zero, d_zero, sigma(0), sigma.n );
            break;

        case MatrixType::identity:
            lapack::laset( "general", A.m, A.n, c_zero, c_one, A(0,0), A.ld );
            lapack::laset( "general", sigma.n, 1, d_one, d_one, sigma(0), sigma.n );
            break;

        case MatrixType::jordan: {
            magma_int_t n1 = A.n - 1;
            lapack::laset( "upper", A.n, A.n, c_zero, c_one, A(0,0), A.ld );  // ones on diagonal
            lapack::laset( "lower", n1,  n1,  c_zero, c_one, A(1,0), A.ld );  // ones on sub-diagonal
            break;
        }

        case MatrixType::rand:
        case MatrixType::randu:
        case MatrixType::randn: {
            magma_int_t idist = (magma_int_t) type;
            magma_int_t sizeA = A.ld * A.n;
            lapack::larnv( idist, opts.iseed, sizeA, A(0,0) );
            if (sigma_max != 1) {
                FloatT scale = blas::traits<FloatT>::make( sigma_max, 0 );
                blas::scal( sizeA, scale, A(0,0), 1 );
            }
            break;
        }

        case MatrixType::diag:
            magma_generate_sigma( opts, dist, false, cond, sigma_max, A, sigma );
            break;

        case MatrixType::svd:
            magma_generate_svd( opts, dist, cond, condD, sigma_max, A, sigma );
            break;

        case MatrixType::poev:
            magma_generate_heev( opts, dist, false, cond, condD, sigma_max, A, sigma );
            break;

        case MatrixType::heev:
            magma_generate_heev( opts, dist, true, cond, condD, sigma_max, A, sigma );
            break;

        case MatrixType::geev:
            magma_generate_geev( opts, dist, cond, condD, sigma_max, A, sigma );
            break;

        case MatrixType::geevx:
            magma_generate_geevx( opts, dist, cond, condD, sigma_max, A, sigma );
            break;
    }

    if (contains( name, "_dominant" )) {
        // make diagonally dominant; strict unless diagonal has zeros
        for (int i = 0; i < minmn; ++i) {
            real_t sum = max( blas::asum( A.m, A(0,i), 1    ),    // i-th col
                              blas::asum( A.n, A(i,0), A.ld ) );  // i-th row
            *A(i,i) = blas::traits<FloatT>::make( sum, 0 );
        }
        // reset sigma to unknown (nan)
        lapack::laset( "general", sigma.n, 1, nan, nan, sigma(0), sigma.n );
    }
}


/******************************************************************************/
// traditional interface with m, n, lda
template< typename FloatT >
void magma_generate_matrix(
    magma_opts& opts,
    magma_int_t m, magma_int_t n,
    FloatT* A_ptr, magma_int_t lda,
    typename blas::traits<FloatT>::real_t* sigma_ptr /* =nullptr */ )
{
    typedef typename blas::traits<FloatT>::real_t real_t;

    // vector & matrix wrappers
    // if sigma is null, create new vector; data is discarded later
    Vector<real_t> sigma( sigma_ptr, min(m,n) );
    if (sigma_ptr == nullptr) {
        sigma = Vector<real_t>( min(m,n) );
    }
    Matrix<FloatT> A( A_ptr, m, n, lda );
    magma_generate_matrix( opts, A, sigma );
}


/******************************************************************************/
// explicit instantiations
template
void magma_generate_matrix(
    magma_opts& opts,
    magma_int_t m, magma_int_t n,
    float* A_ptr, magma_int_t lda,
    float* sigma_ptr );

template
void magma_generate_matrix(
    magma_opts& opts,
    magma_int_t m, magma_int_t n,
    double* A_ptr, magma_int_t lda,
    double* sigma_ptr );

template
void magma_generate_matrix(
    magma_opts& opts,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex* A_ptr, magma_int_t lda,
    float* sigma_ptr );

template
void magma_generate_matrix(
    magma_opts& opts,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex* A_ptr, magma_int_t lda,
    double* sigma_ptr );
