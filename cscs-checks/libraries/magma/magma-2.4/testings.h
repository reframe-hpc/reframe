#ifndef TESTINGS_H
#define TESTINGS_H

#include <stdio.h>
#include <stdlib.h>

#if ! defined(MAGMA_H) && ! defined(MAGMA_V2_H)
#include "magma_v2.h"
#endif

#include <vector>
#include <string>
#include <cmath>

#include "magma_lapack.h"
#include "magma_lapack.hpp"  // C++ bindings; need traits
#include "magma_matrix.hpp"  // experimental Matrix and Vector classes
#include "testing_s.h"
#include "testing_d.h"
#include "testing_c.h"
#include "testing_z.h"

/***************************************************************************//**
 *  For portability to Windows
 */
#if defined( _WIN32 ) || defined( _WIN64 )
    // functions where Microsoft fails to provide C99 or POSIX standard
    // (only with Microsoft, not with nvcc on Windows)
    // in both magma_internal.h and testings.h
    #ifndef __NVCC__
        // note _snprintf has slightly different semantics than snprintf
        #define snprintf      _snprintf
        #define unlink        _unlink
    #endif
#endif


#ifdef __cplusplus
extern "C" {
#endif

void flops_init();


/***************************************************************************//**
    max that propogates nan consistently:
    max_nan( 1,   nan ) = nan
    max_nan( nan, 1   ) = nan

    isnan and isinf are hard to call portably from both C and C++.
    In Windows C,     include float.h, use _isnan as above (before VS 2015)
    In Unix C or C++, include math.h,  use isnan
    In std C++,       include cmath,   use std::isnan
    Sometimes in C++, include cmath,   use isnan is okay (on Linux but not MacOS)
    This makes writing a header inline function a nightmare. For now, do it
    here in testing to avoid potential issues in the MAGMA library itself.
*******************************************************************************/
static inline double magma_max_nan( double x, double y )
{
    #ifdef isnan
        // with include <math.h> macro
        return (isnan(y) || x < y ? y : x);
    #else
        // with include <cmath> function
        return (std::isnan(y) || x < y ? y : x);
    #endif
}


/***************************************************************************//**
 *  Global utilities
 *  in both magma_internal.h and testings.h
 **/
#ifndef max
#define max(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef min
#define min(a, b) ((a) < (b) ? (a) : (b))
#endif

// suppress "warning: unused variable" in a portable fashion
#define MAGMA_UNUSED(var)  ((void)var)

/***************************************************************************//**
 * Macros to handle error checking.
 */

#define TESTING_CHECK( err )                                                 \
    do {                                                                     \
        magma_int_t err_ = (err);                                            \
        if ( err_ != 0 ) {                                                   \
            fprintf( stderr, "Error: %s\nfailed at %s:%d: error %lld: %s\n", \
                     #err, __FILE__, __LINE__,                               \
                     (long long) err_, magma_strerror(err_) );               \
            exit(1);                                                         \
        }                                                                    \
    } while( 0 )


/***************************************************************************//**
 * Functions and data structures used for testing.
 */

void magma_assert( bool condition, const char* msg, ... );

void magma_assert_warn( bool condition, const char* msg, ... );

void magma_flush_cache( size_t cache_size );

#ifdef __cplusplus
}
#endif

/***************************************************************************//**
 * C++ Functions and data structures used for testing.
 */

// for sorting in descending order (e.g., singular values)
template< typename T >
bool greater( T a, T b )
{
    return (a > b);
}

#define MAX_NTEST 1050

typedef enum {
    MagmaOptsDefault = 0,
    MagmaOptsBatched = 1000
} magma_opts_t;

typedef enum {
    MagmaSVD_all,
    MagmaSVD_query,
    MagmaSVD_doc,
    MagmaSVD_doc_old,
    MagmaSVD_min,
    MagmaSVD_min_1,
    MagmaSVD_min_old,
    MagmaSVD_min_old_1,
    MagmaSVD_min_fast,
    MagmaSVD_min_fast_1,
    MagmaSVD_opt,
    MagmaSVD_opt_old,
    MagmaSVD_opt_slow,
    MagmaSVD_max
} magma_svd_work_t;

class magma_opts
{
public:
    // constructor
    magma_opts( magma_opts_t flag=MagmaOptsDefault );
    
    // parse command line
    void parse_opts( int argc, char** argv );
    
    // set range, vl, vu, il, iu for eigen/singular value problems (gesvdx, syevdx, ...)
    void get_range( magma_int_t n, magma_range_t* range,
                    double* vl, double* vu,
                    magma_int_t* il, magma_int_t* iu );
    
    void get_range( magma_int_t n, magma_range_t* range,
                    float* vl, float* vu,
                    magma_int_t* il, magma_int_t* iu );
    
    // deallocate queues, etc.
    void cleanup();
    
    // matrix size
    magma_int_t ntest;
    magma_int_t msize[ MAX_NTEST ];
    magma_int_t nsize[ MAX_NTEST ];
    magma_int_t ksize[ MAX_NTEST ];
    magma_int_t batchcount;
    
    magma_int_t default_nstart;
    magma_int_t default_nend;
    magma_int_t default_nstep;
    
    // scalars
    magma_int_t device;
    magma_int_t cache;
    magma_int_t align;
    magma_int_t nb;
    magma_int_t nrhs;
    magma_int_t nqueue;
    magma_int_t ngpu;
    magma_int_t nsub;
    magma_int_t niter;
    magma_int_t nthread;
    magma_int_t offset;
    magma_int_t itype;     // hegvd: problem type
    magma_int_t version;
    magma_int_t check;
    magma_int_t verbose;
    
    // ranges for eigen/singular values (gesvdx, heevdx, ...)
    double      fraction_lo;
    double      fraction_up;
    magma_int_t irange_lo;
    magma_int_t irange_up;
    double      vrange_lo;
    double      vrange_up;
    
    double      tolerance;
    
    // boolean arguments
    bool magma;
    bool lapack;
    bool warmup;
    
    // lapack options
    magma_uplo_t    uplo;
    magma_trans_t   transA;
    magma_trans_t   transB;
    magma_side_t    side;
    magma_diag_t    diag;
    magma_vec_t     jobz;    // heev:   no eigen vectors
    magma_vec_t     jobvr;   // geev:   no right eigen vectors
    magma_vec_t     jobvl;   // geev:   no left  eigen vectors
    
    // vectors of options
    std::vector< magma_svd_work_t > svd_work;
    std::vector< magma_vec_t > jobu;
    std::vector< magma_vec_t > jobv;

    // LAPACK test matrix generation
    std::string matrix;
    double      cond;
    double      condD;
    magma_int_t iseed[4];

    // queue for default device
    magma_queue_t   queue;
    magma_queue_t   queues2[3];  // 2 queues + 1 extra NULL entry to catch errors
    
    #ifdef HAVE_CUBLAS
    // handle for directly calling cublas
    cublasHandle_t  handle;
    #endif
};

extern const char* g_platform_str;

// -----------------------------------------------------------------------------
template< typename FloatT >
void magma_generate_matrix(
    magma_opts& opts,
    Matrix< FloatT >& A,
    Vector< typename blas::traits<FloatT>::real_t >& sigma );

template< typename FloatT >
void magma_generate_matrix(
    magma_opts& opts,
    magma_int_t m, magma_int_t n,
    FloatT* A, magma_int_t lda,
    typename blas::traits<FloatT>::real_t* sigma=nullptr );

#endif /* TESTINGS_H */
