/*
    -- MAGMA (version 2.4.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date June 2018

       @generated from testing/testing_z.h, normal z -> d, Mon Jun 25 18:24:32 2018
       @author Mark Gates
       
       Utilities for testing.
*/
#ifndef TESTING_MAGMA_D_H
#define TESTING_MAGMA_D_H

#ifdef __cplusplus
extern "C" {
#endif

#define REAL

void magma_dmake_symmetric( magma_int_t N, double* A, magma_int_t lda );
void magma_dmake_symmetric( magma_int_t N, double* A, magma_int_t lda );

void magma_dmake_spd( magma_int_t N, double* A, magma_int_t lda );
void magma_dmake_hpd( magma_int_t N, double* A, magma_int_t lda );

// work around MKL bug in multi-threaded lanhe/lansy
double safe_lapackf77_dlansy(
    const char *norm, const char *uplo,
    const magma_int_t *n,
    const double *A, const magma_int_t *lda,
    double *work );

#ifdef COMPLEX
static inline double magma_dlapy2( double x )
{
    double xr = MAGMA_D_REAL( x );
    double xi = MAGMA_D_IMAG( x );
    return lapackf77_dlapy2( &xr, &xi );
}
#endif

void check_dgesvd(
    magma_int_t check,
    magma_vec_t jobu,
    magma_vec_t jobvt,
    magma_int_t m, magma_int_t n,
    double *A,  magma_int_t lda,
    double *S,
    double *U,  magma_int_t ldu,
    double *VT, magma_int_t ldv,
    double result[4] );

void check_dgeev(
    magma_vec_t jobvl,
    magma_vec_t jobvr,
    magma_int_t n,
    double *A,  magma_int_t lda,
    #ifdef COMPLEX
    double *w,
    #else
    double *wr, double *wi,
    #endif
    double *VL, magma_int_t ldvl,
    double *VR, magma_int_t ldvr,
    double *work, magma_int_t lwork,
    #ifdef COMPLEX
    double *rwork, magma_int_t lrwork,
    #endif
    double result[4] );

//void magma_dgenerate_matrix(
//    magma_int_t matrix,
//    magma_int_t m, magma_int_t n,
//    magma_int_t iseed[4],
//    double* sigma,
//    double* A, magma_int_t lda );

#undef REAL

#ifdef __cplusplus
}
#endif

/******************************************************************************/
// C++ utility functions

//class magma_opts;

//void magma_generate_matrix(
//    magma_opts& opts,
//    magma_int_t iseed[4],
//    magma_int_t m, magma_int_t n,
//    double* sigma_ptr,
//    double* A_ptr, magma_int_t lda );

#endif        //  #ifndef TESTING_MAGMA_D_H
