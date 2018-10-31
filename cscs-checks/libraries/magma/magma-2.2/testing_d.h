/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from testing/testing_z.h, normal z -> d, Sun Nov 20 20:20:46 2016
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

#undef REAL

#ifdef __cplusplus
}
#endif

#endif        //  #ifndef TESTING_MAGMA_D_H
