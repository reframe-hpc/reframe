/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @precisions normal z -> s d c
       @author Mark Gates
       
       Utilities for testing.
*/
#ifndef TESTING_MAGMA_Z_H
#define TESTING_MAGMA_Z_H

#ifdef __cplusplus
extern "C" {
#endif

#define COMPLEX

void magma_zmake_symmetric( magma_int_t N, magmaDoubleComplex* A, magma_int_t lda );
void magma_zmake_hermitian( magma_int_t N, magmaDoubleComplex* A, magma_int_t lda );

void magma_zmake_spd( magma_int_t N, magmaDoubleComplex* A, magma_int_t lda );
void magma_zmake_hpd( magma_int_t N, magmaDoubleComplex* A, magma_int_t lda );

// work around MKL bug in multi-threaded lanhe/lansy
double safe_lapackf77_zlanhe(
    const char *norm, const char *uplo,
    const magma_int_t *n,
    const magmaDoubleComplex *A, const magma_int_t *lda,
    double *work );

#ifdef COMPLEX
static inline double magma_zlapy2( magmaDoubleComplex x )
{
    double xr = MAGMA_Z_REAL( x );
    double xi = MAGMA_Z_IMAG( x );
    return lapackf77_dlapy2( &xr, &xi );
}
#endif

void check_zgesvd(
    magma_int_t check,
    magma_vec_t jobu,
    magma_vec_t jobvt,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex *A,  magma_int_t lda,
    double *S,
    magmaDoubleComplex *U,  magma_int_t ldu,
    magmaDoubleComplex *VT, magma_int_t ldv,
    double result[4] );

void check_zgeev(
    magma_vec_t jobvl,
    magma_vec_t jobvr,
    magma_int_t n,
    magmaDoubleComplex *A,  magma_int_t lda,
    #ifdef COMPLEX
    magmaDoubleComplex *w,
    #else
    double *wr, double *wi,
    #endif
    magmaDoubleComplex *VL, magma_int_t ldvl,
    magmaDoubleComplex *VR, magma_int_t ldvr,
    magmaDoubleComplex *work, magma_int_t lwork,
    #ifdef COMPLEX
    double *rwork, magma_int_t lrwork,
    #endif
    double result[4] );

#undef COMPLEX

#ifdef __cplusplus
}
#endif

#endif        //  #ifndef TESTING_MAGMA_Z_H
