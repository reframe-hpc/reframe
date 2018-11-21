/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from testing/testing_z.h, normal z -> c, Sun Nov 20 20:20:47 2016
       @author Mark Gates
       
       Utilities for testing.
*/
#ifndef TESTING_MAGMA_C_H
#define TESTING_MAGMA_C_H

#ifdef __cplusplus
extern "C" {
#endif

#define COMPLEX

void magma_cmake_symmetric( magma_int_t N, magmaFloatComplex* A, magma_int_t lda );
void magma_cmake_hermitian( magma_int_t N, magmaFloatComplex* A, magma_int_t lda );

void magma_cmake_spd( magma_int_t N, magmaFloatComplex* A, magma_int_t lda );
void magma_cmake_hpd( magma_int_t N, magmaFloatComplex* A, magma_int_t lda );

// work around MKL bug in multi-threaded lanhe/lansy
float safe_lapackf77_clanhe(
    const char *norm, const char *uplo,
    const magma_int_t *n,
    const magmaFloatComplex *A, const magma_int_t *lda,
    float *work );

#ifdef COMPLEX
static inline float magma_clapy2( magmaFloatComplex x )
{
    float xr = MAGMA_C_REAL( x );
    float xi = MAGMA_C_IMAG( x );
    return lapackf77_slapy2( &xr, &xi );
}
#endif

void check_cgesvd(
    magma_int_t check,
    magma_vec_t jobu,
    magma_vec_t jobvt,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex *A,  magma_int_t lda,
    float *S,
    magmaFloatComplex *U,  magma_int_t ldu,
    magmaFloatComplex *VT, magma_int_t ldv,
    float result[4] );

void check_cgeev(
    magma_vec_t jobvl,
    magma_vec_t jobvr,
    magma_int_t n,
    magmaFloatComplex *A,  magma_int_t lda,
    #ifdef COMPLEX
    magmaFloatComplex *w,
    #else
    float *wr, float *wi,
    #endif
    magmaFloatComplex *VL, magma_int_t ldvl,
    magmaFloatComplex *VR, magma_int_t ldvr,
    magmaFloatComplex *work, magma_int_t lwork,
    #ifdef COMPLEX
    float *rwork, magma_int_t lrwork,
    #endif
    float result[4] );

#undef COMPLEX

#ifdef __cplusplus
}
#endif

#endif        //  #ifndef TESTING_MAGMA_C_H
