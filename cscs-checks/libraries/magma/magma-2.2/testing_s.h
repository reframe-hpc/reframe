/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from testing/testing_z.h, normal z -> s, Sun Nov 20 20:20:47 2016
       @author Mark Gates
       
       Utilities for testing.
*/
#ifndef TESTING_MAGMA_S_H
#define TESTING_MAGMA_S_H

#ifdef __cplusplus
extern "C" {
#endif

#define REAL

void magma_smake_symmetric( magma_int_t N, float* A, magma_int_t lda );
void magma_smake_symmetric( magma_int_t N, float* A, magma_int_t lda );

void magma_smake_spd( magma_int_t N, float* A, magma_int_t lda );
void magma_smake_hpd( magma_int_t N, float* A, magma_int_t lda );

// work around MKL bug in multi-threaded lanhe/lansy
float safe_lapackf77_slansy(
    const char *norm, const char *uplo,
    const magma_int_t *n,
    const float *A, const magma_int_t *lda,
    float *work );

#ifdef COMPLEX
static inline float magma_slapy2( float x )
{
    float xr = MAGMA_S_REAL( x );
    float xi = MAGMA_S_IMAG( x );
    return lapackf77_slapy2( &xr, &xi );
}
#endif

void check_sgesvd(
    magma_int_t check,
    magma_vec_t jobu,
    magma_vec_t jobvt,
    magma_int_t m, magma_int_t n,
    float *A,  magma_int_t lda,
    float *S,
    float *U,  magma_int_t ldu,
    float *VT, magma_int_t ldv,
    float result[4] );

void check_sgeev(
    magma_vec_t jobvl,
    magma_vec_t jobvr,
    magma_int_t n,
    float *A,  magma_int_t lda,
    #ifdef COMPLEX
    float *w,
    #else
    float *wr, float *wi,
    #endif
    float *VL, magma_int_t ldvl,
    float *VR, magma_int_t ldvr,
    float *work, magma_int_t lwork,
    #ifdef COMPLEX
    float *rwork, magma_int_t lrwork,
    #endif
    float result[4] );

#undef REAL

#ifdef __cplusplus
}
#endif

#endif        //  #ifndef TESTING_MAGMA_S_H
