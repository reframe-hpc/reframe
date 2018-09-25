/*
    -- MAGMA (version 2.4.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date June 2018

       @generated from testing/testing_z.h, normal z -> s, Mon Jun 25 18:24:32 2018
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

//void magma_sgenerate_matrix(
//    magma_int_t matrix,
//    magma_int_t m, magma_int_t n,
//    magma_int_t iseed[4],
//    float* sigma,
//    float* A, magma_int_t lda );

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
//    float* sigma_ptr,
//    float* A_ptr, magma_int_t lda );

#endif        //  #ifndef TESTING_MAGMA_S_H
