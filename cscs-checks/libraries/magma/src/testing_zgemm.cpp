/*
    -- MAGMA (version 2.4.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date June 2018

       @precisions normal z -> c d s
       @author Mark Gates
*/
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "flops.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#include "magma_operators.h"
#include "testings.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgemm
*/
int main( int argc, char** argv)
{
    #ifdef HAVE_clBLAS
    #define dA(i_, j_)  dA, ((i_) + (j_)*ldda)
    #define dB(i_, j_)  dB, ((i_) + (j_)*lddb)
    #define dC(i_, j_)  dC, ((i_) + (j_)*lddc)
    #else
    #define dA(i_, j_) (dA + (i_) + (j_)*ldda)
    #define dB(i_, j_) (dB + (i_) + (j_)*lddb)
    #define dC(i_, j_) (dC + (i_) + (j_)*lddc)
    #endif
    
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, magma_perf, magma_time, dev_perf, dev_time, cpu_perf, cpu_time;
    double          magma_error, dev_error, work[1];
    magma_int_t M, N, K;
    magma_int_t Am, An, Bm, Bn;
    magma_int_t sizeA, sizeB, sizeC;
    magma_int_t lda, ldb, ldc, ldda, lddb, lddc;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    int status = 0;
    
    magmaDoubleComplex *hA, *hB, *hC, *hCmagma, *hCdev;
    magmaDoubleComplex_ptr dA, dB, dC;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex alpha = MAGMA_Z_MAKE(  0.29, -0.86 );
    magmaDoubleComplex beta  = MAGMA_Z_MAKE( -0.48,  0.38 );
    
    // used only with CUDA
    MAGMA_UNUSED( magma_perf );
    MAGMA_UNUSED( magma_time );
    MAGMA_UNUSED( magma_error );
    
    magma_opts opts;
    opts.parse_opts( argc, argv );
    
    // Allow 3*eps; complex needs 2*sqrt(2) factor; see Higham, 2002, sec. 3.6.
    double eps = lapackf77_dlamch("E");
    double tol = 3*eps;

    #ifdef HAVE_CUBLAS
        // for CUDA, we can check MAGMA vs. CUBLAS, without running LAPACK
        printf("%% If running lapack (option --lapack), MAGMA and %s error are both computed\n"
               "%% relative to CPU BLAS result. Else, MAGMA error is computed relative to %s result.\n\n",
                g_platform_str, g_platform_str );
        printf("%% transA = %s, transB = %s\n",
               lapack_trans_const(opts.transA),
               lapack_trans_const(opts.transB) );
        printf("%%   M     N     K   MAGMA Gflop/s (ms)  %s Gflop/s (ms)   CPU Gflop/s (ms)  MAGMA error  %s error\n",
                g_platform_str, g_platform_str );
    #else
        // for others, we need LAPACK for check
        opts.lapack |= opts.check;  // check (-c) implies lapack (-l)
        printf("%% transA = %s, transB = %s\n",
               lapack_trans_const(opts.transA),
               lapack_trans_const(opts.transB) );
        printf("%%   M     N     K   %s Gflop/s (ms)   CPU Gflop/s (ms)  %s error\n",
                g_platform_str, g_platform_str );
    #endif
    printf("%%========================================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            K = opts.ksize[itest];
            gflops = FLOPS_ZGEMM( M, N, K ) / 1e9;

            if ( opts.transA == MagmaNoTrans ) {
                lda = Am = M;
                An = K;
            } else {
                lda = Am = K;
                An = M;
            }
            
            if ( opts.transB == MagmaNoTrans ) {
                ldb = Bm = K;
                Bn = N;
            } else {
                ldb = Bm = N;
                Bn = K;
            }
            ldc = M;
            
            ldda = magma_roundup( lda, opts.align );  // multiple of 32 by default
            lddb = magma_roundup( ldb, opts.align );  // multiple of 32 by default
            lddc = magma_roundup( ldc, opts.align );  // multiple of 32 by default
            
            sizeA = lda*An;
            sizeB = ldb*Bn;
            sizeC = ldc*N;
            
            TESTING_CHECK( magma_zmalloc_cpu( &hA,       lda*An ));
            TESTING_CHECK( magma_zmalloc_cpu( &hB,       ldb*Bn ));
            TESTING_CHECK( magma_zmalloc_cpu( &hC,       ldc*N  ));
            TESTING_CHECK( magma_zmalloc_cpu( &hCmagma,  ldc*N  ));
            TESTING_CHECK( magma_zmalloc_cpu( &hCdev,    ldc*N  ));
            
            TESTING_CHECK( magma_zmalloc( &dA, ldda*An ));
            TESTING_CHECK( magma_zmalloc( &dB, lddb*Bn ));
            TESTING_CHECK( magma_zmalloc( &dC, lddc*N  ));
            
            /* Initialize the matrices */
            lapackf77_zlarnv( &ione, ISEED, &sizeA, hA );
            lapackf77_zlarnv( &ione, ISEED, &sizeB, hB );
            lapackf77_zlarnv( &ione, ISEED, &sizeC, hC );
            
            magma_zsetmatrix( Am, An, hA, lda, dA(0,0), ldda, opts.queue );
            magma_zsetmatrix( Bm, Bn, hB, ldb, dB(0,0), lddb, opts.queue );
            
            // for error checks
            double Anorm = lapackf77_zlange( "F", &Am, &An, hA, &lda, work );
            double Bnorm = lapackf77_zlange( "F", &Bm, &Bn, hB, &ldb, work );
            double Cnorm = lapackf77_zlange( "F", &M,  &N,  hC, &ldc, work );
            
            /* =====================================================================
               Performs operation using MAGMABLAS (currently only with CUDA)
               =================================================================== */
            #ifdef HAVE_CUBLAS
                magma_zsetmatrix( M, N, hC, ldc, dC, lddc, opts.queue );
                
                magma_flush_cache( opts.cache );
                magma_time = magma_sync_wtime( opts.queue );
                magmablas_zgemm( opts.transA, opts.transB, M, N, K,
                                 alpha, dA, ldda,
                                        dB, lddb,
                                 beta,  dC, lddc,
                                 opts.queue );
                magma_time = magma_sync_wtime( opts.queue ) - magma_time;
                magma_perf = gflops / magma_time;
                
                magma_zgetmatrix( M, N, dC, lddc, hCmagma, ldc, opts.queue );
            #endif
            
            /* =====================================================================
               Performs operation using CUBLAS / clBLAS / Xeon Phi MKL
               =================================================================== */
            magma_zsetmatrix( M, N, hC, ldc, dC(0,0), lddc, opts.queue );
            
            magma_flush_cache( opts.cache );
            dev_time = magma_sync_wtime( opts.queue );
            magma_zgemm( opts.transA, opts.transB, M, N, K,
                         alpha, dA(0,0), ldda,
                                dB(0,0), lddb,
                         beta,  dC(0,0), lddc, opts.queue );
            dev_time = magma_sync_wtime( opts.queue ) - dev_time;
            dev_perf = gflops / dev_time;
            
            magma_zgetmatrix( M, N, dC(0,0), lddc, hCdev, ldc, opts.queue );
            
            /* =====================================================================
               Performs operation using CPU BLAS
               =================================================================== */
            if ( opts.lapack ) {
                magma_flush_cache( opts.cache );
                cpu_time = magma_wtime();
                blasf77_zgemm( lapack_trans_const(opts.transA), lapack_trans_const(opts.transB), &M, &N, &K,
                               &alpha, hA, &lda,
                                       hB, &ldb,
                               &beta,  hC, &ldc );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
            }
            
            /* =====================================================================
               Check the result
               =================================================================== */
            if ( opts.lapack ) {
                // Compute forward error bound (see Higham, 2002, sec. 3.5),
                // modified to include alpha, beta, and input C.
                // ||R_magma - R_ref||_p / (gamma_{K+2} |alpha| ||A||_p ||B||_p + 2 |beta| ||C||_p ) < eps/2.
                // This should work with p = 1, inf, fro, but numerical tests
                // show p = 1, inf are very spiky and sometimes exceed eps.
                // We use gamma_n = sqrt(n)*u instead of n*u/(1-n*u), since the
                // former accurately represents statistical average rounding.
                // We allow a slightly looser tolerance.
                
                // use LAPACK for R_ref
                blasf77_zaxpy( &sizeC, &c_neg_one, hC, &ione, hCdev, &ione );
                dev_error = lapackf77_zlange( "F", &M, &N, hCdev, &ldc, work )
                            / (sqrt(double(K+2))*fabs(alpha)*Anorm*Bnorm + 2*fabs(beta)*Cnorm);
                
                #ifdef HAVE_CUBLAS
                    blasf77_zaxpy( &sizeC, &c_neg_one, hC, &ione, hCmagma, &ione );
                    magma_error = lapackf77_zlange( "F", &M, &N, hCmagma, &ldc, work )
                            / (sqrt(double(K+2))*fabs(alpha)*Anorm*Bnorm + 2*fabs(beta)*Cnorm);
                    
                    bool okay = (magma_error < tol && dev_error < tol);
                    status += ! okay;
                    printf("%5lld %5lld %5lld   %7.2f (%7.2f)    %7.2f (%7.2f)   %7.2f (%7.2f)    %8.2e     %8.2e   %s\n",
                           (long long) M, (long long) N, (long long) K,
                           magma_perf,  1000.*magma_time,
                           dev_perf,    1000.*dev_time,
                           cpu_perf,    1000.*cpu_time,
                           magma_error, dev_error,
                           (okay ? "ok" : "failed"));
                #else
                    bool okay = (dev_error < tol);
                    status += ! okay;
                    printf("%5lld %5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)    %8.2e   %s\n",
                           (long long) M, (long long) N, (long long) K,
                           dev_perf,    1000.*dev_time,
                           cpu_perf,    1000.*cpu_time,
                           dev_error,
                           (okay ? "ok" : "failed"));
                #endif
            }
            else {
                #ifdef HAVE_CUBLAS
                    // use cuBLAS for R_ref (currently only with CUDA)
                    blasf77_zaxpy( &sizeC, &c_neg_one, hCdev, &ione, hCmagma, &ione );
                    magma_error = lapackf77_zlange( "F", &M, &N, hCmagma, &ldc, work )
                            / (sqrt(double(K+2))*fabs(alpha)*Anorm*Bnorm + 2*fabs(beta)*Cnorm);
                    
                    bool okay = (magma_error < tol);
                    status += ! okay;
                    printf("%5lld %5lld %5lld   %7.2f (%7.2f)    %7.2f (%7.2f)     ---   (  ---  )    %8.2e        ---    %s\n",
                           (long long) M, (long long) N, (long long) K,
                           magma_perf,  1000.*magma_time,
                           dev_perf,    1000.*dev_time,
                           magma_error,
                           (okay ? "ok" : "failed"));
                #else
                    printf("%5lld %5lld %5lld   %7.2f (%7.2f)     ---   (  ---  )       ---\n",
                           (long long) M, (long long) N, (long long) K,
                           dev_perf,    1000.*dev_time );
                #endif
            }
            
            magma_free_cpu( hA );
            magma_free_cpu( hB );
            magma_free_cpu( hC );
            magma_free_cpu( hCmagma  );
            magma_free_cpu( hCdev    );
            
            magma_free( dA );
            magma_free( dB );
            magma_free( dC );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
