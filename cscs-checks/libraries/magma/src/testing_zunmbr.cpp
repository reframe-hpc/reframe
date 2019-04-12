/*
    -- MAGMA (version 2.4.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date June 2018

       @author Mark Gates
       @precisions normal z -> c d s
*/
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// includes, project
#include "flops.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#include "magma_operators.h"
#include "testings.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zunmbr
*/
int main( int argc, char** argv )
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();
    
    real_Double_t   gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    double Cnorm, error, dwork[1];
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magma_int_t ione = 1;
    magma_int_t m, n, k, mi, ni, mm, nn, nq, size, info;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t nb, ldc, lda, lwork, lwork_max;
    magmaDoubleComplex *C, *R, *A, *work, *tau, *tauq, *taup;
    double *d, *e;
    int status = 0;
    
    magma_opts opts;
    opts.parse_opts( argc, argv );
    
    // need slightly looser bound (60*eps instead of 30*eps) for some tests
    opts.tolerance = max( 60., opts.tolerance );
    double tol = opts.tolerance * lapackf77_dlamch("E");
    
    // test all combinations of input parameters
    magma_vect_t  vect [] = { MagmaQ,          MagmaP       };
    magma_side_t  side [] = { MagmaLeft,       MagmaRight   };
    magma_trans_t trans[] = { Magma_ConjTrans, MagmaNoTrans };

    printf("%%   M     N     K   vect side   trans   CPU Gflop/s (sec)   GPU Gflop/s (sec)   ||R||_F / ||QC||_F\n");
    printf("%%==============================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
      for( int ivect = 0; ivect < 2; ++ivect ) {
      for( int iside = 0; iside < 2; ++iside ) {
      for( int itran = 0; itran < 2; ++itran ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            m = opts.msize[itest];
            n = opts.nsize[itest];
            k = opts.ksize[itest];
            nb  = magma_get_zgebrd_nb( m, n );
            ldc = m;
            // A is mm x nn == nq x k (vect=Q) or k x nq (vect=P)
            // where nq=m (left) or nq=n (right)
            nq  = (side[iside] == MagmaLeft ? m  : n );
            mm  = (vect[ivect] == MagmaQ    ? nq : k );
            nn  = (vect[ivect] == MagmaQ    ? k  : nq);
            lda = mm;
            
            // MBR calls either MQR or MLQ in various ways
            if ( vect[ivect] == MagmaQ ) {
                if ( nq >= k ) {
                    gflops = FLOPS_ZUNMQR( m, n, k, side[iside] ) / 1e9;
                }
                else {
                    if ( side[iside] == MagmaLeft ) {
                        mi = m - 1;
                        ni = n;
                    }
                    else {
                        mi = m;
                        ni = n - 1;
                    }
                    gflops = FLOPS_ZUNMQR( mi, ni, nq-1, side[iside] ) / 1e9;
                }
            }
            else {
                if ( nq > k ) {
                    gflops = FLOPS_ZUNMLQ( m, n, k, side[iside] ) / 1e9;
                }
                else {
                    if ( side[iside] == MagmaLeft ) {
                        mi = m - 1;
                        ni = n;
                    }
                    else {
                        mi = m;
                        ni = n - 1;
                    }
                    gflops = FLOPS_ZUNMLQ( mi, ni, nq-1, side[iside] ) / 1e9;
                }
            }
            
            // workspace for gebrd is (mm + nn)*nb
            // workspace for unmbr is m*nb or n*nb, depending on side
            lwork_max = max( (mm + nn)*nb, max( m*nb, n*nb ));
            // this rounds it up slightly if needed to agree with lwork query below
            lwork_max = magma_int_t( real( magma_zmake_lwork( lwork_max )));
            
            TESTING_CHECK( magma_zmalloc_cpu( &C,    ldc*n ));
            TESTING_CHECK( magma_zmalloc_cpu( &R,    ldc*n ));
            TESTING_CHECK( magma_zmalloc_cpu( &A,    lda*nn ));
            TESTING_CHECK( magma_zmalloc_cpu( &work, lwork_max ));
            TESTING_CHECK( magma_dmalloc_cpu( &d,    min(mm,nn) ));
            TESTING_CHECK( magma_dmalloc_cpu( &e,    min(mm,nn) ));
            TESTING_CHECK( magma_zmalloc_cpu( &tauq, min(mm,nn) ));
            TESTING_CHECK( magma_zmalloc_cpu( &taup, min(mm,nn) ));
            
            // C is full, m x n
            size = ldc*n;
            lapackf77_zlarnv( &ione, ISEED, &size, C );
            lapackf77_zlacpy( "Full", &m, &n, C, &ldc, R, &ldc );
            
            // A is mm x nn
            magma_generate_matrix( opts, mm, nn, A, lda );
            
            // compute BRD factorization to get Householder vectors in A, tauq, taup
            //lapackf77_zgebrd( &mm, &nn, A, &lda, d, e, tauq, taup, work, &lwork_max, &info );
            magma_zgebrd( mm, nn, A, lda, d, e, tauq, taup, work, lwork_max, &info );
            if (info != 0) {
                printf("magma_zgebrd returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            if ( vect[ivect] == MagmaQ ) {
                tau = tauq;
            } else {
                tau = taup;
            }
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            cpu_time = magma_wtime();
            lapackf77_zunmbr( lapack_vect_const( vect[ivect] ),
                              lapack_side_const( side[iside] ),
                              lapack_trans_const( trans[itran] ),
                              &m, &n, &k,
                              A, &lda, tau, C, &ldc, work, &lwork_max, &info );
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gflops / cpu_time;
            if (info != 0) {
                printf("lapackf77_zunmbr returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            // query for workspace size
            lwork = -1;
            magma_zunmbr( vect[ivect], side[iside], trans[itran],
                          m, n, k,
                          A, lda, tau, R, ldc, work, lwork, &info );
            if (info != 0) {
                printf("magma_zunmbr (lwork query) returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            lwork = (magma_int_t) MAGMA_Z_REAL( work[0] );
            if ( lwork < 0 || lwork > lwork_max ) {
                printf("Warning: optimal lwork %lld > allocated lwork_max %lld\n", (long long) lwork, (long long) lwork_max );
                lwork = lwork_max;
            }
            
            gpu_time = magma_wtime();
            magma_zunmbr( vect[ivect], side[iside], trans[itran],
                          m, n, k,
                          A, lda, tau, R, ldc, work, lwork, &info );
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0) {
                printf("magma_zunmbr returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            /* =====================================================================
               compute relative error |QC_magma - QC_lapack| / |QC_lapack|
               =================================================================== */
            size = ldc*n;
            blasf77_zaxpy( &size, &c_neg_one, C, &ione, R, &ione );
            Cnorm = lapackf77_zlange( "Fro", &m, &n, C, &ldc, dwork );
            error = lapackf77_zlange( "Fro", &m, &n, R, &ldc, dwork ) / (magma_dsqrt(m*n) * Cnorm);
            
            printf( "%5lld %5lld %5lld   %c   %4c   %5c   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %s\n",
                    (long long) m, (long long) n, (long long) k,
                    lapacke_vect_const( vect[ivect] ),
                    lapacke_side_const( side[iside] ),
                    lapacke_trans_const( trans[itran] ),
                    cpu_perf, cpu_time, gpu_perf, gpu_time,
                    error, (error < tol ? "ok" : "failed") );
            status += ! (error < tol);
            
            magma_free_cpu( C );
            magma_free_cpu( R );
            magma_free_cpu( A );
            magma_free_cpu( work );
            magma_free_cpu( d );
            magma_free_cpu( e );
            magma_free_cpu( taup );
            magma_free_cpu( tauq );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
      }}}  // end ivect, iside, itran
      printf( "\n" );
    }
    
    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
