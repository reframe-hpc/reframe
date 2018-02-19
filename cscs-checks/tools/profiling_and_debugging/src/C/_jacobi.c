/*
 ******************************************************************
 * Subroutine HelmholtzJ
 * Solves poisson equation on rectangular grid assuming :
 * (1) Uniform discretization in each direction, and
 * (2) Dirichlect boundary conditions
 *
 * Jacobi method is used in this routine
 *
 * Input : n,m   Number of grid points in the X/Y directions
 *         dx,dy Grid spacing in the X/Y directions
 *         alpha Helmholtz eqn. coefficient
 *         omega Relaxation factor
 *         f(n,m) Right hand side function
 *         u(n,m) Dependent variable/Solution
 *         tolerance Tolerance for iterative solver
 *         maxit  Maximum number of iterations
 *
 * Output : u(n,m) - Solution
 *****************************************************************
 */

#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "jacobi.h"

#define U( j, i ) afU[ ( ( j ) - data->iRowFirst ) * data->iCols + ( i ) ]
#define F( j, i ) afF[ ( ( j ) - data->iRowFirst ) * data->iCols + ( i ) ]
#define UOLD( j, i ) uold[ ( ( j ) - data->iRowFirst ) * data->iCols + ( i ) ]

extern void
ExchangeJacobiMpiData( struct JacobiData* data,
                       double*            uold );

void
Jacobi( struct JacobiData* data )
{
    /*use local pointers for performance reasons*/
    double* afU, * afF;
    int     i, j;
    double  fLRes;

    double  ax, ay, b, residual, tmpResd;

    double* uold = ( double* )malloc(
        data->iCols * ( data->iRowLast - data->iRowFirst + 1 ) * sizeof( double ) );
    afU = data->afU;
    afF = data->afF;

    if ( uold )
    {
        ax       = 1.0 / ( data->fDx * data->fDx );   /* X-direction coef */
        ay       = 1.0 / ( data->fDy * data->fDy );   /* Y_direction coef */
        b        = -2.0 * ( ax + ay ) - data->fAlpha; /* Central coeff */
        residual = 10.0 * data->fTolerance;

        while ( data->iIterCount < data->iIterMax && residual > data->fTolerance )
        {
            residual = 0.0;

            /* copy new solution into old */
            ExchangeJacobiMpiData( data, uold );
#pragma omp parallel
            {
                /* compute stencil, residual and update */
#pragma omp for private(j, i, fLRes) reduction(+:residual)
                for ( j = data->iRowFirst + 1; j <= data->iRowLast - 1; j++ )
                {
                    for ( i = 1; i <= data->iCols - 2; i++ )
                    {
                        fLRes = ( ax * ( UOLD( j, i - 1 ) + UOLD( j, i + 1 ) )
                                  + ay * ( UOLD( j - 1, i ) + UOLD( j + 1, i ) )
                                  +  b * UOLD( j, i ) - F( j, i ) ) / b;

                        /* update solution */
                        U( j, i ) = UOLD( j, i ) - data->fRelax * fLRes;

                        /* accumulate residual error */
                        residual += fLRes * fLRes;
                    }
                }
            } /* end omp parallel */
            tmpResd = residual;
            MPI_Allreduce(
                &tmpResd, &residual, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );

            /* error check */
            ( data->iIterCount )++;
            residual = sqrt( residual ) / ( data->iCols * data->iRows );
        } /* while */

        data->fResidual = residual;
        free( uold );
    }
    else
    {
        fprintf( stderr, "Error: cant allocate memory\n" );
        Finish( data );
        exit( 1 );
    }
}

void
ExchangeJacobiMpiData( struct JacobiData* data,
                       double*            uold )
{
    MPI_Request request[ 4 ];
    MPI_Status  status[ 4 ];
    double*     afU, * afF;
    int         iReqCnt = 0;
    int         i, j;
    const int   iTagMoveLeft  = 10;
    const int   iTagMoveRight = 11;
    afU = data->afU;
    afF = data->afF;

    if ( data->iMyRank != 0 )
    {
        /*  receive stripe mlo from left neighbour blocking */
        MPI_Irecv( &UOLD( data->iRowFirst, 0 ), data->iCols, MPI_DOUBLE,
                   data->iMyRank - 1, iTagMoveRight, MPI_COMM_WORLD,
                   &request[ iReqCnt ] );
        iReqCnt++;
    }

    if ( data->iMyRank != data->iNumProcs - 1 )
    {
        /* receive stripe mhi from right neighbour blocking */
        MPI_Irecv( &UOLD( data->iRowLast, 0 ), data->iCols, MPI_DOUBLE,
                   data->iMyRank + 1, iTagMoveLeft, MPI_COMM_WORLD,
                   &request[ iReqCnt ] );
        iReqCnt++;
    }

    if ( data->iMyRank != data->iNumProcs - 1 )
    {
        /* send stripe mhi-1 to right neighbour async */
        MPI_Isend( &U( data->iRowLast - 1, 0 ), data->iCols, MPI_DOUBLE,
                   data->iMyRank + 1, iTagMoveRight, MPI_COMM_WORLD,
                   &request[ iReqCnt ] );
        iReqCnt++;
    }

    if ( data->iMyRank != 0 )
    {
        /* send stripe mlo+1 to left neighbour async */
        MPI_Isend( &U( data->iRowFirst + 1, 0 ), data->iCols, MPI_DOUBLE,
                   data->iMyRank - 1, iTagMoveLeft, MPI_COMM_WORLD,
                   &request[ iReqCnt ] );
        iReqCnt++;
    }
#pragma omp parallel for private(j, i)
    for ( j = data->iRowFirst + 1; j <= data->iRowLast - 1; j++ )
    {
        for ( i = 0; i < data->iCols; i++ )
        {
            UOLD( j, i ) = U( j, i );
        }
    }

    MPI_Waitall( iReqCnt, request, status );
}
