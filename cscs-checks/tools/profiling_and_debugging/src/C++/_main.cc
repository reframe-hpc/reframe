/*
 * This file is part of the Score-P software (http://www.score-p.org)
 *
 * Copyright (c) 2009-2011,
 *    RWTH Aachen University, Germany
 *    Gesellschaft fuer numerische Simulation mbH Braunschweig, Germany
 *    Technische Universitaet Dresden, Germany
 *    University of Oregon, Eugene, USA
 *    Forschungszentrum Juelich GmbH, Germany
 *    German Research School for Simulation Sciences GmbH, Juelich/Aachen, Germany
 *    Technische Universitaet Muenchen, Germany
 *
 * See the COPYING file in the package base directory for details.
 *
 */

/* *INDENT-OFF* */
//#include <config.h>

#include <mpi.h>
#include <cmath>
#include <iostream>
#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <string.h>
#include <errno.h>
#include <omp.h>
#include "jacobi.h"

//#include <stddef.h>
#ifndef offsetof
#define offsetof( TYPE, MEMBER ) ( ( size_t )&( ( TYPE* )0 )->MEMBER )
#endif

#define U( j, i ) data.afU[ ( ( j ) - data.iRowFirst ) * data.iCols + ( i ) ]
#define F( j, i ) data.afF[ ( ( j ) - data.iRowFirst ) * data.iCols + ( i ) ]

using namespace std;

// setting values, init mpi, omp etc
void
Init( JacobiData &data,
      int &       argc,
      char**      argv )
{
    /* MPI Initialization */
    const int required = MPI_THREAD_FUNNELED;
    int       provided;
    int       ierr = MPI_Init_thread( &argc, &argv, required, &provided );
    if ( ierr != MPI_SUCCESS )
    {
        printf( "Abort: MPI_Init_thread unsuccessful: %s\n", strerror( errno ) );
        MPI_Abort( MPI_COMM_WORLD, ENOSYS );
    }
    else if ( provided < required )
    {
        printf( "Warning: MPI_Init_thread only provided level %d<%d\n", provided, required );
    }

    MPI_Comm_rank( MPI_COMM_WORLD, &data.iMyRank );
    MPI_Comm_size( MPI_COMM_WORLD, &data.iNumProcs );

    if ( data.iMyRank == 0 )
    {
        int   version, subversion;
        int   ITERATIONS = 5;
        char* env        = getenv( "ITERATIONS" );
        if ( env )
        {
            int iterations = atoi( env );
            if ( iterations > 0 )
            {
                ITERATIONS = iterations;
            }
            else
            {
                printf( "Ignoring invalid ITERATIONS=%s!\n", env );
            }
        }

        MPI_Get_version( &version, &subversion );
        printf( "Jacobi %d MPI-%d.%d#%d process(es) with %d OpenMP-%u thread(s)/process\n",
                data.iNumProcs, version, subversion, provided, omp_get_max_threads(), _OPENMP );

// default medium
        data.iCols      = 2000;
        data.iRows      = 2000;
        data.fAlpha     = 0.8;
        data.fRelax     = 1.0;
        data.fTolerance = 1e-10;
        data.iIterMax   = ITERATIONS;
        cout << "\n-> matrix size: " << data.iCols << "x" << data.iRows
             << "\n-> alpha: " << data.fAlpha
             << "\n-> relax: " << data.fRelax
             << "\n-> tolerance: " << data.fTolerance
             << "\n-> iterations: " << data.iIterMax << endl << endl;
    }
    /* Build MPI Datastructure */
    MPI_Datatype typelist[ 8 ] =
    { MPI_INT,    MPI_INT,        MPI_INT,    MPI_INT, MPI_INT,
      MPI_DOUBLE, MPI_DOUBLE,     MPI_DOUBLE };

    int block_lengths[ 8 ];
    for ( int i = 0; i < 8; i++ )
    {
        block_lengths[ i ] = 1;
    }
    MPI_Aint displacements[ 8 ] =
    { ( MPI_Aint )offsetof( struct JacobiData, iRows ),
      ( MPI_Aint )offsetof( struct JacobiData, iCols ),
      ( MPI_Aint )offsetof( struct JacobiData, iRowFirst ),
      ( MPI_Aint )offsetof( struct JacobiData, iRowLast ),
      ( MPI_Aint )offsetof( struct JacobiData, iIterMax ),
      ( MPI_Aint )offsetof( struct JacobiData, fAlpha ),
      ( MPI_Aint )offsetof( struct JacobiData, fRelax ),
      ( MPI_Aint )offsetof( struct JacobiData, fTolerance ) };

    MPI_Datatype MPI_JacobiData;
#if ( MPI_VERSION >= 2 )
    MPI_Type_create_struct( 8, block_lengths, displacements, typelist, &MPI_JacobiData );
#else
    MPI_Type_struct( 8, block_lengths, displacements, typelist, &MPI_JacobiData );
#endif
    MPI_Type_commit( &MPI_JacobiData );

    /* Send input parameters to all procs */
    MPI_Bcast( &data, 1, MPI_JacobiData, 0, MPI_COMM_WORLD );

    /* calculate bounds for the task working area */
    data.iRowFirst = data.iMyRank * ( data.iRows - 2 ) / data.iNumProcs;
    if ( data.iMyRank == data.iNumProcs - 1 )
    {
        data.iRowLast = data.iRows - 1;
    }
    else
    {
        data.iRowLast = ( data.iMyRank + 1 ) * ( data.iRows - 2 ) / data.iNumProcs + 1;
    }

    data.afU = new double[ ( data.iRowLast - data.iRowFirst + 1 ) * data.iCols ];
    data.afF = new double[ ( data.iRowLast - data.iRowFirst + 1 ) * data.iCols ];

    // calculate dx and dy
    data.fDx = 2.0 / ( data.iCols - 1 );
    data.fDy = 2.0 / ( data.iRows - 1 );

    data.iIterCount = 0;

    return;
}

// final cleanup routines
void
Finish( JacobiData &data )
{
    delete[] data.afU;
    delete[] data.afF;

    MPI_Finalize();
    return;
}

// print result summary
void
PrintResults( const JacobiData &data )
{
    if ( data.iMyRank == 0 )
    {
        cout << " Number of iterations : " << data.iIterCount << endl;
        cout << " Residual             : " << data.fResidual << endl;
        cout << " Solution Error       : " << data.fError << endl;
        cout << " Elapsed Time         : " << data.fTimeStop - data.fTimeStart << endl;
        cout << " MFlops               : " << 0.000013 * data.iIterCount
        * ( data.iCols - 2 ) * ( data.iRows - 2 )
        / ( data.fTimeStop - data.fTimeStart ) << endl;
        cout << " SUCCESS " << endl;
    }
    return;
}

// Initializes matrix
// Assumes exact solution is u(x,y) = (1-x^2)*(1-y^2)
void
InitializeMatrix( JacobiData &data )
{
    /* Initilize initial condition and RHS */
#pragma omp parallel for
    for ( int j = data.iRowFirst; j <= data.iRowLast; j++ )
    {
        for ( int i = 0; i < data.iCols; i++ )
        {
            double xx = ( -1.0 + data.fDx * i );
            double yy = ( -1.0 + data.fDy * j );

            double xx2 = xx * xx;
            double yy2 = yy * yy;

            U( j, i ) = 0.0;
            F( j, i ) = -data.fAlpha * ( 1.0 - xx2 ) * ( 1.0 - yy2 )
                        + 2.0 * ( -2.0 + xx2 + yy2 );
        }
    }
}

// Checks error between numerical and exact solution
void
CheckError( JacobiData &data )
{
    double error = 0.0;

    for ( int j = data.iRowFirst; j <= data.iRowLast; j++ )
    {
        if ( ( data.iMyRank != 0 && j == data.iRowFirst ) ||
             ( data.iMyRank != data.iNumProcs - 1 && j == data.iRowLast ) )
        {
            continue;
        }

        for ( int i = 0; i < data.iCols; i++ )
        {
            double xx   = -1.0 + data.fDx * i;
            double yy   = -1.0 + data.fDy * j;
            double temp = U( j, i ) - ( 1.0 - xx * xx ) * ( 1.0 - yy * yy );
            error += temp * temp;
        }
    }

    data.fError = error;
    MPI_Reduce( &data.fError, &error, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD );
    data.fError = sqrt( error ) / ( data.iCols * data.iRows );
}


int
main( int    argc,
      char** argv )
{
    int        retVal = 0; // return value

    JacobiData myData;


    // sets default values or reads from stdin
    // inits MPI and OpenMP if needed
    // distribute MPI data, calculate MPI bounds
    Init( myData, argc, argv );

    if ( myData.afU && myData.afF )
    {
        // matrix init
        InitializeMatrix( myData );

        // starting timer
        myData.fTimeStart = MPI_Wtime();

        // running calculations
        Jacobi( myData );

        // stopping timer
        myData.fTimeStop = MPI_Wtime();

        // error checking
        CheckError( myData );

        // print result summary
        PrintResults( myData );
    }
    else
    {
        cout << " Memory allocation failed ...\n";
        retVal = -1;
    }

    // cleanup
    Finish( myData );
    return retVal;
}
