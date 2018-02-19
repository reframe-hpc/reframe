/*
* Copyright 2012 NVIDIA Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#ifdef USE_MPI
#include <mpi.h>
#endif //USE_MPI
#include <omp.h>
//#include <cuda_runtime.h>

/**
 * @brief Does one Jacobi iteration on A_d writing the results to
 *        Anew_d on all interior points of the domain.
 *
 * The Jacobi iteration solves the poission equation with diriclet
 * boundary conditions and a zero right hand side and returns the max
 * norm of the residue, executes synchronously.
 *
 * @param[in] A_d            pointer to device memory holding the 
 *                           solution of the last iteration including
 *                           boundary.
 * @param[out] Anew_d        pointer to device memory were the updates
 *                           solution should be written
 * @param[in] n              number of points in y direction
 * @param[in] m              number of points in x direction
 * @param[in,out] residue_d  pointer to a single float value in device
 * 			     memory, needed a a temporary storage to
 * 			     calculate the max norm of the residue.
 * @return		     the residue of the last iteration
 */
float launch_jacobi_kernel( const float* const A_d, float* const Anew_d, 
                            const int n, const int m, float* const residue_d );


/**
 * @brief Copies all inner points from Anew_d to A_d, executes
 *        asynchronously.
 *
 * @param[out] A_d    pointer to device memory holding the solution of
 * 		      the last iteration including boundary which
 * 		      should be updated with Anew_d
 * @param[in] Anew_d  pointer to device memory were the updated
 *                    solution is saved
 * @param[in] n       number of points in y direction
 * @param[in] m       number of points in x direction
 */
void launch_copy_kernel( float* const A_d, const float* const Anew_d,
                         const int n, const int m );

void launch_jacobi_kernel_async( const float* const A_d, float* const Anew_d, const int n, const int m, float* const residue_d );
float wait_jacobi_kernel( float* const residue_d );

int  handle_command_line_arguments(int argc, char** argv);
int  init_mpi(int argc, char** argv);
void init_host();
// void init_cuda(int);
void init_cuda( int , float* const , const float* const , float** const , float** const , const int , const int , const int , const int , float** const );
//        const float** const Anew_d, 
void finalize_mpi();
void finalize_host();
void finalize_cuda();

void start_timer();
void stop_timer();

void jacobi();

// CSCS ------------------
void set_device(int);
void memcpy_h2d(float* const, const float* const, 
        float**,float**,float**, 
        const int, const int, const int, const int);
//void memcpy_h2d(float* const, const float* const, 
//        float*,float*,float*, 
//        const int, const int, const int, const int);
void free_device(float*,float*,float*);
void get_info_device(char *, int);
void jacobi_memcpy( float* const ,const float* const , 
        float* const ,const float* const , 
        const int ,const int ,const int ,const int ,const int );
void jacobi_memcpy_final(
        float* const , float** const,
        const int , const int , const int , const int ,
        const int );
//void jacobi_memcpy_final(
//        float* const , const float* const ,
//        const int , const int , const int , const int ,
//        const int );
// CSCS ------------------

int n, m;
int n_global;

int   n_cpu;
float lb;
int   cpu_start, cpu_end;
int   gpu_start, gpu_end;

int rank=0;
int size=1;

int iter = 0;
#define CSCS_ITMAX _CSCS_ITMAX
int iter_max = CSCS_ITMAX;
// int iter_max = 1000;

double starttime;
double runtime;

const float pi = 3.1415926535897932384626f;
const float tol = 1.0e-5f;
float residue = 1.0f;

float* A;
float* Anew;
float* y00; // y0 conflicts with /usr/include/bits/mathcalls.h => use y00 instead

float* A_d;
float* Anew_d;
float* residue_d;

#ifdef USE_MPI
float* sendBuffer;
float* recvBuffer;
#endif //USE_MPI

/********************************/
/****         MAIN            ***/
/********************************/
int main(int argc, char** argv)
{
  int rank=0;
#ifdef USE_MPI
  if ( init_mpi(argc, argv) )
    {
      return 1;
    }
#endif //USE_MPI

  if ( handle_command_line_arguments(argc, argv) )
    {
      return -1;
    }

  init_host();
  // init_cuda(rank);
  init_cuda( rank, A, Anew, &A_d, &Anew_d, n,m,gpu_start,n_cpu, &residue_d );
    
#ifdef USE_MPI
  /* This has do be done after handling command line arguments */
  sendBuffer = (float*) malloc ( (m-2) * sizeof(float) );
  recvBuffer = (float*) malloc ( (m-2) * sizeof(float) );
  
  MPI_Barrier(MPI_COMM_WORLD);
#endif //USE_MPI
  
  start_timer();
  
  // Main calculation
  jacobi();
  
  stop_timer();
  
  finalize_cuda();
  finalize_host();
  
#ifdef USE_MPI
  finalize_mpi();
#endif //USE_MPI
}

/********************************/
/****        JACOBI           ***/
/********************************/
void jacobi()
{
  int i,j;

#ifdef USE_MPI
  int numprocs, rank, namelen;
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int iam = 0, np = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Get_processor_name(processor_name, &namelen);
  #pragma omp parallel default(shared) private(iam, np)
  {
    np = omp_get_num_threads();
    iam = omp_get_thread_num();
    printf("CSCS: thread %d / %d rank %d / %d on %s \n",
           iam, np, rank, numprocs, processor_name);
  }
#endif //USE_MPI

  while ( residue > tol && iter < iter_max )
    {
      residue = 0.0f;
      launch_jacobi_kernel_async( A_d, Anew_d, n-n_cpu, m, residue_d );

#pragma omp parallel
      {
        float my_residue = 0.f;

#pragma omp for nowait
        for( j = cpu_start; j < cpu_end; j++)
          {
            for( i = 1; i < m-1; i++ )
              {
                //Jacobi is Anew[j*m+i] = 1.0/1.0*(rhs[j*m+i] -
                //                        (                           -0.25f*A[(j-1) *m+ i]
                //                          -0.25f*A[j     *m+ (i+1)]                        -0.25f*A[j     *m+ (i-1)]
                //                                                    -0.25f*A[(j+1) *m+ i]));
                //rhs[j*m+i] == 0 for 0 <= j < n and 0 <= i < m
                // =>
                Anew[j *m+ i] = 0.25f * ( A[j     *m+ (i+1)] + A[j     *m+ (i-1)]
                                          +    A[(j-1) *m+ i]     + A[(j+1) *m+ i]);
                //Calculate residue of A
                //residue =
                //   rhs[j*m+i] -  (                           -0.25f*A[(j-1) *m+ i]
                //                   -0.25f*A[j     *m+ (i+1)] +1.00f*A[j     *m+ i]  -0.25f*A[j     *m+ (i-1)]
                //                                             -0.25f*A[(j+1) *m+ i]));
                //rhs[j*m+i] == 0 for 0 <= j < n and 0 <= i < m
                // =>
                //residue =  Anew[j *m+ i]-A[j *m + i]
                my_residue = fmaxf( my_residue, fabsf(Anew[j *m+ i]-A[j *m + i]));
              }
          }

#pragma omp critical
        {
          residue = fmaxf( my_residue, residue);
        }
      }

      residue = fmaxf( residue, wait_jacobi_kernel( residue_d ) );

#ifdef USE_MPI
      float globalresidue = 0.0f;
      MPI_Allreduce( &residue, &globalresidue, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD );
      residue = globalresidue;
#endif //USE_MPI

#ifdef USE_MPI
      if ( size == 2 )
        {
          MPI_Status status;
          if ( rank == 0 )
            {
              MPI_Sendrecv( Anew+(n-2)*m+1, m-2, MPI_FLOAT, 1, 0, A+(n-1)*m+1 , m-2, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, &status );
            } else {
              MPI_Sendrecv( Anew + 1*m+1, m-2, MPI_FLOAT, 0, 0, A+0*m+1, m-2, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status );
            }
        }
#endif //USE_MPI
		
// ----------------------------------------------------------------
      //cudaError_t error;
      if ( n_cpu > 0 && n_cpu < n ) {
        //bug? => no
        jacobi_memcpy(A,Anew, A_d,Anew_d, m, cpu_start, cpu_end, gpu_end, rank);
      }

      launch_copy_kernel(A_d,Anew_d,n-n_cpu,m);
// ----------------------------------------------------------------

#pragma omp parallel for
      for( j = cpu_start; j < cpu_end; j++)
        {
          for( i = 1; i < m-1; i++ )
            {
              A[j *m+ i] = Anew[j *m+ i];
            }
        }

      if(rank == 0 && iter % 100 == 0)
        printf("%5d, %0.6f\n", iter, residue);

      iter++;
    }

// ----------------------------------------------------------------
//bug? => no
  jacobi_memcpy_final(A, &A_d, m, n, n_cpu, cpu_end, rank);
// ----------------------------------------------------------------

}


/********************************/
/**** Initialization routines ***/
/********************************/


#ifdef USE_MPI
// init_mpi ---------------------------------------------------------------
int init_mpi(int argc, char** argv)
{
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if ( size != 1 && size != 2 )
    {
      if ( rank == 0)
        printf("Error: %s can only run with 1 or 2 processes!\n",argv[0]);
      return 1;
    }

  return 0;
}
#endif //USE_MPI


// init_host ---------------------------------------------------------------
void init_host()
{
  iter = 0;
  residue = 1.0f;

  // Index of first gpu element in HOST array
  gpu_start = rank==0 ? 1      : n*lb+1;
  // Index of last gpu element in DEVICE array
  gpu_end   = n - n*lb - 1;
  cpu_start = rank==0 ? n-n*lb : 1;
  cpu_end   = rank==0 ? n-1    : n*lb;

  A	= (float*) malloc( n*m * sizeof(float) );
  Anew	= (float*) malloc( n*m * sizeof(float) );
  y00 	= (float*) malloc( n   * sizeof(float) );
  int i,j;

#ifdef OMP_MEMLOCALITY
#pragma omp parallel for shared(A,Anew,m,n)
  for( j = cpu_start; j < cpu_end; j++)
    {
      for( i = 0; i < m; i++ )
        {
          Anew[j *m+ i] 	= 0.0f;
          A[j *m+ i] 		= 0.0f;
        }
    }
#endif //OMP_MEMLOCALITY

  memset(A, 0, n * m * sizeof(float));
  memset(Anew, 0, n * m * sizeof(float));

// set boundary conditions
#pragma omp parallel for
  for (i = 0; i < m; i++)
    {
      //Set top boundary condition only for rank 0 (rank responsible of the upper halve of the domain)
      if ( rank == 0 )
        A[0	    *m+ i] = 0.f;
      //Set bottom boundary condition only for rank 1 (rank responsible of the lower halve of the domain)
      if ( rank == 0 || size == 1 )
        A[(n-1) *m+ i] = 0.f;
    }

  int j_offset = 0;
  if ( size == 2 && rank == 1 )
    {
      j_offset = n-2;
    }
  for (j = 0; j < n; j++)
    {
      y00[j] = sinf(pi * (j_offset + j) / (n-1));
      A[j *m+ 0] = y00[j];
      A[j *m+ (m-1)] = y00[j]*expf(-pi);
    }

#pragma omp parallel for
  for (i = 1; i < m; i++)
    {
      if (rank == 0)
        Anew[0     *m+ i] = 0.f;
      if (rank == 1 || size == 1)
        Anew[(n-1) *m+ i] = 0.f;
    }
#pragma omp parallel for
  for (j = 1; j < n; j++)
    {
      Anew[j *m+ 0] = y00[j];
      Anew[j *m+ (m-1)] = y00[j]*expf(-pi);
    }
}

/* ok-ok-ok-ok-ok-ok-ok-ok-ok-ok-ok-ok-ok-ok-ok-ok
// init_cuda -------------------------------------------------------
// CSCS
#ifndef DEVS_PER_NODE
#define DEVS_PER_NODE 1  // Devices per node
#endif
void init_cuda()
{
  //int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
  int dev = rank % DEVS_PER_NODE;
  // printf("rk=%d dev=%d\n",rank,dev);

  cudaError_t error;
  error = cudaSetDevice(dev);
  cuda_check_status(error, __LINE__);

  //bug? ==> yes
  //bug: 
  // fix in place:
  //     for A_d: &A_d (=pointer to pointer to really pass, not only a copy) instead of A_d
  //     for A: A
  memcpy_h2d(A, Anew, &A_d, &Anew_d, &residue_d, n, m, gpu_start, n_cpu);
  //memcpy_h2d(A, Anew, &A_d, &Anew_d, &residue_d, n, m, gpu_start, n_cpu);
  //bug: memcpy_h2d(A, Anew, A_d, Anew_d, residue_d, n, m, gpu_start, n_cpu);

  //cudaMemcpy( A_d,    A+gpu_start-1,    m*(n-n_cpu)*sizeof(float), cudaMemcpyHostToDevice ); cuda_check_status(error, __LINE__);
  //cudaMemcpy( Anew_d, Anew+gpu_start-1, m*(n-n_cpu)*sizeof(float), cudaMemcpyHostToDevice ); cuda_check_status(error, __LINE__);

}
*/

// handle_command_line_arguments ---------------------------------------------------------------
int handle_command_line_arguments(int argc, char** argv)
{
  if ( argc > 4 )
    {
      if ( rank == 0)
        printf( "usage: %s [n] [m] [lb]\n", argv[0] );
      return -1;
    }

  n = 4096;
  if ( argc >= 2 )
    {
      n = atoi( argv[1] );
      if ( n <= 0 )
        {
          if ( rank == 0 )
            printf("Error: The number of rows (n=%i) needs to positive!\n",n);
          return 1;
        }
    }
  if ( size == 2 && n%2 != 0 )
    {
      if ( rank == 0)
        printf("Error: The number of rows (n=%i) needs to be devisible by 2 if two processes are used!\n",n);
      return 1;
    }
  m = n;
  if ( argc >= 3 )
    {
      m = atoi( argv[2] );
      if ( m <= 0 )
        {
          if ( rank == 0 )
            printf("Error: The number of columns (m=%i) needs to positive!\n",m);
          return 1;
        }
    }
  if ( argc == 4 )
    {
      lb = atof( argv[3] );
      if ( lb < 0.0f || lb > 1.0f )
    	{
          if ( rank == 0 )
            printf("Error: The load balancing factor (lb=%0.2f) needs to be in [0:1]!\n",lb);
          return -1;
    	}
    }
  
  n_global = n;

  if ( size == 2 )
    {
      //Do a domain decomposition and add one row for halo cells
      n = n/2 + 1;
    }

  n_cpu = lb*n;


// ---------------------
#ifndef DEVS_PER_NODE
#define DEVS_PER_NODE 1  // Devices per node
#endif
  if ( rank == 0 )
    {
      char gpu_str[256] = "";
      int dev = rank % DEVS_PER_NODE;
      get_info_device(gpu_str, dev);
      printf("=== /proc/driver/nvidia/version ===\n");
      static const char filename[] = "/proc/driver/nvidia/version";
      FILE *file = fopen ( filename, "r" );
      if ( file != NULL ) {
          fgets ( gpu_str, sizeof gpu_str, file ) ;
          fputs ( gpu_str, stdout ); 
      }
      fclose(file);

#pragma omp parallel
      {
#pragma omp master
        {
          if ( n_cpu > 0 )
            {
              printf("Jacobi relaxation Calculation: %d x %d mesh "
                     "with %d processes and %d threads + one %s for "
                     "each process.\n", 
                     n_global, m,size,omp_get_num_threads(),gpu_str);
            }
          else
            {
              printf("Jacobi relaxation Calculation: %d x %d mesh "
                     "with %d processes and one %s for each process.\n"
                     , n_global, m,size,gpu_str);
            }
          printf("\t%d of %d local rows are calculated on the "
                 "CPU to balance the load between the CPU and "
                 "the GPU. (%d iterations max)\n", 
                 n_cpu, n, iter_max);
        }
      }
    }
// ---------------------
  return 0;
}


/********************************/
/****  Finalization routines  ***/
/********************************/


#ifdef USE_MPI
// finalize_mpi ---------------------------------------------------------------
void finalize_mpi()
{
  free( recvBuffer );
  free( sendBuffer );

  MPI_Finalize();
  printf("SUCCESS\n");
}
#endif //USE_MPI


// finalize_host ---------------------------------------------------------------
void finalize_host()
{
  free(y00);
  free(Anew);
  free(A);
}

/********************************/
/****    Timing functions     ***/
/********************************/

// start_timer ---------------------------------------------------------------
void start_timer()
{
#ifdef USE_MPI
  starttime = MPI_Wtime();
#else
  starttime = omp_get_wtime();
#endif //USE_MPI
}


// stop_timer ---------------------------------------------------------------
void stop_timer()
{
#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
  runtime = MPI_Wtime() - starttime;
#else
  runtime = omp_get_wtime() - starttime;
#endif //USE_MPI

  if (rank == 0)
    printf(" total: %f s\n", runtime);
}

