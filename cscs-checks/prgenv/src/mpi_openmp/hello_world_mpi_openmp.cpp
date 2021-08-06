#include <stdio.h>
#include <omp.h>
#include "mpi.h"

int main(int argc, char *argv[])
{
  int size, rank;
  // int namelen;
  // char processor_name[MPI_MAX_PROCESSOR_NAME];
  int tid = 0;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // MPI_Get_processor_name(processor_name, &namelen);

  #pragma omp parallel default(shared) private(tid)
  {
    int nthreads = omp_get_num_threads();
    tid = omp_get_thread_num();
    printf("Hello, World from thread %d out of %d from process %d out of %d\n",
           tid, nthreads, rank, size);
  }

  MPI_Finalize();

  return 0;
}
