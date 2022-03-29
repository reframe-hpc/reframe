#include <stdio.h>
#include <omp.h>
#include "mpi.h"

int main(int argc, char **argv) {
    int thread_safety;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &thread_safety);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
	int num_threads = omp_get_num_threads();
	printf("Hello from rank %d running omp thread %d/%d\n", rank, tid, num_threads);
    }

    return 0;
}
