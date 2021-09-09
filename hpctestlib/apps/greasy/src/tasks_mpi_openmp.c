#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef _MPI
#include "mpi.h"
#endif

#define STRINGIZE_MACRO(A) #A
#define STRINGIZE(A) STRINGIZE_MACRO(A)

int main(int argc, char *argv[])
{
    int size = 1;
    int rank = 0;
    int tid = 0;
    FILE *outputfile;

    if (argc > 1)
    {
        outputfile = fopen(argv[1], "a");
        if (outputfile == NULL) {
            fprintf(stdout, "Error. Unable to open output file %s", argv[1]);
        }
    }
    else
    {
        outputfile = stdout;
    }

#ifdef _MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

#ifdef _OPENMP
    #pragma omp parallel default(shared) private(tid)
#endif
    {
#ifdef _OPENMP
        int nthreads = omp_get_num_threads();
        tid = omp_get_thread_num();
#else
        int nthreads = 1;
#endif
        // sleep for as long as it is necessary to distinguish whether we are running in parallel or not
        // if GREASY is running correctly the test should take approximatelly this amount of time to run
        sleep(atoi(STRINGIZE(SLEEP_TIME)));
        fprintf(outputfile, "Hello, World from thread %d out of %d from process %d out of %d\n",
                tid, nthreads, rank, size);
    }

#ifdef _MPI
    MPI_Finalize();
#endif

    return 0;
}
