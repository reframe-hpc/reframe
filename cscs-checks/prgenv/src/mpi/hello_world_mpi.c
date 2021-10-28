#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
   int rank, size;
   int mpiversion, mpisubversion;
   int resultlen = -1;
   char mpilibversion[MPI_MAX_LIBRARY_VERSION_STRING];
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   printf("Hello, World from thread %d out of %d from process %d out of %d\n",
       0, 1, rank, size);

   MPI_Get_version( &mpiversion, &mpisubversion );
   MPI_Get_library_version(mpilibversion, &resultlen);
   printf( "# MPI-%d.%d = %s\n", mpiversion, mpisubversion, mpilibversion);

   MPI_Finalize();
   return 0;
}
