/* requires console i/o on all mpi processes, so might fail, twr */
#include <stdio.h>  
#include <mpi.h>   

int main(int argc, char *argv[]) {
  int rank, size;
  int mpiversion, mpisubversion;
  int resultlen = -1;
  char mpilibversion[MPI_MAX_LIBRARY_VERSION_STRING];

  MPI::Init(argc, argv);

  rank = MPI::COMM_WORLD.Get_rank();
  size = MPI::COMM_WORLD.Get_size();
  printf("Hello World from thread %d out of %d from process %d out of %d\n",
       0, 1, rank, size);

  MPI_Get_version( &mpiversion, &mpisubversion );
  MPI_Get_library_version(mpilibversion, &resultlen);
  printf( "# MPI-%d.%d = %s\n", mpiversion, mpisubversion, mpilibversion);

  MPI::Finalize();

  return 0;
} /* end func main */
