// testing MPI_Init_thread
#include <iostream>
#include <mpi.h>
#include <stdio.h>
using namespace std;

int main(int argc, char **argv) {
  int rank, size, mpiversion, mpisubversion;
  int resultlen = -1, mpi_thread_supported = -1;
  char mpilibversion[MPI_MAX_LIBRARY_VERSION_STRING];

  // --------------------------------------------------------------------------
  // int MPI_Init_thread( int *argc, char ***argv, int required, int *provided )
  //
  // { MPI_THREAD_SINGLE}
  // Only one thread will execute.
  //
  // { MPI_THREAD_FUNNELED}
  // The process may be multi-threaded, but only the main thread will make MPI
  // calls (all MPI calls are funneled to the main thread).
  //
  // { MPI_THREAD_SERIALIZED}
  // The process may be multi-threaded, and multiple threads may make MPI calls,
  // but only one at a time: MPI calls are not made concurrently from two
  // distinct threads (all MPI calls are serialized).
  //
  // { MPI_THREAD_MULTIPLE}
  // Multiple threads may call MPI, with no restrictions.
  // --------------------------------------------------------------------------

#if defined(_MPI_THREAD_SINGLE)
  cout << "mpi_thread_required=MPI_THREAD_SINGLE ";
  int ev = MPI_Init_thread(0, 0, MPI_THREAD_SINGLE, &mpi_thread_supported);
#elif defined(_MPI_THREAD_FUNNELED)
  cout << "mpi_thread_required=MPI_THREAD_FUNNELED ";
  int ev = MPI_Init_thread(0, 0, MPI_THREAD_FUNNELED, &mpi_thread_supported);
#elif defined(_MPI_THREAD_SERIALIZED)
  cout << "mpi_thread_required=MPI_THREAD_SERIALIZED ";
  int ev = MPI_Init_thread(0, 0, MPI_THREAD_SERIALIZED, &mpi_thread_supported);
#elif defined(_MPI_THREAD_MULTIPLE)
  cout << "mpi_thread_required=MPI_THREAD_MULTIPLE ";
  int ev = MPI_Init_thread(0, 0, MPI_THREAD_MULTIPLE, &mpi_thread_supported);
#else
  cout << "mpi_thread_required=none ";
  int ev = MPI_Init(0, 0);
#endif

  switch (mpi_thread_supported) {
  case MPI_THREAD_SINGLE:
    cout << "mpi_thread_supported=MPI_THREAD_SINGLE";
    break;
  case MPI_THREAD_FUNNELED:
    cout << "mpi_thread_supported=MPI_THREAD_FUNNELED";
    break;
  case MPI_THREAD_SERIALIZED:
    cout << "mpi_thread_supported=MPI_THREAD_SERIALIZED";
    break;
  case MPI_THREAD_MULTIPLE:
    cout << "mpi_thread_supported=MPI_THREAD_MULTIPLE";
    break;
  default:
    cout << "mpi_thread_supported=UNKNOWN";
  }

  // Return the level of thread support provided by the MPI library:
  int mpi_thread_required = -1;
  MPI_Query_thread(&mpi_thread_required);
  switch (mpi_thread_supported) {
  case MPI_THREAD_SINGLE:
    cout << " mpi_thread_queried=MPI_THREAD_SINGLE " << mpi_thread_required
         << std::endl;
    break;
  case MPI_THREAD_FUNNELED:
    cout << " mpi_thread_queried=MPI_THREAD_FUNNELED " << mpi_thread_required
         << std::endl;
    break;
  case MPI_THREAD_SERIALIZED:
    cout << " mpi_thread_queried=MPI_THREAD_SERIALIZED " << mpi_thread_required
         << std::endl;
    break;
  case MPI_THREAD_MULTIPLE:
    cout << " mpi_thread_queried=MPI_THREAD_MULTIPLE " << mpi_thread_required
         << std::endl;
    break;
  default:
    cout << " mpi_thread_queried=UNKNOWN " << mpi_thread_required << std::endl;
  }

  MPI_Get_version(&mpiversion, &mpisubversion);
  MPI_Get_library_version(mpilibversion, &resultlen);
  printf("# MPI-%d.%d = %s", mpiversion, mpisubversion, mpilibversion);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  cout << "tid=0 out of 1 from rank " << rank << " out of " << size << "\n";

  // std::cout << " mpi_thread_queried=" << mpi_thread_required << std::endl;

  MPI_Finalize();

  return 0;
} /* end func main */
