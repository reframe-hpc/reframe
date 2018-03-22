#include <stdio.h>
#include <string.h>
#include <mpi.h>

int
main(int argc, char **argv)
{
  int rank;
  char msg[20];
  MPI_Status status;
  int num_proc;
  int i;
  int dest = 0;
  int tag = 0;

  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if(num_proc < 2)
  {
    MPI_Finalize();
    return 0;
  }

  if (rank!=0)
  {
    // printf ("I am slave. I am sending the message.\n");
    strcpy(msg,"Hello World!");
    MPI_Send(msg, 13, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
  }
  else
  {
    //printf ("I am master. I am receiving the message.\n");
    for(i=1; i < num_proc ; i++)
    {
      MPI_Recv(msg, 13, MPI_CHAR, i, tag, MPI_COMM_WORLD, &status);
      // printf ("The message is: %s\n", msg);
    }
    printf ("Received messages from %d processes.\n", num_proc);
  }

  MPI_Finalize();
}
