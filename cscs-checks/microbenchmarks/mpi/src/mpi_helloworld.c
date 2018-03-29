#include <stdio.h>
#include <string.h>
#include <mpi.h>

#define MSG_SIZE_MAX 255


int main(int argc, char **argv)
{
    const char *msg = "Hello, World!";
    char msg_buff[MSG_SIZE_MAX+1];
    size_t msg_len = strnlen(msg, MSG_SIZE_MAX);
    int rank, num_tasks, i;
    int dest = 0;
    int tag  = 0;
    int nr_correct = 0;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (num_tasks < 2) {
        fprintf(stderr, "Not enough tasks to run the test.\n");
        MPI_Finalize();
        return 1;
    }

    if (rank != 0) {
        strncpy(msg_buff, msg, MSG_SIZE_MAX);
        MPI_Send(msg_buff, msg_len+1, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
    } else {
        for (i = 1; i < num_tasks; i++) {
            MPI_Recv(msg_buff, msg_len+1, MPI_CHAR,
                     i, tag, MPI_COMM_WORLD, &status);
            if (!strncmp(msg, msg_buff, MSG_SIZE_MAX))
                nr_correct++;
        }
        printf("Received correct messages from %d processes.\n", nr_correct);
    }

    MPI_Finalize();
    return 0;
}
