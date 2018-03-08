#include <iostream>
#include <mpi.h>

using std::cout;
using std::endl;

int main(int argc, char** argv){
    MPI_Status status;
    int mpi_size, mpi_rank;
    int host_data, *device_data;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    if (mpi_size!=2){
        if (mpi_rank==0) cout << "2 MPI ranks required" << endl;
        MPI_Finalize();
        return(1);
    }

    host_data = mpi_rank;
    cudaMalloc((void **)&device_data, sizeof(int));

    cudaMemcpy(device_data, &host_data, sizeof(int), cudaMemcpyHostToDevice);

    if (mpi_rank==0){
        MPI_Recv(device_data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, &status);
    }else{
        MPI_Send(device_data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    cudaMemcpy(&host_data, device_data, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_data);

    if (mpi_rank==0){
        cout << "Result : " << host_data << endl;
    }

    MPI_Finalize();

    return(0);
}
