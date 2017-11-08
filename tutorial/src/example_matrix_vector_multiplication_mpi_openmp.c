#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>

int main (int argc, char *argv[]){
    double *matrix, *vector_in, *vector_out, out;
    long dim_mn, iterations, i, j, iteration;
    struct timeval start, stop, tdiff;
    int mpi_size, mpi_rank;
    double exec_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    if (argc!=3){
        if (mpi_rank==0){
            fprintf(stderr, "%s matrixdimension numberofiterations\n",
                    argv[0]);
        }
        exit(1);
    }
    dim_mn = atoi(argv[1]);
    iterations = atoi(argv[2]);

    if ((dim_mn<1)||(iterations<1)){
        if (mpi_rank==0){
            fprintf(stderr, "matrixdimension and numberofiterations must be "
                            "positive integers\n");
        }
        exit(2);
    }
    if (dim_mn%mpi_size!=0){
        if (mpi_rank==0){
            fprintf(stderr, "matrixdimension must be a multiple of number of "
                            "MPI ranks\n");
        }
        exit(3);
    }

    matrix = (double*) malloc(dim_mn/mpi_size*dim_mn*sizeof(double));
    vector_in = (double*) malloc(dim_mn*sizeof(double));
    vector_out = (double*) malloc(dim_mn/mpi_size*sizeof(double));
    for (i=0; i<dim_mn/mpi_size; i++){
        for (j=0; j<dim_mn; j++){
            matrix[i*dim_mn+j] = 0.;
        }
    }
    for (i=0; i<dim_mn; i++){
        vector_in[i] = 1.;
    }
    for (i=0; i<dim_mn/mpi_size; i++){
        matrix[i*dim_mn+i+mpi_rank*dim_mn/mpi_size] = 1.;
    }

    gettimeofday(&start, NULL);
    for (iteration=0; iteration<iterations; iteration++){
        #pragma omp parallel private(i, j, out)
        {
            #pragma omp for
            for (i=0; i<dim_mn/mpi_size; i++){
                out = 0.;
                for (j=0; j<dim_mn; j++){
                    out += matrix[i*dim_mn+j]*vector_in[j];
                }
                vector_out[i] = out;
            }
        }
        MPI_Allgather(vector_out, dim_mn/mpi_size, MPI_LONG, vector_in,
            dim_mn/mpi_size, MPI_LONG, MPI_COMM_WORLD);
    }
    gettimeofday(&stop, NULL);

    timersub(&stop, &start, &tdiff);
    
    exec_time = ((double)tdiff.tv_sec+((double)tdiff.tv_usec)/1000000.)
                /iterations;
    if (mpi_rank==0){
        MPI_Reduce(MPI_IN_PLACE, &exec_time, 1, MPI_DOUBLE, MPI_MAX, 0,
            MPI_COMM_WORLD);
        printf("time for single matrix vector multiplication %E s\n",
            exec_time);
    }else{
        MPI_Reduce(&exec_time, &exec_time, 1, MPI_DOUBLE, MPI_MAX, 0,
            MPI_COMM_WORLD);
    }

    double l2_norm = 0.0;
    for (i=0; i < dim_mn/mpi_size; i++){
        l2_norm += vector_out[i] * vector_out[i];
    }

    if (mpi_rank==0){
        MPI_Reduce(MPI_IN_PLACE, &l2_norm, 1, MPI_DOUBLE, MPI_SUM, 0,
            MPI_COMM_WORLD);
        printf("The L2 norm of the resulting vector is: %E\n", l2_norm);

    }else{
        MPI_Reduce(&l2_norm, &l2_norm, 1, MPI_DOUBLE, MPI_SUM, 0,
            MPI_COMM_WORLD);
    }

    free(vector_out);
    free(vector_in);
    free(matrix);

    MPI_Finalize();

    return(0);
}
