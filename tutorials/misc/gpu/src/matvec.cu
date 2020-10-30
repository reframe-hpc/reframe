#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

__global__ void matrix_multiply_kernel(double *matrix, double *vector_in,
                                       double *vector_out, long dim_mn){
    double out;
    long i, j;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i<dim_mn){
        out = 0.;
        for (j=0; j<dim_mn; j++){
            out += matrix[i*dim_mn+j] * vector_in[j];
        }
        vector_out[i] = out;
    }
}

int main (int argc, char *argv[]){
    double *matrix, *vector_in, *vector_out;
    double *matrix_device, *vector_in_device, *vector_out_device;
    long dim_mn, iterations, i, j, iteration;
    struct timeval start, stop, tdiff;

    if (argc!=3){
        fprintf(stderr, "%s matrixdimension numberofiterations\n", argv[0]);
        exit(1);
    }
    dim_mn = atoi(argv[1]);
    iterations = atoi(argv[2]);

    if ((dim_mn<1)||(iterations<1)){
        fprintf(stderr, "matrixdimension and numberofiterations must be "
                        "positive integers\n");
        exit(2);
    }

    matrix = (double*) malloc(dim_mn*dim_mn*sizeof(double));
    vector_in = (double*) malloc(dim_mn*sizeof(double));
    vector_out = (double*) malloc(dim_mn*sizeof(double));
    for (i=0; i<dim_mn; i++){
        for (j=0; j<dim_mn; j++){
            matrix[i*dim_mn+j] = 0.;
        }
    }
    for (i=0; i<dim_mn; i++){
        vector_in[i] = 1.;
        matrix[i*dim_mn+i] = 1.;
    }

    cudaMalloc((void**)&matrix_device, dim_mn*dim_mn*sizeof(double));
    cudaMalloc((void**)&vector_in_device, dim_mn*sizeof(double));
    cudaMalloc((void**)&vector_out_device, dim_mn*sizeof(double));
    cudaMemcpy(matrix_device, matrix, dim_mn*dim_mn*sizeof(double),
               cudaMemcpyHostToDevice);
    cudaMemcpy(vector_in_device, vector_in, dim_mn*sizeof(double),
               cudaMemcpyHostToDevice);

    gettimeofday(&start, NULL);
    for (iteration=0; iteration<iterations; iteration++){
        matrix_multiply_kernel<<<dim_mn-1/128+1, 128>>>(matrix_device,
            vector_in_device, vector_out_device, dim_mn);
        cudaMemcpy(vector_in_device, vector_out_device, dim_mn*sizeof(double),
                   cudaMemcpyDeviceToDevice);
    }
    cudaDeviceSynchronize();
    gettimeofday(&stop, NULL);

    cudaMemcpy(vector_out, vector_out_device, dim_mn*sizeof(double),
               cudaMemcpyDeviceToHost);
    cudaFree(vector_out_device);
    cudaFree(vector_in_device);
    cudaFree(matrix_device);

    timersub(&stop, &start, &tdiff);
    double execution_time = ((double)tdiff.tv_sec+((double)tdiff.tv_usec)/1000000.)/iterations;
    printf("time for single matrix vector multiplication %E s\n", execution_time);

    double l2_norm = 0.0;
    for (i=0; i < dim_mn; i++){
        l2_norm += vector_out[i] * vector_out[i];
    }

    printf("The L2 norm of the resulting vector is: %E\n", l2_norm);

    double gflops = (2.0*dim_mn*dim_mn/1.0e+09) / execution_time;
    printf("Performance: %f Gflop/s\n", gflops);

    free(vector_out);
    free(vector_in);
    free(matrix);

    return(0);
}
