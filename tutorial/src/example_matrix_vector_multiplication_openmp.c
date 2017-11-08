#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

int main (int argc, char *argv[]){
    double *matrix, *vector_in, *vector_out, out;
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

    gettimeofday(&start, NULL);
    for (iteration=0; iteration<iterations; iteration++){
        #pragma omp parallel private(i, j, out)
        {
            #pragma omp for
            for (i=0; i<dim_mn; i++){
                out = 0.;
                for (j=0; j<dim_mn; j++){
                    out += matrix[i*dim_mn+j]*vector_in[j];
                }
                vector_out[i] = out;
            }
            #pragma omp for
            for (i=0; i<dim_mn; i++){
                vector_in[i] = vector_out[i];
            }
        }
    }
    gettimeofday(&stop, NULL);

    timersub(&stop, &start, &tdiff);
    printf("time for single matrix vector multiplication %E s\n",
        ((double)tdiff.tv_sec+((double)tdiff.tv_usec)/1000000.)/iterations);

    double l2_norm = 0.0;
    for (i=0; i < dim_mn; i++){
        l2_norm += vector_out[i] * vector_out[i];
    }

    printf("The L2 norm of the resulting vector is: %E\n", l2_norm);

    free(vector_out);
    free(vector_in);
    free(matrix);

    return(0);
}
