#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <complex.h>
#include <fftw3.h>
#include <mpi.h>

fftw_complex *deri_temp_x, *deri_temp_y, *deri_temp_z;
fftw_plan plan_forward_x, plan_backward_x, plan_forward_y, plan_backward_y, plan_forward_z, plan_backward_z;

void init_derivatives(double *func, double *deri, int npx, int npy, int npz, int npy2, int npz2){
    int nnn;
    deri_temp_x = (fftw_complex *) malloc(npy*npz*(npx/2+1)*sizeof(fftw_complex));
    deri_temp_y = (fftw_complex *) malloc(npx*(npy/2+1)*sizeof(fftw_complex));
    deri_temp_z = (fftw_complex *) malloc(npx*npy2*(npz2/2+1)*sizeof(fftw_complex));
    nnn = npx;
    plan_forward_x = fftw_plan_many_dft_r2c(1, &nnn, npy*npz, func, &nnn, 1, npx, deri_temp_x, &nnn, 1, npx/2+1, FFTW_MEASURE+FFTW_UNALIGNED);
    nnn = npy;
    plan_forward_y = fftw_plan_many_dft_r2c(1, &nnn, npx, func, &nnn, npx, 1, deri_temp_y, &nnn, 1, npy/2+1, FFTW_MEASURE+FFTW_UNALIGNED);
    nnn = npz2;
    plan_forward_z = fftw_plan_many_dft_r2c(1, &nnn, npx*npy2, func, &nnn, npx*npy2, 1, deri_temp_z, &nnn, 1, npz2/2+1, FFTW_MEASURE+FFTW_UNALIGNED);
    nnn = npx;
    plan_backward_x = fftw_plan_many_dft_c2r(1, &nnn, npy*npz, deri_temp_x, &nnn, 1, npx/2+1, deri, &nnn, 1, npx, FFTW_MEASURE+FFTW_UNALIGNED);
    nnn = npy;
    plan_backward_y = fftw_plan_many_dft_c2r(1, &nnn, npx, deri_temp_y, &nnn, 1, npy/2+1, deri, &nnn, npx, 1, FFTW_MEASURE+FFTW_UNALIGNED);
    nnn = npz2;
    plan_backward_z = fftw_plan_many_dft_c2r(1, &nnn, npx*npy2, deri_temp_z, &nnn, 1, npz2/2+1, deri, &nnn, npx*npy2, 1, FFTW_MEASURE+FFTW_UNALIGNED);
}

void done_derivatives(){
    fftw_destroy_plan(plan_backward_z);
    fftw_destroy_plan(plan_backward_y);
    fftw_destroy_plan(plan_backward_x);
    fftw_destroy_plan(plan_forward_z);
    fftw_destroy_plan(plan_forward_y);
    fftw_destroy_plan(plan_forward_x);
    free(deri_temp_z);
    free(deri_temp_y);
    free(deri_temp_x);
}

void derivative_x1(double *func, double *deri, int npx, int npy, int npz){
    int i, jk;
    fftw_execute_dft_r2c(plan_forward_x, func, deri_temp_x);
    fftw_execute_dft_c2r(plan_backward_x, deri_temp_x, deri);
}

void derivative_y1(double *func, double *deri, int npx, int npy, int npz){
    int i, j, k;
    for (k = 0; k<npz; k++){
        fftw_execute_dft_r2c(plan_forward_y, func+k*npy*npx, deri_temp_y);
        fftw_execute_dft_c2r(plan_backward_y, deri_temp_y, deri+k*npy*npx);
    }
}

void derivative_z1(double *func, double *deri, int npx, int npy, int npz){
    int k, ij;
    fftw_execute_dft_r2c(plan_forward_z, func, deri_temp_z);
    fftw_execute_dft_c2r(plan_backward_z, deri_temp_z, deri);
}

int main(int argc, char *argv[]){
    int mpi_size, mpi_rank;
    int npoints, nproc, iter, withmpi;
    double *fvalue, *dvalue;
    int npx, npy, npz, npy2, npz2;
    int i, j, k;
    double my_time;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    if (argc != 5){
        if (mpi_rank == 0){
            printf("Usage: %s npoints nproc niter withmpi\n", argv[0]);
        }
        MPI_Finalize();
        exit(1);
    }
    npoints = atoi(argv[1]);
    nproc = atoi(argv[2]);
    iter = atoi(argv[3]);
    withmpi = atoi(argv[4]);
    if ((npoints <= 0) || (nproc <= 0) || (iter <= 0) || (withmpi < 0)){
        if (mpi_rank == 0){
            printf("%s: invalid input arguments\n", argv[0]);
        }
        MPI_Finalize();
        exit(1);
    }
    if (mpi_size != nproc){
        if (mpi_rank == 0){
            printf("number of MPI processes must be %d\n", nproc);
        }
        MPI_Finalize();
        exit(1);
    }
    npx = npy = npz2 = npoints;
    npz = npy2 = npoints/nproc;
    fvalue = (double *) malloc(npz*npy*npx*sizeof(double));
    dvalue = (double *) malloc(npz*npy*npx*sizeof(double));
    init_derivatives(fvalue, dvalue, npx, npy, npz, npy2, npz2);
    MPI_Barrier(MPI_COMM_WORLD);
    my_time = MPI_Wtime();
    for (i = 0; i<iter; i++){
        derivative_x1(fvalue, dvalue, npx, npy, npz);
        derivative_y1(fvalue, dvalue, npx, npy, npz);
        if (withmpi){
            MPI_Alltoall(fvalue, npx*npy2*npz, MPI_DOUBLE, dvalue, npx*npy2*npz, MPI_DOUBLE, MPI_COMM_WORLD);
        }
        derivative_z1(fvalue, dvalue, npx, npy, npz);
        if (withmpi){
            MPI_Alltoall(fvalue, npx*npy2*npz, MPI_DOUBLE, dvalue, npx*npy2*npz, MPI_DOUBLE, MPI_COMM_WORLD);
        }
    }
    my_time = MPI_Wtime()-my_time;
    if (mpi_rank == 0){
        MPI_Reduce(MPI_IN_PLACE, &my_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        printf("npoints: %d nproc: %d iter: %d withmpi: %d execution time: %e\n", npoints, nproc, iter, withmpi, my_time);
    }else{
        MPI_Reduce(&my_time, &my_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    }
    done_derivatives();
    MPI_Finalize();
    return(0);
}
