#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

extern void dgemm_(char*, char*, int*, int*, int*, double*, double*,
                   int*, double*, int*, double*, double*, int*);

int main(int argc, char* argv[])
{
    double alpha = 1.2;
    double beta = 1.0e-3;
    double gflop;
    double time_avg;

    int m = 1024;
    int n = 2048;
    int k = 512;
    int LOOP_COUNT = 10;
    int i;

    char ta='N';
    char tb='N';

    struct timeval start_time, end_time, duration[LOOP_COUNT];


#ifndef HOST_NAME_MAX
#define HOST_NAME_MAX sysconf (_SC_HOST_NAME_MAX)
#endif

    char hostname[HOST_NAME_MAX];
    gethostname(hostname, sizeof(hostname));

    if (argc >= 2) m = atoi(argv[1]);
    if (argc >= 3) n = atoi(argv[2]);
    if (argc >= 4) k = atoi(argv[3]);
    if (argc >= 5) LOOP_COUNT = atoi(argv[4]);

    double perf[LOOP_COUNT];
    double time[LOOP_COUNT];
    double* A = (double*)malloc(sizeof(double)*m*k);
    double* B = (double*)malloc(sizeof(double)*k*n);
    double* C = (double*)malloc(sizeof(double)*m*n);

    printf("%s: Size of Matrix A(mxk)\t\t:\t%d x %d\n", hostname, m, k);
    printf("%s: Size of Matrix B(kxn)\t\t:\t%d x %d\n", hostname, k, n);
    printf("%s: Size of Matrix C(mxn)\t\t:\t%d x %d\n", hostname, m, n);
    printf("%s: LOOP COUNT\t\t\t:\t%d \n", hostname, LOOP_COUNT);
    printf("\n");

#pragma omp parallel for
    for (i=0; i<m*k ; ++i) A[i] = i%3+1;
#pragma omp parallel for
    for (i=0; i<k*n ; ++i) B[i] = i%3+1;
#pragma omp parallel for
    for (i=0; i<m*n ; ++i) C[i] = i%3+1;

    gflop = (2.0 * m * n * k + 3.0 * m * n) * 1E-9;

    /* CALL DGEMM ONCE TO INITIALIZE THREAD/BUFFER */
    dgemm_(&ta, &tb, &m, &n, &k, &alpha, A, &m, B, &k, &beta, C, &m);

    /* LOOP OVER DGEMM IN ORDER TO SMOOTHEN THE RESULTS */
    for (i=0; i<LOOP_COUNT; ++i)
    {
        gettimeofday(&start_time, NULL);
        dgemm_(&ta, &tb, &m, &n, &k, &alpha, A, &m, B, &k, &beta, C, &m);
        gettimeofday(&end_time,NULL);
        timersub(&end_time, &start_time, &duration[i]);
    }

    time_avg = 0.0;
    for (i=0; i<LOOP_COUNT; ++i)
    {
        time[i] = (duration[i].tv_sec * 1.e3 +
                   duration[i].tv_usec * 1.e-3) * 1.e-3;
        perf[i] = gflop / time[i];
        time_avg += time[i];
        printf("%s: Run %d \t\t\t\t:\t%.5f GFlops/sec\n", hostname, i, perf[i]);
    }


    printf("\n");
    printf("%s: Flops based on given dimensions\t:\t%.5f Gflops\n", hostname, gflop);
    printf("%s: Avg. performance               \t:\t%.5f Gflop/s\n", hostname, gflop * LOOP_COUNT / time_avg);
    printf("%s: Avg. time / DGEMM operation\t:\t%f secs \n", hostname, time_avg / LOOP_COUNT);
    printf("%s: Time for %d DGEMM operations\t:\t%f secs \n", hostname, LOOP_COUNT, time_avg);
    printf("\n");

    return 0;
}
