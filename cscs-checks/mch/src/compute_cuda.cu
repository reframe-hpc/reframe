#include  <stdio.h>
#include "cuda.h"

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

extern "C" {

__global__  void simple_add(float* a, float* b, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x; 
  if(i < n) {
    a[i] = a[i] + b[i];
  }
}

void cuda_kernel_no_copy(float* a, float* b, int n)
{
  const int THREADS_PER_BLOCK = 1;
  const int NUMBER_OF_BLOCKS = 10;

  cudaThreadSynchronize();
  simple_add<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(a, b, n);
  cudaThreadSynchronize();

  cudaCheckErrors("cuda error");
}

void cuda_kernel_with_copy(float* a, float* b, int n)
{
  const int THREADS_PER_BLOCK = 1;
  const int NUMBER_OF_BLOCKS = 10;

  float* d_a;
  float* d_b;
  cudaMalloc(&d_a, n*sizeof(float));
  cudaMalloc(&d_b, n*sizeof(float));
  cudaMemcpy(d_a, a, n*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, n*sizeof(float), cudaMemcpyHostToDevice);
    
  cudaThreadSynchronize();
  simple_add<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(d_a, d_b, n);
  cudaThreadSynchronize();

  cudaMemcpy(a, d_a, n*sizeof(float), cudaMemcpyDeviceToHost);
  
  cudaCheckErrors("cuda error");

}
};
