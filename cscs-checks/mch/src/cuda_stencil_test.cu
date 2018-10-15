#include "cuda.h"
#include <cassert>
#include <chrono>
#include <iostream>
#include <stdio.h>

#define N 800
#define TPB 16
#define RED_NB 50

#define BigN 4000
#define BigTPB 20

#define memN 200000000
#define memTPB 500

using namespace std;
using chrono_clock = std::chrono::high_resolution_clock;
using sec_dur = std::chrono::duration<double, std::ratio<1, 1>>;

__global__ void bigstencil(int* a) {
  unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
  a[i] = 7;
}

__global__ void indexfill(int* a) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  a[i] = i;
}

__global__ void dotProduct(int* a, int* b, int* c) {
  __shared__ int temp[N];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  temp[i] = a[i] * b[i];

  __syncthreads();
  if(i == 0) {
    int sum = 0;
    for(int i = 0; i < N; i++)
      sum += temp[i];

    *c = sum;
  }
}

__global__ void reduction(int* input, int* output) {
  __shared__ int tmp[TPB];

  tmp[threadIdx.x] = input[threadIdx.x + blockIdx.x * blockDim.x];

  __syncthreads();

  if(threadIdx.x < blockDim.x / 2)
    tmp[threadIdx.x] += tmp[threadIdx.x + blockDim.x / 2];

  __syncthreads();

  if(threadIdx.x < blockDim.x / 4)
    tmp[threadIdx.x] += tmp[threadIdx.x + blockDim.x / 4];

  __syncthreads();

  if(threadIdx.x < blockDim.x / 8)
    tmp[threadIdx.x] += tmp[threadIdx.x + blockDim.x / 8];

  __syncthreads();

  if(threadIdx.x == 0) {
    tmp[threadIdx.x] += tmp[threadIdx.x + 1];
    output[blockIdx.x] = tmp[threadIdx.x];
  }
}
__global__ void bigstencil(int* in, int* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  out[i] = in[i] + 2;
}

int main() {

  // CSCS: Initialize CUDA runtime outside measurement region
  cudaFree(0);
  chrono_clock::time_point time_start, time_end;
  time_start = chrono_clock::now();

  // ===------------------------------------------------===
  // Setup for the kernels
  // ===------------------------------------------------===
  // Indexing
  int h_indices[N];
  int* d_indices;
  cudaMalloc((void**)&d_indices, N * sizeof(int));

  // Dot Product
  int h_a[N], h_b[N], h_c;
  int *d_a, *d_b, *d_c;

  cudaMalloc((void**)&d_a, N * sizeof(int));
  cudaMalloc((void**)&d_b, N * sizeof(int));
  cudaMalloc((void**)&d_c, sizeof(int));

  for(int i = 0; i < N; i++) {
    h_a[i] = 1;
    h_b[i] = i;
  }
  cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);

  // Reduction
  int h_input[N], h_output[RED_NB];
  int *d_input, *d_output;

  for(int i = 0; i < N; i++)
    h_input[i] = 1;

  cudaMalloc((void**)&d_input, N * sizeof(int));
  cudaMalloc((void**)&d_output, RED_NB * sizeof(int));

  cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

  // Time consuming Stencil

  int *d_in, *d_out;
  int h_in[BigN], h_out[BigN];
  for(int i = 0; i < BigN; ++i) {
    h_in[i] = 1;
  }

  cudaMalloc((void**)&d_in, BigN * sizeof(int));
  cudaMalloc((void**)&d_out, BigN * sizeof(int));
  cudaMemcpy(d_in, h_in, BigN * sizeof(int), cudaMemcpyHostToDevice);

  // ===------------------------------------------------===
  // Running the kernels
  // ===------------------------------------------------===

  indexfill<<<N / TPB, TPB>>>(d_indices);

  dotProduct<<<1, N>>>(d_a, d_b, d_c);

  reduction<<<RED_NB, TPB>>>(d_input, d_output);

  bigstencil<<<BigN / BigTPB, BigTPB>>>(d_in, d_out);

  // ===------------------------------------------------===
  // Retrieving GPU Data
  // ===------------------------------------------------===
  // Indexing
  cudaMemcpy(&h_indices, d_indices, N * sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_indices);

  // Dot Product
  cudaMemcpy(&h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  // Reduction
  cudaMemcpy(h_output, d_output, RED_NB * sizeof(int), cudaMemcpyDeviceToHost);

  // Bigger stencil
  cudaMemcpy(h_out, d_out, BigN * sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(h_out);
  cudaFree(h_in);

  // do a lot of memcpy to see timing stability
  int nIterMemCpy = 1000000;
  int* mem_h;
  int mem_d[memN];
  for(int i = 0; i < memN; ++i) {
    mem_d[i] = i;
  }
  cudaMalloc((void**)&mem_h, memN * sizeof(int));
  for(int i = 0; i < nIterMemCpy; ++i) {
    cudaMemcpy(mem_d, mem_h, memN * sizeof(int), cudaMemcpyHostToDevice);
    bigstencil<<<memN / memTPB, memTPB>>>(mem_d);
    cudaMemcpy(&mem_h, mem_d, memN * sizeof(int), cudaMemcpyDeviceToHost);
    mem_d[i]++;
  }
  cudaFree(mem_h);

  time_end = chrono_clock::now();
  // ===------------------------------------------------===
  // Verify results
  // ===------------------------------------------------===
  bool testpass = true;

  // Indexing
  int sum = 0;
  for(int i = 0; i < N; ++i) {
    sum += h_indices[i];
  }
  int ref = N * (N - 1) / 2;
  testpass = testpass * (ref == sum);

  // Dot Product
  testpass = testpass * (ref == h_c);

  // Reduction
  for(int i = 0; i < RED_NB; ++i) {
    testpass = testpass * (16 == h_output[i]);
  }
  int res = 0;
  for(int i = 0; i < BigN; ++i) {
    res += h_out[i];
  }
  testpass = testpass * (3 * BigN == res);

  // ===-------------------------------------------------------===
  // Output
  // ===-------------------------------------------------------===
  std::string result = (testpass) ? "OK" : "ERROR";
  auto secs = std::chrono::duration_cast<sec_dur>(time_end - time_start);

  std::cout << "Result: " << result << std::endl;
  std::cout << "Timing: " << secs.count() << std::endl;

  return 0;
}
