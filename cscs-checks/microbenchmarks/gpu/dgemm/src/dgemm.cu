/*
 * Basic DGEMM test
 *
 * Multiply two matrices of dimensions SIZE*SIZE filled with ones. Therefore,
 * all the elements of the resulting matrix will be just SIZE.
 */

#define SIZE 1024
#define REPEAT 30

#include <iostream>
#include <unistd.h>
#include <thread>
#include <mutex>
#include <vector>
#include <algorithm>
#include <functional>

#include "Xdevice/runtime.hpp"
#include "Xdevice/blas.hpp"


namespace kernels
{
  template<class T>
  __global__ void init_as_ones(T * arr, size_t size)
  {
    unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
    if (tid < size)
    {
      arr[tid] = (T)1.0;
    }
  }

  template<class T>
  __global__ void verify(T * arr, size_t size, int * err)
  {
    unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
    if (tid < size)
    {
      if (int(arr[tid]) != SIZE)
        atomicAdd(err, 1);
    }
  }
}

/*
 * This code uses a thread per device in the node.
 * For simplicity, we define the variables below as global.
 */

#define HOST_NAME_SIZE 128
char hostname[HOST_NAME_SIZE];
double tflops = SIZE*SIZE*SIZE*2.0 * 1E-12;
int totalErrors = 0;
std::mutex mtx;

#define BLOCK_SIZE 128
void dgemm(int device)
{
    XSetDevice(device);

    double * A;
    double * B;
    double * C;
    const double alpha = 1.0;
    const double beta = 0.0;

    XMalloc((void**)&A, sizeof(double)*SIZE*SIZE);
    XMalloc((void**)&B, sizeof(double)*SIZE*SIZE);
    XMalloc((void**)&C, sizeof(double)*SIZE*SIZE);

    kernels::init_as_ones<double><<<(SIZE*SIZE+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(A, SIZE*SIZE);
    kernels::init_as_ones<double><<<(SIZE*SIZE+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(B, SIZE*SIZE);
    XDeviceSynchronize();

    XStream_t stream;
    XStreamCreate(&stream);
    XblasHandle_t blas_handle;
    XblasCreate(&blas_handle);
    XblasSetStream(blas_handle, stream);

    // Warmup call
    XblasDgemm(blas_handle,
               XBLAS_OP_N, XBLAS_OP_N,
               SIZE, SIZE, SIZE,
               &alpha,
               (const double*)A, SIZE,
               (const double*)B, SIZE,
               &beta,
               C, SIZE);
    XDeviceSynchronize();

    // Time the execution
    XTimer t(stream);
    t.start();
    for (int i = 0; i < REPEAT; i++)
    {
        XblasDgemm(blas_handle,
                   XBLAS_OP_N, XBLAS_OP_N,
                   SIZE, SIZE, SIZE,
                   &alpha,
                   (const double*)A, SIZE,
                   (const double*)B, SIZE,
                   &beta,
                   C, SIZE);
    }

    // Calc the performance data in TFlops/sec
    double perf = tflops/(t.stop()/REPEAT/1000.0);

    XblasDestroy(blas_handle);
    XStreamDestroy(stream);

    // Verify that the final values of C are correct.
    int * err, h_err = 0;
    XMalloc((void**)&err, sizeof(int));
    XMemcpy( err, &h_err, sizeof(int), XMemcpyHostToDevice);
    kernels::verify<double><<<(SIZE+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(C, SIZE*SIZE, err);
    XMemcpy(&h_err, err, sizeof(int), XMemcpyDeviceToHost);
    {
      std::lock_guard<std::mutex> lg(mtx);
      totalErrors += h_err;

      // Print the performance results
      printf("[%s] GPU %d: %4.2f TF/s\n", hostname, device, (float)perf);
    }
    XFree(A);
    XFree(B);
    XFree(C);

}

int main(int argc, char **argv)
{

    gethostname(hostname, sizeof(hostname));

    int num_devices;
    XGetDeviceCount(&num_devices);

    // Print device count
    printf("[%s] Found %d device(s).\n", hostname, num_devices);

    // Create vector of threads.
    std::vector<std::thread> threads;

    // Do the dgemm for all devices in the node.
    for (int device = 0; device < num_devices; device++)
    {
        threads.push_back(std::thread(dgemm,device));
    }

    // Join all threads
    std::for_each(threads.begin(), threads.end(), std::mem_fn(&std::thread::join));

    // Test if there were any errors and print the test result.
    printf("[%s] Test %s\n", hostname, totalErrors == 0 ? "passed" : "failed");

    return 0;
}
