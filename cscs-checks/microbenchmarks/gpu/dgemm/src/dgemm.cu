/*
 * Basic DGEMM test
 *
 * Multiply two matrices of dimensions SIZE*SIZE filled with ones. Therefore,
 * all the elements of the resulting matrix will be just SIZE.
 */

#define SIZE 2048
#define REPEAT 30

#include <iostream>

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

#define BLOCK_SIZE 128
int main(int argc, char **argv)
{
    double * A;
    double * B;
    double * C;

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
    const double alpha = 1.0;
    const double beta = 0.0;

    // Warmup call
    XblasDgemm(blas_handle,
               XBLAS_OP_N, XBLAS_OP_N,
               SIZE, SIZE, SIZE,
               &alpha,
               (const double*)A, SIZE,
               (const double*)B, SIZE,
               &beta,
               C, SIZE);

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

    std::cout << "Elapsed time: " << t.stop() << std::endl;

    XblasDestroy(blas_handle);
    XStreamDestroy(stream);

    int * err, h_err = 0;
    XMalloc((void**)&err, sizeof(int));
    XMemset(&err, 0, sizeof(int));
    kernels::verify<double><<<(SIZE+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(C, SIZE*SIZE, err);

    XDeviceSynchronize();
    XMemcpy(&h_err, err, sizeof(int), XMemcpyDeviceToHost);
    XDeviceSynchronize();
    std::cout << "Number of errors:" << h_err << std::endl;

    return 0;
}
