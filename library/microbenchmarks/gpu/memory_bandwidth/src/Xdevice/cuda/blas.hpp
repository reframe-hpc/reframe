#ifndef __INCLUDED_CUDA_BLAS__
#define __INCLUDED_CUDA_BLAS__

#include <iostream>
#include <unistd.h>
#include <cuda.h>
#include <cublas_v2.h>

static inline void checkError(cublasStatus_t errorCode)
{
#  ifdef DEBUG
#    ifndef HOSTNAME_SIZE
#      define HOSTNAME_SIZE 80
#    endif

   if (errorCode != CUBLAS_STATUS_SUCCESS)
   {
      char nid[HOSTNAME_SIZE];
      gethostname(nid, HOSTNAME_SIZE);
      std::cerr << "[" << nid << "] A call to the cuBLAS API returned an error." << std::endl;
      exit(1);
   }
#  endif
}

using XblasHandle_t = cublasHandle_t;
using XblasOperation_t = cublasOperation_t;
XblasOperation_t XBLAS_OP_N = CUBLAS_OP_N;
XblasOperation_t XBLAS_OP_T = CUBLAS_OP_T;
XblasOperation_t XBLAS_OP_C = CUBLAS_OP_C;

void XblasCreate(cublasHandle_t * handle)
{
  checkError( cublasCreate(handle) );
}

void XblasDestroy(cublasHandle_t handle)
{
  checkError( cublasDestroy(handle) );
}

void XblasSetStream(cublasHandle_t h, cudaStream_t s)
{
  checkError ( cublasSetStream(h, s) );
}

auto XblasDgemm = cublasDgemm;
auto XblasSgemm = cublasSgemm;

#endif
