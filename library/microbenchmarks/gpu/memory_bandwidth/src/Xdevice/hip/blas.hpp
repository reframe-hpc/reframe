#ifndef __INCLUDED_CUDA_BLAS__
#define __INCLUDED_CUDA_BLAS__

#include <iostream>
#include <unistd.h>
#include "hip/hip_runtime_api.h"
#include "rocblas.h"

static inline void checkError(rocblas_status errorCode)
{
#  ifdef DEBUG
#    ifndef HOSTNAME_SIZE
#      define HOSTNAME_SIZE 80
#    endif

   if (errorCode != CUBLAS_STATUS_SUCCESS)
   {
      char nid[HOSTNAME_SIZE];
      gethostname(nid, HOSTNAME_SIZE);
      std::cerr << "[" << nid << "] A call to the rocBLAS API returned an error." << std::endl;
      exit(1);
   }
#  endif
}

using XblasHandle_t = rocblas_handle;
using XblasOperation_t = rocblas_operation;
XblasOperation_t XBLAS_OP_N = rocblas_operation_none;
XblasOperation_t XBLAS_OP_T = rocblas_operation_transpose;
XblasOperation_t XBLAS_OP_C = rocblas_operation_conjugate_transpose;

void XblasCreate(XblasHandle_t * handle)
{
  checkError( rocblas_create_handle(handle) );
}

void XblasDestroy(XblasHandle_t handle)
{
  checkError( rocblas_destroy_handle(handle) );
}

void XblasSetStream(XblasHandle_t handle, hipStream_t stream)
{
  checkError( rocblas_set_stream(handle, stream) );
}

auto XblasDgemm = rocblas_dgemm;
auto XblasSgemm = rocblas_sgemm;

#endif
