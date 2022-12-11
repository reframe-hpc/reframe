#ifndef __DEFINED_CUDA_TYPES__
#define __DEFINED_CUDA_TYPES__

#include<cuda.h>

using XStream_t = cudaStream_t;

// Memcpy types
using XMemcpyKind = cudaMemcpyKind;
XMemcpyKind XMemcpyHostToDevice = cudaMemcpyHostToDevice;
XMemcpyKind XMemcpyDeviceToHost = cudaMemcpyDeviceToHost;
XMemcpyKind XMemcpyDeviceToDevice = cudaMemcpyDeviceToDevice;
XMemcpyKind XMemcpyHostToHost = cudaMemcpyHostToHost;
XMemcpyKind XMemcpyDefault = cudaMemcpyDefault;

#define XHostAllocMapped cudaHostAllocMapped

#endif
