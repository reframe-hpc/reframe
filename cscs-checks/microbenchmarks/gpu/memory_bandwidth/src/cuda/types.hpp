#ifndef __DEFINED_CUDA_TYPES__
#define __DEFINED_CUDA_TYPES__

#include<cuda.h>

using XStream_t = cudaStream_t;

struct XMemcpyBase {};

struct XMemcpyDefault : XMemcpyBase
{
  static const cudaMemcpyKind value;
};
const cudaMemcpyKind XMemcpyDefault::value = cudaMemcpyDefault; 

struct XMemcpyHostToDevice : XMemcpyBase
{
  static const cudaMemcpyKind value;
};
const cudaMemcpyKind XMemcpyHostToDevice::value = cudaMemcpyHostToDevice; 

struct XMemcpyDeviceToHost : XMemcpyBase
{
  static const cudaMemcpyKind value;
};
const cudaMemcpyKind XMemcpyDeviceToHost::value = cudaMemcpyDeviceToHost;
 
struct XMemcpyDeviceToDevice : XMemcpyBase
{
  static const cudaMemcpyKind value;
};
const cudaMemcpyKind XMemcpyDeviceToDevice::value = cudaMemcpyDeviceToDevice;
 
struct XMemcpyHostToHost : XMemcpyBase
{
  static const cudaMemcpyKind value;
};
const cudaMemcpyKind XMemcpyHostToHost::value = cudaMemcpyHostToHost;
 
#endif
