#ifndef __DEFINED_HIP_TYPES__
#define __DEFINED_HIP_TYPES__

#include <hip/hip_runtime.h>

using XStream_t = hipStream_t;

// Memcpy types
using XMemcpyKind = hipMemcpyKind;
XMemcpyKind XMemcpyHostToDevice = hipMemcpyHostToDevice;
XMemcpyKind XMemcpyDeviceToHost = hipMemcpyDeviceToHost;
XMemcpyKind XMemcpyDeviceToDevice = hipMemcpyDeviceToDevice;
XMemcpyKind XMemcpyHostToHost = hipMemcpyHostToHost;
XMemcpyKind XMemcpyDefault = hipMemcpyDefault;

// This flag is ignored by ROCm
#define XHostAllocMapped 0x02

#endif
