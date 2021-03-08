#ifndef __INCLUDED_CUDA_TOOLS__
#define __INCLUDED_CUDA_TOOLS__

#include <iostream>
#include <unistd.h>
#include <nvml.h>

/*
 * NVML - SMI tools
 */

static inline void nvmlCheck(nvmlReturn_t err)
{
# ifdef DEBUG
  if (err != NVML_SUCCESS)
  {
    std::cerr << "Call to the nvml API failed!" << std::endl;
    exit(1);
  }
# endif
}

Smi::Smi()
{
  if (!(this->smiIsActive))
  {
    nvmlCheck( nvmlInit() );
    this->smiIsActive = 1;
    nvmlCheck( nvmlDeviceGetCount(&numberOfDevices) );
  }

  this->activeSmiInstances += 1;
}

void Smi::setCpuAffinity(int id)
{
  checkGpuIdIsSensible(id);

  nvmlDevice_t device;
  nvmlCheck( nvmlDeviceGetHandleByIndex(id, &device) );
  nvmlCheck( nvmlDeviceSetCpuAffinity(device) );
}

void Smi::getGpuTemp(int id, float * temp) const
{
  // Check that the gpu id is sensible
  checkGpuIdIsSensible(id);

  // Get device handle
  nvmlDevice_t device;
  nvmlCheck( nvmlDeviceGetHandleByIndex(id, &device) );

  // Get the temperature
  unsigned int temperature;
  nvmlCheck( nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temperature) );
  *temp = (float)temperature;
}

void Smi::getDeviceMemorySize(int id, size_t * sMem) const
{
  nvmlDevice_t device;
  nvmlCheck( nvmlDeviceGetHandleByIndex(id, &device) );
  nvmlMemory_t memory;
  nvmlCheck( nvmlDeviceGetMemoryInfo(device, &memory) );
  *sMem = memory.total;
}

void Smi::getDeviceAvailMemorySize(int id, size_t * sMem) const
{
  nvmlDevice_t device;
  nvmlCheck( nvmlDeviceGetHandleByIndex(id, &device) );
  nvmlMemory_t memory;
  nvmlCheck( nvmlDeviceGetMemoryInfo(device, &memory) );
  *sMem = memory.free;
}

Smi::~Smi()
{
  this->activeSmiInstances -= 1;
  if (!(this->activeSmiInstances))
  {
    nvmlCheck( nvmlShutdown() );
    this->smiIsActive = 0;
  }
}


/*
 * ASM tools
 */

__device__ __forceinline__ uint32_t XClock()
{
  // Clock counter
  uint32_t x;
  asm volatile ("mov.u32 %0, %%clock;" : "=r"(x) :: "memory");
  return x;
}

__device__ __forceinline__ uint64_t XClock64()
{
  // Clock counter
  uint64_t x;
  asm volatile ("mov.u64 %0, %%clock64;" : "=l"(x) :: "memory");
  return x;
}

__device__ __forceinline__ uint32_t XSyncClock()
{
  // Clock counter with a preceeding barrier.
  uint32_t x;
  asm volatile ("bar.sync	0;\n\t"
                "mov.u32 %0, %%clock;" : "=r"(x) :: "memory");
  return x;
}

__device__ __forceinline__ uint64_t XSyncClock64()
{
  // Clock counter with a preceeding barrier.
  uint64_t x;
  asm volatile ("bar.sync	0;\n\t"
                "mov.u64 %0, %%clock64;" : "=l"(x) :: "memory");
  return x;
}


template<class T = uint32_t>
class __XClocks
{
  /*
   * XClocks timer tool
   * Tracks the number of clock cycles between a call to the start
   * and end member functions.
   */
public:
  T startClock;
  __device__ void start()
  {
    startClock = XSyncClock();
  }
  __device__ T end()
  {
    return XClock() - startClock;
  }
};

template<>
void __XClocks<uint64_t>::start()
{
  this->startClock = XSyncClock64();
}

template<>
uint64_t __XClocks<uint64_t>::end()
{
  return XClock64() - this->startClock;
}

using XClocks64 = __XClocks<uint64_t>;
using XClocks = __XClocks<>;


template<class T>
__device__ T XClockLatency()
{
  uint64_t start = XClock64();
  uint64_t end   = XClock64();
  return (T)(end-start);
}

__device__ __forceinline__ int __smId()
{
  // SM ID
  uint32_t x;
  asm volatile ("mov.u32 %0, %%smid;" : "=r"(x) :: "memory");
  return (int)x;
}

#endif
