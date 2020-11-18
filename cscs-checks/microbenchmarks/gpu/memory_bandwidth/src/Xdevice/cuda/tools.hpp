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

class Smi
{
private:
  static int nvmlIsActive;
  static int activeSmiInstances;
  unsigned int numberOfDevices;

public:
  Smi();
  void setCpuAffinity( int );
  ~Smi();
};

int Smi::nvmlIsActive = 0;
int Smi::activeSmiInstances = 0;

Smi::Smi()
{
  if (!(this->nvmlIsActive))
  {
    nvmlCheck( nvmlInit() );
    this->nvmlIsActive = 1;
    nvmlCheck( nvmlDeviceGetCount(&numberOfDevices) );
  }

  this->activeSmiInstances += 1;
}

void Smi::setCpuAffinity(int id)
{
  if (id < 0 || id >= numberOfDevices)
  {
    std::cerr << "Requested device ID is out of range of the existing devices." << std::endl;
    return;
  }

  nvmlDevice_t device;
  nvmlCheck( nvmlDeviceGetHandleByIndex(id, &device) );
  nvmlCheck( nvmlDeviceSetCpuAffinity(device) );
}

Smi::~Smi()
{
  this->activeSmiInstances -= 1;
  if (this->nvmlIsActive)
  {
    nvmlCheck( nvmlShutdown() );
    this->nvmlIsActive = 0;
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
  // Use a data dependency (i.e. store the address of a given variable) to force the compiler to wait.

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


__device__ __forceinline__ int __smId()
{
  // SM ID
  uint32_t x;
  asm volatile ("mov.u32 %0, %%smid;" : "=r"(x) :: "memory");
  return (int)x;
}

#endif
