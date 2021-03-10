#ifndef __DEFINED_HIP_TOOLS__
#define __DEFINED_HIP_TOOLS__

#include <iostream>
#include <unistd.h>
#include <numa.h>
#include "rocm_smi/rocm_smi.h"

static inline void rsmiCheck(rsmi_status_t err)
{
# ifdef DEBUG
  if(err != RSMI_STATUS_SUCCESS)
  {
    std::cerr << "Call to the rsmi API failed!" << std::endl;
    exit(1);
  }
# endif
}

Smi::Smi()
{
  if (!(this->smiIsActive))
  {
    rsmiCheck( rsmi_init(0) );
    this->smiIsActive = 1;
    rsmiCheck( rsmi_num_monitor_devices(&numberOfDevices) );
  }

  this->activeSmiInstances += 1;
}

void Smi::setCpuAffinity(int id)
{
  checkGpuIdIsSensible(id);

  uint32_t numa_node;
  rsmiCheck( rsmi_topo_numa_affinity_get( id, &numa_node) );
  numa_run_on_node(numa_node);
}

void Smi::getGpuTemp(int id, float * temp) const
{
  // Check that the GPU id is correct
  checkGpuIdIsSensible(id);

  // Get the temperature reading
  int64_t temperature;
  rsmiCheck( rsmi_dev_temp_metric_get(id, RSMI_TEMP_TYPE_FIRST, RSMI_TEMP_CURRENT, &temperature) );
  *temp = float(temperature)/1000.f;
}

void Smi::getDeviceMemorySize(int id, size_t * sMem) const
{
  uint64_t total;
  rsmiCheck( rsmi_dev_memory_total_get((uint32_t)id, RSMI_MEM_TYPE_FIRST, &total) );
  *sMem = (size_t)total;
}

void Smi::getDeviceAvailMemorySize(int id, size_t * sMem) const
{
  uint64_t used;
  this->getDeviceMemorySize(id, sMem);
  rsmiCheck( rsmi_dev_memory_usage_get((uint32_t)id, RSMI_MEM_TYPE_FIRST, &used) );
  sMem[0] -= (size_t)used;
}

Smi::~Smi()
{
  this->activeSmiInstances -= 1;
  if (!(this->activeSmiInstances))
  {
    rsmiCheck( rsmi_shut_down() );
    this->smiIsActive = 0;
  }
}


/*
 * ASM tools
 */

template< class T >
__device__ __forceinline__ T __XSyncClock()
{
  // Force the completion of other pending operations before requesting the
  // clock counter. The clock counter is read "asyncrhonously" and its value
  // is not guaranteed to be present in "x" on return.
  uint64_t x;
  asm volatile (
                "s_waitcnt vmcnt(0) & lgkmcnt(0) & expcnt(0);\n\t"
                "s_memtime %0;"
                : "=s"(x)
               );
  return (T)x;
}

__device__ __forceinline__ uint32_t XSyncClock()
{
  return __XSyncClock<uint32_t>();
}

__device__ __forceinline__ uint64_t XSyncClock64()
{
  return __XSyncClock<uint64_t>();
}

template< class T >
__device__ __forceinline__ T __XClock()
{
  // Retrieve the clock couner and forces a wait on the associated
  // memory operation.
  uint64_t x;
  asm volatile ("s_memtime %0; \t\n"
                "s_waitcnt lgkmcnt(0);"
                : "=s"(x)
               );
  return (T)x;
}

__device__ uint32_t XClock()
{
  return __XClock<uint32_t>();
}

__device__ uint64_t XClock64()
{
  return  __XClock<uint64_t>();
}


template < class T = uint32_t>
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
    startClock = __XSyncClock<T>();
  }
  __device__ T end()
  {
    return __XClock<T>() - startClock;
  }
};

using XClocks = __XClocks<>;
using XClocks64 = __XClocks<uint64_t>;


__device__ void __clockLatency64( uint64_t * clk)
{
  /*
   * Expose the latency of a clock read.
   */
  asm volatile (
                "s_waitcnt lgkmcnt(0);\n\t"
                "s_memtime %[a];\n\t"
                "s_waitcnt lgkmcnt(0);\n\t"
                "s_memtime %[b];\n\t"
                "s_waitcnt lgkmcnt(0);\n\t"
                :[a]"=r"(clk[0]), [b]"=r"(clk[1]) :: "memory");
}


template <class T>
__device__ T XClockLatency()
{
  uint64_t c[2];
  __clockLatency64(c);
  return (T)(c[1]-c[0]);
}


__device__ __forceinline__ int __smId()
{
  // NOT possible to retrieve the workgroup ID with AMD GPUs
  return -1;
}

#endif
