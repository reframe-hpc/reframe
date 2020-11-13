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

class Smi
{
private:
  static int rsmiIsActive;
  static int activeSmiInstances;
  unsigned int numberOfDevices;

public:
  Smi();
  void setCpuAffinity(int);
  ~Smi();
};

int Smi::rsmiIsActive = 0;
int Smi::activeSmiInstances = 0;

Smi::Smi()
{
  if (!(this->rsmiIsActive))
  {
    rsmiCheck( rsmi_init(0) );
    this->rsmiIsActive = 1;
    rsmiCheck( rsmi_num_monitor_devices(&numberOfDevices) );
  }

  this->activeSmiInstances += 1;
}

void Smi::setCpuAffinity(int id)
{
  if (id < 0 || id >= numberOfDevices)
  {
    std::cerr << "Requested device ID is out of range from the existing devices." << std::endl;
    return;
  }

  uint32_t numa_node;
  rsmiCheck( rsmi_topo_numa_affinity_get( id, &numa_node) );
  numa_run_on_node(numa_node);
}

Smi::~Smi()
{
  this->activeSmiInstances -= 1;
  if (this->rsmiIsActive)
  {
    rsmiCheck( rsmi_shut_down() );
    this->rsmiIsActive = 0;
  }
}


/*
 * ASM tools
 */

template< class T >
__device__ __forceinline__ T __XClock()
{
  // Clock counter
  uint64_t x;
  asm volatile ("s_memtime %0; \t\n"
                "s_waitcnt lgkmcnt(0);"
                : "=r"(x)
               );
  return (T)x;
}

__device__ __forceinline__ uint32_t XClock()
{
  return __XClock<uint32_t>();
}

__device__ __forceinline__ uint64_t XClock64()
{
  return __XClock<uint64_t>();
}


__device__ __forceinline__ int __smId()
{
  // NOT possible to retrieve the workgroup ID with AMD GPUs
  return -1;
}

#endif
