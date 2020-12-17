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

#endif
