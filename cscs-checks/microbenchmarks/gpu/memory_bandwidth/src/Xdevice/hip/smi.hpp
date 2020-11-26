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
  if (!(this->rsmiIsActive))
  {
    rsmiCheck( rsmi_init(0) );
    this->rsmiIsActive = 1;
    rsmiCheck( rsmi_num_monitor_devices(&numberOfDevices) );
  }

  this->activeSmiInstances += 1;
}

void Smi::checkGpuIdIsSensible(int id)
{
  if (id < 0 || id >= numberOfDevices)
  {
    std::cerr << "Requested device ID is out of range from the existing devices." << std::endl;
    exit(1);
  }
}

void Smi::setCpuAffinity(int id)
{
  checkGpuIdIsSensible(id);

  uint32_t numa_node;
  rsmiCheck( rsmi_topo_numa_affinity_get( id, &numa_node) );
  numa_run_on_node(numa_node);
}

float Smi::getGpuTemp(int id)
{
  // Check that the GPU id is correct
  checkGpuIdIsSensible(id);

  // Get the temperature reading
  int64_t temperature;
  rsmiCheck( rsmi_dev_temp_metric_get(id, RSMI_TEMP_TYPE_FIRST, RSMI_TEMP_CURRENT, &temperature) );
  return float(temperature)/1000.f;
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

#endif
