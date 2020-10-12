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
    char nid[80];
    gethostname(nid, 80);
    std::cerr << "Call to the rsmi API failed!" << std::endl;
    exit(err);
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

#endif
