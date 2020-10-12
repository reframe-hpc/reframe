#ifndef __INCLUDED_CUDA_TOOLS__
#define __INCLUDED_CUDA_TOOLS__

#include<iostream>
#include <unistd.h>
#include<nvml.h>

static inline void nvmlCheck(nvmlReturn_t err)
{
# ifdef DEBUG
  if (err != NVML_SUCCESS) 
  {
    char nid[80];
    gethostname(nid, 80);
    std::cerr << "Call to the nvml API failed!" << std::endl;
    exit(err);
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

#endif
