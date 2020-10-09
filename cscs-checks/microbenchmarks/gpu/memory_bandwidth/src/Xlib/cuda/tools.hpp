#ifndef __INCLUDED_CUDA_TOOLS__
#define __INCLUDED_CUDA_TOOLS__

#include<iostream>
#include<nvml.h>

static inline void nvmlCheck(nvmlReturn_t err)
{
      if (err != NVML_SUCCESS) 
      {
         char nid[80];
         gethostname(nid,80);
         std::cerr << "Call to the nvml API failed!" << std::endl;
         exit(err);
      }
}


class DeviceSet
{
private:
  static int nvmlIsActive;
  static int activeDeviceSets;
  unsigned int numberOfDevices;

public:
  DeviceSet();
  void setCpuAffinity(int);
  ~DeviceSet();

};

int DeviceSet::nvmlIsActive = 0;
int DeviceSet::activeDeviceSets = 0;

DeviceSet::DeviceSet()
{
  if (!(this->nvmlIsActive))
  {
    nvmlCheck( nvmlInit() );
    this->nvmlIsActive = 1;
    this->activeDeviceSets += 1;
    nvmlCheck( nvmlDeviceGetCount( &numberOfDevices ) ); 
  }
}

void DeviceSet::setCpuAffinity(int id)
{
  nvmlDevice_t device;
  nvmlCheck( nvmlDeviceGetHandleByIndex_v2(id, &device) );
  nvmlCheck( nvmlDeviceSetCpuAffinity( device ) );
}

DeviceSet::~DeviceSet()
{
  this->activeDeviceSets -= 1;
  if (!(this->activeDeviceSets))
  {
    nvmlCheck( nvmlShutdown() );
  }
}

#endif
