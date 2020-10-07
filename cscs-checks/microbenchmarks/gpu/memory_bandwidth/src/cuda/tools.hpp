#ifndef __INCLUDED_CUDA_TOOLS__
#define __INCLUDED_CUDA_TOOLS__

#include<nvml.h>

#ifndef NO_ERROR_CHECK
#  define nvmlCheck(err) { (err); }
#else
#  define nvmlCheck(err) { nvmlAssert((err), __FILE__, __LINE__); }
   inline void nvmlAssert(nvmlResult_t err, const char *file, int line)
   {
      if (err != NVML_SUCCESS) 
      {
         char nid[80];
         gethostname(nid,80);
         fprintf(stderr,"A call to the nvml failed on node %s, at %s, %s %d\n", file, line);
         exit(err);
      }
   }
#endif


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
