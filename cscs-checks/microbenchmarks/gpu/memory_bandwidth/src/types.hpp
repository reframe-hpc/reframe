#ifndef __INCLUDED_TYPES__
#define __INCLUDED_TYPES__

#include "Xdevice/runtime.hpp"

// Class managing host data.
class HostData
{
private:
  size_t size;

public:
  void * data;
  HostData() = delete;
  HostData(size_t s) : size(s)
  {
    XMallocHost(&data, size);
    memset(data, 0, size);
  }
  ~HostData()
  {
    XFreeHost(data);
  }
};

// Class managing device data
class DeviceData
{
private:
  size_t size;

public:
  void * data;
  DeviceData() = delete;
  DeviceData(size_t s) : size(s)
  {
    XMalloc(&data, size);
    XMemset(data, 0, size);
    XDeviceSynchronize();
  }
  ~DeviceData()
  {
    XFree(data);
  }
};
#endif
