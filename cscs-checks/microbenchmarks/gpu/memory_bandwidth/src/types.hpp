#ifndef __INCLUDED_TYPES__
#define __INCLUDED_TYPES__

#include "Xlib/runtime.hpp"

class RAIIData
{
public:
  RAIIData() {};
  virtual ~RAIIData() {};
};

// Class managing host data.
class HostData : public RAIIData
{
private:
  size_t size;

public:
  void * data;
  HostData() = delete;
  HostData(size_t s) : size(s)
  {
    XMallocHost(&data, size);
  }
  ~HostData()
  {
    XFreeHost(data);
  }
};

// Class managing device data
class DeviceData : public RAIIData
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
