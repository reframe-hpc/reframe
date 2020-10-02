#ifndef __INCLUDED_TYPES__
#define __INCLUDED_TYPES__

#if defined TARGET_CUDA
#  include "cuda/utils.hpp"
#elif defined TARGET_AMD
#  include "rocm/utils.hpp"
#endif

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
  }
  ~DeviceData()
  {
    XFree(data);
  }
};
#endif
