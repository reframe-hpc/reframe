#ifndef __INCLUDED_TYPES__
#define __INCLUDED_TYPES__

#include<cuda.h>

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
    cudaMallocHost(&data, size);
  }
  ~HostData()
  {
    cudaFreeHost(data);
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
    cudaMalloc(&data, size);
  }
  ~DeviceData()
  {
    cudaFree(data);
  }
};
#endif
