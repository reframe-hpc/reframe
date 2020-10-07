#include <iostream>
#include <unistd.h>

// Set default platform
#if (!defined TARGET_CUDA || !defined TARGET_AMD)
#  define TARGET_CUDA
#endif

#ifdef TARGET_CUDA
# include "cuda/include.hpp"
#else
# error "TARGET NOT IMPLEMENTED"
#endif

#include "bandwidth.hpp"

#ifndef COPY_SIZE
# define COPY_SIZE 1073741824
#endif
#ifndef NUMBER_OF_COPIES
# define NUMBER_OF_COPIES 20
#endif

int main()
{

  char * nid_name = (char *)calloc(80,sizeof(char));
  gethostname(nid_name,80);

  int number_of_devices;
  XGetDeviceCount(number_of_devices);

  // Initialise the a DeviceSet instance to manage the devices.
  DeviceSet dSet;

  // Make sure we've got devices aboard.g
  if (number_of_devices == 0) 
  {
    std::cout << "No devices found on host " << nid_name << std::endl;
    return 1;
  }
  else
  {
    printf("[%s] Found %d device(s).\n", nid_name, number_of_devices); 
  }

  /*
   Test parameters
  */
  size_t copy_size = COPY_SIZE;
  int copy_repeats = NUMBER_OF_COPIES;
  float fact = (float)copy_size/(float)1e3;

  /*
   Test the Host to Device bandwidth.
  */
  for (int d = 0; d < number_of_devices; d++)
  {
    XSetDevice(d);
    dSet.setCpuAffinity(d);
    float bw = fact / copyBandwidth<HostData, DeviceData, XMemcpyHostToDevice>(copy_size, d, copy_repeats);
    printf("[%s] Host to device bandwidth on device %d is %.2f Mb/s.\n", nid_name, d, bw);
  }

  /*
   Test the Device to Host bandwidth.
  */
  for (int d = 0; d < number_of_devices; d++)
  {
    XSetDevice(d);
    dSet.setCpuAffinity(d);
    float bw = fact / copyBandwidth<DeviceData, HostData, XMemcpyDeviceToHost>(copy_size, d, copy_repeats);
    printf("[%s] Device to host bandwidth on device %d is %.2f Mb/s.\n", nid_name, d, bw);
  }

  /*
   Test the Device to Device bandwidth.
  */
  for (int d = 0; d < number_of_devices; d++)
  {
    XSetDevice(d);
    dSet.setCpuAffinity(d);
    float bw = (float)2 * fact / copyBandwidth<DeviceData, DeviceData, XMemcpyDeviceToDevice>(copy_size, d, copy_repeats);
    printf("[%s] Device to device bandwidth on device %d is %.2f Mb/s.\n", nid_name, d, bw);
  }

  // Do some basic error checking
  if (!XGetLastError())
  {
    printf("[%s] Test Result = PASS\n", nid_name);
  } 
  else
  {
    printf("[%s] Test Result = FAIL\n", nid_name);
  }

  return 0;
}
