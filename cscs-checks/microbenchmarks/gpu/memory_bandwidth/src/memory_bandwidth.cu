#include <iostream>
#include <unistd.h>
#include <cuda_runtime.h>
#include <cuda.h>

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
  cudaError_t cudaError = cudaGetDeviceCount(&number_of_devices);

  // Do some error checking
  if (cudaError != cudaSuccess)
  {
    std::cout << "On host " << nid_name << ", cudaGetDeviceCount returned an error with ID " << (int)cudaError << " (" <<
                 cudaGetErrorString(cudaError) << ").\n";
    return 1; 
  }

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
  /*
   Test the Host to Device bandwidth.
  */
  for (int d = 0; d < number_of_devices; d++)
  {
    cudaSetDevice(d);
    float bw = copyBandwidth<HostData, DeviceData, cudaMemcpyHostToDevice>(copy_size, d, copy_repeats);
    printf("[%s] Host to device bandwidth on device %d is %f Mb/s.\n", nid_name, d, bw);
  }

  /*
   Test the Device to Host bandwidth.
  */
  for (int d = 0; d < number_of_devices; d++)
  {
    cudaSetDevice(d);
    float bw = copyBandwidth<DeviceData, HostData, cudaMemcpyDeviceToHost>(copy_size, d, copy_repeats);
    printf("[%s] Device to host bandwidth on device %d is %f Mb/s.\n", nid_name, d, bw);
  }

  /*
   Test the Device to Device bandwidth.
  */
  for (int d = 0; d < number_of_devices; d++)
  {
    cudaSetDevice(d);
    float bw = copyBandwidth<DeviceData, DeviceData, cudaMemcpyDeviceToDevice>(copy_size, d, copy_repeats) * 2.0;

    printf("[%s] Device to device bandwidth on device %d is %f Mb/s.\n", nid_name, d, bw);
  }

  // Do some basic error checking
  if (cudaGetLastError() == cudaSuccess)
  {
    printf("[%s] Test Result = PASS", nid_name);
  } 
  else
  {
    printf("[%s] Test Result = FAIL", nid_name);
  }
  return 0;
}
