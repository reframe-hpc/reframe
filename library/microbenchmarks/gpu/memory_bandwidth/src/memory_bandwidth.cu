#include "testHeaders.hpp"

int main()
{

  char nid_name[HOSTNAME_SIZE];
  gethostname(nid_name, HOSTNAME_SIZE);

  int number_of_devices;
  XGetDeviceCount(&number_of_devices);

  // Initialise the Smi to manage the devices.
  Smi smiHandle;

  // Make sure we've got devices aboard.
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
  float fact = (float)copy_size/1e3;

  /*
   Test the Host to Device bandwidth.
  */
  for (int d = 0; d < number_of_devices; d++)
  {
    XSetDevice(d);
    smiHandle.setCpuAffinity(d);
    float bw = fact / copyBandwidth<HostData, DeviceData>(copy_size, d, copy_repeats, XMemcpyHostToDevice);
    printf("[%s] Host to device bandwidth on device %d is %.2f Mb/s.\n", nid_name, d, bw);
  }

  /*
   Test the Device to Host bandwidth.
  */
  for (int d = 0; d < number_of_devices; d++)
  {
    XSetDevice(d);
    smiHandle.setCpuAffinity(d);
    float bw = fact / copyBandwidth<DeviceData, HostData>(copy_size, d, copy_repeats, XMemcpyDeviceToHost);
    printf("[%s] Device to host bandwidth on device %d is %.2f Mb/s.\n", nid_name, d, bw);
  }

  /*
   Test the Device to Device bandwidth.
  */
  for (int d = 0; d < number_of_devices; d++)
  {
    XSetDevice(d);
    smiHandle.setCpuAffinity(d);
    float bw = (float)2 * fact / copyBandwidth<DeviceData, DeviceData>(copy_size, d, copy_repeats, XMemcpyDeviceToDevice);
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
