#include <iostream>
#include <unistd.h>

#include "Xdevice/runtime.hpp"
#include "Xdevice/smi.hpp"

#include "bandwidth.hpp"

#ifndef HOSTNAME_SIZE
# define HOSTNAME_SIZE 80
#endif

void copy_bandwidth_single(size_t n_devices, size_t copy_size, size_t n_copies, char *nid_name, Smi &smi_handle)
{
  /*
   * For each device in the node, compute the host-to-device, device-to-host and device-to-device bandwidths.
   */

  float fact = (float)copy_size/(1024.0*1024.0*1024.0/1e3);

  // Host to Device bandwidth.
  for (int d = 0; d < n_devices; d++)
  {
    XSetDevice(d);
    smi_handle.setCpuAffinity(d);
    float bw = fact / copyBandwidth<HostData, DeviceData>(copy_size, d, n_copies, XMemcpyHostToDevice);
    printf("[%s] Host to device bandwidth on device %d is %.2f GB/s.\n", nid_name, d, bw);
  }

  // Device to Host bandwidth.
  for (int d = 0; d < n_devices; d++)
  {
    XSetDevice(d);
    smi_handle.setCpuAffinity(d);
    float bw = fact / copyBandwidth<DeviceData, HostData>(copy_size, d, n_copies, XMemcpyDeviceToHost);
    printf("[%s] Device to host bandwidth on device %d is %.2f GB/s.\n", nid_name, d, bw);
  }

  // Device to Device bandwidth.
  for (int d = 0; d < n_devices; d++)
  {
    XSetDevice(d);
    smi_handle.setCpuAffinity(d);
    float bw = (float)2 * fact / copyBandwidth<DeviceData, DeviceData>(copy_size, d, n_copies, XMemcpyDeviceToDevice);
    printf("[%s] Device to device bandwidth on device %d is %.2f GB/s.\n", nid_name, d, bw);
  }
}

void copy_bandwidth_multi(size_t n_devices, size_t copy_size, int n_copies, bool p2p, bool symm, char * nid_name, Smi &smi_handle)
{
  /*
   This function evaluates the GPU to GPU copy bandwith in all GPU to GPU combinations.
   If bandwidth symmetry is assumed (i.e. the bandwidth in both directions is the same),
   the symm argument must be passed as true. P2P memory access is controlled through the
   function argument p2p.

   This function prints all the results into a table, where the "Totals" column excludes
   the diagonal terms from the sum.

   This function sets the CPU affinity for the sending GPU.
  */

  float fact = (float)copy_size/(1024*1024.0*1024.0/1e3);

  const char * p2p_out = (p2p ? "enabled" : "disabled");
  printf("[%s] P2P Memory bandwidth (GB/s) with remote direct memory access %s\n", nid_name, p2p_out);
  printf("[%s] %10s", nid_name, "From \\ To ");
  for (int ds = 0; ds < n_devices; ds++)
  {
    printf("%4sGPU %2d", "", ds);
  } printf("%10s\n", "Totals");

  for (int ds = 0; ds < n_devices; ds++)
  {
    // Track the sum of the bandwidths
    float totals = 0;

    // Set the CPU affinity to the sending device.
    smi_handle.setCpuAffinity(ds);

    int inner_range = symm ? ds : 0;
    printf("[%s] GPU %2d%4s", nid_name, ds, " ");
    for (int dr = 0; dr < inner_range; dr++)
    {
      printf("%10s", "X");
    }

    for (int dr = inner_range; dr < n_devices; dr++)
    {
      float same = ( (ds==dr) ? float(2) : float(1) );
      float bw = fact / p2pBandwidth(copy_size, ds, dr, n_copies, p2p) * same;
      if (ds != dr)
      {
        totals += bw;
      }
      printf("%10.2f", bw);
    } printf("%10.2f\n", totals);

  }
}

int main(int argc, char ** argv)
{

  // Default values.
  size_t copy_size = 1073741824;
  size_t num_copies = 20;
  bool multi_gpu = false;
  bool p2p = false;
  bool symm = false;

  // Parse the command line args.
  for (int i = 0; i < argc; i++)
  {
    std::string str = argv[i];
    if (str == "--help" || str == "-h")
    {
      std::cout << "--size #    : Set the size of the copy buffer (in bytes). Default is 1GB." << std::endl;
      std::cout << "--copies #  : Number of repetitions per copy operation. Default is 20." << std::endl;
      std::cout << "--multi-gpu : Compute the copy bandwidth amongst all devices in the node." << std::endl;
      std::cout << "--p2p       : Enables peer-to-peer copy across devices. This only has an " << std::endl;
      std::cout << "              when used with the --multi-gpu argument." << std::endl;
      std::cout << "--symm      : Assume the bandwidth across devices are the same and computes" << std::endl;
      std::cout << "              only half of the matrix." << std::endl;
      std::cout << "--help (-h) : Prints this help menu." << std::endl;
      return 0;
    }
    else if (str == "--size")
    {
      copy_size = std::stol((std::string)argv[++i]);
    }
    else if (str == "--copies")
    {
      num_copies = std::stol((std::string)argv[++i]);
    }
    else if (str == "--multi-gpu")
    {
      multi_gpu = true;
    }
    else if (str == "--p2p")
    {
      p2p = true;
    }
    else if (str == "--symm")
    {
      symm = true;
    }
  }

  char nid_name[HOSTNAME_SIZE];
  gethostname(nid_name, HOSTNAME_SIZE);

  int num_devices;
  XGetDeviceCount(&num_devices);

  // Initialise the Smi to manage the devices.
  Smi smi_handle;

  // Make sure we've got devices aboard.
  if (num_devices == 0)
  {
    std::cout << "No devices found on host " << nid_name << std::endl;
    return 1;
  }
  else
  {
    printf("[%s] Found %d device(s).\n", nid_name, num_devices);
  }

  if (!multi_gpu)
  {
    copy_bandwidth_single(num_devices, copy_size, num_copies, nid_name, smi_handle);
  }
  else
  {
    copy_bandwidth_multi(num_devices, copy_size, num_copies, p2p, symm, nid_name, smi_handle);
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
