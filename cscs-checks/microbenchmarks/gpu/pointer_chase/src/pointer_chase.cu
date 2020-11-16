#include <cstdint>
#include <cassert>
#include <iostream>
#include <string>
#include <random>
#include <chrono>
#include <set>
#include <memory>
#include <queue>

/*
 ~~ GPU Linked list pointer chase algorithm ~~
 Times in clock cycles the time it takes to jump from one node to the next
 in a singly linked list.

 The list can be initialized sequentially or with a random node ordering. This
 can be controlled passing the command line argument "--rand".

 The stride and the full buffer size can be set with "--stride" and "--buffer",
 both in number of nodes.

 The macro NODES sets the total number of nodes in the list. Note that the
 list traversal is 'unrolled' inlining a recursive template, and this will
 not work if you use a large number of nodes.

 The nodes can be padded with an arbitrary size controlled by the NODE_PADDING
 macro (in Bytes).

 The LIST_TYPE macro dictates where the list is allocated. If DeviceList is used
 (default option) the linked list is allocated in device memory. In contrast, if
 HostList is used, the list is allocated as host's pinned memory.

 The links of the list can be made volatile defining the macro VOLATILE.

 By default, the code returns the aveage number of cycles per jump, but this can
 be changed to return the cycle count on a per-jump basis by defining the flag
 TIME_EACH_STEP.
*/

#define NODES 64
#define NODE_PADDING 0

#ifndef LIST_TYPE
# define LIST_TYPE DeviceList
#endif

#ifndef HOSTNAME_SIZE
# define HOSTNAME_SIZE 80
#endif


// Include the CUDA/HIP wrappers from the other test for now.
#include "Xdevice/runtime.hpp"

// List structure
#include "pChase_list.hpp"


template < class LIST >
uint32_t * generalPointerChase(int local_device, int remote_device, int init_mode, size_t buffSize, size_t stride)
{
  /*
   * Driver to manage the whole allocation, list traversal, etc.
   * It returns the array containing the timings. Note that these values will depend on whether the
   * flag -DTIME_EACH_STEP was defined or not (see top of the file).
   *
   * - local_device: ID of the device where the allocation of the list takes place
   * - remote_device: ID of the device doing the pointer chase.
   * - init_mode: see the class List.
   * - buff_size: Size (in nodes) of the buffer.
   * - stride: Gap (in nodes) between two consecutive nodes. This only applies if init_mode is 0.
   */

  XSetDevice(remote_device);
  LIST l(NODES, buffSize, stride);
  l.initialize(init_mode);

  // Check if we have remote memory access.
  XSetDevice(local_device);
  bool peerAccessSet = false;
  if (local_device!=remote_device)
  {
    int hasPeerAccess;
    XDeviceCanAccessPeer(&hasPeerAccess, local_device, remote_device);
    if (!hasPeerAccess)
    {
      printf("Devices have no peer access.\n");
      exit(1);
    }

    // Enable the peerAccess access.
    peerAccessSet = true;
    XDeviceEnablePeerAccess(remote_device, 0);
  }

  // Warm-up kernel
  l.traverse();

  // Time the pointer chase
  l.time_traversal();

  if (peerAccessSet)
    XDeviceDisablePeerAccess(remote_device);

   // Set again the device where the allocations were placed, so it can take care of it's
   // own deallocations in the List destructor.
   XSetDevice(remote_device);

   return l.timer;
}


template < class LIST >
void localPointerChase(int num_devices, int init_mode, size_t buffSize, size_t stride, char * nid)
{
  /*
   * Specialised pointer chase on a single device.
   */
  for (int gpu_id = 0; gpu_id < num_devices; gpu_id++)
  {
    uint32_t* timer = generalPointerChase< LIST >(gpu_id, gpu_id, init_mode, buffSize, stride);

    // Print the timings of the pointer chase
#   ifndef TIME_EACH_STEP
    printf("[%s] On device %d, the chase took on average %d cycles per node jump.\n", nid, gpu_id, timer[0]/(NODES-1));
#   else
    printf("[%s] Latency for each node jump (device %d):\n", nid, gpu_id);
    for (uint32_t i = 0; i < NODES-1; i++)
    {
      printf("[%s][device %d] %d\n", nid, gpu_id, timer[i]);
    }
#   endif
    delete [] timer;
  }
}


#ifdef SYMM
# define LIMITS j
#else
# define LIMITS 0
#endif

void print_device_table(int num_devices, std::queue<uint32_t> q, const char * what, const char * nid)
{
  printf("[%s] %s memory latency (in clock cycles) with remote direct memory access\n", nid, what);
  printf("[%s] %10s", nid, "From \\ To ");
  for (int ds = 0; ds < num_devices; ds++)
  {
    printf("%4sGPU %2d", "", ds);
  } printf("%10s\n", "Totals");

  for (int j = 0; j < num_devices; j++)
  {
    // Track the sum of the latencies
    uint32_t totals = 0;

    printf("[%s] GPU %2d%4s", nid, j, " ");
    for (int i = 0; i < LIMITS; i++)
    {
      printf("%10s", "X");
    }

    for (int i = LIMITS; i < num_devices; i++)
    {
      uint32_t timer = q.front();
      q.pop();
      if (i != j)
      {
        totals += timer;
      }
      printf("%10d", timer);
    } printf("%10d\n", totals);
  }
}

template < class LIST >
void remotePointerChase(int num_devices, int init_mode, size_t buffSize, size_t stride, char * nid)
{
  /*
   * Specialised pointer chase to allocate the list in one device, and do the pointer chase from another device.
   */

# ifndef TIME_EACH_STEP
  std::queue<uint32_t> q_average;
  auto fetch = [](uint32_t* t){return t[0]/(NODES-1);};
# else
  std::queue<uint32_t> q_max;
  std::queue<uint32_t> q_min;
  auto fetchMax = [](uint32_t* t)
  {
    uint32_t max = 0;
    for (int i = 0; i < NODES-1; i++)
    {
      if (t[i] > max)
        max = t[i];
    }
    return max;
  };
  auto fetchMin = [](uint32_t* t)
  {
    uint32_t min = ~0;
    for (int i = 0; i < NODES-1; i++)
    {
      if (t[i] < min)
        min = t[i];
    }
    return min;
  };
# endif

  // Do the latency measurements
  for (int j = 0; j < num_devices; j++)
  {
    for (int i = LIMITS; i < num_devices; i++)
    {
      uint32_t * timer_ptr = generalPointerChase< LIST >(i, j, init_mode, buffSize, stride);

      // Store the desired values for each element of the matrix in queues
#     ifndef TIME_EACH_STEP
      q_average.push(fetch(timer_ptr));
#     else
      q_min.push(fetchMin(timer_ptr));
      q_max.push(fetchMax(timer_ptr));
#     endif
      delete [] timer_ptr;
    }
  }

  std::string what;
# ifndef TIME_EACH_STEP
  what = "Average";
  print_device_table(num_devices, q_average, what.c_str(), nid);
# else
  what = "Min.";
  print_device_table(num_devices, q_min, what.c_str(), nid);
  printf("\n");
  what = "Max.";
  print_device_table(num_devices, q_max, what.c_str(), nid);
# endif

}


int main(int argc, char ** argv)
{
  // Set program defaults before parsing the command line args.
  int list_init_mode = 0;
  size_t stride = 1;
  size_t buffSize = NODES*stride;
  int multiGPU = 0;

  // Parse the command line args.
  for (int i = 0; i < argc; i++)
  {
    std::string str = argv[i];
    if (str == "--help" || str == "-h")
    {
      std::cout << "--rand      : Initializes the linked list with nodes in random order." << std::endl;
      std::cout << "--stride #  : Sets the stride between the nodes in the list (in number of nodes)." << std::endl;
      std::cout << "              If --rand is used, this parameter just changes the buffer size." << std::endl;
      std::cout << "--buffer #  : Sets the size of the buffer where the linked list is allocated on. " << std::endl;
      std::cout << "              The number indicates the size of the buffer in list nodes." << std::endl;
      std::cout << "--multiGPU  : Runs the pointer chase algo using all device-pair combinations." << std::endl;
      std::cout << "              This measures the device-to-device memory latency." << std::endl;
      std::cout << "--help (-h) : I guess you figured what this does already ;)" << std::endl;
      return 0;
    }
    else if (str == "--rand")
    {
      list_init_mode = 1;
    }
    else if (str == "--stride")
    {
      stride = std::stoi((std::string)argv[++i]);
      if (buffSize < NODES*stride)
          buffSize = NODES*stride;
    }
    else if (str == "--buffer")
    {
      buffSize = std::stoi((std::string)argv[++i]);
    }
    else if (str == "--multiGPU")
    {
      multiGPU = 1;
    }
  }

  // Sanity of the command line args.
  if (buffSize < NODES*stride)
  {
    std::cerr << "Buffer is not large enough to fit the list." << std::endl;
    return 1;
  }

  // Get the node name
  char nid_name[HOSTNAME_SIZE];
  gethostname(nid_name, HOSTNAME_SIZE);

  // Make sure we've got devices aboard.
  int num_devices;
  XGetDeviceCount(num_devices);
  if (num_devices == 0)
  {
    std::cout << "No devices found on host " << nid_name << std::endl;
    return 1;
  }
  else
  {
    printf("[%s] Found %d device(s).\n", nid_name, num_devices);
  }

  if (!multiGPU)
  {
    localPointerChase<LIST_TYPE>(num_devices, list_init_mode, buffSize, stride, nid_name);
  }
  else
  {
    remotePointerChase<LIST_TYPE>(num_devices, list_init_mode, buffSize, stride, nid_name);
  }

  printf("[%s] Pointer chase complete.\n", nid_name);
  return 0;
}
