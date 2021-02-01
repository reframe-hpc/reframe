#include <cstdint>
#include <cassert>
#include <iostream>
#include <string>
#include <random>
#include <chrono>
#include <memory>
#include <algorithm>
#include <queue>

/*
 ~~ GPU Linked list pointer chase algorithm ~~
 Times in clock cycles the time it takes to jump from one node to the next
 in a singly linked list.

 The list can be initialized sequentially or with a random node ordering. This
 can be controlled passing the command line argument "--rand".

 The stride and the full buffer size can be set with "--stride" and "--buffer",
 both in number of nodes.

 The nodes can be padded with an arbitrary size controlled by the NODE_PADDING
 macro (in Bytes).

 If the ALLOC_ON_HOST macro is defined, the list will be allocated in host
 pinned memory. Otherwise, the list is allocated in device memory.

 The links of the list can be made volatile defining the macro VOLATILE.

 By default, the code returns the aveage number of cycles per jump, but this can
 be changed to return the cycle count on a per-jump basis by defining the flag
 TIME_EACH_STEP.
*/


#define REPEAT2(x) x; x;
#define REPEAT4(x) REPEAT2(x) REPEAT2(x)
#define REPEAT8(x) REPEAT4(x) REPEAT4(x)
#define REPEAT16(x) REPEAT8(x) REPEAT8(x)
#define REPEAT32(x) REPEAT16(x) REPEAT16(x)
#define REPEAT64(x) REPEAT32(x) REPEAT32(x)
#define REPEAT128(x) REPEAT64(x) REPEAT64(x)
#define REPEAT256(x) REPEAT128(x) REPEAT128(x)

#define JUMPS 256
#define REPEAT_JUMPS(x) REPEAT256(x)

#define NODE_PADDING 0

#ifndef HOSTNAME_SIZE
# define HOSTNAME_SIZE 80
#endif


// Include the CUDA/HIP wrappers from the other test for now.
#include "Xdevice/runtime.hpp"

// List structure
#include "linked_list.hpp"
#ifdef ALLOC_ON_HOST
using list_type = HostList;
#else
using list_type = DeviceList;
#endif



template < class List >
uint32_t * general_pointer_chase(int local_device, int remote_device, int init_mode, size_t num_nodes, size_t stride)
{
  /*
   * Driver to manage the whole allocation, list traversal, etc.
   * It returns the array containing the timings. Note that these values will depend on whether the
   * flag -DTIME_EACH_STEP was defined or not (see top of the file).
   *
   * - local_device: ID of the device where the allocation of the list takes place
   * - remote_device: ID of the device doing the pointer chase.
   * - init_mode: see the List class.
   * - num_nodes: nodes in the liked list.
   * - stride: Gap (in nodes) between two consecutive nodes. This only applies if init_mode is 0.
   */

  XSetDevice(remote_device);
  List l(num_nodes, stride);
  l.initialize(init_mode);

  // Check if we have remote memory access.
  XSetDevice(local_device);
  bool peer_access_set = false;
  if (local_device!=remote_device)
  {
    int has_peer_access;
    XDeviceCanAccessPeer(&has_peer_access, local_device, remote_device);
    if (!has_peer_access)
    {
      printf("Devices have no peer access.\n");
      exit(1);
    }

    // Enable the peerAccess access.
    peer_access_set = true;
    XDeviceEnablePeerAccess(remote_device, 0);
  }

  // Warm-up kernel
  l.traverse();

  // Time the pointer chase
  l.time_traversal();

  if (peer_access_set)
    XDeviceDisablePeerAccess(remote_device);

   // Set again the device where the allocations were placed, so it can take care of it's
   // own deallocations in the List destructor.
   XSetDevice(remote_device);

   return l.timer;
}


template < class List >
void local_pointer_chase(int num_devices, int init_mode, size_t num_nodes, size_t stride, char * nid)
{
  /*
   * Specialised pointer chase on a single device.
   */
  for (int gpu_id = 0; gpu_id < num_devices; gpu_id++)
  {
    uint32_t* timer = general_pointer_chase< List >(gpu_id, gpu_id, init_mode, num_nodes, stride);

    // Print the timings of the pointer chase
#   ifndef TIME_EACH_STEP
    printf("[%s] On device %d, the chase took on average %d cycles per node jump.\n", nid, gpu_id, timer[0]/JUMPS);
#   else
    printf("[%s] Latency for each node jump (device %d):\n", nid, gpu_id);
    for (uint32_t i = 0; i < JUMPS; i++)
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
  /*
   * Print the data in a table format - useful when doing P2P list traversals.
   */

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


template < class List >
void remote_pointer_chase(int num_devices, int init_mode, size_t num_nodes, size_t stride, char * nid, int summarize)
{
  /*
   * Specialised pointer chase to allocate the list in one device, and do the pointer chase from another device.
   * - summarize: if different than zero, the results will be printed in a table format with the function above.
   *   Otherwise, every single result will be printed out.
   */

# ifndef TIME_EACH_STEP
  std::queue<uint32_t> q_average;
  auto fetch = [](uint32_t* t){return t[0]/(JUMPS);};
# else
  std::queue<uint32_t> q_max;
  std::queue<uint32_t> q_min;
  auto fetch_max = [](uint32_t* t)
  {
    uint32_t max = 0;
    for (int i = 0; i < JUMPS; i++)
    {
      if (t[i] > max)
        max = t[i];
    }
    return max;
  };
  auto fetch_min = [](uint32_t* t)
  {
    uint32_t min = ~0;
    for (int i = 0; i < JUMPS; i++)
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
      uint32_t * timer_ptr = general_pointer_chase< List >(i, j, init_mode, num_nodes, stride);

      // Store the desired values for each element of the matrix in queues
#     ifndef TIME_EACH_STEP
      q_average.push(fetch(timer_ptr));
#     else
      if (summarize)
      {
        q_min.push(fetch_min(timer_ptr));
        q_max.push(fetch_max(timer_ptr));
      }
      else
      {
        for (int n = 0; n < JUMPS; n++)
        {
          printf("[%s][device %d][device %d] %d\n", nid, j, i, timer_ptr[n]);
        }
      }
#     endif
      delete [] timer_ptr;
    }
  }

  std::string what;
# ifndef TIME_EACH_STEP
  what = "Average";
  print_device_table(num_devices, q_average, what.c_str(), nid);
# else
  if (summarize)
  {
    what = "Min.";
    print_device_table(num_devices, q_min, what.c_str(), nid);
    printf("\n");
    what = "Max.";
    print_device_table(num_devices, q_max, what.c_str(), nid);
  }
# endif

}


int main(int argc, char ** argv)
{
  // Set program defaults before parsing the command line args.
  int list_init_random = 0;
  size_t sparsity = 1;
  size_t num_nodes = JUMPS;
  int multi_gpu = 0;
  int print_summary_only = 0;
  int clock = 0;

  // Parse the command line args.
  for (int i = 0; i < argc; i++)
  {
    std::string str = argv[i];
    if (str == "--help" || str == "-h")
    {
      std::cout << "--nodes #    : Number of nodes in the linked list. If no value is specified, it " << std::endl;
      std::cout << "               defaults to the number of node jumps set at compile time." << std::endl;
      std::cout << "--rand       : Places the linked list nodes into the buffer in random order (i.e." << std::endl;
      std::cout << "               consecutive list nodes are not consecutive in memory). If this option" << std::endl;
      std::cout << "               is not used, the nodes are placed in sequential order." << std::endl;
      std::cout << "--sparsity # : Controls the sparsity of the list nodes in the buffer. This sets the" << std::endl;
      std::cout << "               buffer size where the list is placed as sparsity*num_nodes. If the" << std::endl;
      std::cout << "               list is initialized in sequential order, this effectively sets the stride." << std::endl;
      std::cout << "--multi-gpu  : Runs the pointer chase algo using all device-pair combinations." << std::endl;
      std::cout << "               This measures the device-to-device memory latency." << std::endl;
      std::cout << "--summary    : When timing each node jump individually and used alongside --multi-gpu, " << std::endl;
      std::cout << "               this collapses the output into two tables with the min and max latencies." << std::endl;
      std::cout << "--clock      : Skip all the above and just print the clock latency for all devices." << std::endl;
      std::cout << "--help (-h)  : I guess you figured what this does already ;)" << std::endl;
      return 0;
    }
    else if (str == "--nodes")
    {
      num_nodes = std::stoi((std::string)argv[++i]);
    }
    else if (str == "--rand")
    {
      list_init_random = 1;
    }
    else if (str == "--sparsity")
    {
      sparsity = std::stoi((std::string)argv[++i]);
    }
    else if (str == "--multi-gpu")
    {
      multi_gpu = 1;
    }
    else if (str == "--summary")
    {
      print_summary_only = 1;
    }
    else if (str == "--clock")
    {
      clock = 1;
    }
  }

  // Get the node name
  char nid_name[HOSTNAME_SIZE];
  gethostname(nid_name, HOSTNAME_SIZE);

  // Make sure we've got devices aboard.
  int num_devices;
  XGetDeviceCount(&num_devices);
  if (num_devices == 0)
  {
    std::cout << "No devices found on host " << nid_name << std::endl;
    return 1;
  }
  else
  {
    printf("[%s] Found %d device(s).\n", nid_name, num_devices);
  }

  if (clock)
  {
    for (int i = 0; i < num_devices; i++)
    {
      print_clock_latency(nid_name,i);
    }
  }
  else
  {
    if (!multi_gpu)
    {
      local_pointer_chase<list_type>(num_devices, list_init_random, num_nodes, sparsity, nid_name);
    }
    else
    {
      remote_pointer_chase<list_type>(num_devices, list_init_random, num_nodes, sparsity, nid_name, print_summary_only);
    }
  }

  printf("[%s] Pointer chase complete.\n", nid_name);
  return 0;
}
