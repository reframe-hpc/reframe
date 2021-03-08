#include <cstdint>
#include <cassert>
#include <iostream>
#include <string>
#include <random>
#include <chrono>
#include <memory>
#include <algorithm>
#include <queue>
#include <thread>
#include <mutex>
#include <vector>
#include <functional>

/*
 ~~ GPU Linked list pointer chase algorithm ~~
 Times in clock cycles the time it takes to jump from one node to the next
 in a singly linked list.

 The number of nodes in the list and the stride amongst nodes can be set with
 "--nodes" and "--stride".

 The the nodes can be placed in the list in either sequential or random order.
 This can be controlled passing the command line argument "--rand".

 The nodes can be padded with an arbitrary size controlled by the NODE_PADDING
 macro (in Bytes).

 If the ALLOC_ON_HOST macro is defined, the list will be allocated in host
 pinned memory. Otherwise, the list is allocated in device memory.
*/


#ifndef NODE_PADDING
# define NODE_PADDING 0
#endif

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
uint64_t general_pointer_chase(int local_device, int remote_device, int init_mode, size_t num_nodes, size_t stride, size_t num_jumps)
{
  /*
   * Driver to manage the whole allocation, list traversal, etc.
   * Before any timings are done, this function traverses the full list. This "fills up" the device
   * caches and removes any spurious latencies on the first few node jumps. This means that there is no
   * need to even traverse the full list when performing the timed traversal.
   * - local_device: ID of the device where the allocation of the list takes place
   * - remote_device: ID of the device doing the pointer chase.
   * - init_mode: see the List class.
   * - num_nodes: nodes in the liked list.
   * - stride: Gap (in nodes) between two consecutive nodes. This only applies if init_mode is 0.
   * - num_jumps: Number of node jumps to carry out on the timed traversal.
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
  l.time_traversal(num_jumps);

  if (peer_access_set)
    XDeviceDisablePeerAccess(remote_device);

   // Set again the device where the allocations were placed, so it can take care of it's
   // own deallocations in the List destructor.
   XSetDevice(remote_device);

   return l.timer;
}

std::mutex mtx;
template < class L >
void loc_ptr_ch(int gpu_id, int init_mode, size_t num_nodes, size_t stride, size_t num_jumps, char * nid)
{
  /*
   * Low-level thread-safe local pointer chase function.
   */
  uint64_t total_cycles = general_pointer_chase< L >(gpu_id, gpu_id, init_mode, num_nodes, stride, num_jumps);

  // Print the timings of the pointer chase
  {
    std::lock_guard<std::mutex> lg(mtx);
    printf("[%s] On device %d, the chase took on average %d cycles per node jump.\n", nid, gpu_id, total_cycles/num_jumps);
  }
}

template < class List >
void local_pointer_chase(int num_devices, int init_mode, size_t num_nodes, size_t stride, size_t num_jumps, char * nid)
{
  /*
   * Specialised pointer chase on a single device.
   */
  std::vector<std::thread> threads;
  for (int gpu_id = 0; gpu_id < num_devices; gpu_id++)
  {
    threads.push_back(std::thread(loc_ptr_ch<List>,
                                  gpu_id, init_mode,
                                  num_nodes, stride, num_jumps, nid
                     )
    );
  }

  // Join all threads
  std::for_each(threads.begin(), threads.end(), std::mem_fn(&std::thread::join));
}


#ifdef SYMM
# define LIMITS j
#else
# define LIMITS 0
#endif

void print_device_table(int num_devices, std::queue<uint32_t> q, const char * nid)
{
  /*
   * Print the data in a table format - useful when doing P2P list traversals.
   */

  printf("[%s] Average memory latency (in clock cycles) with remote direct memory access\n", nid);
  printf("[%s] %10s", nid, "From \\ To ");
  for (int ds = 0; ds < num_devices; ds++)
  {
    printf("%4sGPU %2d", "", ds);
  } printf("%10s\n", "Max");

  for (int j = 0; j < num_devices; j++)
  {
    // Track the max latency
    uint32_t max_latency = 0;

    printf("[%s] GPU %2d%4s", nid, j, " ");
    for (int i = 0; i < LIMITS; i++)
    {
      printf("%10s", "X");
    }

    for (int i = LIMITS; i < num_devices; i++)
    {
      uint32_t cycles = q.front();
      q.pop();
      if (cycles > max_latency)
      {
        max_latency = cycles;
      }
      printf("%10d", cycles);
    } printf("%10d\n", max_latency);
  }
}


template < class List >
void remote_pointer_chase(int num_devices, int init_mode, size_t num_nodes, size_t stride, size_t num_jumps, char * nid, int summarize)
{
  /*
   * Specialised pointer chase to allocate the list in one device, and do the pointer chase from another device.
   * - summarize: if different than zero, the results will be printed in a table format with the function above.
   *   Otherwise, every single result will be printed out.
   */

  std::queue<uint32_t> timings;

  // Do the latency measurements
  for (int j = 0; j < num_devices; j++)
  {
    for (int i = LIMITS; i < num_devices; i++)
    {
      uint64_t total_cycles = general_pointer_chase< List >(i, j, init_mode, num_nodes, stride, num_jumps);

      // Store the desired values for each element of the matrix in queues
      timings.push(total_cycles/num_jumps);
    }
  }

  print_device_table(num_devices, timings, nid);
}


int main(int argc, char ** argv)
{
  // Set program defaults before parsing the command line args.
  int list_init_random = 0;
  size_t stride = 1;
  size_t num_nodes = 1024;
  size_t num_jumps = num_nodes;
  int multi_gpu = 0;
  int print_summary_only = 0;
  int clock = 0;

  // Parse the command line args.
  for (int i = 0; i < argc; i++)
  {
    std::string str = argv[i];
    if (str == "--help" || str == "-h")
    {
      std::cout << "--nodes #    : Number of nodes in the linked list. Default value is set to 1024." << std::endl;
      std::cout << "--stride #   : Distance (in number of nodes) between two consecutive nodes. Default is 1." << std::endl;
      std::cout << "               This effectively sets the buffer size where the list is allocated to nodes*stride." << std::endl;
      std::cout << "--num-jumps #: Number of jumps during the timing of the list traversal. Default is 1024." << std::endl;
      std::cout << "--rand       : Place the nodes in the buffer in random order (default is sequential ordering)." << std::endl;
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
    else if (str == "--stride")
    {
      stride = std::stoi((std::string)argv[++i]);
    }
    else if (str == "--num-jumps")
    {
      num_jumps = std::stoi((std::string)argv[++i]);
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
      local_pointer_chase<list_type>(num_devices, list_init_random, num_nodes, stride, num_jumps, nid_name);
    }
    else
    {
      remote_pointer_chase<list_type>(num_devices, list_init_random, num_nodes, stride, num_jumps, nid_name, print_summary_only);
    }
  }

  printf("[%s] Pointer chase complete.\n", nid_name);
  return 0;
}
