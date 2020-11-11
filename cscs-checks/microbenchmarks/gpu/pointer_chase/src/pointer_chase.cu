#include <cstdint>
#include <cassert>
#include <iostream>
#include <string>
#include <random>
#include <chrono>
#include <set>
#include <memory>

// Include the CUDA/HIP wrappers from the other test for now.
#include "../../memory_bandwidth/src/Xdevice/runtime.hpp"

/*
 ~~ GPU Linked list pointer chase algorithm ~~
 Times in clock cycles the time it takes to jump from one node to the next
 in a singly linked list.

 The list can be initialized sequentially or with a random node ordering. This
 can be controlled passing the command line argument "--rand".

 The stride and the full buffer size can be set with "--stride" and "--buffer",
 both in number of nodes.

 The macro NODES sets the total number of nodes in the list. Node that the
 list traversal is 'unrolled' inlining a recursive template, and this will
 not work if you use a large number of nodes.

 The nodes can be padded with an arbitrary size controlled by the NODE_PADDING
 macro (in Bytes).

 The LIST_TYPE macro dictates where the list is allocated. If DeviceList is used
 (default option) the linked list is allocated in device memory. In contrast, if
 HostList is used, the list is allocated as host's pinned memory.

 The links of the list can be made vlatile defining the macro VOLATILE.

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
__device__ uint32_t __clockLatency()
{
  uint32_t start = __ownClock();
  uint32_t end = __ownClock();
  return end-start;
}


__global__ void clockLatency()
{
  uint32_t clkLatency = __clockLatency();
  printf(" - Clock latency is %d.\n", clkLatency);
}


/*
 * Linked list definitions
 */

struct Node
{
  /* The node */
  Node * next = nullptr;
  char _padding[8*NODE_PADDING];
};


/*
 *  Kernels and device functions
 */

__global__ void initialize_list(Node * buffer, int stride = 1)
{
  /* List serial initializer.
   * - buffer: where the list is to be placed.
   * - stride: argument controls the number of empty node spaces
   *   in between two consecutive list nodes.
   */

  // Set the head
  Node * prev = new (&(buffer[0])) Node();

  // Init the rest of the list
  for (int n = 1; n < NODES; n++)
  {
    Node * temp = new (&(buffer[n*stride])) Node();
    prev->next = temp;
    prev = temp;
  }

}

__global__ void initialize_random_list(Node * buffer, uint32_t *indices)
{
  /* List random initializer
   * - buffer: where the list is to be placed.
   * - indices: array containing the node ordering indices as offsets in
   *     the buffer.
   */

  // Set the head
  Node * prev = new (&(buffer[indices[0]])) Node();

  // Init the rest of the list
  for (int n = 1; n < NODES; n++)
  {
    Node * temp = new (&(buffer[indices[n]])) Node();
    prev->next = temp;
    prev = temp;
  }

}

__global__ void simple_traverse(Node * __restrict__ buffer, uint32_t headIndex)
{
  /* Simple list traverse - no timing is done here
   * - buffer: where the list is
   * - headIndex: index in the buffer where the head of the list is
   */

  uint32_t count = 0;
  Node * head = &(buffer[headIndex]);
  Node * ptr = head;
  while(ptr->next != nullptr || count < NODES-1)
  {
    ptr = ptr->next;
    count++;
  }

  // Silly dep. to tell the compiler not to throw away this kernel.
  if (ptr->next == head)
  {
    printf("You had a circular list :(\n");
  }

}


#ifdef VOLATILE
# define __VOLATILE__ volatile
#else
# define __VOLATILE__
#endif

/*
 * Timed list traversal. This implementation is recursive (because it's less code) so you have to
 * watch out to not exceed the recursion limits. The functions are force-inlined, so the PTX code
 * looks identical as if you were to unwrap the recursion manually.
 *
 * Depending on the compiler flags used, the timing can either measure each node jump, or the entire
 * list traversal as a whole.
 */
template < unsigned int repeat >
__device__ __forceinline__ void nextNode( __VOLATILE__ Node ** ptr, uint32_t * timer, Node ** ptrs)
{
  /*
   * Go to the next node in the list
   */

# ifdef TIME_EACH_STEP
  uint32_t t1 = __ownClock();
# endif
  (*ptr) = (*ptr)->next;
# ifdef TIME_EACH_STEP
  (*ptrs) = (Node*)(*ptr);  // Data dep. to prevent ILP.
  *timer = __ownClock() - t1; // Time the jump
# endif

  // Keep traversing the list.
  nextNode<repeat-1>(ptr, timer+1, ptrs+1);
}

// Specialize the function to break the recursion.
template<>
__device__ __forceinline__ void  nextNode<0>( __VOLATILE__ Node ** ptr, uint32_t * timer, Node ** ptrs){}


__global__ void timed_list_traversal(Node * __restrict__ buffer, uint32_t headIndex, uint32_t * timer)
{
  /* Timed List traversal - we make a singly-linked list circular just to have a data dep. and
   * prevent from compiler optimisations.
   */

  // These are used to prevent ILP when timing each jump.
  __shared__ uint32_t s_timer[NODES-1];
  __shared__ Node * ptrs[NODES-1];

  // Create a pointer to iterate through the list
  __VOLATILE__ Node * ptr = &(buffer[headIndex]);

#ifndef TIME_EACH_STEP
  // start timer
  uint32_t start = __ownClock();
#endif

  nextNode<NODES-1>(&ptr, s_timer, ptrs);

#ifndef TIME_EACH_STEP
  // end cycle count
  uint32_t end = __ownClock();
  timer[0] = end - start;
#else
  for (uint32_t i = 0; i < NODES-1; i++)
  {
    timer[i] = s_timer[i];
  }
  if (ptr == ptrs[0])
  {
    printf("This is some data dependency that will never be executed.");
  }
#endif

  // Join the tail with the head (just for the data dependency).
  if (ptr->next == nullptr)
  {
    ptr->next = &(buffer[headIndex]);
  }

}


/*
 * List structure definitions
 */

struct List
{
  /*
   * Contains the buffer where the list is stored, the index in this buffer where the head
   * of the list is, the buffer size, and the stride in between nodes (this last one is only
   * meaningful if the list is not initialised as random).
   *
   * The member functions are:
   *  - info: prints the list details.
   *  - initialize: populatest the buffer with the list nodes.
   *  - traverse: simple list traversal.
   *  - timed_traverse: traverses the list and measures the number of cycles per node jump.
   */

  Node * buffer = nullptr;
  uint32_t headIndex = 0;
  uint32_t * timer = nullptr;
  uint32_t * d_timer = nullptr;
  size_t buffSize;
  size_t stride;

  List(size_t bSize, size_t st) : buffSize(bSize), stride(st)
  {
    // Allocate the buffers to store the timings measured in the kernel
    timer = new uint32_t[NODES];
    XMalloc((void**)&d_timer, sizeof(uint32_t)*(NODES));
  };

  virtual ~List()
  {
    XFree(d_timer);
  }

  void info(size_t n, size_t buffSize)
  {
    printf("Creating Linked list:\n");
    printf(" - Node size: %lu\n", sizeof(Node));
    printf(" - Number of nodes: %lu:\n", n);
    printf(" - Total buffer size: %10.2f MB:\n", float(sizeof(Node)*buffSize)/1024.0/1024);
    clockLatency<<<1,1>>>();
    XDeviceSynchronize();
  }

  void initialize(int mode=0)
  {
    /*
     * mode 0 initializes the list as serial.
     * mode 1 initializes the list in a random order.
     */

    if (mode < 0 || mode > 1)
    {
      printf("Unknown list initialization scheme. Defaulting back to 0.");
      mode = 0;
    }

    if (mode == 0)
    {
      initialize_list<<<1,1>>>(buffer, stride);
      XDeviceSynchronize();
    }
    else
    {
      // Random number engine.
      std::mt19937_64 rng;
      uint64_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
      std::seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed>>32)};
      rng.seed(ss);
      std::uniform_real_distribution<double> unif(0, 1);

      uint32_t * nodeIndices = (uint32_t*)malloc(sizeof(uint32_t)*NODES);
      // Create set to keep track of the assigned indices.
      std::set<uint32_t> s = {};
      for (int i = 0; i < NODES; i++)
      {
        // Get a random index.
        uint32_t currentIndex = (uint32_t)(unif(rng)*buffSize);

        // If already present in the set, find another alternative index.
        while (s.find(currentIndex) != s.end())
        {
          if (currentIndex < NODES-1)
          {
            currentIndex++;
          }
          else
          {
            currentIndex = 0;
          }
        }

        nodeIndices[i] = currentIndex;
        s.insert(currentIndex);
      }

      // Copy the node indices to the device and init the random list
      uint32_t * d_nodeIndices;
      XMalloc((void**)&d_nodeIndices, sizeof(uint32_t)*NODES);
      XMemcpy(d_nodeIndices, nodeIndices, sizeof(uint32_t)*NODES, XMemcpyHostToDevice);
      initialize_random_list<<<1,1>>>(buffer, d_nodeIndices);
      headIndex = nodeIndices[0];
      free(nodeIndices);
      XFree(d_nodeIndices);
    }

    XDeviceSynchronize();
  }

  void traverse()
  {
    /*
     * Simple list traversal - NOT timed.
     */
    simple_traverse<<<1,1>>>(buffer, headIndex);
    XDeviceSynchronize();
  }

  void time_traversal()
  {
    /*
     * Timed list traversal
     */

    timed_list_traversal<<<1,1>>>(buffer, headIndex, d_timer);
    XDeviceSynchronize();

    // Copy the timing data back to the host
    XMemcpy(timer, d_timer, sizeof(uint32_t)*(NODES-1), XMemcpyDeviceToHost);
  }

};


struct DeviceList : public List
{
  /*
   * List allocated in device memory
   */

  DeviceList(size_t n, size_t buffSize, size_t stride) : List(buffSize, stride)
  {
#   ifdef DEBUG
    List::info(n, buffSize);
#   endif
    XMalloc((void**)&buffer, sizeof(Node)*buffSize);
  }

  ~DeviceList()
  {
    XFree(buffer);
  }
};


struct HostList : public List
{
  /*
   * List allocated in pinned host memory
   */

  Node * h_buffer;
  HostList(size_t n, size_t buffSize, size_t stride) : List(buffSize,stride)
  {
#   ifdef DEBUG
    List::info(n, buffSize);
#   endif
    XHostMalloc((void**)&h_buffer, sizeof(Node)*buffSize, XHostAllocMapped);
    XHostGetDevicePointer((void**)&buffer, (void*)h_buffer, 0);
  }

  ~HostList()
  {
    XFreeHost(buffer);
  }
};


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


template < class LIST >
void remotePointerChase(int num_devices, int init_mode, size_t buffSize, size_t stride, char * nid)
{
  /*
   * Specialised pointer chase to allocate the list in one device, and do the pointer chase from another device.
   */

#ifdef SYMM
# define LIMITS j
#else
# define LIMITS 0
#endif

  auto fetch = [](uint32_t* t){return t[0]/(NODES-1);};

  printf("[%s] Memory latency (cycles) with remote direct memory access\n", nid);
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
      uint32_t timer = fetch(generalPointerChase< LIST >(i, j, init_mode, buffSize, stride));
      if (i != j)
      {
        totals += timer;
      }
      printf("%10d", timer);
    } printf("%10d\n", totals);
  }
}


int main(int argc, char ** argv)
{
  // Set program defaults before parsing the command line args.
  int list_init_mode = 0;
  size_t stride = 1;
  size_t buffSize = NODES*stride;

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

  localPointerChase<LIST_TYPE>(num_devices, list_init_mode, buffSize, stride, nid_name);
  remotePointerChase<LIST_TYPE>(num_devices, list_init_mode, buffSize, stride, nid_name);

  printf("[%s] Pointer chase complete.\n", nid_name);
  return 0;
}
