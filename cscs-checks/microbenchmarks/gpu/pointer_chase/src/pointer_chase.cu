#include <cstdint>
#include <cassert>
#include <iostream>
#include <string>
#include <random>
#include <chrono>
#include <set>

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

void checkErrors()
{
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err)
  {
    std::cerr << cudaGetErrorString(err) << std::endl;
  }
}


static __device__ uint32_t __clockLatency()
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

// The node
struct Node
{
  Node * next = nullptr;
  char _padding[8*NODE_PADDING];
};

// List serial initializer
__global__ void initialize_list(Node * head, int stride = 1)
{
  // Set the head
  Node * prev = new (&(head[0])) Node();

  // Init the rest of the list
  for (int n = 1; n < NODES; n++)
  {
    Node * temp = new (&(head[n*stride])) Node();
    prev->next = temp;
    prev = temp;
  }

}

// List random initializer
__global__ void initialize_random_list(Node * buffer, uint32_t *indices)
{
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

// Simple list traverse without any timers
__global__ void simple_traverse(Node * __restrict__ buffer, uint32_t headIndex)
{
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
__device__ __forceinline__ void nextNode( __VOLATILE__ Node ** ptr, uint32_t * timings, Node ** ptrs)
{
# ifdef TIME_EACH_STEP
  uint32_t t1 = __ownClock();
# endif
  (*ptr) = (*ptr)->next;
# ifdef TIME_EACH_STEP
  (*ptrs) = (Node*)(*ptr);  // Data dep. to prevent ILP.
  *timings = __ownClock() - t1; // Time the jump
# endif

  // Keep traversing the list.
  nextNode<repeat-1>(ptr, timings+1, ptrs+1);
}

// Specialize the function to break the recursion.
template<>
__device__ __forceinline__ void  nextNode<0>( __VOLATILE__ Node ** ptr, uint32_t * timings, Node ** ptrs){}


/* List traversal to make a singly-linked list circular. This is just to have a data dependency and
 * cover from a potential compiler optimization that might throw the list traversal away.
 */
__global__ void make_circular(Node * __restrict__ buffer, uint32_t headIndex)
{

  // These are used to prevent ILP when timing each jump.
  __shared__ uint32_t timings[NODES-1];
  __shared__ Node * ptrs[NODES-1];
  uint32_t sum = 0;

  // Create a pointer to iterate through the list
  __VOLATILE__ Node * ptr = &(buffer[headIndex]);

#ifndef TIME_EACH_STEP
  // start timer
  uint32_t start = __ownClock();
#endif

  nextNode<NODES-1>(&ptr, timings, ptrs);

#ifndef TIME_EACH_STEP
  // end cycle count
  uint32_t end = __ownClock();
  sum = end - start;
#else
  printf("Latency for each node jump:\n");
  for (uint32_t i = 0; i < NODES-1; i++)
  {
    printf("%d\n", timings[i]);
    sum += timings[i];
  }
  if (ptr == ptrs[0])
  {
    printf("This is some data dependency that will never be executed.");
  }
#endif

  printf("Chase took on average %d cycles per node jump (SM %d).\n", sum/(NODES-1), __smId());

  // Join the tail with the head (just for the data dependency).
  if (ptr->next == nullptr)
  {
    ptr->next = &(buffer[headIndex]);
  }

}


struct List
{
  Node * buffer = nullptr;
  uint32_t headIndex = 0;
  size_t buffSize;
  size_t stride;

  List(size_t bSize, size_t st) : buffSize(bSize), stride(st) {};

  static void info(size_t n, size_t buffSize)
  {
    printf("Creating Linked list:\n");
    printf(" - Node size: %d\n", sizeof(Node));
    printf(" - Number of nodes: %d:\n", n);
    printf(" - Total buffer size: %10.2f MB:\n", float(sizeof(Node)*buffSize)/1024.0/1024);
    clockLatency<<<1,1>>>();
    XDeviceSynchronize();
  }

  void initialize(int mode=0)
  {
    if (mode < 0 || mode > 1)
    {
      printf("Unknown list initialization scheme. Default to 0.");
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
        if(s.find(currentIndex) != s.end())
        {
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

        }
        nodeIndices[i] = currentIndex;
        s.insert(currentIndex);
      }
      uint32_t * d_nodeIndices;
      XMalloc((void**)&d_nodeIndices, sizeof(uint32_t)*NODES);
      XMemcpy(d_nodeIndices, nodeIndices, sizeof(uint32_t)*NODES, XMemcpyHostToDevice);
      initialize_random_list<<<1,1>>>(buffer, d_nodeIndices);
      headIndex = nodeIndices[0];
      free(nodeIndices);
      XFree(d_nodeIndices);
    }

    XDeviceSynchronize();
    //checkErrors();
  }

  void traverse()
  {
    simple_traverse<<<1,1>>>(buffer, headIndex);
    XDeviceSynchronize();
    //checkErrors();
  }
  void time_traversal()
  {
    make_circular<<<1,1>>>(buffer, headIndex);
    XDeviceSynchronize();
    //checkErrors();
  }

};


struct DeviceList : public List
{
  DeviceList(size_t n, size_t buffSize, size_t stride) : List(buffSize, stride)
  {
    List::info(n, buffSize);
    XMalloc((void**)&buffer, sizeof(Node)*buffSize);
  }
  ~DeviceList()
  {
    XFree(buffer);
  }
};

struct HostList : public List
{
  Node * h_buffer;
  HostList(size_t n, size_t buffSize, size_t stride) : List(buffSize,stride)
  {
    List::info(n, buffSize);
    XHostAlloc((void**)&h_buffer, sizeof(Node)*buffSize, XHostAllocMapped);
    XHostGetDevicePointer((void**)&buffer, (void*)h_buffer, 0);
  }
  ~HostList()
  {
    XFreeHost(buffer);
  }
};


template < class LIST >
void devicePointerChase(int m, size_t buffSize, size_t stride)
{
  LIST l(NODES, buffSize, stride);

  l.initialize(m);
  l.traverse(); // warmup kernel
  l.time_traversal();

}

int main(int argc, char ** argv)
{
  // Set program defaults before parsing the command line args.
  int list_init = 0;
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
      list_init = 1;
    }
    else if (str == "--stride")
    {
      stride = std::stoi((std::string)argv[++i]);
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


  // Run the pointer chase.
  devicePointerChase<LIST_TYPE>(list_init, buffSize, stride);
}
