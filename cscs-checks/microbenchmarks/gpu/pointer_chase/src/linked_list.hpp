#include "Xdevice/smi.hpp"

/*
 *
 * Singly linked list implementation for GPUs
 *
 */


__global__ void clock_latency(int * clk)
{
  // This returns the clock latency when reading the 64-bit clock counter.
  clk[0] = XClockLatency<int>();
}


void print_clock_latency(char * nid, int dev)
{
  /* Prints the latency of reading the clock cycles */
  int * clk_d;
  int clk;
  XSetDevice(dev);
  XMalloc((void**)&clk_d, sizeof(int));
  clock_latency<<<1,1>>>(clk_d);
  XDeviceSynchronize();
  XMemcpy(&clk, clk_d, sizeof(int), XMemcpyDeviceToHost);
  XFree(clk_d);
  printf("[%s] The clock latency on device %d is %d cycles.\n", nid, dev, clk);
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

__global__ void initialize_list(Node * buffer, int num_nodes, int stride = 1)
{
  /* List serial initializer.
   * - buffer: where the list is to be placed.
   * - stride: argument controls the number of empty node spaces
   *   in between two consecutive list nodes.
   */

  // Set the head
  Node * prev = buffer;

  // Init the rest of the list
  for (int n = 1; n < num_nodes; n++)
  {
    Node * temp = buffer + n*stride;
    prev->next = temp;
    prev = temp;
  }
  prev->next = buffer;

}

__global__ void initialize_random_list(Node * buffer, int num_nodes, uint32_t *indices)
{
  /* List random initializer
   * - buffer: where the list is to be placed.
   * - indices: array containing the node ordering indices as offsets in
   *     the buffer.
   */

  // Set the head
  Node * prev = buffer + indices[0];

  // Init the rest of the list
  for (int n = 1; n < num_nodes; n++)
  {
    Node * temp = buffer + indices[n];
    prev->next = temp;
    prev = temp;
  }
  prev->next = buffer + indices[0];

}

__global__ void simple_traverse(Node * __restrict__ buffer, uint32_t head_index)
{
  /* Simple list traverse - no timing is done here
   * - buffer: where the list is
   * - head_index: index in the buffer where the head of the list is
   */

  Node * head = &(buffer[head_index]);
  Node * ptr = head;
  while(ptr->next != head)
  {
    ptr = ptr->next;
  }

  // Silly dep. to tell the compiler not to throw away this kernel.
  if (ptr->next->next == head)
  {
    printf("The impossible just happened\n");
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
__device__ __forceinline__ void next_node( __VOLATILE__ Node ** ptr, uint32_t * timer, Node ** ptrs, int & jump)
{
  /*
   * Recursive function to traverse the list.
   * - ptr: Pointer of a pointer to a node in the linked list.
   * - timer: Array to store the timings of each individual node jump.
   *   Only used if this option is activated (-DTIME_EACH_STEP)
   * - ptrs: Just used to have a data dependency to block ILP.
   * - jump: Int to keep track of the number of jumps
   */

# ifdef TIME_EACH_STEP
  XClocks64 clocks;
  clocks.start();
# endif
  (*ptr) = (*ptr)->next;
# ifdef TIME_EACH_STEP
  *(ptrs+jump) = (*ptr);  // Data dep. to prevent ILP.
  *(timer+jump) = clocks.end();    // Time the jump
# endif

  jump++;
}


__global__ void timed_list_traversal(Node * __restrict__ buffer, uint32_t head_index, uint32_t * timer)
{
  /* Timed List traversal - we make a singly-linked list circular just to have a data dep. and
   * cover from compiler optimisations.
   */

  // These are used to prevent ILP when timing each jump.
  __shared__ uint32_t s_timer[JUMPS];
  __shared__ Node * ptrs[JUMPS];

  // Create a pointer to iterate through the list
  __VOLATILE__ Node * ptr = &(buffer[head_index]);

  // Node jump counter
  int jump = 0;

#ifndef TIME_EACH_STEP
  // start timer
  XClocks64 clocks;
  clocks.start();
#endif

  // Traverse the list
  REPEAT_JUMPS(next_node(&ptr, s_timer, ptrs, jump);)

#ifndef TIME_EACH_STEP
  // end cycle count
  timer[0] = clocks.end();
#else
  for (uint32_t i = 0; i < JUMPS; i++)
  {
    timer[i] = s_timer[i];
  }
  if (ptrs[1] == ptrs[0])
  {
    printf("This is some data dependency that will never be executed.");
  }
#endif

  // Just for the data dependency - the list is already circular.
  if (ptr->next == nullptr)
  {
    ptr->next = &(buffer[head_index]);
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
   *  - initialize: populate the buffer with the list nodes.
   *  - traverse: simple list traversal.
   *  - timed_traverse: traverses the list and measures the number of cycles per node jump.
   */

  uint32_t num_nodes;
  Node * buffer = nullptr;
  uint32_t head_index = 0;
  uint32_t * timer = nullptr;
  uint32_t * d_timer = nullptr;
  size_t stride;

  List(int n, size_t st) : num_nodes(n), stride(st)
  {
    // Allocate the buffers to store the timings measured in the kernel
    timer = new uint32_t[JUMPS];
    XMalloc((void**)&d_timer, sizeof(uint32_t)*(JUMPS));
  };

  virtual ~List()
  {
    XFree(d_timer);
  }

  void info()
  {
    printf("Creating Linked list:\n");
    printf(" - Node size: %lu\n", sizeof(Node));
    printf(" - Number of nodes: %lu:\n", num_nodes);
    printf(" - Total buffer size: %10.2f MB:\n", float(sizeof(Node)*num_nodes*stride)/1024.0/1024);
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
      initialize_list<<<1,1>>>(buffer, num_nodes, stride);
      XDeviceSynchronize();
    }
    else
    {
      // Random number engine.
      std::random_device rd;
      std::mt19937_64 gen(rd());
      uint32_t * node_indices = (uint32_t*)malloc(sizeof(uint32_t)*num_nodes);
      std::iota(node_indices, node_indices + num_nodes, 0);
      std::shuffle(node_indices, node_indices + num_nodes, gen);

      // Copy the node indices to the device and init the random list
      uint32_t * d_node_indices;
      XMalloc((void**)&d_node_indices, sizeof(uint32_t)*num_nodes);
      XMemcpy(d_node_indices, node_indices, sizeof(uint32_t)*num_nodes, XMemcpyHostToDevice);
      initialize_random_list<<<1,1>>>(buffer, num_nodes, d_node_indices);
      head_index = node_indices[0];
      free(node_indices);
      XFree(d_node_indices);
    }

    XDeviceSynchronize();
  }

  void traverse()
  {
    /*
     * Simple list traversal - NOT timed.
     */
    simple_traverse<<<1,1>>>(buffer, head_index);
    XDeviceSynchronize();
  }

  void time_traversal()
  {
    /*
     * Timed list traversal
     */

    timed_list_traversal<<<1,1>>>(buffer, head_index, d_timer);
    XDeviceSynchronize();

    // Copy the timing data back to the host
    XMemcpy(timer, d_timer, sizeof(uint32_t)*JUMPS, XMemcpyDeviceToHost);
  }

};


struct DeviceList : public List
{
  /*
   * List allocated in device memory
   */

  DeviceList(size_t num_nodes, size_t stride) : List(num_nodes, stride)
  {
#   ifdef DEBUG
    List::info();
#   endif
    XMalloc((void**)&buffer, sizeof(Node)*num_nodes*stride);
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
  HostList(size_t num_nodes, size_t stride) : List(num_nodes, stride)
  {
#   ifdef DEBUG
    List::info();
#   endif
    XHostMalloc((void**)&h_buffer, sizeof(Node)*num_nodes*stride, XHostAllocMapped);
    XHostGetDevicePointer((void**)&buffer, (void*)h_buffer, 0);
  }

  ~HostList()
  {
    XFreeHost(buffer);
  }
};


