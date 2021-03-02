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

__global__ void simple_traversal(Node * __restrict__ buffer)
{
  Node * ptr = buffer;
  while(ptr->next != buffer)
  {
    ptr = ptr->next;
  }

  // Silly dep. to tell the compiler not to throw away this kernel.
  if (ptr->next->next == buffer)
  {
    printf("The list has only 1 node\n");
  }
}


__global__ void timed_traversal(Node * __restrict__ buffer, size_t num_jumps, uint64_t * timer)
{
  // start timer
  XClocks64 clocks;
  clocks.start();

  // Traverse the list
  while(num_jumps--)
  {
    buffer = buffer->next;
  }

  // end cycle count
  timer[0] = clocks.end();

  // Just for the data dependency - the list is already circular.
  if (buffer->next == nullptr)
  {
    buffer->next = buffer;
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
  uint64_t timer;
  uint64_t * d_timer = nullptr;
  size_t stride;

  List(int n, size_t st) : num_nodes(n), stride(st)
  {
    // Allocate the buffers to store the timings measured in the kernel
    XMalloc((void**)&d_timer, sizeof(uint64_t));
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
    simple_traversal<<<1,1>>>(buffer);
    XDeviceSynchronize();
  }

  void time_traversal(size_t num_jumps)
  {
    /*
     * Timed list traversal
     */

    timed_traversal<<<1,1>>>(buffer, num_jumps, d_timer);
    XDeviceSynchronize();

    // Copy the timing data back to the host
    XMemcpy(&timer, d_timer, sizeof(uint64_t), XMemcpyDeviceToHost);
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
