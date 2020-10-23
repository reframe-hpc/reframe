#include <cstdint>
#include <iostream>
#include <random>
#include <chrono>
#include <set>

#define NODES 128
#define NODE_PADDING 0

#ifndef NODE_STRIDE
#define NODE_STRIDE 1
#endif

#ifndef BUFFER_SIZE
# define BUFFER_SIZE NODES*NODE_STRIDE
#endif

#if (BUFFER_SIZE < NODES*NODE_STRIDE)
# error "Buffer size cannot be lower than the number of nodes."
#endif

void checkErrors()
{
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err)
  {
    std::cerr << cudaGetErrorString(err) << std::endl;
  }
}


static __device__ __forceinline__ uint32_t __clock()
{
  uint32_t x;
  asm volatile ("mov.u32 %0, %%clock;" : "=r"(x) :: "memory");
  return x;
}


static __device__ __forceinline__ uint32_t __smId()
{
  uint32_t x;
  asm volatile ("mov.u32 %0, %%smid;" : "=r"(x) :: "memory");
  return x;
}


struct Node
{
  Node * next = nullptr;
  char _padding[8*NODE_PADDING];
};


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

#ifdef VOLATILE
# define __VOLATILE__ volatile
#else
# define __VOLATILE__
#endif


template < unsigned int repeat >
__device__ __forceinline__ void nextNode( __VOLATILE__ Node ** ptr)
{
#ifdef TIME_EACH_STEP
  uint32_t t1 = __clock();
#endif
  (*ptr) = (*ptr)->next;
#ifdef TIME_EACH_STEP
  uint32_t t2 = __clock();
  printf("Single jump took %d cycles.\n" , t2-t1);
#endif
  nextNode<repeat-1>(ptr);
}

template<>
__device__ __forceinline__ void  nextNode<0>( __VOLATILE__ Node ** ptr){}


__global__ void make_circular(Node * __restrict__ buffer, uint32_t headIndex)
{
  // Create a pointer to iterate through the list
  __VOLATILE__ Node * ptr = &(buffer[headIndex]);

  // start timer
  uint32_t start = __clock();

  nextNode<NODES-1>(&ptr);
  
  // end cycle count
  uint32_t end = __clock();
  uint32_t smId = __smId();
  printf("Chase took on average %d cycles per node jump (SM %d).\n", (end - start)/(NODES-1), smId);

  // Join the tail with the head.
  if (ptr->next == nullptr)
  {
    ptr->next = &(buffer[headIndex]);
  }

}


struct List
{
  Node * buffer = nullptr;
  uint32_t headIndex = 0;

  static void info(int n)
  {
    printf("Creating Linked list:\n");
    printf(" - Node size: %d\n", sizeof(Node));
    printf(" - Number of nodes: %d:\n", n);
    printf(" - Total buffer size: %10.2f MB:\n", float(sizeof(Node)*BUFFER_SIZE)/1024.0/1024);
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
      initialize_list<<<1,1>>>(buffer, NODE_STRIDE);
      cudaDeviceSynchronize();
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
        uint32_t currentIndex = (uint32_t)(unif(rng)*BUFFER_SIZE);
        
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
      cudaMalloc((void**)&d_nodeIndices, sizeof(uint32_t)*NODES);
      cudaMemcpy(d_nodeIndices, nodeIndices, sizeof(uint32_t)*NODES, cudaMemcpyHostToDevice);
      initialize_random_list<<<1,1>>>(buffer, d_nodeIndices);
      headIndex = nodeIndices[0];
      free(nodeIndices);
      cudaFree(d_nodeIndices); 
    }

    cudaDeviceSynchronize();
    checkErrors();
  }

  void traverse()
  {
    make_circular<<<1,1>>>(buffer, headIndex);
    cudaDeviceSynchronize();
    checkErrors();
  }

};


struct DeviceList : public List
{
  DeviceList(int n) 
  {
    List::info(n);
    cudaMalloc((void**)&buffer, sizeof(Node)*BUFFER_SIZE);
  }
  ~DeviceList()
  {
    cudaFree(buffer);
  }
};

struct HostList : public List
{
  Node * h_buffer;
  HostList(int n) 
  {
    List::info(n);
    cudaHostAlloc((void**)&h_buffer, sizeof(Node)*BUFFER_SIZE, cudaHostAllocMapped);
    cudaHostGetDevicePointer((void**)&buffer, (void*)h_buffer, 0);
  }
  ~HostList()
  {
    cudaFreeHost(buffer);
  }
};


template < class LIST >
void devicePointerChase(int m)
{
  LIST l(NODES);

  l.initialize(m);
  l.traverse(); 

}

int main()
{
  devicePointerChase<DeviceList>(0);
  devicePointerChase<DeviceList>(1);
  devicePointerChase<HostList>(0);
  devicePointerChase<HostList>(1);
}
