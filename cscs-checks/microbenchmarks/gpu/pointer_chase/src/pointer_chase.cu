#define EXPAND2(x) x; x; 
#define EXPAND4(x) EXPAND2(x) EXPAND2(x) 
#define EXPAND8(x) EXPAND4(x) EXPAND4(x) 
#define EXPAND16(x) EXPAND8(x) EXPAND8(x) 
#define EXPAND32(x) EXPAND16(x) EXPAND16(x) 
#define EXPAND64(x) EXPAND32(x) EXPAND32(x) 
#define EXPAND128(x) EXPAND24(x) EXPAND64(x)
#define EXPAND256(x) EXPAND128(x) EXPAND128(x)

#include <cstdint>
#include <iostream>


#define NODES 200
#define NODE_PADDING 0

static __device__ __forceinline__ uint32_t __clock()
{
  uint32_t x;
  asm volatile ("mov.u32 %0, %%clock;" : "=r"(x) :: "memory");
  return x;
}


struct Node
{
  Node * next = nullptr;
  char _padding[NODE_PADDING];
};


__global__ void initList(Node * head)
{
  // Set the head
  Node * prev = new (head) Node();

  // Init the rest of the list
  for (int n = 1; n < NODES+1; n++)
  {
    Node * temp = new (&(head[n])) Node();
    prev->next = temp;
    prev = temp;
  }

}


template < unsigned int repeat >
__device__ __forceinline__ void ptrChase(Node ** ptr)
{
  (*ptr) = (*ptr)->next;
  ptrChase<repeat-1>(ptr);
}

template<>
__device__ __forceinline__ void  ptrChase<0>(Node ** ptr){}


__global__ void pointer_chase(Node * head)
{
  // Create a pointer to iterate through the list
  Node * ptr = head;

  // start timer
  uint32_t start = __clock();

  ptrChase<10>(&ptr);
  
  // end cycle count
  uint32_t end = __clock();

  printf("Chase took %d cycles per node jump.\n", (end - start)/(NODES-1));
  head[0] = (*ptr);
}

void devicePointerChase()
{
  // Allocate device buffer for the list
  Node * listBuffer;
  cudaMalloc((void**)&listBuffer, sizeof(Node)*NODES);

  // Initilize the list
  initList<<<1,1>>>(listBuffer);

  // Do the chase
  pointer_chase<<<1,1>>>(listBuffer);
  cudaDeviceSynchronize();

  cudaFree(listBuffer);
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err)
  {
    std::cerr << cudaGetErrorString(err) << std::endl;
  }
}

int main()
{
  for (int i = 0; i < 10; i++)
  {
    devicePointerChase();
  }
  printf("end.\n");
}
