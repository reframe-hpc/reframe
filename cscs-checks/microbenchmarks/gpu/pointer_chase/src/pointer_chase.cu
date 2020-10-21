#define EXPAND2(x) x; x; 
#define EXPAND4(x) EXPAND2(x) EXPAND2(x) 
#define EXPAND8(x) EXPAND4(x) EXPAND4(x) 
#define EXPAND16(x) EXPAND8(x) EXPAND8(x) 
#define EXPAND32(x) EXPAND16(x) EXPAND16(x) 
#define EXPAND64(x) EXPAND32(x) EXPAND32(x) 
#define EXPAND128(x) EXPAND24(x) EXPAND64(x)
#define EXPAND256(x) EXPAND128(x) EXPAND128(x)

#include <stdio.h>
#include <cstdint>
#include <iostream>

//using namespace std;

#define NUM_NODES 200
#define BUFFER_SIZE 400
#define NODE_PADDING 128

#if BUFFER_SIZE < NUM_NODES
# error "NUM_NODES cannot exceed BUFFER_SIZE."
#endif

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
  for (int n = 1; n <= NUM_NODES; n++)
  {
    Node * temp = new (&(head[n])) Node();
    prev->next = temp;
    prev = temp;
  }

}

__global__ void pointer_chase(Node * head, Node * nodeOut)
{
  // Create a pointer to iterate through the list
  Node * ptr = head;

  // start timer
  uint32_t start = __clock();

  // Traverse the list
  EXPAND64(ptr = ptr->next)

  // end cycle count
  uint32_t end = __clock();

  printf("Chase took %d cycles.\n", end - start);
  *nodeOut = *ptr;
}

int main()
{
  // Allocate device buffer for the list
  Node * listBuffer;
  cudaMalloc((void**)&listBuffer, sizeof(Node)*BUFFER_SIZE);
  initList<<<1,1>>>(listBuffer);
  Node * result = nullptr;
  pointer_chase<<<1,1>>>(listBuffer, result);
  cudaDeviceSynchronize();

  unsigned char buf[sizeof(int)*2] ; 
  
  // placement new in buf 
  int *pInt = new (buf) int(3);

}
