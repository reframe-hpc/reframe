#ifndef __INCLUDED_CUDA_UTILS__
#define __INCLUDED_CUDA_UTILS__

#include <iostream>
#include <unistd.h>
#include <cuda.h>

static inline void checkError(cudaError_t errorCode)
{
#  ifdef DEBUG
#    ifndef HOSTNAME_SIZE
#      define HOSTNAME_SIZE 80
#    endif

   if (errorCode != cudaSuccess)
   {
      char nid[HOSTNAME_SIZE];
      gethostname(nid, HOSTNAME_SIZE);
      std::cerr << "[" << nid << "] A call to the CUDA API returned an error :" <<
        cudaGetErrorString(errorCode) << std::endl;
      exit(1);
   }
#  endif
}

void XMallocHost(void ** data, size_t size)
{
  checkError( cudaMallocHost(data, size) );
}

void XHostMalloc(void** pHost, size_t size, unsigned int flags)
{
  checkError( cudaHostAlloc(pHost, size, flags) );
}

void XFreeHost(void * data)
{
  checkError( cudaFreeHost(data) );
}

void XMalloc(void ** data, size_t size)
{
  checkError( cudaMalloc(data, size) );
}

void XMemcpy(void * out, void * in, size_t size, cudaMemcpyKind dir)
{
  checkError( cudaMemcpy(out, in, size, dir) );
}

void XMemcpyAsync(void * out, void * in, size_t size, cudaMemcpyKind dir, cudaStream_t stream)
{
  checkError( cudaMemcpyAsync(out, in, size, dir, stream) );
}

void XMemset( void * in, int val, size_t size)
{
  checkError( cudaMemset(in, val, size) );
}

void XFree(void * data)
{
  checkError( cudaFree(data) );
}

void XDeviceSynchronize()
{
  checkError( cudaDeviceSynchronize() );
}

void XGetDeviceCount(int * devices)
{
  checkError( cudaGetDeviceCount(devices) );
}

void XSetDevice(int device)
{
  checkError( cudaSetDevice(device) );
}

void XStreamCreate(cudaStream_t * stream)
{
  checkError( cudaStreamCreate(stream) );
}

void XStreamDestroy(cudaStream_t stream)
{
  checkError( cudaStreamDestroy(stream) );
}

void XDeviceCanAccessPeer(int * hasAccess, int self, int peer)
{
  checkError( cudaDeviceCanAccessPeer(hasAccess, self, peer) );
}

void XDeviceEnablePeerAccess(int peer, unsigned int flags)
{
  checkError( cudaDeviceEnablePeerAccess(peer, flags) );
}

void XDeviceDisablePeerAccess(int peer)
{
  checkError( cudaDeviceDisablePeerAccess(peer) );
}

void XMemcpyPeerAsync(void * dst, int peerDevId, void * src, int srcDevId, size_t size, cudaStream_t stream)
{
  checkError( cudaMemcpyPeerAsync(dst, peerDevId, src, srcDevId, size, stream) );
}

void XHostGetDevicePointer(void** device, void* host, unsigned int flags)
{
  checkError( cudaHostGetDevicePointer(device, host, flags) );
}

int XGetLastError()
{
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("cudaGetLastError returned an error ID %d (%s)", (int)err, cudaGetErrorString(err));
    return 1;
  }
  return 0;
}

class XTimer
{
private:
  cudaStream_t stream;
  cudaEvent_t startEvent, stopEvent;

public:
  XTimer(cudaStream_t st = 0) : stream(st)
  {
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
  }

  void start()
  {
    cudaEventRecord(startEvent, stream);
  }

  float stop()
  {
    cudaEventRecord(stopEvent, stream);
    cudaEventSynchronize(stopEvent);
    float execTime = 0;
    cudaEventElapsedTime(&execTime, startEvent, stopEvent);
    return execTime;
  }

  ~XTimer()
  {
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
  }

};

#endif
