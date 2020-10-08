#ifndef __INCLUDED_CUDA_UTILS__
#define __INCLUDED_CUDA_UTILS__


#include <unistd.h>
#include <cuda.h>

#ifndef NO_ERROR_CHECK
#  define gpuErrorCheck(err) { (err); }
#else
#  define gpuErrorCheck(err) { gpuAssert((err), __FILE__, __LINE__); }
   inline void gpuAssert(cudaError_t errorCode, const char *file, int line, bool abort=true)
   {
      if (errorCode != cudaSuccess) 
      {
         char nid[80];
         gethostname(nid,80);
         fprintf(stderr,"On node %s, at %s, %s %d\n", cudaGetErrorString(errorCode), file, line);
         if (abort) 
         {
           exit(errorCode);
         }
      }
   }
#endif

void XMallocHost(void ** data, size_t size)
{
  gpuErrorCheck( cudaMallocHost(data, size) );
}

void XFreeHost(void * data)
{
  gpuErrorCheck( cudaFreeHost(data) );
}

void XMalloc(void ** data, size_t size)
{
  gpuErrorCheck( cudaMalloc(data, size) );
}

void XMemcpyAsync(void * in, void * out, size_t size, cudaMemcpyKind dir, cudaStream_t stream)
{
  gpuErrorCheck ( cudaMemcpyAsync(out, in, size, dir, stream) );
}

void XFree(void * data)
{
  gpuErrorCheck( cudaFree(data) );
}

void XGetDeviceCount(int &devices)
{
  gpuErrorCheck( cudaGetDeviceCount(&devices) ); 
}

void XSetDevice(int device)
{
  gpuErrorCheck( cudaSetDevice(device) );
}

void XStreamCreate(cudaStream_t * stream)
{
  gpuErrorCheck( cudaStreamCreate(stream) );
}

void XStreamDestroy(cudaStream_t stream)
{ 
  cudaStreamDestroy(stream);
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
