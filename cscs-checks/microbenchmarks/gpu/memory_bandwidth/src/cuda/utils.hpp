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


#endif
