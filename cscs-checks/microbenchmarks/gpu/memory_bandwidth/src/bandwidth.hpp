#ifndef __INCLUDED_BANDWIDTH__
#define __INCLUDED_BANDWIDTH__

#include<iostream>
#include<cuda.h>
#include<cuda_runtime.h>

#include "types.hpp"

template<class dataFrom, class dataTo, cudaMemcpyKind cpyDir>
float copyBandwidth(size_t size, int device, int repeat)
{
  /*
   Returns the average time per memory copy from the structure data_in to data_out.
   These structures can be either host or device arrays (see types.hpp).
  */

  // Declare and allocate the buffers.
  dataFrom data_in(size);
  dataTo data_out(size);

  // Create a cuda stream.
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // Declare and create the cuda Events to do the timing,
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, stream);
  for ( int i = 0; i < repeat; i++ )
  {
    cudaMemcpyAsync(data_out.data, data_in.data, size, cpyDir, stream);
  }

  // Do the timing
  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);
  float execution_time = 0;
  cudaEventElapsedTime(&execution_time, start, stop);
  
  // Destroy the cuda stream
  cudaStreamDestroy(stream);

  // Return the average time per copy.
  return size/(execution_time/(float)repeat)/1e3;
}
#endif
