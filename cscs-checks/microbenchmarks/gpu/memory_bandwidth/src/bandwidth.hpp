#ifndef __INCLUDED_BANDWIDTH__
#define __INCLUDED_BANDWIDTH__

#include<iostream>
#include<cuda.h>
#include<cuda_runtime.h>

#include "types.hpp"
#include "cuda/types.hpp"

template<class dataFrom, class dataTo, typename memcpy>
float copyBandwidth(size_t size, int device, int repeat)
{
  /*
   Returns the average time taken for a copy from data_in to data_out done
   several times. These types can represent either host data or device data.
   dataFrom and dataTo are RAII classes to handle both host and device data 
   in a much easier way (see types.hpp).
   The template parameter memcpy is a stora class, where memcpy::value has the
   right arguments for the copy, depending on whether dataFrom and dataTo are 
   host or device targeted classes.
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
    cudaMemcpyAsync(data_out.data, data_in.data, size, memcpy::value, stream);
  }

  // Do the timing
  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);
  float execution_time = 0;
  cudaEventElapsedTime(&execution_time, start, stop);
  
  // Destroy the cuda stream
  cudaStreamDestroy(stream);

  // Return the average time per copy.
  return (execution_time/(float)repeat);
}
#endif
