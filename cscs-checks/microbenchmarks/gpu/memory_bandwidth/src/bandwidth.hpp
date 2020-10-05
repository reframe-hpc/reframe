#ifndef __INCLUDED_BANDWIDTH__
#define __INCLUDED_BANDWIDTH__

#include<iostream>

#include "types.hpp"
#include "cuda/include.hpp"

template<class dataFrom, class dataTo, typename memcpy>
float copyBandwidth(size_t size, int device, int repeat)
{
  /*
   Returns the average time taken for a copy from data_in to data_out done
   several times. These types can represent either host data or device data.
   dataFrom and dataTo are RAII classes to handle both host and device data 
   in a much easier way (see types.hpp).
   The template parameter memcpy is a storage class, where memcpy::value has the
   right arguments for the copy, depending on whether dataFrom and dataTo are 
   host or device targeted classes.
  */

  // Declare and allocate the buffers.
  dataFrom data_in(size);
  dataTo data_out(size);

  // Create a cuda stream.
  XStream_t stream;
  XStreamCreate(&stream);

  // Instantiate the timer.
  XTimer t(stream);

  // Start the timer and run the copy
  t.start();
  for ( int i = 0; i < repeat; i++ )
  {
    XMemcpyAsync(data_out.data, data_in.data, size, memcpy::value, stream);
  }

  // Do the timing
  float execution_time = t.stop(); 

  // Destroy the cuda stream
  XStreamDestroy(stream);

  // Return the average time per copy.
  return (execution_time/(float)repeat);
}
#endif
