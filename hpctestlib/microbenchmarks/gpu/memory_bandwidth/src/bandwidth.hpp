#ifndef __INCLUDED_BANDWIDTH__
#define __INCLUDED_BANDWIDTH__

#include "types.hpp"
#include "Xdevice/runtime.hpp"

template<class dataFrom, class dataTo>
float copyBandwidth(size_t size, int device, int repeat, XMemcpyKind copy_dir)
{
  /*
   Returns the average time (ms) taken for a copy from data_dst to data_src
   done several times. These types can represent either host data or device
   data. dataFrom and dataTo are RAII classes to handle both host and device
   data in a much easier way (see types.hpp).
  */

  // Declare and allocate the buffers.
  dataFrom data_src(size);
  dataTo data_dst(size);

  // Create a X stream.
  XStream_t stream;
  XStreamCreate(&stream);

  // Instantiate the timer.
  XTimer t(stream);

  // Start the timer and run the copy
  t.start();
  for ( int i = 0; i < repeat; i++ )
  {
    XMemcpyAsync(data_dst.data, data_src.data, size, copy_dir, stream);
  }

  // Do the timing
  float execution_time = t.stop();

  // Destroy the device stream
  XStreamDestroy(stream);

  // Return the average time per copy.
  return (execution_time/(float)repeat);
}


float p2pBandwidth(size_t size, int send_device, int recv_device, int repeat, int peer_access)
{
  /*
   Time the data transfer across different devices. The peer_access argument enables or
   disables the direct memory access to another device.
  */

  // Set the sending device.
  XSetDevice(send_device);

  // Check whether the sending device has peer access to the recv_device.
  if (peer_access && recv_device!=send_device)
  {
    int hasPeerAccess;
    XDeviceCanAccessPeer(&hasPeerAccess, send_device, recv_device);
    if (!hasPeerAccess)
    {
      return (float)-1;
    }

    // Enable the peer access.
    XDeviceEnablePeerAccess(recv_device, 0);
  }

  // Allocate the send buffer.
  DeviceData data_src(size);

  // Allocate the receive buffer.
  XSetDevice(recv_device);
  DeviceData data_dst(size);

  // Set the sending device again.
  XSetDevice(send_device);

  // Create the stream.
  XStream_t stream;
  XStreamCreate(&stream);

  // Instantiate the timer.
  XTimer t(stream);

  // Start the timer and run the copy
  t.start();
  if (peer_access)
  {
    for ( int i = 0; i < repeat; i++ )
    {
      XMemcpyAsync(data_dst.data, data_src.data, size, XMemcpyDeviceToDevice, stream);
    }
  }
  else
  {
    for ( int i = 0; i < repeat; i++ )
    {
      XMemcpyPeerAsync(data_dst.data, recv_device, data_src.data, send_device, size, stream);
    }
  }

  // Do the timing
  float execution_time = t.stop();

  // Destroy the X stream
  XStreamDestroy(stream);

  // Unset the peer access
  if (peer_access && recv_device!=send_device)
  {
    XDeviceDisablePeerAccess(recv_device);
  }

  // Return the average time per copy.
  return (execution_time/(float)repeat);
}
#endif
