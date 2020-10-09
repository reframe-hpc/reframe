#ifndef __DEFINED__XLIB_RUNTIME__
#define __DEFINED__XLIB_RUNTIME__

// Set default platform
#if (!defined TARGET_CUDA && !defined TARGET_HIP)
#  define TARGET_CUDA
#endif

#ifdef TARGET_CUDA
# include "cuda/include.hpp"
#elif defined TARGET_HIP
# include "hip/include.hpp"
#else
# error "TARGET NOT IMPLEMENTED"
#endif

#endif // __DEFINED__XLIB_RUNTIME__
