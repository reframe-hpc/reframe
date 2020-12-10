#ifndef __INCLUDED_XDEV_RUNTIME__
#define __INCLUDED_XDEV_RUNTIME__

#include "defaults.hpp"

#ifdef TARGET_CUDA
# include "cuda/include.hpp"
#elif defined TARGET_HIP
# include "hip/include.hpp"
#else
# error "TARGET NOT IMPLEMENTED"
#endif

#endif // __DEFINED__XDEV_RUNTIME__
