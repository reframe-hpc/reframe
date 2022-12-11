#ifndef INCLUDED_BLAS
#define INCLUDED_BLAS

#include "defaults.hpp"

/*
 * Interface to Nvidia's cuBLAS-v2 and ROCm's rocBLAS.
 */

#ifdef TARGET_CUDA
# include "cuda/blas.hpp"
#elif defined TARGET_HIP
# include "hip/blas.hpp"
#else
# error "TARGET NOT IMPLEMENTED"
#endif

#endif
