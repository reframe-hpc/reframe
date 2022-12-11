#ifndef INCLUDED_SMI
#define INCLUDED_SMI

#include "defaults.hpp"

/*
 *
 * This is meant to be a bit of a friendlier interface for both
 * Nvidia's NVML and ROCm's RSMI.
 *
 * Dependencies:
 * If compiled with TARGET_HIP, this library must be linked with libnuma.
 *
 * See common/smi.hpp for the library description.
 *
 */

#include "common/smi.hpp"

#ifdef TARGET_CUDA
# include "cuda/smi.hpp"
#elif defined TARGET_HIP
# include "hip/smi.hpp"
#else
# error "TARGET NOT IMPLEMENTED"
#endif

#endif
