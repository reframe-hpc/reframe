#ifndef __INCLUDED_TEST_HEADERS__
#define __INCLUDED_TEST_HEADERS__

#include <iostream>
#include <unistd.h>

// Set default platform
#if (!defined TARGET_CUDA || !defined TARGET_AMD)
#  define TARGET_CUDA
#endif

#ifdef TARGET_CUDA
# include "cuda/include.hpp"
#else
# error "TARGET NOT IMPLEMENTED"
#endif

#include "bandwidth.hpp"

#ifndef COPY_SIZE
# define COPY_SIZE 1073741824
#endif
#ifndef NUMBER_OF_COPIES
# define NUMBER_OF_COPIES 20
#endif

#endif
