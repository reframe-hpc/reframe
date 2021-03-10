#ifndef __INCLUDED_TEST_HEADERS__
#define __INCLUDED_TEST_HEADERS__

#include <iostream>
#include <unistd.h>

#include "Xdevice/runtime.hpp"
#include "Xdevice/smi.hpp"

#include "bandwidth.hpp"

#ifndef COPY_SIZE
# define COPY_SIZE 1073741824
#endif
#ifndef NUMBER_OF_COPIES
# define NUMBER_OF_COPIES 20
#endif
#ifndef HOSTNAME_SIZE
# define HOSTNAME_SIZE 80
#endif

#endif
