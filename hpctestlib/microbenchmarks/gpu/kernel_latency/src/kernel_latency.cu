#include <iostream>
#include <chrono>
#include <ratio>
#include <unistd.h>

#include "Xdevice/runtime.hpp"

__global__ void null_kernel() {
};

int main(int argc, char* argv[]) {

    char hostname[256];
    hostname[255]='\0';
    gethostname(hostname, 255);

    int gpu_count = 0;
    XGetDeviceCount(&gpu_count);

    if (gpu_count <= 0) {
        std::cout << "[" << hostname << "] " << "Could not find any gpu\n";
        return 1;
    }
    std::cout << "[" << hostname << "] " << "Found " << gpu_count << " gpu(s)\n";

    for (int i = 0; i < gpu_count; i++)
    {
        XSetDevice(i);

        // Warm-up kernel
        null_kernel<<<1, 1>>>();

        auto t_start = std::chrono::system_clock::now();
        const int kernel_count = 5000;

        for (int j = 0; j < kernel_count; ++j) {
            null_kernel<<<1, 1>>>();
            #if SYNCKERNEL == 1
            XDeviceSynchronize();
            #endif
        }

        #if SYNCKERNEL != 1
        XDeviceSynchronize();
        #endif

        // End the timing
        auto t_end = std::chrono::system_clock::now();
        std::cout << "[" << hostname << "] " << "[gpu " << i << "] " <<
            "Kernel launch latency: " <<
            std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(t_end - t_start).count() / kernel_count <<
            " us\n";
    }

    return 0;
}

