#include <iostream>
#include <chrono>
#include <ratio>
#include <unistd.h>
#include <cuda.h>

__global__ void null_kernel() {
};

int main(int argc, char* argv[]) {

    char hostname[256];
    hostname[255]='\0';
    gethostname(hostname, 255);

    cudaError_t error;
    int gpu_count = 0;

    error = cudaGetDeviceCount(&gpu_count);

    if (error == cudaSuccess) {
        if (gpu_count <= 0) {
            std::cout << "[" << hostname << "] " << "Could not find any gpu\n";
            return 1;
        }
        std::cout << "[" << hostname << "] " << "Found " << gpu_count << " gpu(s)\n";
    }
    else{
        std::cout << "[" << hostname << "] " << "Error getting gpu count, exiting...\n";
        return 1;
    }

    for (int i = 0; i < gpu_count; i++) {

        cudaSetDevice(i);
        // Single kernel launch to initialize cuda runtime
        null_kernel<<<1, 1>>>();

        auto t_start = std::chrono::system_clock::now();
        const int kernel_count = 1000;

        for (int i = 0; i < kernel_count; ++i) {
            null_kernel<<<1, 1>>>();
            #if SYNCKERNEL == 1
            cudaDeviceSynchronize();
            #endif
        }

        #if SYNCKERNEL != 1
        cudaDeviceSynchronize();
        #endif

        auto t_end = std::chrono::system_clock::now();
        std::cout << "[" << hostname << "] " << "[gpu " << i << "] " <<  "Kernel launch latency: " << std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(t_end - t_start).count() / kernel_count << " us\n";
    }

    return 0;
}

