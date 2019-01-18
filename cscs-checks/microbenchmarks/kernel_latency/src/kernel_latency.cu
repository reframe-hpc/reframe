#include <iostream>
#include <chrono>
#include <ratio>
#include <cuda.h>

__global__ void null_kernel() {
};

int main(int argc, char* argv[]) {

    cudaError_t error;
    int gpu_count = 0;

    error = cudaGetDeviceCount(&gpu_count);

    if (error == cudaSuccess) {
        if (gpu_count <= 0) {
            std::cout << "Could not found any gpu\n";
            return 1;
        }
        std::cout << "Found " << gpu_count << " gpu(s)\n";
    }
    else{
        std::cout << "Error getting gpu count, exiting...\n";
        return 1;
    }

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
    std::cout << "Kernel launch latency: " << std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(t_end - t_start).count() / kernel_count << " us\n";

    return 0;
}

