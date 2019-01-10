#include <iostream>
#include <chrono>
#include <cuda.h>

__global__ void null_kernel() {
};

int main(int argc, char* argv[]) {

    cudaError_t error;
    int gpu_count;

    error = cudaGetDeviceCount(&gpu_count);

    if (error == cudaSuccess) {
        std::cout << "Found " << gpu_count << " gpu(s)" << std::endl;
    }
    else {
        std::cout << "Error getting gpu count, exiting..." << std::endl;
        return -1;
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

    std::cout << "Kernel launch latency: " << std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count() / kernel_count << " seconds" << std::endl;
}

