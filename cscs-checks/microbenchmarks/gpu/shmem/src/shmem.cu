
// Shared memory bandwidth benchmark
// contributed by Sebastian Keller
//
// Relevant nvprof metrics:
// nvprof -m shared_load_throughput,shared_store_throughput

#include <iostream>
#include <malloc.h>
#include <unistd.h>

#include "Xdevice/runtime.hpp"

#define NTHREADS 256
#define NITER    4096
// length of the thread block swap chain (must be even)
#define SHARED_SEGMENTS 4

template <class T>
__device__ void swap(T* a, T* b)
{
    T tmp = *a;
    *a = *b;
    *b = tmp;
}

template <class T>
__global__ void test_shmem(T* glob_mem)
{
    __shared__ T smem[NTHREADS*SHARED_SEGMENTS];

    int tid = threadIdx.x;

    smem[tid] = T{0};
    for (int i = 0; i < NITER; ++i)
    {
        // even shared segments
        for (int j = 0; j < SHARED_SEGMENTS-1; j+=2)
            swap(smem + tid + j*blockDim.x, smem + tid + (j+1)*blockDim.x);

        // uneven shared segments
        for (int j = 1; j < SHARED_SEGMENTS-2; j+=2)
            swap(smem + tid + j*blockDim.x, smem + tid + (j+1)*blockDim.x);
    }

    glob_mem[blockIdx.x * blockDim.x + tid] = smem[tid];
}

template <class T>
double test_bw(long size)
{
    T* dev_buffer;
    XMalloc((void**)&dev_buffer, size);
    int nblocks = size / (NTHREADS * sizeof(T));

    // Create a stream to attach the timer to.
    XStream_t stream;
    XStreamCreate(&stream);

    // Instantiate the timer
    XTimer t(stream);

    t.start();
    test_shmem<<<nblocks, NTHREADS, 0, stream>>>(dev_buffer);

    // convert to seconds
    double gpu_time = t.stop()/double(1000);

    // 2 writes + 2 reads per swap
    double nbytes = NITER * size * (SHARED_SEGMENTS-1) * 4;

    XStreamDestroy(stream);
    XFree(dev_buffer);

    return nbytes / gpu_time;
}

int main()
{
    long size = 1024 * 1024 * 64; // 64 MB global buffer

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
        // warmup
        test_bw<int>(size);

        // test
        std::cout << "[" << hostname << "] GPU " << i << ": Bandwidth(int) " << test_bw<int>(size) / 1024 / 1024 / 1024 << " GB/s" << std::endl;
        std::cout << "[" << hostname << "] GPU " << i << ": Bandwidth(double) " << test_bw<double>(size) / 1024 / 1024 / 1024 << " GB/s" << std::endl;
    }
}
