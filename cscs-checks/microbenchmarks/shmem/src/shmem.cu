
// Shared memory bandwidth benchmark
// contributed by Sebastian Keller
// 
// Relevant nvprof metrics:
// nvprof -m shared_load_throughput,shared_store_throughput

#include <iostream>

#include <malloc.h>
#include <cuda.h>
#include <cuda_runtime.h>


#define NTHREADS 256
#define NITER    4096
// length of the thread block swap chain (must be even)
#define SHARED_SEGMENTS 4

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

template <class T>
__device__ void swap(T* a, T* b)
{
    T tmp;
    tmp = *a;
    *a = *b;
    // +1 isn't needed to prevent code elimination by the
    // compiler, but is added in case it gets smarter in
    // a future version
    *b = tmp + T{1};
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
    T* buffer = (T*)malloc(size);
    T* dev_buffer; 
    HANDLE_ERROR( cudaMalloc((void**)&dev_buffer, size) );
    int nblocks = size / (NTHREADS * sizeof(T));

    cudaEvent_t start, stop;
    HANDLE_ERROR( cudaEventCreate(&start) );
    HANDLE_ERROR( cudaEventCreate(&stop) );
    HANDLE_ERROR( cudaEventRecord(start,0) );

    test_shmem<<<nblocks, NTHREADS>>>(dev_buffer);

    HANDLE_ERROR( cudaEventRecord(stop,0) );
    HANDLE_ERROR( cudaEventSynchronize(stop) );
    float gpu_time;
    HANDLE_ERROR( cudaEventElapsedTime( &gpu_time, start, stop ) );
    // convert to seconds
    gpu_time /= 1000;

    // 2 writes + 2 reads per swap
    double nbytes = NITER * size * (SHARED_SEGMENTS-1) * 4;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(buffer);
    cudaFree(dev_buffer);

    return nbytes / gpu_time;
}

int main()
{
    long size = 1024 * 1024 * 64; // 64 MB global buffer

    // warmup
    test_bw<int>(size);

    std::cout << "Bandwidth(int) " << test_bw<int>(size) / 1024 / 1024 / 1024 << " GB/s" << std::endl;
    std::cout << "Bandwidth(double) " << test_bw<double>(size) / 1024 / 1024 / 1024 << " GB/s" << std::endl;
}
