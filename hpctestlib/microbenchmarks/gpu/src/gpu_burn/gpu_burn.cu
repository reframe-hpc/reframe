/*
 * Modified for CSCS by
 *
 * - Mark Klein (kleinm@cscs.ch) for adding support of NVML bindings and reduced
 *   output
 * - Javier Otero (javier.otero@cscs.ch) to support both HIP and CUDA.
 * - Vasileios Karakasis (vasileios.karakasis@cscs.ch) to support selection of
 *   the devices to be burnt.
 *
 * original gpu_burn
 * Copyright (c) 2016, Ville Timonen
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the FreeBSD Project.
 */

#define SIZE 2048ul // Matrices are SIZE*SIZE..  2048^2 should be efficiently implemented in CUBLAS
#define USEMEM 0.9  // Try to allocate 90% of memory

// Operations per matrix multiply
#define OPS_PER_MUL (2*SIZE*SIZE*SIZE-SIZE*SIZE)

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <condition_variable>
#include <type_traits>
#include <vector>
#include <array>
#include <mutex>
#include <algorithm>
#include <functional>
#include <memory>

#include "Xdevice/runtime.hpp"
#include "Xdevice/smi.hpp"
#include "Xdevice/blas.hpp"

// Actually, there are no rounding errors due to results being accumulated in an arbitrary order.
// Therefore EPSILON = 0.0f is OK
#define EPSILON 0.001f
#define EPSILOND 0.0000001

namespace kernels
{
  template<class T>
  __global__ void compare(T *C, int *numberOfErrors, size_t iters) {
      size_t iterStep = blockDim.x*blockDim.y*gridDim.x*gridDim.y;
      size_t myIndex = (blockIdx.y*blockDim.y + threadIdx.y)* // Y
                        gridDim.x*blockDim.x + // W
                        blockIdx.x*blockDim.x + threadIdx.x; // X

      int localErrors = 0;
      for (size_t i = 1; i < iters; ++i)
          if (fabs(C[myIndex] - C[myIndex + i*iterStep]) > EPSILOND)
              localErrors++;

      atomicAdd(numberOfErrors, localErrors);
  }
}

template <class T> class GemmTest
{
private:
  int deviceId;

  // SMI handle to do system queries.
  Smi * smi_handle;

  // Iterations per call to this->compute
  size_t iters;

  long long int totalErrors;

  // Work arrays
  T * d_C;
  T * d_A;
  T * d_B;

  int * d_numberOfErrors;

  XblasHandle_t d_blas;
  static const int g_blockSize = 16;

public:
  GemmTest(int id, Smi * smi_hand) : deviceId(id), smi_handle(smi_hand)
  {
      // Set the device and pin thread to CPU with
      XSetDevice(deviceId);
      smi_handle->setCpuAffinity(deviceId);

      // Create blas plan
      XblasCreate(&d_blas);

      totalErrors = 0;
  }
  ~GemmTest()
  {
      XFree(d_C);
      XFree(d_A);
      XFree(d_B);
      XFree(d_numberOfErrors);
      XblasDestroy(d_blas);
      XDeviceSynchronize();
  }

  unsigned long long int getErrors()
  {
      unsigned long long int tempErrs = totalErrors;
      totalErrors = 0;
      return tempErrs;
  }

  size_t getIters()
  {
      return iters;
  }

  size_t availMemory()
  {
      size_t freeMem;
      smi_handle->getDeviceAvailMemorySize(deviceId, &freeMem);
      return freeMem;
  }

  void initBuffers(T * h_A, T * h_B)
  {
      size_t useBytes = (size_t)((double)availMemory()*USEMEM);
      size_t d_resultSize = sizeof(T)*SIZE*SIZE;
      iters = (useBytes - 2*d_resultSize)/d_resultSize; // We remove A and B sizes
      XMalloc((void**)&d_C, iters*d_resultSize);
      XMalloc((void**)&d_A, d_resultSize);
      XMalloc((void**)&d_B, d_resultSize);
      XMalloc((void**)&d_numberOfErrors, sizeof(int));

      // Populating matrices A and B
      XMemcpy(d_A, h_A, d_resultSize, XMemcpyHostToDevice);
      XMemcpy(d_B, h_B, d_resultSize, XMemcpyHostToDevice);
  }

  void compute() = delete;

  void compare() {
      int numberOfErrors;
      XMemset(d_numberOfErrors, 0, sizeof(int));
      dim3 block(g_blockSize,g_blockSize);
      dim3 grid(SIZE/g_blockSize,SIZE/g_blockSize);
      kernels::compare<T><<<grid,block>>>((T*)d_C,(int*)d_numberOfErrors,(size_t)iters);

      XMemcpy(&numberOfErrors, d_numberOfErrors, sizeof(int), XMemcpyDeviceToHost);
      if (numberOfErrors) {
          totalErrors += (long long int) numberOfErrors;
          printf("WE FOUND %d FAULTY ELEMENTS from GPU %d\n", numberOfErrors, deviceId);
      }
  }
};

template<>
void GemmTest<double>::compute()
{
    static const double alpha = 1.0;
    static const double beta = 0.0;
    for (size_t i = 0; i < iters; ++i)
    {
        XblasDgemm(d_blas,
                   XBLAS_OP_N, XBLAS_OP_N,
                   SIZE, SIZE, SIZE,
                   &alpha,
                   (const double*)d_A, SIZE,
                   (const double*)d_B, SIZE,
                   &beta,
                   d_C + i*SIZE*SIZE, SIZE);
    }
}

template<>
void GemmTest<float>::compute()
{
    static const float alpha = 1.0;
    static const float beta = 0.0;
    for (size_t i = 0; i < iters; ++i)
    {
        XblasSgemm(d_blas,
                   XBLAS_OP_N, XBLAS_OP_N,
                   SIZE, SIZE, SIZE,
                   &alpha,
                   (const float*)d_A, SIZE,
                   (const float*)d_B, SIZE,
                   &beta,
                   d_C + i*SIZE*SIZE, SIZE);
    }
}


class BurnTracker
{
    /* Timing class that keeps track of the progress made by a single thread
     * through the burn process.
     *
     * All the member functions are thread-safe. These could be accessed by the
     * master/slave thread at any time to read/write data.
     *
     * When the read function, the counters are reset.
     */
public:
    std::mutex mtx;
    size_t iters, reps, err;
    float devTemp;
    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::duration<double> compare_time;

    BurnTracker()
    {
        std::lock_guard<std::mutex> lg(mtx);
        err = 0; iters = 0; reps = 0;
        devTemp = 0.0;
    };

    void set_iters(size_t it)
    {
        std::lock_guard<std::mutex> lg(mtx);
        iters = it;
    }

    void set_compare_time(std::chrono::duration<double> t)
    {
        std::lock_guard<std::mutex> lg(mtx);
        compare_time = t;
    }

    void start_timer()
    {
        std::lock_guard<std::mutex> lg(mtx);
        start = std::chrono::high_resolution_clock::now();
    }

    void log(size_t e, Smi *smi_handle, int devId)
    {
        float temp;

        std::lock_guard<std::mutex> lg(mtx);

        end = std::chrono::high_resolution_clock::now();

        smi_handle->getGpuTemp(devId, &temp);
        if (temp > devTemp) {
            devTemp = temp;
        }

        reps++;
        err += e;
    }

    float getTemp()
    {
        std::lock_guard<std::mutex> lg(mtx);
        float temp = devTemp;
        devTemp = 0.0;
        return temp;
    }

    double read()
    {
        std::lock_guard<std::mutex> lg(mtx);

        // Failure checking
        if (err)
            return -1;

        if (reps == 0) {
            printf("Warning: duration is too short, didn't finish a single repetition\n");
            return -1;
        }

        // Get the time difference and return the flops
        std::chrono::duration<double> diff = end-start;
        double Gflops = 1e-9 * iters * reps * OPS_PER_MUL / (diff - compare_time*reps).count();

        // Reset the counters
        err = 0;
        reps = 0;
        start = end;
        return Gflops;
    }
};


// Global vars for inter-thread communication.
std::condition_variable cv;
std::mutex cv_m;
volatile bool burn = false;
volatile int startUpCounter = 0;
int devCount;


template<class T>
void startBurn(int devId,
               Smi * smi_handle, T *A, T *B,
               BurnTracker * bt)
{
    std::chrono::high_resolution_clock::time_point warmup_start, warmup_end;
    std::chrono::duration<double> warmup_diff;

    GemmTest<T> test(devId, smi_handle);
    test.initBuffers(A, B);

    // Log the number of iterations per compute call
    bt->set_iters(test.getIters());

    // Warmup burn
    warmup_start = std::chrono::high_resolution_clock::now();
    test.compute();
    XDeviceSynchronize();
    warmup_end = std::chrono::high_resolution_clock::now();
    warmup_diff = warmup_end-warmup_start;
    printf("GPU %2d: Warmup computation takes %g seconds, duration must be larger than that to get any results\n", devId, warmup_diff.count());
    fflush(stdout);
    warmup_start = std::chrono::high_resolution_clock::now();
    for (int i=0; i < 100; i++) {
        test.compare();
        test.getErrors();
    }
    warmup_end = std::chrono::high_resolution_clock::now();
    warmup_diff = (warmup_end-warmup_start)/100;
    bt->set_compare_time(warmup_diff);

    XDeviceSynchronize();
    {
        // Flag that this thread is done with the warmup.
        std::lock_guard<std::mutex> lg(cv_m);
        ++startUpCounter;
        cv.notify_all();
    }

    // Hold off any computation until all threads are go.
    {
        std::unique_lock<std::mutex> lk(cv_m);
        cv.wait(lk, []{return burn;});
    }
    bt->start_timer();

    // The actual work
    while (burn) {
        test.compute();
        test.compare();

        // Update the results
        bt->log(test.getErrors(), smi_handle, devId);
    }
}


template<class T> void launch(int duration, const std::vector<int> &devices)
{
    // Initializing A and B with random data
    T *A = (T*) malloc(sizeof(T)*SIZE*SIZE);
    T *B = (T*) malloc(sizeof(T)*SIZE*SIZE);
    srand(10);
    for (size_t i = 0; i < SIZE*SIZE; ++i) {
        A[i] = (T)((double)(rand()%1000000)/100000.0);
        B[i] = (T)((double)(rand()%1000000)/100000.0);
    }

    // Initialise the SMI
    Smi smi_handle;

    // Here burn is a switch that holds and breaks the work done by the slave threads.
    burn = false;
    std::vector<std::thread> threads;

    // All the burn info is stored in instances of the BurnTracker class.
    size_t devCount = devices.size();
    BurnTracker **trackThreads = new BurnTracker*[devCount];

    // Create one thread per device - burn is still off here.
    for (int i = 0; i < devCount; i++)
    {
        trackThreads[i] = new BurnTracker();
        threads.push_back(std::thread(startBurn<T>,
                                      devices[i], &smi_handle,
                                      A, B,
                                      trackThreads[i])
        );
    }

    // Hold until all the threads are done with the init.
    {
        std::unique_lock<std::mutex> lk(cv_m);
        cv.wait(lk, [devCount]{return startUpCounter == devCount;});
    }

    // Burn-time.
    burn = true;
    cv.notify_all();
    std::this_thread::sleep_for(std::chrono::seconds(duration));

    // Burn-time done.
    burn = false;

    // Process output
    for (int i = 0; i < devCount; ++i)
    {
        double flops = trackThreads[i]->read();
        float devTemp = trackThreads[i]->getTemp();
        printf("GPU %2d(%s): %4.0f GF/s  %d Celsius\n", devices[i],
               flops < 0.0 ? "FAULTY" : "OK", flops, (int) devTemp);
    }

    // Join all threads
    std::for_each(threads.begin(), threads.end(), std::mem_fn(&std::thread::join));

    // Cleanup
    free(A);
    free(B);
    for (int i = 0; i < devCount; i++)
    {
        delete trackThreads[i];
    }
    delete [] trackThreads;
}


const char *ProgName = nullptr;


void print_usage()
{
    if (!ProgName) {
        return;
    }

    std::cerr << "Usage\n";
    std::cerr << "\t" << ProgName << " [-dh] "
              << "[-D DEVLIST] BURN_DURATION\n";
}


int main(int argc, char **argv)
{
    /*
     * The time of the burn can be set by passing the time in seconds as an as
     * an executable argument. If this value is prepended with the `-d` option,
     * the matrix operations will be double-precesion.
     *
     * By default, the code will run for 10s in single-precision mode.
     */

    ProgName = argv[0];

    int burn_duration = 10;
    bool use_doubles = false;
    std::vector<int> devices;
    int opt;
    while ( (opt = getopt(argc, argv, "dhD:")) != -1) {
        switch(opt) {
        case 'd':
            use_doubles = true;
            break;
        case 'h':
            print_usage();
            exit(0);
        case 'D':
            for (char *tok = std::strtok(optarg, ","); tok != nullptr;
                 tok = std::strtok(nullptr, ",")) {
                devices.push_back(std::atoi(tok));
            }
            break;
        case '?':
            print_usage();
            exit(1);
        }
    }

    if (optind < argc) {
        // Consume the burn duration argument
        burn_duration = std::atoi(argv[optind]);
        if (!burn_duration) {
            std::cerr << "invalid burn duration specified: "
                      << argv[optind] << "\n";
            exit(1);
        }
    } else {
        std::cout << "Burn duration not specified; burning for "
                  << burn_duration << "s\n";
    }

    // Get the device list
    if (!devices.size()) {
        // No devices specified
        std::cout << "No devices specified by the user; "
                  << "using all available devices\n";
        int num_devices;
        XGetDeviceCount(&num_devices);
        std::cout << "Found " << num_devices << " device(s)\n";
        for (int i = 0; i < num_devices; ++i) {
            devices.push_back(i);
        }
    }

    std::cout << "==> double precision: " << use_doubles << "\n";
    std::cout << "==> burn duration: " << burn_duration << "s\n";
    std::cout << "==> devices selected (" << devices.size() << "): ";
    for (size_t i = 0; i < devices.size(); ++i) {
        std::cout << devices[i] << " ";
    }
    std::cout << "\n";
    if (use_doubles) {
        launch<double>(burn_duration, devices);
    } else {
        launch<float>(burn_duration, devices);
    }
    return 0;
}
