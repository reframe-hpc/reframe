/*
 * Modified for CSCS by Javier Otero (javier.otero@cscs.ch) to
 * support both HIP and CUDA.
 *
 * Modifications for CSCS by Mark Klein (klein@cscs.ch)
 * - NVML bindings
 * - Reduced output
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
#define USEMEM 0.9 // Try to allocate 90% of memory

// Used to report op/s, measured through Visual Profiler, CUBLAS from CUDA 7.5
// (Seems that they indeed take the naive dim^3 approach)
#define OPS_PER_MUL 17188257792ul

#include <cstdio>
#include <string>
#include <map>
#include <vector>
#include <sys/types.h>
#include <signal.h>
#include <sys/wait.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <cstdlib>

#include "Xdevice/runtime.hpp"
#include "Xdevice/smi.hpp"
#include <cuda.h>
#include "cublas_v2.h"

#include <thread>
#include <type_traits>
#include <vector>
#include <array>
#include <mutex>
#include <algorithm>
#include <functional>
#include <memory>

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

template <class T> class FirePit
{
private:
  int deviceId;

  // SMI handle to do system queries.
  Smi * smi_handle;

  size_t AvailDeviceMemory;
  size_t iters;
  size_t d_resultSize;

  long long int totalErrors;

  static const int g_blockSize = 16;
  T * d_C;
  T * d_A;
  T * d_B;
  int * d_numberOfErrors;

  cublasHandle_t d_cublas;

public:
  FirePit(int id, Smi * smi_hand) : deviceId(id), smi_handle(smi_hand)
  {
      // Set the device and pin thread to CPU with
      XSetDevice(deviceId);
      smi_handle->setCpuAffinity(deviceId);

      // Create blas plan
      cublasCreate(&d_cublas);
      cublasSetPointerMode(d_cublas, CUBLAS_POINTER_MODE_HOST);

      totalErrors = 0;
  }
  ~FirePit()
  {
      XFree(d_C);
      XFree(d_A);
      XFree(d_B);
      XFree(d_numberOfErrors);
      cublasDestroy(d_cublas);
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

  void compute()
  {
      // See function specialisations below.
      std::cout << "compute function not implemented for this type." << std::endl;
      exit(1);
  }

  void compare() {
      int numberOfErrors;
      XMemset(d_numberOfErrors, 0, sizeof(int));
      dim3 block(g_blockSize,g_blockSize);
      dim3 grid(SIZE/g_blockSize,SIZE/g_blockSize);
      kernels::compare<T><<<grid,block>>>((T*)d_C,(int*)d_numberOfErrors,(size_t)iters);

      XMemcpy(&numberOfErrors, d_numberOfErrors, sizeof(int), XMemcpyDeviceToHost);
      if (numberOfErrors) {
          totalErrors += (long long int)numberOfErrors;
          printf("WE FOUND %d FAULTY ELEMENTS from GPU %d\n", numberOfErrors, deviceId);
      }
  }
};

template<>
void FirePit<double>::compute()
{
    static const double alpha = 1.0;
    static const double beta = 0.0;
    for (size_t i = 0; i < iters; ++i)
    {
        cublasDgemm(d_cublas,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    SIZE, SIZE, SIZE,
                    &alpha,
                    (const double*)d_A, SIZE,
                    (const double*)d_B, SIZE,
                    &beta,
                    d_C + i*SIZE*SIZE, SIZE);
    }
}

template<>
void FirePit<float>::compute()
{
    static const float alpha = 1.0;
    static const float beta = 0.0;
    for (size_t i = 0; i < iters; ++i)
    {
        cublasSgemm(d_cublas,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    SIZE, SIZE, SIZE,
                    &alpha,
                    (const float*)d_A, SIZE,
                    (const float*)d_B, SIZE,
                    &beta,
                    d_C + i*SIZE*SIZE, SIZE);
    }
}


void checkError(int rCode, std::string desc = "") {
}
void checkError(cublasStatus_t rCode, std::string desc = "") {
}

template <class T> class GPU_Test {
    public:
        GPU_Test(int dev, bool doubles) : d_devNumber(dev), d_doubles(doubles) {
            checkError(cuDeviceGet(&d_dev, d_devNumber));
            checkError(cuCtxCreate(&d_ctx, 0, d_dev));
            bind();
            checkError(cublasCreate(&d_cublas), "init");
            d_error = 0;
        }
	~GPU_Test() {
            bind();
            checkError(cuMemFree(d_Cdata), "Free A");
            checkError(cuMemFree(d_Adata), "Free B");
            checkError(cuMemFree(d_Bdata), "Free C");
            cublasDestroy(d_cublas);
        }

        unsigned long long int getErrors() {
            unsigned long long int tempErrs = d_error;
            d_error = 0;
            return tempErrs;
        }

        size_t getIters() {
            return d_iters;
        }

        void bind() {
            checkError(cuCtxSetCurrent(d_ctx), "Bind CTX");
        }

        size_t totalMemory() {
            bind();
            size_t freeMem, totalMem;
            checkError(cuMemGetInfo(&freeMem, &totalMem));
            return totalMem;
        }

        size_t availMemory() {
            bind();
            size_t freeMem, totalMem;
            checkError(cuMemGetInfo(&freeMem, &totalMem));
            return freeMem;
        }

        void initBuffers(T *A, T *B) {
            bind();
            size_t useBytes = (size_t)((double)availMemory()*USEMEM);
            size_t d_resultSize = sizeof(T)*SIZE*SIZE;
            d_iters = (useBytes - 2*d_resultSize)/d_resultSize; // We remove A and B sizes
            // printf("Results are %d bytes each, thus performing %d iterations\n", d_resultSize, d_iters);
            checkError(cuMemAlloc(&d_Cdata, d_iters*d_resultSize), "C alloc");
            checkError(cuMemAlloc(&d_Adata, d_resultSize), "A alloc");
            checkError(cuMemAlloc(&d_Bdata, d_resultSize), "B alloc");
            checkError(cuMemAlloc(&d_faultyElemData, sizeof(int)), "faulty data");
            // Populating matrices A and B
            checkError(cuMemcpyHtoD(d_Adata, A, d_resultSize), "A -> device");
            checkError(cuMemcpyHtoD(d_Bdata, B, d_resultSize), "A -> device");
            // initCompareKernel();
        }

	void compute() {
            bind();
            static const float alpha = 1.0f;
            static const float beta = 0.0f;
            static const double alphaD = 1.0;
            static const double betaD = 0.0;

            for (size_t i = 0; i < d_iters; ++i) {
                if (d_doubles)
                    checkError(cublasDgemm(d_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                           SIZE, SIZE, SIZE, &alphaD,
                                           (const double*)d_Adata, SIZE,
                                           (const double*)d_Bdata, SIZE,
                                           &betaD,
                                           (double*)d_Cdata + i*SIZE*SIZE, SIZE), "DGEMM");
                else
                    checkError(cublasSgemm(d_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                           SIZE, SIZE, SIZE, &alpha,
                                           (const float*)d_Adata, SIZE,
                                           (const float*)d_Bdata, SIZE,
                                           &beta,
                                           (float*)d_Cdata + i*SIZE*SIZE, SIZE), "SGEMM");
            }
        }

        void compare() {
            int faultyElems;
            checkError(cuMemsetD32(d_faultyElemData, 0, 1), "memset");
            dim3 block(g_blockSize,g_blockSize);
            dim3 grid(SIZE/g_blockSize,SIZE/g_blockSize);
            //checkError(cuLaunchGrid(d_function, SIZE/g_blockSize, SIZE/g_blockSize), "Launch grid");
            if(d_doubles)
                kernels::compare<double><<<grid,block>>>((double*)d_Cdata,(int*)d_faultyElemData,(size_t)d_iters);
            else
                kernels::compare<float><<<grid,block>>>((float*)d_Cdata,(int*)d_faultyElemData,(size_t)d_iters);

            checkError(cuMemcpyDtoH(&faultyElems, d_faultyElemData, sizeof(int)), "Read faultyelemdata");
            if (faultyElems) {
                d_error += (long long int)faultyElems;
                printf("WE FOUND %d FAULTY ELEMENTS from GPU %d\n", faultyElems, d_devNumber);
            }
        }

        private:
            bool d_doubles;
            int d_devNumber;
            size_t d_iters;
            size_t d_resultSize;

            long long int d_error;

            static const int g_blockSize = 16;

            CUdevice d_dev;
            CUcontext d_ctx;
            CUmodule d_module;
            CUfunction d_function;

            CUdeviceptr d_Cdata;
            CUdeviceptr d_Adata;
            CUdeviceptr d_Bdata;
            CUdeviceptr d_faultyElemData;

            cublasHandle_t d_cublas;
};

int getNumDevices() {
    int deviceCount = 0;
    XGetDeviceCount(&deviceCount);

    if (!deviceCount)
        throw std::string("No CUDA devices");

        #ifdef USEDEV
        if (USEDEV >= deviceCount)
            throw std::string("Not enough devices for USEDEV");
        #endif

    return deviceCount;
}

int initCuda() {
    int deviceCount = 0;
    XGetDeviceCount(&deviceCount);

    if (!deviceCount)
        throw std::string("No CUDA devices");

        #ifdef USEDEV
        if (USEDEV >= deviceCount)
            throw std::string("Not enough devices for USEDEV");
        #endif

    return deviceCount;
}

template<class T>
void startFire(int devId,
               Smi * smi_handle, T *A, T *B,
               volatile bool & burn,
               std::mutex & mtx,
               volatile std::pair<size_t, unsigned long long int> & ops,
               volatile unsigned long long int & err
               )
{
    FirePit<T> *fp;
    try {
        fp = new FirePit<T>(devId, smi_handle);
        fp->initBuffers(A, B);
    }
    catch (std::string e) {
        fprintf(stderr, "Couldn't init a GPU test: %s\n", e.c_str());
        exit(124);
    }

    {   // Store the number of iterations
        std::lock_guard<std::mutex> lg(mtx);
        ops.first = fp->getIters();
    }

    // Hold off any computation until master says go.
    while(!burn){};

    // The actual work
    try {
        while (burn) {
            fp->compute();
            fp->compare();

            // Make the rest a critical section
            std::lock_guard<std::mutex> lg(mtx);
            ops.second++;
            err += fp->getErrors();
        }
    }
    catch (std::string e) {
        fprintf(stderr, "Failure during compute: %s\n", e.c_str());
        std::lock_guard<std::mutex> lg(mtx);
        ops.first = 0;
        exit(111);
    }
}

template<class T> void startBurn(int index, int writeFd, T *A, T *B, bool doubles) {
    GPU_Test<T> *our;
    try {
        our = new GPU_Test<T>(index, doubles);
        our->initBuffers(A, B);
    }
    catch (std::string e) {
        fprintf(stderr, "Couldn't init a GPU test: %s\n", e.c_str());
        exit(124);
    }

    // The actual work
    try {
        while (true) {
            our->compute();
            our->compare();
            int ops = our->getIters();
            write(writeFd, &ops, sizeof(int));
            ops = our->getErrors();
            write(writeFd, &ops, sizeof(int));
        }
    }
    catch (std::string e) {
    fprintf(stderr, "Failure during compute: %s\n", e.c_str());
    int ops = -1;
    // Signalling that we failed
    write(writeFd, &ops, sizeof(int));
    write(writeFd, &ops, sizeof(int));
    exit(111);
    }
}

void refreshTemperatures(Smi * smi_handle, std::vector<int> *temps)
{
    static int gpuIter = 0;
    int device_count;
    smi_handle->getNumberOfDevices(&device_count);
    for (unsigned int i = 0; i < device_count; i++)
    {
        temps->at(gpuIter) = (int)(smi_handle->getGpuTemp(i));
        gpuIter = (gpuIter+1)%(temps->size());
    }
}

void updateTemps(std::vector<int> *temps)
{
    static int gpuIter = 0;
    int device_count;
    Smi smi_handle = Smi();
    smi_handle.getNumberOfDevices(&device_count);
    for (unsigned int i = 0; i < device_count; i++)
    {
        temps->at(gpuIter) = (int)(smi_handle.getGpuTemp(i));
        gpuIter = (gpuIter+1)%(temps->size());
    }
}

void listenClients(std::vector<int> clientFd, std::vector<pid_t> clientPid, int runTime) {
    fd_set waitHandles;

    // pid_t tempPid;
    char hostname[256];
    hostname[255]='\0';
    gethostname(hostname,255);
    int maxHandle = 0;
    FD_ZERO(&waitHandles);

    for (size_t i = 0; i < clientFd.size(); ++i) {
        if (clientFd.at(i) > maxHandle)
            maxHandle = clientFd.at(i);
        FD_SET(clientFd.at(i), &waitHandles);
    }

    std::vector<int> clientTemp;
    std::vector<int> clientErrors;
    std::vector<int> clientCalcs;
    std::vector<struct timespec> clientUpdateTime;
    std::vector<float> clientGflops;
    std::vector<bool> clientFaulty;

    time_t startTime = time(0);

    for (size_t i = 0; i < clientFd.size(); ++i) {
        clientTemp.push_back(0);
        clientErrors.push_back(0);
        clientCalcs.push_back(0);
        struct timespec thisTime;
        clock_gettime(CLOCK_REALTIME, &thisTime);
        clientUpdateTime.push_back(thisTime);
        clientGflops.push_back(0.0f);
        clientFaulty.push_back(false);
    }

    float nextReport = 2.0f;
    bool childReport = false;
    while ((select(maxHandle+1, &waitHandles, NULL, NULL, NULL))) {
        size_t thisTime = time(0);
        struct timespec thisTimeSpec;
        clock_gettime(CLOCK_REALTIME, &thisTimeSpec);

        // Going through all descriptors
        for (size_t i = 0; i < clientFd.size(); ++i)
            if (FD_ISSET(clientFd.at(i), &waitHandles)) {
                // First, reading processed
                int processed, errors;
                read(clientFd.at(i), &processed, sizeof(int));
                // Then errors
                read(clientFd.at(i), &errors, sizeof(int));

                clientErrors.at(i) += errors;
                if (processed == -1)
                    clientCalcs.at(i) = -1;
                else {
                    double flops = (double)processed * (double)OPS_PER_MUL;
                    struct timespec clientPrevTime = clientUpdateTime.at(i);
                    double clientTimeDelta = (double)thisTimeSpec.tv_sec + (double)thisTimeSpec.tv_nsec / 1000000000.0 - ((double)clientPrevTime.tv_sec + (double)clientPrevTime.tv_nsec / 1000000000.0);
                    clientUpdateTime.at(i) = thisTimeSpec;

                    clientGflops.at(i) = (double)((unsigned long long int)processed * OPS_PER_MUL) / clientTimeDelta / 1000.0 / 1000.0 / 1000.0;
                    clientCalcs.at(i) += processed;
                }

                childReport = true;
            }

            updateTemps(&clientTemp);

            // Resetting the listeners
            FD_ZERO(&waitHandles);
            // FD_SET(tempHandle, &waitHandles);
            for (size_t i = 0; i < clientFd.size(); ++i)
                FD_SET(clientFd.at(i), &waitHandles);

            // Printing progress (if a child has initted already)
            if (childReport) {
                float elapsed = fminf((float)(thisTime-startTime)/(float)runTime*100.0f, 100.0f);
                for (size_t i = 0; i < clientErrors.size(); ++i) {
                    std::string note = "%d ";
                }

                fflush(stdout);

                if (nextReport < elapsed) {
                    nextReport = elapsed + 2.0f;
                    for (size_t i = 0; i < clientErrors.size(); ++i) {
                        if (clientErrors.at(i))
                            clientFaulty.at(i) = true;
                        clientErrors.at(i) = 0;
                    }
                }
            }

            // Checking whether all clients are dead
            bool oneAlive = false;
            for (size_t i = 0; i < clientCalcs.size(); ++i)
                if (clientCalcs.at(i) != -1)
                    oneAlive = true;
            if (!oneAlive) {
                fprintf(stderr, "\n\nNo clients are alive!  Aborting\n");
                exit(123);
            }

            if (startTime + runTime < thisTime)
                break;
    }

    fflush(stdout);
    for (size_t i = 0; i < clientPid.size(); ++i)
        kill(clientPid.at(i), 15);

    while (wait(NULL) != -1);
    printf("Node %s:\n", hostname);

    for (size_t i = 0; i < clientPid.size(); ++i) {
        printf("  GPU %2d(%s): %4.0f GF/s  %i Celsius\n", (int)i,clientFaulty.at(i) ? "FAULTY" : "OK", clientGflops.at(i), clientTemp.at(i));
    }
    printf("\n");
}

template<class T> void lighter(int duration)
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
    Smi * smi_handle = new Smi();

    // Here burn is a switch that holds and breaks the work done by the slave threads.
    bool burn = false;

    // Schedule the work in all threads, but don't do any just yet.
    int devCount = getNumDevices();

    std::vector<std::thread> threads;
    std::mutex * mutexes = new std::mutex[devCount];

    // The number of ops are stored in a pair, where the first element has the number of iterations per
    // compute call, and the second element counts the number of times this compute function was called.
    std::pair<size_t, unsigned long long int> * ops = new std::pair<size_t, unsigned long long int>[devCount];

    // Error counter.
    unsigned long long int * err = new unsigned long long int[devCount];

    for (int i = 0; i < devCount; i++)
    {
        /* Init the counters for each thread
         * The ops & errors are counted separately on a per-device
         * basis. Therefore, we assign a different mutex to each thread.
         * Note that we don't really need mutexes if we're just counting
         * the ops & err at the end of the burn, but they are included to
         * give this function full control over the output from each thread.
         */
        ops[i] = std::pair<size_t, unsigned long long int>(0, 0);
        err[i] = 0;
        std::mutex * m = new (&mutexes[i]) std::mutex();

        // Launch the thread
        threads.push_back(std::thread(startFire<T>,
                                      i, smi_handle,
                                      A, B,
                                      std::ref(burn),
                                      std::ref(*m),
                                      std::ref(ops[i]),
                                      std::ref(err[i])
                          )
        );
    }

    // Burn-time.
    burn = true;
    std::this_thread::sleep_for( std::chrono::seconds(duration) );

    // Burn-time done.
    burn = false;

    // Join all threads
    std::for_each(threads.begin(), threads.end(), std::mem_fn(&std::thread::join));

    // Cleanup
    free(A);
    free(B);
    delete smi_handle;
    delete [] mutexes;
    delete [] ops;
    delete [] err;
}


template<class T> void launch(int runLength, bool useDoubles) {

    // Initializing A and B with random data
    T *A = (T*) malloc(sizeof(T)*SIZE*SIZE);
    T *B = (T*) malloc(sizeof(T)*SIZE*SIZE);
    srand(10);
    for (size_t i = 0; i < SIZE*SIZE; ++i) {
        A[i] = (T)((double)(rand()%1000000)/100000.0);
        B[i] = (T)((double)(rand()%1000000)/100000.0);
    }

    // Forking a process..  This one checks the number of devices to use,
    // returns the value, and continues to use the first one.
    int mainPipe[2];
    pipe(mainPipe);
    int readMain = mainPipe[0];
    std::vector<int> clientPipes;
    std::vector<pid_t> clientPids;
    clientPipes.push_back(readMain);

    pid_t myPid = fork();
    if (!myPid) {
        // Child
        close(mainPipe[0]);
        int writeFd = mainPipe[1];
        int devCount = initCuda();
        write(writeFd, &devCount, sizeof(int));
        startBurn<T>(0, writeFd, A, B, useDoubles);
        close(writeFd);
        return;
    }
    else {
        clientPids.push_back(myPid);
        close(mainPipe[1]);
        int devCount;
        read(readMain, &devCount, sizeof(int));

        if (!devCount) {
            fprintf(stderr, "No CUDA devices\n");
        }
        else {
            for (int i = 1; i < devCount; ++i) {
                int slavePipe[2];
                pipe(slavePipe);
                clientPipes.push_back(slavePipe[0]);
                pid_t slavePid = fork();
                if (!slavePid) {
                    // Child
                    close(slavePipe[0]);
                    initCuda();
                    startBurn<T>(i, slavePipe[1], A, B, useDoubles);
                    close(slavePipe[1]);
                    return;
                }
                else {
                    clientPids.push_back(slavePid);
                    close(slavePipe[1]);
                }
            }
            listenClients(clientPipes, clientPids, runLength);
        }
    }

    for (size_t i = 0; i < clientPipes.size(); ++i)
        close(clientPipes.at(i));

    free(A);
    free(B);
}

int main(int argc, char **argv) {
    int runLength = 10;
    bool useDoubles = false;
    int thisParam = 0;
    if (argc >= 2 && std::string(argv[1]) == "-d") {
        useDoubles = true;
        thisParam++;
    }
    if (argc-thisParam < 2)
        printf("Run length not specified in the command line.  Burning for 10 secs\n");
    else
        runLength = atoi(argv[1+thisParam]);

    if (useDoubles)
        launch<double>(runLength, useDoubles);
    else
        launch<float>(runLength, useDoubles);

    lighter<double>(runLength);

    return 0;
}
