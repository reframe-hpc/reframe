#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#define MAX_SOURCE_SIZE (0x100000)

int main(int argc, char ** argv) {

    int SIZE = 1024;

    // Allocate memories for input arrays and output array.
    float *A = (float*)malloc(sizeof(float)*SIZE);
    float *B = (float*)malloc(sizeof(float)*SIZE);

    // Output
    float *C = (float*)malloc(sizeof(float)*SIZE);

    // Initialize values for array members.
    int i = 0;
    for (i=0; i<SIZE; ++i) {
        A[i] = i+1;
        B[i] = (i+1)*2;
    }

    // Load kernel from file vecAddKernel.cl
    FILE *kernelFile;
    char *kernelSource;
    size_t kernelSize;

    kernelFile = fopen("vecAddKernel.cl", "r");

    if (!kernelFile) {
        fprintf(stderr, "No file named vecAddKernel.cl was found\n");
        exit(-1);
    }

    kernelSource = (char*)malloc(MAX_SOURCE_SIZE);
    kernelSize = fread(kernelSource, 1, MAX_SOURCE_SIZE, kernelFile);
    fclose(kernelFile);

    // Getting platform and device information
    cl_platform_id platformId = NULL;
    cl_device_id deviceID = NULL;
    cl_uint retNumDevices;
    cl_uint retNumPlatforms;
    cl_int ret = clGetPlatformIDs(1, &platformId, &retNumPlatforms);
    ret = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_DEFAULT, 1, &deviceID, &retNumDevices);

    // Creating context.
    cl_context context = clCreateContext(NULL, 1, &deviceID, NULL, NULL,  &ret);

    // Creating command queue
    cl_command_queue commandQueue = clCreateCommandQueue(context, deviceID, 0, &ret);

    // Memory buffers for each array
    cl_mem aMemObj = clCreateBuffer(context, CL_MEM_READ_ONLY, SIZE * sizeof(float), NULL, &ret);
    cl_mem bMemObj = clCreateBuffer(context, CL_MEM_READ_ONLY, SIZE * sizeof(float), NULL, &ret);
    cl_mem cMemObj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, SIZE * sizeof(float), NULL, &ret);

    // Copy lists to memory buffers
    ret = clEnqueueWriteBuffer(commandQueue, aMemObj, CL_TRUE, 0, SIZE * sizeof(float), A, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(commandQueue, bMemObj, CL_TRUE, 0, SIZE * sizeof(float), B, 0, NULL, NULL);

    // Create program from kernel source
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, (const size_t *)&kernelSize, &ret);

    // Build program
    ret = clBuildProgram(program, 1, &deviceID, NULL, NULL, NULL);

    // Create kernel
    cl_kernel kernel = clCreateKernel(program, "addVectors", &ret);

    // Set arguments for kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&aMemObj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&bMemObj);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&cMemObj);

    // Execute the kernel
    size_t globalItemSize = SIZE;
    size_t localItemSize = 64; // globalItemSize has to be a multiple of localItemSize. 1024/64 = 16
    ret = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &globalItemSize, &localItemSize, 0, NULL, NULL);

    // Read from device back to host.
    ret = clEnqueueReadBuffer(commandQueue, cMemObj, CL_TRUE, 0, SIZE * sizeof(float), C, 0, NULL, NULL);

    // Test if correct answer
    for (i=0; i<SIZE; ++i) {
        if (C[i] != (A[i] + B[i])) {
            printf("FAILURE\n");
            break;
	}
    }

    if (i == SIZE) {
        printf("SUCCESS\n");
    }

    // Clean up, release memory.
    ret = clFlush(commandQueue);
    ret = clFinish(commandQueue);
    ret = clReleaseCommandQueue(commandQueue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(aMemObj);
    ret = clReleaseMemObject(bMemObj);
    ret = clReleaseMemObject(cMemObj);
    ret = clReleaseContext(context);
    free(A);
    free(B);
    free(C);

    return 0;
}
