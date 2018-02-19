/*
* Copyright 2012 NVIDIA Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
#include <stdio.h>

// cuda_check_status_cu ----------------------------------------------------
// extern "C"
static void cuda_check_status_cu(cudaError_t status, int whiam) 
{
        if(status != cudaSuccess) {
                printf("cuda error: code %d, %s, line%d\n", status, cudaGetErrorString(status), whiam);
                exit(EXIT_FAILURE);
        }
}

// atomicMax ---------------------------------------------------------------
__device__ float atomicMax(float* const address, const float val)
{
	if ( *address >= val )
		return *address;
	int* const address_as_i = (int*)address;
	int old = *address_as_i;
	int assumed = old;
	do {
		assumed = old;
		if ( __int_as_float(assumed) >= val )
			break;
		old = atomicCAS(address_as_i, assumed, __float_as_int(val) );
	} while (assumed != old);
	return __int_as_float(old);
}


// jacobi_kernel ---------------------------------------------------------------
__global__ void jacobi_kernel( const float* const A_d, float* const Anew_d, const int n, const int m, float* const residue_d )
{
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	int j = blockIdx.y*blockDim.y+threadIdx.y;
	float residue = 0.0f;
	if ( j >= 1 && j < n-1 && i >= 1 && i < m-1 )
	{
		Anew_d[j *m+ i] = 0.25f * ( A_d[j     *m+ (i+1)] + A_d[j     *m+ (i-1)]
    			               +    A_d[(j-1) *m+ i]     + A_d[(j+1) *m+ i]);
		residue = fabsf(Anew_d[j *m+ i]-A_d[j *m+ i]);
		atomicMax( residue_d, residue );
	}
}


// copy_kernel ---------------------------------------------------------------
__global__ void copy_kernel( float* const A_d, const float* const Anew_d, const int n, const int m )
{
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	int j = blockIdx.y*blockDim.y+threadIdx.y;
	if ( j >= 1 && j < n-1 && i >= 1 && i < m-1 )
	{
		A_d[j *m + i] = Anew_d[j *m + i];
	}
}


// launch_jacobi_kernel ---------------------------------------------------------------
extern "C" 
float launch_jacobi_kernel( const float* const A_d, float* const Anew_d, const int n, const int m, float* const residue_d )
{
	const dim3 dimBlock(16,16,1);
	const dim3 dimGrid((m/dimBlock.x)+1,(n/dimBlock.y)+1,1);
	float residue = 0.f;
	cudaMemcpy( residue_d, &residue, sizeof(float), cudaMemcpyHostToDevice );

	jacobi_kernel<<<dimGrid,dimBlock>>>(A_d,Anew_d,n,m,residue_d);
	
	cudaMemcpy( &residue, residue_d, sizeof(float), cudaMemcpyDeviceToHost );
	return residue;
}


// launch_jacobi_kernel_async ---------------------------------------------------------------
extern "C" 
void launch_jacobi_kernel_async( const float* const A_d, float* const Anew_d, const int n, const int m, float* const residue_d  )
{
	const dim3 dimBlock(16,16,1);
	const dim3 dimGrid((m/dimBlock.x)+1,(n/dimBlock.y)+1,1);
	float residue = 0.f;
	cudaMemcpy( residue_d, &residue, sizeof(float), cudaMemcpyHostToDevice );
	if ( n > 0 && m > 0 )
	{
		jacobi_kernel<<<dimGrid,dimBlock>>>(A_d,Anew_d,n,m,residue_d);
	}
}


// wait_jacobi_kernel ---------------------------------------------------------------
extern "C" 
float wait_jacobi_kernel( const float* const residue_d )
{
	float residue = 0.f;
	cudaMemcpy( &residue, residue_d, sizeof(float), cudaMemcpyDeviceToHost );
	return residue;
}


// launch_copy_kernel ---------------------------------------------------------------
extern "C" 
void launch_copy_kernel( float* const A_d, const float* const Anew_d, const int n, const int m )
{
	const dim3 dimBlock(16,16,1);
	const dim3 dimGrid((m/dimBlock.x)+1,(n/dimBlock.y)+1,1);
	copy_kernel<<<dimGrid,dimBlock>>>(A_d,Anew_d,n,m);
}


// set_device ---------------------------------------------------------------
extern "C"
void set_device(int dev)
{
        cudaSetDevice(dev);
}

 
// memcpy_h2d ---------------------------------------------------------------
extern "C"
void memcpy_h2d(
        float* const A, 
        const float* const Anew, 
        float** A_d, 
        float** Anew_d, 
        float** residue_d,
        const int n, const int m, const int gpu_start, const int n_cpu
        )
{
        cudaError_t rcm;
        // printf("%d \n",cudaGetLastError());
        rcm = cudaMalloc( (void**)A_d,       n*m * sizeof(float) ); cuda_check_status_cu(rcm, __LINE__);
        rcm = cudaMalloc( (void**)Anew_d,    n*m * sizeof(float) ); cuda_check_status_cu(rcm, __LINE__);
        rcm = cudaMalloc( (void**)residue_d, sizeof(float) );       cuda_check_status_cu(rcm, __LINE__);

        rcm=cudaMemcpy( *A_d,    A+gpu_start-1,    m*(n-n_cpu)*sizeof(float), cudaMemcpyHostToDevice ); cuda_check_status_cu(rcm, __LINE__);
        //cuda error: code 11, invalid argument, line150 ==> Ben: dereference the pointer to pointer !

        rcm=cudaMemcpy( *Anew_d, Anew+gpu_start-1, m*(n-n_cpu)*sizeof(float), cudaMemcpyHostToDevice ); cuda_check_status_cu(rcm, __LINE__);
        //cuda error: code 11, invalid argument, line150 ==> Ben: dereference the pointer to pointer !

//bug     rcm = cudaMemcpy( &A_d,    A+gpu_start-1,    m*(n-n_cpu)*sizeof(float), cudaMemcpyHostToDevice );cuda_check_status_cu(rcm, __LINE__);
//bug     rcm = cudaMemcpy( &Anew_d, Anew+gpu_start-1, m*(n-n_cpu)*sizeof(float), cudaMemcpyHostToDevice );cuda_check_status_cu(rcm, __LINE__);
//bug     rcm = cudaMalloc( (void**)&A_d,       n*m * sizeof(float) ); cuda_check_status_cu(rcm, __LINE__);
//bug     rcm = cudaMalloc( (void**)&Anew_d,    n*m * sizeof(float) ); cuda_check_status_cu(rcm, __LINE__);
//bug     rcm = cudaMalloc( (void**)&residue_d, sizeof(float) ); cuda_check_status_cu(rcm, __LINE__);
//bug     rcm = cudaMemcpy( A_d,    A+gpu_start-1,    m*(n-n_cpu)*sizeof(float), cudaMemcpyHostToDevice ); cuda_check_status_cu(rcm, __LINE__);
//bug     rcm = cudaMemcpy( Anew_d, Anew+gpu_start-1, m*(n-n_cpu)*sizeof(float), cudaMemcpyHostToDevice ); cuda_check_status_cu(rcm, __LINE__);
}


// jacobi_memcpy -----------------------------------------------------------------
extern "C"
void jacobi_memcpy(
        float* const A  , const float* const Anew  , 
        float* const A_d, const float* const Anew_d, 
        const int m, const int cpu_start, const int cpu_end, const int gpu_end,
        const int rank )
{
        cudaError_t error;
        if ( rank == 0 )
        {
                error = cudaMemcpy( A_d+(gpu_end+1)*m+1,   Anew  +cpu_start*m+1, (m-2)*sizeof(float), cudaMemcpyHostToDevice );
                cuda_check_status_cu(error, __LINE__);
                // cuda error: code 11, invalid argument, line173
                error = cudaMemcpy( A  +(cpu_start-1)*m+1, Anew_d+gpu_end*m+1,   (m-2)*sizeof(float), cudaMemcpyDeviceToHost );
                cuda_check_status_cu(error, __LINE__);

        } else {
                error = cudaMemcpy( A_d+0*m+1,         Anew+cpu_end*m+1, (m-2)*sizeof(float), cudaMemcpyHostToDevice );
                cuda_check_status_cu(error, __LINE__);
                error = cudaMemcpy( A+(cpu_end+1)*m+1, Anew_d+1*m+1,     (m-2)*sizeof(float), cudaMemcpyDeviceToHost );
                cuda_check_status_cu(error, __LINE__);
        }

}
/*
cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind )
    Copies data between host and device.
    dst - Destination memory address 
    src - Source memory address 
    count - Size in bytes to copy 
    kind - Type of transfer

    Returns cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevicePointer, cudaErrorInvalidMemcpyDirection

Copies count bytes from the memory area pointed to by src to the
memory area pointed to by dst, where kind is one of
cudaMemcpyHostToHost, 
cudaMemcpyHostToDevice, 
cudaMemcpyDeviceToHost, or
cudaMemcpyDeviceToDevice, 
and specifies the direction of the copy. The
memory areas may not overlap. Calling cudaMemcpy() with dst and src
pointers that do not match the direction of the copy results in an
undefined behavior.
*/


// jacobi_memcpy_final -----------------------------------------------------------------
extern "C"
void jacobi_memcpy_final(
        float* const A, float** const A_d,
        const int m, const int n, const int n_cpu, const int cpu_end, 
        const int rank )
{
//        float* const A, const float* const A_d,
        cudaError_t error;
        if ( rank == 0 )
        {
                error = cudaMemcpy( A+1*m+1, *(A_d)+1*m+1, (m*(n-n_cpu-1)-2)*sizeof(float), cudaMemcpyDeviceToHost );
                cuda_check_status_cu(error, __LINE__);
                // cuda error: code 11, invalid argument, line227
        } else {
                 error = cudaMemcpy( A+cpu_end*m+1, *(A_d)+1*m+1, (m*(n-n_cpu-1)-2)*sizeof(float), cudaMemcpyDeviceToHost );
                 cuda_check_status_cu(error, __LINE__);
        }

}


// get_info_device ---------------------------------------------------------------
extern "C"
void get_info_device(char *gpu_string, int dev)
{
	int driverVersion = 0, runtimeVersion = 0;
	struct cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	strcpy(gpu_string, deviceProp.name);
	
	// printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
	cudaDriverGetVersion(&driverVersion);
	cudaRuntimeGetVersion(&runtimeVersion);
	printf("\n  CUDA Driver Version / Runtime Version     %d.%d / %d.%d\n", driverVersion/1000, (driverVersion%100)/10, runtimeVersion/1000, (runtimeVersion%100)/10);
	printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);
	
	printf("  Maximum number of threads per multiprocessor:  %d\n", deviceProp.maxThreadsPerMultiProcessor);
	
	printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
	
	printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
	                deviceProp.maxThreadsDim[0],
	                deviceProp.maxThreadsDim[1],
	                deviceProp.maxThreadsDim[2]);
	printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
	                deviceProp.maxGridSize[0],
	                deviceProp.maxGridSize[1],
	                deviceProp.maxGridSize[2]);
}

// init_cuda -------------------------------------------------------
#ifndef DEVS_PER_NODE
#define DEVS_PER_NODE 1  // Devices per node
#endif
// void init_cuda(int rank)
extern "C"
void init_cuda( int rank, 
        float* const A,
        const float* const Anew, 
        float** const A_d,
        float** const Anew_d, 
        const int n,
        const int m,
        const int gpu_start,
        const int n_cpu,
        float** const residue_d )
{
//        const float** const Anew_d, 
  // int rank;
  // MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
  int dev = rank % DEVS_PER_NODE;
  // printf("rk=%d dev=%d\n",rank,dev);

  cudaError_t error;
  error = cudaSetDevice(dev);
  cuda_check_status_cu(error, __LINE__);

  //bug? ==> yes
  //bug: 
  // fix in place:
  //     for A_d: &A_d (=pointer to pointer to really pass, not only a copy) instead of A_d
  //     for A: A
  //OKOKOKOK: memcpy_h2d(A, Anew, &A_d, &Anew_d, &residue_d, n, m, gpu_start, n_cpu);
  //memcpy_h2d(A, Anew, &A_d, &Anew_d, &residue_d, n, m, gpu_start, n_cpu);
  //bug: memcpy_h2d(A, Anew, A_d, Anew_d, residue_d, n, m, gpu_start, n_cpu);

  error = cudaMalloc( (void**)A_d,       n*m * sizeof(float) ); cuda_check_status_cu(error, __LINE__);
  error = cudaMalloc( (void**)Anew_d,    n*m * sizeof(float) ); cuda_check_status_cu(error, __LINE__);
  error = cudaMalloc( (void**)residue_d, sizeof(float) );       cuda_check_status_cu(error, __LINE__);

  error=cudaMemcpy( *A_d,    A+gpu_start-1,    m*(n-n_cpu)*sizeof(float), cudaMemcpyHostToDevice ); cuda_check_status_cu(error, __LINE__);
  //cuda error: code 11, invalid argument, line150 ==> Ben: dereference the pointer to pointer !

  error=cudaMemcpy( *Anew_d, 
        Anew+gpu_start-1, 
        m*(n-n_cpu)*sizeof(float), 
        cudaMemcpyHostToDevice ); cuda_check_status_cu(error, __LINE__);

}
 
// free_device -----------------------------------------------------
extern "C"
void free_device(float* A_d, float* Anew_d, float* residue_d)
{
        // cudaDeviceSynchronize();
        cudaError_t error;
        error = cudaFree( residue_d ); cuda_check_status_cu(error, __LINE__);
        error = cudaFree(Anew_d); cuda_check_status_cu(error, __LINE__);
        error = cudaFree(A_d); cuda_check_status_cu(error, __LINE__);
        error = cudaDeviceReset(); cuda_check_status_cu(error, __LINE__);
}

// finalize_cuda ---------------------------------------------------------------
extern "C"
void finalize_cuda()
{
  // cudaDeviceSynchronize();
  cudaError_t error;
  error = cudaDeviceSynchronize();
  cuda_check_status_cu(error, __LINE__);

/* TODO:
  cudaFree( residue_d );
  cudaFree(Anew_d);
  cudaFree(A_d);
*/
  // free_device(A_d, Anew_d, residue_d);
}

