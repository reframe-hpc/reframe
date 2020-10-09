ROCBLAS_ROOT=/opt/rocm-3.8.0/rocblas
ROCM_ROOT=/opt/rocm-3.8.0

memorybandwidth:
	hipcc -O3 memory_bandwidth.cu -DTARGET_HIP -std=c++11 --amdgpu-target=gfx906,gfx908 -I${ROCM_ROOT}/include -I${ROCTRACER_ROOT}/include

p2pbandwidth:
	hipcc -O3 p2p_bandwidth.cu -DTARGET_HIP -std=c++11 --amdgpu-target=gfx906,gfx908 -I${ROCM_ROOT}/include -I${ROCTRACER_ROOT}/include
