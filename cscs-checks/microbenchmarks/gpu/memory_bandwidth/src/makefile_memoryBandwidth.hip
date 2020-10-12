ROCBLAS_ROOT=/opt/rocm-3.8.0/rocblas
ROCM_ROOT=/opt/rocm-3.8.0
RSMI_ROOT=/opt/rocm-3.8.0/rocm_smi
AMDGPU_TARGET=gfx900

test:
	hipcc -O3 memory_bandwidth.cu -DTARGET_HIP ${CXXFLAGS} -std=c++11 -lnuma --amdgpu-target=${AMDGPU_TARGET} -I${ROCM_ROOT}/include -I${ROCTRACER_ROOT}/include -I${RSMI_ROOT}/include -L${RSMI_ROOT}/lib -lrocm_smi64
