ROCM_ROOT=/opt/rocm-3.9.0
RSMI_ROOT=/opt/rocm-3.9.0/rocm_smi
CXXFLAGS?=--amdgpu-target=gfx906

test:
	hipcc -O3 memory_bandwidth.cu -o memory_bandwidth.x -DTARGET_HIP ${CXXFLAGS} -std=c++11 -lnuma -I${ROCM_ROOT}/include -I${ROCTRACER_ROOT}/include -I${RSMI_ROOT}/include -L${RSMI_ROOT}/lib -lrocm_smi64
