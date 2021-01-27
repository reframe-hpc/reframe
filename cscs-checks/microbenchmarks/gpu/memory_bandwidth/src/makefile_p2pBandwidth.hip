ROCM_ROOT=/opt/rocm
RSMI_ROOT=/opt/rocm/rocm_smi
CXXFLAGS?=--amdgpu-target=gfx906

p2p_bandwidth:
	hipcc -O3 $@.cu -DTARGET_HIP ${CXXFLAGS} -std=c++11 -lnuma -I${ROCM_ROOT}/include -I${ROCTRACER_ROOT}/include -I${RSMI_ROOT}/include -L${RSMI_ROOT}/lib -lrocm_smi64 -o $@.x
