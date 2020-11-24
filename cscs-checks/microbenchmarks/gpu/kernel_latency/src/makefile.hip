RSMI_ROOT?=/opt/rocm-3.9.0/rocm_smi
CXXFLAGS?=--amdgpu-target=gfx906,gfx908

test:
	hipcc -O3 kernel_latency.cu -o kernel_latency.x -DTARGET_HIP ${CXXFLAGS} -std=c++11 -lnuma -I${RSMI_ROOT}/include -L${RSMI_ROOT}/lib -lrocm_smi64
