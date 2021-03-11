CXXFLAGS?=--amdgpu-target=gfx906,gfx908
ROCM_ROOT?=/opt/rocm
RSMI_ROOT?=/opt/rocm/rocm_smi

dgemm:
	hipcc -O3 $@.cu -o $@.x -DTARGET_HIP ${CXXFLAGS} -std=c++14 -I${ROCM_ROOT} -I${RSMI_ROOT}/include -lnuma -lrocm_smi64 -lrocblas
