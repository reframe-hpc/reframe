RSMI_ROOT?=/opt/rocm/rocm_smi
CXXFLAGS?=--amdgpu-target=gfx906,gfx908,gfx900

pointerChase:
	hipcc -o pChase.x -O3 pointer_chase.cu -DTARGET_HIP ${CXXFLAGS} -std=c++11 -lnuma -I${RSMI_ROOT}/include -L${RSMI_ROOT}/lib -lrocm_smi64
