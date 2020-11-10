RSMI_ROOT=/opt/rocm-3.9.0/rocm_smi
AMDGPU_TARGET=gfx906,gfx908

pointerChase:
	hipcc -o pChase.x -O3 pointer_chase.cu -DTARGET_HIP ${CXXFLAGS} -std=c++11 -lnuma --amdgpu-target=${AMDGPU_TARGET}  -I${RSMI_ROOT}/include -L${RSMI_ROOT}/lib -lrocm_smi64
