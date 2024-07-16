RSMI_ROOT?=/opt/rocm/rocm_smi

gpu_burn:
	hipcc -O3 $@.cu -o $@.x -DTARGET_HIP ${CXXFLAGS} -std=c++14 -I${RSMI_ROOT}/include -lnuma -lrocm_smi64 -lrocblas -pthreads
