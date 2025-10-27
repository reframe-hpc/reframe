RSMI_ROOT?=/opt/rocm/rocm_smi
ROCM_ROOT?=/opt/rocm/

gpu_burn:
	hipcc -O3 $@.cu -o $@.x -DTARGET_HIP ${CXXFLAGS} -std=c++14 -I${ROCM_ROOT}/include/rocblas/ -I${RSMI_ROOT}/include -lnuma -lrocm_smi64 -lrocblas -pthreads
