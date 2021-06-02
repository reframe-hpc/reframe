RSMI_ROOT?=/opt/rocm/rocm_smi

memory_bandwidth:
	hipcc -O3 $@.cu -o $@.x -DTARGET_HIP ${CXXFLAGS} -std=c++11 -lnuma -I${RSMI_ROOT}/include -L${RSMI_ROOT}/lib -lrocm_smi64
