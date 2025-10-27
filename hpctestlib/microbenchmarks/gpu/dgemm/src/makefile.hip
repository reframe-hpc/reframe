ROCM_ROOT?=/opt/rocm
RSMI_ROOT?=/opt/rocm/rocm_smi

all: dgemm sgemm

dgemm:
	hipcc -O3 xgemm.cu -o $@.x -DTARGET_HIP -DGEMM_TYPE=double -DXBLAS_GEMM=XblasDgemm ${CXXFLAGS} -std=c++14 -I${ROCM_ROOT}/include/rocblas/ -I${RSMI_ROOT}/include -lnuma -lrocm_smi64 -lrocblas -lpthread

sgemm:
	hipcc -O3 xgemm.cu -o $@.x -DTARGET_HIP -DGEMM_TYPE=float -DXBLAS_GEMM=XblasSgemm ${CXXFLAGS} -std=c++14 -I${ROCM_ROOT}/include/rocblas/ -I${RSMI_ROOT}/include -lnuma -lrocm_smi64 -lrocblas -lpthread
