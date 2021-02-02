CXXFLAGS?=--amdgpu-target=gfx906,gfx908

kernel_latency:
	hipcc -O3 $@.cu -o $@.x -DTARGET_HIP ${CXXFLAGS} -std=c++11
