vecAdd_opencl: vecAdd_opencl.o
	${CC} -L${CUDATOOLKIT_HOME}/lib64 -lOpenCL $^ -o $@

vecAdd_opencl.o: vecAdd_opencl.c vecAddKernel.cl
	${CC} ${CFLAGS} -c -I${CUDATOOLKIT_HOME}/include $< -o $@
