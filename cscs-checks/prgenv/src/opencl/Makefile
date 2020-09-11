vecAdd_opencl: vecAdd.o
	${CC} -L${CUDATOOLKIT_HOME}/lib64 -lOpenCL $^ -o $@

vecAdd_opencl.o: vecAdd.c vecAddKernel.cl
	${CC} ${CFLAGS} -c -I${CUDATOOLKIT_HOME}/include $< -o $@
