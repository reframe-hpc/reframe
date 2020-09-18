vecAdd: vecAdd.o
	${CC} -L${CUDATOOLKIT_HOME}/lib64 -lOpenCL $^ -o $@

vecAdd.o: vecAdd.c vecAddKernel.cl
	${CC} ${CFLAGS} -c -I${CUDATOOLKIT_HOME}/include $< -o $@
