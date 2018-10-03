all: vectorize_fma_c vectorize_fma_cplusplus vectorize_fma_ftn

vectorize_fma_c: vectorize_fma_c.c
	$(CC) $(CFLAGS) vectorize_fma_c.c

vectorize_fma_cplusplus: vectorize_fma_cplusplus.cc
	$(CXX) $(CXXFLAGS) vectorize_fma_cplusplus.cc

vectorize_fma_ftn: vectorize_fma_ftn.f90
	$(FC) $(FCFLAGS) vectorize_fma_ftn.f90

clean:
	rm *.s
