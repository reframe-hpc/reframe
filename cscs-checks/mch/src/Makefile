RM := rm -f
EXECUTABLE := openacc_cuda_mpi_cppstd

all: $(EXECUTABLE)
LD = $(FC)

OBJS = compute_cuda.o openacc_cuda_mpi_cppstd.o std_cpp_call.o
# OBJ2 = $(subst _,$(PE_ENV)_,$(OBJ))
LIB  = 

.SUFFIXES: .o .cu .cpp .F90

%.o: %.cu
	$(PREP) nvcc $(CXXFLAGS) $(DDTFLAGS) -c $< -o $@

%.o: %.cpp
	$(PREP) $(CXX) $(CFLAGS) $(DDTFLAGS) -c $< -o $@

%.o: %.F90
	$(PREP) $(FC) $(FFLAGS) $(DDTFLAGS) -c $< -o $@

$(EXECUTABLE): $(OBJS)
	$(PREP) $(LD) $(FFLAGS) $(LDFLAGS) $(OBJS) $(LIB) -o $@

clean:
	-$(RM) $(OBJS)

distclean:
	-$(RM) $(OBJS) $(EXECUTABLE)
