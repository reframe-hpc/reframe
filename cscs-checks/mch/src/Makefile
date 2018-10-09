RM := rm -f
EXECUTABLE := openacc_cuda_mpi_cppstd

all: $(EXECUTABLE)
LD = $(FC)

OBJS = compute_cuda.o openacc_cuda_mpi_cppstd.o std_cpp_call.o
# OBJ2 = $(subst _,$(PE_ENV)_,$(OBJ))
LIB  =

.SUFFIXES: .o .cu .cpp .F90

%.o: %.cu
	$(NVCC) $(CPPFLAGS) $(NVCC_FLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

%.o: %.F90
	$(FC) $(CPPFLAGS) $(FCFLAGS) -c $< -o $@

$(EXECUTABLE): $(OBJS)
	$(LD) $(OBJS) -o $@ $(LDFLAGS) $(LIB)

clean:
	-$(RM) $(OBJS)

distclean:
	-$(RM) $(OBJS) $(EXECUTABLE)
