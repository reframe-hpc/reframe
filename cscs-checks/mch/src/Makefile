RM := rm -f
EXECUTABLE := openacc_cuda_mpi_cppstd

all: $(EXECUTABLE)
LD = $(FC)

OBJS = compute_cuda.o openacc_cuda_mpi_cppstd.o std_cpp_call.o
# OBJ2 = $(subst _,$(PE_ENV)_,$(OBJ))
LIB  =

.SUFFIXES: .o .cu .cpp .F90

%.o: %.cu
	$(PREP) nvcc $(CPPFLAGS) $(NVCC_FLAGS) -c $< -o $@

%.o: %.cpp
	$(PREP) $(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

%.o: %.F90
	$(PREP) $(FC) $(FCFLAGS) -c $< -o $@

$(EXECUTABLE): $(OBJS)
	$(PREP) $(LD) $(OBJS) -o $@ $(LDFLAGS) $(LIB)

clean:
	-$(RM) $(OBJS)

distclean:
	-$(RM) $(OBJS) $(EXECUTABLE)
