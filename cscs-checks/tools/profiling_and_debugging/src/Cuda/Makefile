RM := rm -f
EXECUTABLE := jacobi

all: $(EXECUTABLE)
LD = $(CC)

OBJS = _jacobi-cuda-kernel.o _jacobi-cuda.o
# OBJ2 = $(subst _,$(PE_ENV)_,$(OBJ))
LIB  = 

.SUFFIXES: .o .cu .c

%.o: %.cu
	$(PREP) nvcc $(NVCCFLAGS) $(DDTFLAGS) -c $< -o $@

%.o: %.c
	$(PREP) $(CC) $(CPPFLAGS) $(CFLAGS) $(DDTFLAGS) -c $< -o $@

$(EXECUTABLE): $(OBJS)
	$(PREP) $(LD) $(CFLAGS) $(LDFLAGS) $(OBJS) $(LIB) -o $@

clean:
	-$(RM) $(OBJS)

distclean:
	-$(RM) $(OBJS) $(EXECUTABLE)
