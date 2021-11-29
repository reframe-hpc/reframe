RM := rm -f
EXECUTABLE := jacobi
all: jacobi
OBJS :=  _variableDef.o _jacobi.o _main.o

ifeq ($(PE_ENV),GNU)
ifeq ($(word 1, $(subst ., ,$(GNU_VERSION))),10)
MISMATCH = -fallow-argument-mismatch
endif
ifeq ($(word 1, $(subst ., ,$(GNU_VERSION))),11)
MISMATCH = -fallow-argument-mismatch -w
endif
endif

$(OBJS):
	$(PREP) $(FC) $(MISMATCH) $(FCFLAGS) -c -o $(@) $(@:.o=.F90)

$(EXECUTABLE): $(OBJS)
	$(PREP) $(FC) $(FCFLAGS) $(LDFLAGS) -o $(@) $(OBJS)

clean:
	-$(RM) $(OBJS) *.mod

distclean:
	-$(RM) $(OBJS) $(EXECUTABLE)
