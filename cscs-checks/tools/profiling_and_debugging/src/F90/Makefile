RM := rm -f
EXECUTABLE := jacobi

all: jacobi

OBJS :=  _variableDef.o _jacobi.o _main.o

$(OBJS):
	$(PREP) $(FC) $(FFLAGS) -c -o $(@) $(@:.o=.F90)

$(EXECUTABLE): $(OBJS)
	$(PREP) $(FC) $(FFLAGS) $(LDFLAGS) -o $(@) $(OBJS)

clean:
	-$(RM) $(OBJS) *.mod

distclean:
	-$(RM) $(OBJS) $(EXECUTABLE)
