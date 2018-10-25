RM := rm -f
EXECUTABLE := jacobi

all: jacobi

OBJS :=  _variableDef.o _jacobi.o _main.o

$(OBJS):
	$(PREP) $(FC) $(FCFLAGS) -c -o $(@) $(@:.o=.F90)

$(EXECUTABLE): $(OBJS)
	$(PREP) $(FC) $(FCFLAGS) $(LDFLAGS) -o $(@) $(OBJS)

clean:
	-$(RM) $(OBJS) *.mod

distclean:
	-$(RM) $(OBJS) $(EXECUTABLE)
