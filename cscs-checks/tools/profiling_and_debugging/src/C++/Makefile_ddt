RM := rm -f
EXECUTABLE := jacobi

all: jacobi

OBJS := _main.o _jacobi.o

$(OBJS):
	$(PREP) $(CXX) $(CXXFLAGS) -c -o $(@) $(@:.o=.cc)

$(EXECUTABLE): $(OBJS)
	$(PREP) $(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(@) $(OBJS)

clean:
	-$(RM) $(OBJS)

distclean:
	-$(RM) $(OBJS) $(EXECUTABLE)
