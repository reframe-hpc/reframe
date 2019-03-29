RM := rm -f
EXECUTABLE := jacobi

all: jacobi

OBJS := _main.o _jacobi.o

$(OBJS):
	$(PREP) $(CXX) $(CXXFLAGS) -c -o $(@) $(@:.o=.cc)

$(EXECUTABLE): $(OBJS)
	$(PREP) $(CXX) $(CXXFLAGS) -o $(@) $(OBJS) $(LDFLAGS)

clean:
	-$(RM) $(OBJS)

distclean:
	-$(RM) $(OBJS) $(EXECUTABLE)
