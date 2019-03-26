RM := rm -f
EXECUTABLE := roof.exe
all: $(EXECUTABLE)
OBJS := _roofline.o

$(OBJS):
	$(PREP) $(CXX) $(CXXFLAGS) -c -o $(@) $(@:.o=.cpp)

$(EXECUTABLE): $(OBJS)
	$(PREP) $(CXX) $(CXXFLAGS) -o $(@) $(OBJS) $(LDFLAGS)

clean:
	-$(RM) $(OBJS)

distclean:
	-$(RM) $(OBJS) $(EXECUTABLE)
