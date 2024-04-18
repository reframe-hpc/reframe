EXECUTABLE := dotprod

.SUFFIXES: .o .cpp

CXXFLAGS ?= -std=c++11
OBJS := dotprod.o
$(EXECUTABLE): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^

$(OBJS): dotprod.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(LDFLAGS) -o $@ $^
