.PHONY: all clean

PROGRAMS = hello_c hello_cpp hello_fort

all: $(PROGRAMS)

hello_c: hello.c
	$(CC) $(CFLAGS) -o $@ $<

hello_cpp: hello.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

hello_fort: hello.f90
	$(FC) $(FFLAGS) -o $@ $<

clean:
	/bin/rm -f $(PROGRAMS)
