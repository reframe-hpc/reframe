OBJECTS = $(filter-out %.c,$(notdir $(patsubst %.cc,%.o,$(SOURCES)))) \
	  $(filter-out %.cc,$(notdir $(patsubst %.c,%.o,$(SOURCES))))

# to be used on dom with the new lib
#PATH_TO_LIB = /opt/cray/nvidia/440.33.01_3.1.43-7.0.1.1_2.2__gc90b954.ari/lib64

# to be used on daint
PATH_TO_LIB = /opt/cray/nvidia/default/lib64

LDFLAGS +=   -Wl,-rpath ${PATH_TO_LIB} -L${PATH_TO_LIB} -lOpenGL -lEGL

# Comment out to build without debug info.
CXXFLAGS += -g -I. 

SOURCES = tinyegl.cc

BIN = tinyegl

$(BIN): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) $(LDFLAGS) -o $@

.PHONY: clean
clean:
	rm -f $(BIN) *.o
