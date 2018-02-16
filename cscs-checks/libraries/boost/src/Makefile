TARGET = hello_boost_python

$(TARGET).so: $(TARGET).o
	$(CXX) -shared -Wl,--export-dynamic $^ -l$(PYTHON_BOOST_LIB) -o $@
 
$(TARGET).o: $(TARGET).cpp
	$(CXX) -fPIC -c -I$(PYTHON_INCLUDE) $^ -o $@
