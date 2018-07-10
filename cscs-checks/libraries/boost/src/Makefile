TARGET = hello_boost_python
PYTHONDIR=$(shell echo $(PYTHONPATH) | cut -d ':' -f 1)

$(TARGET).so: $(TARGET).o
	$(CXX) -shared -Wl,--export-dynamic $^ -l$(PYTHON_BOOST_LIB) -o $@
 
$(TARGET).o: $(TARGET).cpp
	$(CXX) -fPIC -c -I"$(PYTHONDIR)/$(PYTHON_INCLUDE)" $^ -o $@
