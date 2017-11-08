EXECUTABLE := advanced_example1 

.SUFFIXES: .o .c

OBJS := advanced_example1.o

$(EXECUTABLE): $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^
 
$(OBJS): advanced_example1.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $(LDFLAGS) -o $@ $^ 
