RM := rm -f
EXECUTABLE := jacobi

all: $(EXECUTABLE)

OBJS := _main.o _jacobi.o
.SUFFIXES: .o .c

$(OBJS):
	$(PREP) $(CC) $(CFLAGS) -c -o $(@) $(@:.o=.c)

$(EXECUTABLE): $(OBJS)
	$(PREP) $(CC) $(CFLAGS) $(LDFLAGS) -o $(@) $(OBJS)

clean:
	-$(RM) $(OBJS)

distclean:
	-$(RM) $(OBJS) $(EXECUTABLE)
