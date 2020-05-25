.PHONY: all clean

PROGRAMS = hello_c

all: $(PROGRAMS)

hello_c: hello.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	/bin/rm -f $(PROGRAMS)
