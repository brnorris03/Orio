all: nBodyMain

# makefile for N Body codes
# Alex Kaiser, LBNL, 7/2010

include ../../../Makefile.in

LOCALOBJS = nBodyMain.o nBodyArithmetic.o nBodyUtil.o generators.o


nBodyMain: $(LOCALOBJS)
	$(CC) $(LOCALOBJS) $(CFLAGS) $(CLIBS) -o nBodyMain

nBodyMain.o:
	$(CC) $(CFLAGS) -c nBodyMain.c
	
nBodyArithmetic.o:
	$(CC) $(CFLAGS) -c nBodyArithmetic.c

nBodyUtil.o: 
	$(CC) $(CFLAGS) -c nBodyUtil.c

generators.o: 
	$(CC) $(CFLAGS) -c generators.c

clean:
	rm *.o nBodyMain


