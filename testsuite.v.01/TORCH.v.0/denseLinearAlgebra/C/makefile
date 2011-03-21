all: denseMain

include ../../Makefile.in

# makefile for dense linear algebra codes
# Alex Kaiser, LBNL, 7/2010



LOCALOBJS = denseMain.o denseArithmetic.o denseUtil.o generators.o


denseMain: $(LOCALOBJS)
	$(CC) $(LOCALOBJS) $(CFLAGS) $(CLIBS) -o denseMain

denseMain.o:
	$(CC) $(CFLAGS) -c denseMain.c
	
denseArithmetic.o:
	$(CC) $(CFLAGS) -c denseArithmetic.c

denseUtil.o: 
	$(CC) $(CFLAGS) -c denseUtil.c

generators.o: 
	$(CC) $(CFLAGS) -c generators.c

clean:
	rm *.o denseMain


