all: charArraySort integerSort spatialSort

include ../Makefile.in

# makefile for Sort kernels
# Alex Kaiser, LBNL, 9/2010



charArraySort: charArraySort.o
	$(CC) $(CFLAGS) $(CLIBS) charArraySort.o -o charArraySort

charArraySort.o:
	$(CC) $(CFLAGS) charArraySort.c -c
	
integerSort: integerSort.o
	$(CC) $(CFLAGS) $(CLIBS) integerSort.o -o integerSort

integerSort.o:
	$(CC) $(CFLAGS) integerSort.c -c
	
spatialSort: spatialSort.o
	$(CC) $(CFLAGS) $(CLIBS) spatialSort.o -o spatialSort

spatialSort.o:
	$(CC) $(CFLAGS) spatialSort.c -c

clean:
	rm *.o charArraySort integerSort spatialSort
