all: sparseMain

# makefile for sparse linear algebra
# Alex Kaiser, LBNL, 7/2010

include ../../Makefile.in

sparseMain: sparseMain.o generators.o sparseArithmetic.o sparseUtil.o
	$(CC) $(CFLAGS) $(CLIBS) sparseMain.o generators.o sparseArithmetic.o sparseUtil.o -o sparseMain

sparseMain.o: 
	$(CC) $(CFLAGS) -c sparseMain.c

generators.o: 
	$(CC) $(CFLAGS) -c generators.c

sparseArithmetic.o: 
	$(CC) $(CFLAGS) -c sparseArithmetic.c
	
sparseUtil.o: 
	$(CC) $(CFLAGS) -c sparseUtil.c

clean:
	rm *.o sparseMain
