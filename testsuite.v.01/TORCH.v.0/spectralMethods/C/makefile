all: spectralMain

include ../../Makefile.in

# makefile for structured grid codes
# Alex Kaiser, LBNL, 7/2010


spectralMain: spectralMain.o ffts.o complexUtil.o
	$(CC) $(CFLAGS) $(CLIBS) spectralMain.o ffts.o complexUtil.o -o spectralMain

spectralMain.o: ffts.o complexUtil.o
	$(CC) $(CFLAGS) -c spectralMain.c

ffts.o: ffts.c complexUtil.h
	$(CC) $(CFLAGS) -c ffts.c

complexUtil.o: complexUtil.h complexUtil.c
	$(CC) $(CFLAGS) -c complexUtil.c


clean:
	rm *.o spectralMain