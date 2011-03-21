include ../Makefile.in

TARGETS = breadth_first_search betweenness_centrality

all: $(TARGETS)

.PHONY: clean

BFS_OBJS = breadth_first_search.o graph_gen.o
BC_OBJS = betweenness_centrality.o graph_gen.o

INCLUDEFILES = graph.h 

%.o: %.c $(INCLUDEFILES)
	$(CC) $(CFLAGS) -c $<

breadth_first_search: $(BFS_OBJS) $(INCLUDEFILES) 
	$(CC) $(CFLAGS) -o breadth_first_search $(BFS_OBJS)

betweenness_centrality: $(BC_OBJS) $(INCLUDEFILES)
	$(CC) $(CFLAGS) -o betweenness_centrality $(BC_OBJS)

clean:
	rm -f *.o *~ $(TARGETS) *.exe core*
