EXAMPLES = ex1 ex2
TESTS    = $(EXAMPLES:%=_%.c)

all: $(TESTS)

$(TESTS):
	@orcc -v $(@:_%=%);\
	if (diff reference/$@ $@) then true; \
	else echo "Possible problem with example $@, diffs above \n========================================="; fi;\
	rm -f tuning*.log $@ timer* *.exe

clean:
	rm -f tuning*.log _*.* timer* *.exe

