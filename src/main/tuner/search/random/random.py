#
# Implementation of the random search algorithm
#

import sys, time
import main.tuner.search.search

#-----------------------------------------------------

class Random(main.tuner.search.search.Search):
    '''
    The search engine that uses a random search approach, enhanced with a local search that finds
    the best neighboring coordinate.

    Below is a list of algorithm-specific arguments used to steer the search algorithm.
      local_distance            the distance number used in the local search to find the best
                                neighboring coordinate located within the specified distance
    '''

    # algorithm-specific argument names
    __LOCAL_DIST = 'local_distance'       # default: 0
    
    #--------------------------------------------------
    
    def __init__(self, cfrags, axis_names, axis_val_ranges, constraint, time_limit, total_runs, 
                 search_opts, cmd_line_opts, ptcodegen, ptdriver, odriver, use_parallel_search):
        '''To instantiate a random search engine'''

        main.tuner.search.search.Search.__init__(self, cfrags, axis_names, axis_val_ranges,
                                                 constraint, time_limit, total_runs, search_opts,
                                                 cmd_line_opts, ptcodegen, ptdriver, odriver,
                                                 use_parallel_search)

        # set all algorithm-specific arguments to their default values
        self.local_distance = 0

        # read all algorithm-specific arguments
        self.__readAlgoArgs()
        
        # complain if both the search time limit and the total number of search runs are undefined
        if self.time_limit <= 0 and self.total_runs <= 0:
            print (('error: %s search requires either (both) the search time limit or (and) the ' +
                    'total number of search runs to be defined') % self.__class__.__name__)
            sys.exit(1)

    #--------------------------------------------------
    
    def __readAlgoArgs(self):
        '''To read all algorithm-specific arguments'''

        # check for algorithm-specific arguments
        for vname, rhs in self.search_opts.iteritems():

            # local search distance
            if vname == self.__LOCAL_DIST:
                if not isinstance(rhs, int) or rhs < 0:
                    print ('error: %s argument "%s" must be a positive integer or zero'
                           % (self.__class__.__name__, vname))
                    sys.exit(1)
                self.local_distance = rhs

            # unrecognized algorithm-specific argument
            else:
                print ('error: unrecognized %s algorithm-specific argument: "%s"' %
                       (self.__class__.__name__, vname))
                sys.exit(1)

    #--------------------------------------------------

    def __getNextCoord(self, coord_records, neigh_coords):
        '''Get the next coordinate to be empirically tested'''

        # check if all coordinates have been explored
        if len(coord_records) >= self.space_size:
            return None

        # pick the next neighbor coordinate in the list (if exists)
        while len(neigh_coords) > 0:
            coord = neigh_coords.pop(0)
            if str(coord) not in coord_records:
                coord_records[str(coord)] = None
                return coord

        # randomly pick a coordinate that has never been explored before
        while True:
            coord = self.getRandomCoord()
            if str(coord) not in coord_records:
                coord_records[str(coord)] = None
                return coord
    
    #--------------------------------------------------
    
    def searchBestCoord(self):
        '''
        To explore the search space and retun the coordinate that yields the best performance
        (i.e. minimum performance cost).
        '''

        if self.verbose: print '\n----- begin random search -----'

        # get the total number of coordinates to be tested at the same time
        coord_count = 1
        if self.use_parallel_search:
            coord_count = self.num_procs

        # initialize a storage to remember all coordinates that have been explored
        coord_records = {}

        # initialize a list to store the neighboring coordinates
        neigh_coords = []

        # record the best coordinate and its best performance cost
        best_coord = None
        best_perf_cost = self.MAXFLOAT
        old_perf_cost = best_perf_cost

        # record the number of runs
        runs = 0
        
        # start the timer
        start_time = time.time()

        # execute the randomized search method
        while True:

            # randomly pick a set of coordinates to be empirically tested
            coords = []
            while len(coords) < coord_count:
                coord = self.__getNextCoord(coord_records, neigh_coords)
                if coord:
                    coords.append(coord)
                else:
                    break

            # check if all coordinates in the search space have been explored
            if len(coords) == 0:
                break

            # determine the performance cost of all chosen coordinates
            perf_costs = self.getPerfCosts(coords)

            # compare to the best result
            pcost_items = perf_costs.items()
            pcost_items.sort(lambda x,y: cmp(eval(x[0]),eval(y[0])))
            for i, (coord_str, perf_cost) in enumerate(pcost_items):
                coord_val = eval(coord_str)
                if self.verbose:
                    print '(run %s) coordinate: %s, cost: %s' % (runs+i+1, coord_val, perf_cost)
                if perf_cost < best_perf_cost:
                    best_coord = coord_val
                    best_perf_cost = perf_cost
                    if self.verbose:
                        print '>>>> best coordinate found: %s, cost: %s' % (coord_val, perf_cost)

            # if a better coordinate is found, explore the neighboring coordinates
            if old_perf_cost != best_perf_cost:
                neigh_coords.extend(self.getNeighbors(best_coord, self.local_distance))
                old_perf_cost = best_perf_cost

            # increment the number of runs
            runs += len(perf_costs)
                        
            # check if the time is up
            if self.time_limit > 0 and (time.time()-start_time) > self.time_limit:
                break

            # check if the maximum limit of runs is reached
            if self.total_runs > 0 and runs >= self.total_runs:
                break

        # compute the total search time
        search_time = time.time() - start_time
        
        if self.verbose: print '----- end random search -----'
        if self.verbose: print '----- begin summary -----'
        if self.verbose: print ' best coordinate: %s, cost: %s' % (best_coord, best_perf_cost)
        if self.verbose: print ' total search time: %.2f seconds' % search_time
        if self.verbose: print ' total completed runs: %s' % runs
        if self.verbose: print '----- end summary -----'
        
        # return the best coordinate
        return best_coord
            
