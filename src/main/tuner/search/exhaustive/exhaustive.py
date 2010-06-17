#
# Implementation of the exhaustive search algorithm 
#

import sys, time
import main.tuner.search.search

#-----------------------------------------------------

class Exhaustive(main.tuner.search.search.Search):
    '''The search engine that uses an exhaustive search approach'''

    def __init__(self, params):
        '''
        Instantiate an exhaustive search engine given a dictionary of options.
        @param params: dictionary of options needed to configure the search algorithm.
        '''

        main.tuner.search.search.Search.__init__(self, params)

        # read all algorithm-specific arguments
        self.__readAlgoArgs()

        # complain if the total number of search runs is defined (i.e. exhaustive search
        # only needs to be run once)
        if self.total_runs > 1:
            print ('error: the total number of %s search runs must be one (or can be undefined)' %
                   self.__class__.__name__)
            sys.exit(1)
            

    #-----------------------------------------------------
    # Method required by the search interface
    def searchBestCoord(self):
        '''
        To explore the search space and retun the coordinate that yields the best performance
        (i.e. minimum performance cost).
        '''

        if self.verbose: print '\n----- begin exhaustive search -----'

        # get the total number of coordinates to be tested at the same time
        coord_count = 1
        if self.use_parallel_search:
            coord_count = self.num_procs
        
        # record the best coordinate and its best performance cost
        best_coord = None
        best_perf_cost = self.MAXFLOAT
        
        # start the timer
        start_time = time.time()

        # start from the origin coordinate (i.e. [0,0,...])
        coord = [0] * self.total_dims 
        coords = [coord]
        while len(coords) < coord_count:
            coord = self.__getNextCoord(coord)
            if coord:
                coords.append(coord)
            else:
                break

        # evaluate every coordinate in the search space
        while True:

            # determine the performance cost of all chosen coordinates
            perf_costs = self.getPerfCosts(coords)

            # compare to the best result
            pcost_items = perf_costs.items()
            pcost_items.sort(lambda x,y: cmp(eval(x[0]),eval(y[0])))
            for coord_str, perf_cost in pcost_items:
                coord_val = eval(coord_str)
                if self.verbose: print 'coordinate: %s, cost: %s' % (coord_val, perf_cost)
                if perf_cost < best_perf_cost:
                    best_coord = coord_val
                    best_perf_cost = perf_cost
                    if self.verbose:
                        print '>>>> best coordinate found: %s, cost: %s' % (coord_val, perf_cost)

            # check if the time is up
            if self.time_limit > 0 and (time.time()-start_time) > self.time_limit:
                break

            # get to all the next coordinates in the search space
            coords = []
            while len(coords) < coord_count:
                if coord:
                    coord = self.__getNextCoord(coord)
                if coord:
                    coords.append(coord)
                else:
                    break

            # check if all coordinates have been visited
            if len(coords) == 0:
                break

        # compute the total search time
        search_time = time.time() - start_time

        if self.verbose: print '----- end exhaustive search -----'
        if self.verbose: print '----- begin summary -----'
        if self.verbose: print ' best coordinate: %s, cost: %s' % (best_coord, best_perf_cost)
        if self.verbose: print ' total search time: %.2f seconds' % search_time
        if self.verbose: print '----- end summary -----'
        
        # return the best coordinate
        return best_coord
    
    # Private methods       
    #--------------------------------------------------
        
    def __readAlgoArgs(self):
        '''To read all algorithm-specific arguments'''
                
        for vname, rhs in self.search_opts.iteritems():
            print ('error: unrecognized %s algorithm-specific argument: "%s"' %
                   (self.__class__.__name__, vname))
            sys.exit(1)

    #--------------------------------------------------

    def __getNextCoord(self, coord):
        '''
        Return the next neighboring coordinate to be considered in the search space.
        Return None if all coordinates in the search space have been visited.
        '''
        next_coord = coord[:]
        for i in range(0, self.total_dims):
            ipoint = next_coord[i]
            iuplimit = self.dim_uplimits[i]
            if ipoint < iuplimit-1:
                next_coord[i] += 1
                break
            else:
                next_coord[i] = 0
                if i == self.total_dims - 1:
                    return None
        return next_coord

