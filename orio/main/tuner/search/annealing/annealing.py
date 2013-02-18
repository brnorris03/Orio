#
# Implementation of the simulated-annealing search algorithm
#

import math, sys, time
import orio.main.tuner.search.search
from orio.main.util.globals import *

#-----------------------------------------------------

class Annealing(orio.main.tuner.search.search.Search):
    '''
    The search engine that uses a simulated-annealing search approach, enhanced with a local search
    that finds the best neighboring coordinate.

    Below is a list of algorithm-specific arguments used to steer the search algorithm.
      local_distance            the distance number used in the local search to find the best
                                neighboring coordinate located within the specified distance
      cooling_factor            the temperature reduction factor
      final_temperature_ratio   the percentage of the temperature used as the final temperature
      trials_limit              the maximum limit of numbers of trials at each temperature
      moves_limit               the maximum limit of numbers of successful moves at each temperature
    '''

    # algorithm-specific argument names
    __LOCAL_DIST = 'local_distance'             # default: 0
    __COOL_FACT = 'cooling_factor'              # default: 0.95
    __FTEMP_RATIO = 'final_temperature_ratio'   # default: 0.05
    __TR_LIMIT = 'trials_limit'                 # default: 100
    __MV_LIMIT = 'moves_limit'                  # default: 20
    
    #--------------------------------------------------
    
    def __init__(self, params):
        '''To instantiate a simulated-annealing search engine'''

        orio.main.tuner.search.search.Search.__init__(self, params)

        # set all algorithm-specific arguments to their default values
        self.local_distance = 0
        self.cooling_factor = 0.95
        self.final_temp_ratio = 0.05
        self.trials_limit = 100
        self.moves_limit = 20
        self.bignum = 13124314.0

        # read all algorithm-specific arguments
        self.__readAlgoArgs()
        

            
    #--------------------------------------------------
    # Method required by the search interface

    def searchBestCoord(self, startCoord=None):
        '''
        To explore the search space and return the coordinate that yields the best performance
        (i.e. minimum performance cost).
        '''
        
        # TODO: implement startCoord support

        info('\n----- begin simulated annealing search -----')

        # check for parallel search
        if self.use_parallel_search:
            err('orio.main.tuner.search.annealing.annealing: simulated annealing search does not support parallel search')

        # initialize a storage to remember all initial coordinates that have been explored
        coord_records = {}
                
        # record the best global coordinate and its best performance cost
        best_global_coord = None
        best_global_perf_cost = self.MAXFLOAT

        info('--> begin temperature initialization')
        
        # calculate the initial and final temperatures
        init_temperature = self.__initTemperature()
        final_temperature = self.final_temp_ratio * init_temperature

        info('--> end temperature initialization')

        # record the number of runs
        runs = 0
        
        # start the timer
        start_time = time.time()
        
        # execute the simulated annealing procedure
        while True:

            # initialize the temperature
            temperature = init_temperature

            # randomly pick an initial coordinate in the search space
            coord = self.__initRandomCoord(coord_records)

            # if all initial coordinates in the search space have been used before
            if coord == None:
                break

            # get the performance cost of the current initial coordinate
            perf_cost = self.getPerfCost(coord)

            # record the best coordinate and its best performance cost
            best_coord = coord
            best_perf_cost = perf_cost
            
            info('\n(run %s) initial coord: %s, cost: %e' % (runs+1, coord, perf_cost))
            
            # the annealing loop
            while temperature > final_temperature:

                
                info('-> anneal step: temperature: %.2f%%, final temperature: %.2f%%' %
                     (100.0 * temperature / init_temperature,
                      100.0 * final_temperature / init_temperature))

                # initialize the number of good moves
                good_moves = 0
                
                # the trial loop (i.e. the Metropolis Monte Carlo simulation loop)
                for trial in range(0, self.trials_limit):
                
                    # get a new coordinate (i.e. a random neighbor)
                    new_coord = self.__getRandomNeighbor(coord)
                    
                    # check if no neighboring coordinate can be found
                    if new_coord == None:
                        break

                    # get the performance cost of the new coordinate
                    new_perf_cost = self.getPerfCost(new_coord)
                    
                    # compare to the best result so far
                    if new_perf_cost < best_perf_cost and new_perf_cost > 0.0:
                        best_coord = new_coord
                        best_perf_cost = new_perf_cost
                        info('--> best annealing coordinate found: %s, cost: %e' %
                             (best_coord, best_perf_cost))

                    # calculate the performance cost difference
                    delta = new_perf_cost - perf_cost
                        
                    # if the new coordinate has a better performance cost
                    if delta < 0 and new_perf_cost > 0.0:
                        coord = new_coord
                        perf_cost = new_perf_cost
                        good_moves += 1
                        info('--> move to BETTER coordinate: %s, cost: %e' %
                             (coord, perf_cost))

                    # compute the acceptance probability (i.e. the Boltzmann probability or
                    # the Metropolis criterion) to see whether a move to the new coordinate is
                    # needed
                    # the acceptance probability formula: p = e^(-delta/temperature)
                    else:

                        # count the probability of moving to the new coordinate
                        delta = self.bignum
                        p = math.exp(-delta / temperature)
                        if self.getRandomReal(0,1) < p:
                            coord = new_coord
                            perf_cost = new_perf_cost
                            good_moves += 1
                            info('--> move to WORSE coordinate: %s, cost: %e' % (coord, perf_cost))

                    # check if the maximum limit of the good moves is reached
                    if good_moves > self.moves_limit:
                        break

                    # check if the time is up
                    if self.time_limit > 0 and (time.time()-start_time) > self.time_limit:
                        break
                
                # reduce the temperature (i.e. the cooling/annealing schedule)
                temperature *= self.cooling_factor

                # check if the time is up
                if self.time_limit > 0 and (time.time()-start_time) > self.time_limit:
                    break

            info('-> best annealing coordinate: %s, cost: %e' % (best_coord, best_perf_cost))

            # record the current best performance cost
            old_best_perf_cost = best_perf_cost
            
            # check if the time is not up yet
            if self.time_limit <= 0 or (time.time()-start_time) <= self.time_limit:
                
                # perform a local search on the best annealing coordinate
                best_coord, best_perf_cost = self.searchBestNeighbor(best_coord, self.local_distance)

                # if the neighboring coordinate has a better performance cost
                if best_perf_cost < old_best_perf_cost:
                    info('---> better neighbor found: %s, cost: %s' % (best_coord, best_perf_cost))
                
            # compared to the best global result so far
            if best_perf_cost < best_global_perf_cost:
                best_global_coord = best_coord
                best_global_perf_cost = best_perf_cost
                info('>>>> best coordinate found: %s, cost: %s' % (best_global_coord, best_global_perf_cost))
                            
            # increment the number of runs
            runs += 1
            
            # check if the time is up
            if self.time_limit > 0 and (time.time()-start_time) > self.time_limit:
                break

            # check if the maximum limit of runs is reached
            if self.total_runs > 0 and runs >= self.total_runs:
                break

        # compute the total search time
        search_time = time.time() - start_time
        
        info('----- end simulated annealing search -----')
        
        # return the best coordinate
        return best_global_coord, best_global_perf_cost, search_time, runs
       
       
        #--------------------------------------------------
    
    def __readAlgoArgs(self):
        '''To read all algorithm-specific arguments'''

        # check for algorithm-specific arguments
        for vname, rhs in self.search_opts.iteritems():

            # local search distance
            if vname == self.__LOCAL_DIST:
                if not isinstance(rhs, int) or rhs < 0:
                    err('orio.main.tuner.search.annealing: %s argument "%s" must be a positive integer or zero'
                        % (self.__class__.__name__, vname))
                self.local_distance = rhs

            # the temperature reduction factor
            elif vname == self.__COOL_FACT:
                if not isinstance(rhs, float) or rhs <= 0 or rhs >= 1:
                    err('orio.main.tuner.search.annealing: %s argument "%s" must be a real number between zero and one'
                           % (self.__class__.__name__, vname))
                self.cooling_factor = rhs

            # the final temperature ratio
            elif vname == self.__FTEMP_RATIO:
                if not isinstance(rhs, float) or rhs <= 0 or rhs >= 1:
                    err('orio.main.tuner.search.annealing: %s argument "%s" must be a real number between zero and one'
                           % (self.__class__.__name__, vname))
                self.final_temp_ratio = rhs

            # the maximum limit of numbers of trials at each temperature 
            elif vname == self.__TR_LIMIT:
                if not isinstance(rhs, int) or rhs <= 0:
                    err('orio.main.tuner.search.annealing: %s argument "%s" must be a positive integer'
                           % (self.__class__.__name__, vname))
                self.trials_limit = rhs

            # the maximum limit of numbers of successful moves at each temperature
            elif vname == self.__MV_LIMIT:
                if not isinstance(rhs, int) or rhs <= 0:
                    err('orio.main.tuner.search.annealing: %s argument "%s" must be a positive integer'
                           % (self.__class__.__name__, vname))
                self.moves_limit = rhs

            # unrecognized algorithm-specific argument
            else:
                err('orio.main.tuner.search.annealing: unrecognized %s algorithm-specific argument: "%s"' %
                       (self.__class__.__name__, vname))

    # Private methods
    #--------------------------------------------------

    def __initTemperature(self):
        '''
        Provide an estimation of the initial temperature by taking the average of
        the performance-cost differences among randomly chosen coordinates
        '''

        # set some useful variables
        cur_coord_records = {}
        max_distinct_coords = min(self.space_size, 5000)
        max_random_coords = min(self.space_size, 10)

        # randomly pick several random coordinates with their performance costs
        random_coords = []
        perf_costs = []
        while True:
            if len(cur_coord_records) >= max_distinct_coords:
                break
            coord = self.getRandomCoord()
            if str(coord) not in cur_coord_records:
                cur_coord_records[str(coord)] = None
                perf_cost = self.getPerfCost(coord)
                if perf_cost != self.MAXFLOAT:
                    random_coords.append(coord)
                    perf_costs.append(perf_cost)
                    if len(random_coords) >= max_random_coords:
                        break

        # check if not enough random coordinates are found
        if len(random_coords) == 0:
            err('orio.main.tuner.search.annealing: initialization of Simulated Annealing failed: no valid values of ' +
                   'performance parameters can be found. the performance parameter constraints ' +
                   'might prune out the entire search space.')
        
        # sort the random coordinates in an increasing order of performance costs
        sorted_coords = zip(random_coords, perf_costs)
        sorted_coords.sort(lambda x,y: cmp(x[1],y[1]))
        random_coords, perf_costs = zip(*sorted_coords)

        # take the best coordinate
        best_coord = random_coords[0]
        best_perf_cost = perf_costs[0]

        # compute the average performance-cost difference
        total_cost_diff = reduce(lambda x,y: x+y, map(lambda x: x-best_perf_cost, perf_costs), 0)
        avg_cost_diff = 0
        if total_cost_diff > 0:
            avg_cost_diff = total_cost_diff / (len(random_coords)-1)

        # select an initial temperature value that results in a 80% acceptance probability
        # the acceptance probability formula: p = e^(-delta/temperature)
        # hence, temperature = -delta/ln(p)
        # where the delta is the average performance-cost difference relative to the minimum cost
        init_temperature = -avg_cost_diff / math.log(0.8, math.e)

        # return the initial temperature
        return init_temperature

    #--------------------------------------------------

    def __initRandomCoord(self, coord_records):
        '''Randomly initialize a coordinate in the search space'''

        # check if all coordinates have been explored
        if len(coord_records) >= self.space_size:
            return None
        
        # randomly pick a coordinate that has never been explored before
        while True:
            coord = self.getRandomCoord()
            if str(coord) not in coord_records:
                coord_records[str(coord)] = None
                return coord
            
    #--------------------------------------------------

    def __getRandomNeighbor(self, coord):
        '''
        Return a random neighboring coordinate, that is different from the given coordinate.
        If no neighboring coordinate is found (after many attempts), return the given coordinate.
        '''

        neigh_coord = None
        total_trials = 1000
        for trial in range(0, total_trials):
            n_coord = coord[:]
            for i in range(0, self.total_dims):
                ipoint = coord[i] + self.getRandomInt(-1,1)
                if 0 <= ipoint < self.dim_uplimits[i]:
                    n_coord[i] = ipoint
            if n_coord != coord:
                return n_coord
        return neigh_coord 
