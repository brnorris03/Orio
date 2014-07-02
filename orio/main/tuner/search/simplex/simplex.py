#
# Implementation of the Nelder-Mead Simplex algorithm
#
# The detailed algorithm is described in the following paper.
#   "Convergence Properties of the Nelder-Mead Simplex Method in Low Dimensions"
#   by Jeffrey C. Lagarias
#

import random, sys, time
import orio.main.tuner.search.search
from orio.main.util.globals import *

#-----------------------------------------------------

class Simplex(orio.main.tuner.search.search.Search):
    '''
    The search engine that uses the Nelder-Mead Simplex algorithm, enhanced with a local search
    that finds the best neighboring coordinate.

    Below is a list of algorithm-specific arguments used to steer the search algorithm.
      local_distance            the distance number used in the local search to find the best
                                neighboring coordinate located within the specified distance
      reflection_coef           the reflection coefficient
      expansion_coef            the expansion coefficient
      contraction_coef          the contraction coefficient
      shrinkage_coef            the shrinkage coefficient
    '''

    # algorithm-specific argument names
    __LOCAL_DIST = 'local_distance'       # default: 0
    __REFL_COEF = 'reflection_coef'       # default: 1.0
    __EXP_COEF = 'expansion_coef'         # default: 2.0
    __CONT_COEF = 'contraction_coef'      # default: 0.5
    __SHRI_COEF = 'shrinkage_coef'        # default: 0.5
    __X0       =  'x0'                    # default: all 0's

    #-----------------------------------------------------

    def __init__(self, params):
        '''To instantiate a Nelder-Mead simplex search engine'''
        
        orio.main.tuner.search.search.Search.__init__(self, params)

        if self.use_parallel_search:
            err('parallel search for simplex is not supported yet.\n')
            
        # other private class variables
        self.__simplex_size = self.total_dims + 1

        # set all algorithm-specific arguments to their default values
        self.local_distance = 0
        self.refl_coefs = [1.0]
        self.exp_coefs = [2.0]
        self.cont_coefs = [0.5]
        self.shri_coef = 0.5
        self.x0 = [0] * self.total_dims
        self.sim_size = max(self.dim_uplimits)

        # read all algorithm-specific arguments
        self.__readAlgoArgs()

        # complain if both the search time limit and the total number of search runs are undefined
        if self.time_limit <= 0 and self.total_runs <= 0:
            err(('orio.main.tuner.search.simplex.simplex:  %s search requires either (both) the search time limit or (and) the ' +
                    'total number of search runs to be defined') % self.__class__.__name__)

    #-----------------------------------------------------
    # Method required by the search interface
    def searchBestCoord(self, startCoord=None):
        '''
        Search for the coordinate that yields the best performance parameters.
        
        @param startCoord: Starting coordinate (optional)
        
        '''
        # TODO: implement startCoord support
        
        if len(self.x0) != self.total_dims:
            err('orio.main.tuner.search.simplex: initial coordiniate x0 has to match the total dimensions')

        info('\n----- begin simplex search -----')

        # check for parallel search
        if self.use_parallel_search:
            err('orio.main.tuner.search.simplex: simplex search does not support parallel search')

        # check if the size of the search space is valid for this search
        self.__checkSearchSpace()

        # initialize a storage to remember all initial simplexes that have been explored
        simplex_records = {}

        # record the global best coordinate and its performance cost
        best_global_coord = None
        best_global_perf_cost = self.MAXFLOAT
        
        # record the number of runs
        runs = 0
        
        # start the timer
        start_time = time.time()
        
        simplex = None

        # execute the Nelder-Mead Simplex method
        while True:
            
            # list of the last several moves (used for termination criteria)
            last_simplex_moves = []
            
            
            # initialize a simplex in the search space
            if simplex == None:
                simplex = self.__initSimplex()
            else:
                simplex = self.__initRandomSimplex(simplex_records)

            info('\n(run %s) initial simplex: %s' % (runs+1, simplex))

            # get the performance cost of each coordinate in the simplex
            perf_costs = map(self.getPerfCost, simplex)
            perf_costs = map(lambda x: x[0] if len(x)==1 else sum(x[1:])/(len(x)-1), perf_costs)
            
            

            while True:

                # sort the simplex coordinates in an increasing order of performance costs
                sorted_simplex_cost = zip(simplex, perf_costs)
                sorted_simplex_cost.sort(lambda x,y: cmp(x[1],y[1]))
 
                # unbox the coordinate-cost tuples
                simplex, perf_costs = zip(*sorted_simplex_cost)
                simplex = list(simplex)
                perf_costs = list(perf_costs)
                
                
                # record time elapsed vs best perf cost found so far in a format that could be read in by matlab/octave
                #progress = 'init' if best_global_coord == None else 'continue'
                #if best_global_coord == None:
                    #best_global_coord = 'notNone'
                #result = perf_costs[0] if perf_costs[0] < best_global_perf_cost else best_global_perf_cost
                #best_coord_thus_far = simplex[0] if perf_costs[0] < best_global_perf_cost else best_global_coord
                #IOtime = Globals().stats.record(time.time()-start_time, result, best_coord_thus_far, progress)
                # don't include time on recording data in the tuning time
                #start_time += IOtime
                
                
                # remove bogus values (0 time)
                indicestoremove = []
                for i in range(0,len(perf_costs)):
                    if perf_costs[i] > 0.0: continue
                    else: indicestoremove.append(i)

                for i in indicestoremove:
                    del perf_costs[i]
                    del simplex[i]
                
                info('-> simplex: %s' % simplex)

                # check if the time is up
                if self.time_limit > 0 and (time.time()-start_time) > self.time_limit:
                    info('simplex: time is up')
                    break
                
                # termination criteria: a loop is present
                if str(simplex) in last_simplex_moves:
                    info('-> converged with simplex: %s' % simplex)
                    break

                # record the last several simplex moves (used for the termination criteria)
                last_simplex_moves.append(str(simplex))
                while len(last_simplex_moves) > 10:
                    last_simplex_moves.pop(0)
                
                # best coordinate
                best_coord = simplex[0]
                best_perf_cost = perf_costs[0]

                # worst coordinate
                worst_coord = simplex[len(simplex)-1]
                worst_perf_cost = perf_costs[len(perf_costs)-1]

                # 2nd worst coordinate
                second_worst_coord = simplex[len(simplex)-2]
                second_worst_perf_cost = perf_costs[len(perf_costs)-2]

                # calculate centroid
                centroid = self.__getCentroid(simplex[:len(simplex)-1])

                # reflection
                refl_coords = self.__getReflection(worst_coord, centroid)
                refl_perf_costs = map(self.getPerfCost, refl_coords)
                refl_perf_costs = map(lambda x: x[0] if len(x)==1 else sum(x[1:])/(len(x)-1), refl_perf_costs)
                
                refl_perf_cost = min(refl_perf_costs)
                ipos = refl_perf_costs.index(refl_perf_cost)
                refl_coord = refl_coords[ipos]

                # the replacement of the worst coordinate
                next_coord = None
                next_perf_cost = None
            
                # if cost(best) <= cost(reflection) < cost(2nd_worst)
                if best_perf_cost <= refl_perf_cost < second_worst_perf_cost:
                    next_coord = refl_coord
                    next_perf_cost = refl_perf_cost
                    info('--> reflection to %s' % next_coord )

                # if cost(reflection) < cost(best)
                elif refl_perf_cost < best_perf_cost:

                    # expansion
                    exp_coords = self.__getExpansion(refl_coord, centroid)
                    exp_perf_costs = map(self.getPerfCost, exp_coords)
                    exp_perf_costs = map(lambda x: x[0] if len(x)==1 else sum(x[1:])/(len(x)-1), exp_perf_costs)
                    
                    exp_perf_cost = min(exp_perf_costs)
                    ipos = exp_perf_costs.index(exp_perf_cost)
                    exp_coord = exp_coords[ipos]

                    # if cost(expansion) < cost(reflection)
                    if exp_perf_cost < refl_perf_cost:
                        next_coord = exp_coord
                        next_perf_cost = exp_perf_cost
                        info('--> expansion to %s' % next_coord )
                    else:
                        next_coord = refl_coord
                        next_perf_cost = refl_perf_cost
                        info('--> reflection to %s' % next_coord )
                        
                # if cost(reflection) < cost(worst)
                elif refl_perf_cost < worst_perf_cost:

                    # outer contraction
                    cont_coords = self.__getContraction(refl_coord, centroid)
                    cont_perf_costs = map(self.getPerfCost, cont_coords)
                    cont_perf_costs = map(lambda x: x[0] if len(x)==1 else sum(x[1:])/(len(x)-1), cont_perf_costs)
                    
                    cont_perf_cost = min(cont_perf_costs)
                    ipos = cont_perf_costs.index(cont_perf_cost)
                    cont_coord = cont_coords[ipos]
                    
                    # if cost(contraction) < cost(reflection)
                    if cont_perf_cost < refl_perf_cost:
                        next_coord = cont_coord
                        next_perf_cost = cont_perf_cost
                        info('--> outer contraction to %s' % next_coord )

                # if cost(reflection) >= cost(worst)
                else:
                
                    # inner contraction
                    cont_coords = self.__getContraction(worst_coord, centroid)
                    cont_perf_costs = map(self.getPerfCost, cont_coords)
                    cont_perf_costs = map(lambda x: x[0] if len(x)==1 else sum(x[1:])/(len(x)-1), cont_perf_costs)
                    
                    cont_perf_cost = min(cont_perf_costs)
                    ipos = cont_perf_costs.index(cont_perf_cost)
                    cont_coord = cont_coords[ipos]

                    # if cost(contraction) < cost(worst)
                    if cont_perf_cost < worst_perf_cost:
                        next_coord = cont_coord
                        next_perf_cost = cont_perf_cost
                        info('--> inner contraction to %s' % next_coord )

                # if shrinkage is needed
                if next_coord == None and next_perf_cost == None:

                    # shrinkage
                    simplex = self.__getShrinkage(best_coord, simplex)
                    perf_costs = map(self.getPerfCost, simplex)
                    perf_costs = map(lambda x: x[0] if len(x)==1 else sum(x[1:])/(len(x)-1), perf_costs)
                    
                    info('--> shrinkage on %s' % best_coord )
                    
                # replace the worst coordinate with the better coordinate
                else:
                    simplex.pop()
                    perf_costs.pop()
                    simplex.append(next_coord)
                    perf_costs.append(next_perf_cost)
                
            # get the best simplex coordinate and its performance cost
            best_simplex_coord = simplex[0]
            best_simplex_perf_cost = perf_costs[0]
            old_best_simplex_perf_cost = best_simplex_perf_cost

            info('-> best simplex coordinate: %s, cost: %e' %
                 (best_simplex_coord, best_simplex_perf_cost))
            
            # check if the time is not up yet
            if self.time_limit <= 0 or (time.time()-start_time) <= self.time_limit:

                # perform a local search on the best simplex coordinate
                (best_simplex_coord,
                 best_simplex_perf_cost) = self.searchBestNeighbor(best_simplex_coord,
                                                                   self.local_distance)
                 
                 
                best_simplex_perf_cost = best_simplex_perf_cost[0]
                
                # if the neighboring coordinate has a better performance cost
                if best_simplex_perf_cost < old_best_simplex_perf_cost:
                    info('---> better neighbor found: %s, cost: %e' %
                         (best_simplex_coord, best_simplex_perf_cost))
                else:
                    best_simplex_coord = simplex[0]
                    best_simplex_perf_cost = old_best_simplex_perf_cost

            # compare to the global best coordinate and its performance cost
            if best_simplex_perf_cost < best_global_perf_cost:
                best_global_coord = best_simplex_coord
                best_global_perf_cost = best_simplex_perf_cost
                info('>>>> best coordinate found: %s, cost: %e' %
                     (best_global_coord, best_global_perf_cost))

            # increment the number of runs
            runs += 1
            
            # check if the time is up
            if self.time_limit > 0 and (time.time()-start_time) > self.time_limit:
                info('simplex: time is up')
                break
            
            # check if the maximum limit of runs is reached
            if self.total_runs > 0 and runs >= self.total_runs:
                info('simplex: total runs reached')
                break
            
        # compute the total search time
        search_time = time.time() - start_time
                                                                     
        info('----- end simplex search -----')
        
        # record time elapsed vs best perf cost found so far in a format that could be read in by matlab/octave
        #Globals().stats.record(time.time()-start_time, best_global_perf_cost, best_global_coord, 'done')
 
        # return the best coordinate
        return best_global_coord, best_global_perf_cost, search_time, runs

    # Private methods
    #-----------------------------------------------------

    def __readAlgoArgs(self):
        '''To read all algorithm-specific arguments'''

        # check for algorithm-specific arguments
        for vname, rhs in self.search_opts.iteritems():
            
            # local search distance
            if vname == self.__LOCAL_DIST:
                if not isinstance(rhs, int) or rhs < 0:
                    err('orio.main.tuner.search.simplex.simplex: %s argument "%s" must be a positive integer or zero'
                           % (self.__class__.__name__, vname))
                    
                self.local_distance = rhs

            # reflection coefficient
            elif vname == self.__REFL_COEF:
                if isinstance(rhs, int) or isinstance(rhs, float):
                    rhs = [rhs]
                if isinstance(rhs, list):
                    for n in rhs:
                        if (not isinstance(n, int) and not isinstance(n, float)) or n <= 0:
                            err('man.tuner.search.simplex.simplex: %s argument "%s" must be number(s) greater than zero'
                                   % (self.__class__.__name__, vname))
                            
                    self.refl_coefs = rhs
                else:
                    err('man.tuner.search.simplex.simplex: %s argument "%s" must be number(s) greater than zero'
                           % (self.__class__.__name__, vname))
                    
                        
            # expansion coefficient
            elif vname == self.__EXP_COEF:
                if isinstance(rhs, int) or isinstance(rhs, float):
                    rhs = [rhs]
                if isinstance(rhs, list):
                    for n in rhs:
                        if (not isinstance(n, int) and not isinstance(n, float)) or n <= 1:
                            err('man.tuner.search.simplex.simplex: %s argument "%s" must be number(s) greater than one'
                                   % (self.__class__.__name__, vname))
                            
                    self.exp_coefs = rhs
                else:
                    err('man.tuner.search.simplex.simplex: %s argument "%s" must be number(s) greater than one'
                           % (self.__class__.__name__, vname))
                    
            
            # contraction coefficient
            elif vname == self.__CONT_COEF:
                if isinstance(rhs, int) or isinstance(rhs, float):
                    rhs = [rhs]
                if isinstance(rhs, list):
                    for n in rhs:
                        if (not isinstance(n, int) and not isinstance(n, float)) or n <= 0 or n >= 1:
                            err('man.tuner.search.simplex.simplex: %s argument "%s" must be number(s) between zero and one'
                                   % (self.__class__.__name__, vname))
                            
                    self.cont_coefs = rhs
                else:
                    err('man.tuner.search.simplex.simplex: %s argument "%s" must be number(s) between zero and one'
                           % (self.__class__.__name__, vname))
                    
            
            # shrinkage coefficient
            elif vname == self.__SHRI_COEF:
                if (not isinstance(rhs, int) and not isinstance(rhs, float)) or rhs <= 0 or rhs >= 1:
                    err('man.tuner.search.simplex.simplex: %s argument "%s" must be a single number between zero and one'
                           % (self.__class__.__name__, vname))
                    
                self.shri_coef = rhs
                
            # x0
            elif vname == self.__X0:
                if isinstance(rhs, list):
                    for n in rhs:
                        if not isinstance(n, int) or rhs < 0:
                            err('%s argument "%s" must be integers greater than or equal to 0'
                                   % (self.__class__.__name__, vname))
                else:
                    err('%s argument "%s" must be integers greater than or equal to 0'
                                   % (self.__class__.__name__, vname))
                self.x0 = rhs  
                
            # unrecognized algorithm-specific argument
            else:
                err('man.tuner.search.simplex.simplex: unrecognized %s algorithm-specific argument: "%s"' %
                       (self.__class__.__name__, vname))
                
        
    #-----------------------------------------------------

    def __checkSearchSpace(self):
        '''Check the size of the search space if it is valid for this search method'''

        # is the search space size too small?
        # Nelder-Mead requires to initialize a simplex that has N+1 vertices, where N is the
        # number of dimensions
        if self.space_size < self.__simplex_size:
            err(('orio.main.tuner.search.simplex.simplex:  the search space is too small for %s algorithm. ' +
                    'please use the exhaustive search.') % self.__class__.__name__)
            
    #-----------------------------------------------------
    
    
    def __dupCoord(self, simplex):
        '''check whether or not simplex has two coords that are identical'''
        simplex = map(lambda x: tuple(x), simplex)
        result = len(simplex) != len(set(simplex))
        if result:
            info('simplex with dup coords: %s' % (simplex))
        return result

    #-----------------------------------------------------
    
    
    def __initSimplex(self):
        '''initialize a right-angled simplex in the search space'''

        
        coord = list(self.x0)
        for i in range(0, self.total_dims):
            iuplimit = self.dim_uplimits[i]
            #if coord[i] >= iuplimit:
             #   coord[i] = iuplimit-1
            #elif coord[i] < 0:
             #   coord[i] = 0
            if coord[i] >= iuplimit or coord[i] < 0:
                err('msimplex: initial point x0 out of bound!')
                
        simplex = [coord]
        
        
        
        for i in range(0, self.total_dims):
            coord = list(self.x0)
            
            axis = coord[i]
            iuplimit = self.dim_uplimits[i]
            pos = iuplimit - axis - 1
            neg = axis
            
            prefer = self.sim_size-1
            
            if prefer <= pos:
                coord[i] += prefer
            elif prefer <= neg:
                coord[i] -= prefer
            elif pos >= neg:
                coord[i] += pos
            else:
                coord[i] -= neg
            
            
            
            #coord[i] += self.sim_size-1
            #iuplimit = self.dim_uplimits[i]
            #if coord[i] >= iuplimit:
            #    coord[i] = iuplimit-1
            
            simplex.append(coord)
            
        if self.__dupCoord(simplex):
            err('msimplex: simplex created has duplicate!!')
        
        return simplex
    #-----------------------------------------------------

    def __initRandomSimplex(self, simplex_records):
        '''Randomly initialize a simplex in the search space'''

        # remove some simplex records, if necessary
        max_num_records = 100000
        if len(simplex_records) > max_num_records:
            for i in range(i, int(max_num_records*0.05)):
                simplex_records.popitem()

        # randomly create a new simplex that has never been used before
        while True:

            # randomly pick (N+1) vertices to form a simplex, where N is the number of dimensions
            simplex = []
            while True:
                coord = self.getRandomCoord()
                if coord not in simplex:
                    simplex.append(coord)
                    if len(simplex) == self.__simplex_size:
                        break

            # check if the new simplex has never been used before
            simplex.sort()
            if str(simplex) not in simplex_records:
                simplex_records[str(simplex)] = None
                return simplex
            
    #-----------------------------------------------------

    def __getCentroid(self, coords):
        '''Return a centroid coordinate'''
        total_coords = len(coords)
        centroid = coords[0]
        for c in coords[1:]:
            centroid = self.addCoords(centroid, c)
        centroid = self.mulCoords((1.0/total_coords), centroid)
        return centroid

    def __getReflection(self, coord, centroid):
        '''Return a reflection coordinate'''
        sub_coord = self.subCoords(centroid, coord)
        return map(lambda x: self.addCoords(centroid, self.mulCoords(x, sub_coord)),
                   self.refl_coefs)
    
    def __getExpansion(self, coord, centroid):
        '''Return an expansion coordinate'''
        sub_coord = self.subCoords(coord, centroid)
        return map(lambda x: self.addCoords(centroid, self.mulCoords(x, sub_coord)),
                   self.exp_coefs)
    
    def __getContraction(self, coord, centroid):
        '''Return a contraction coordinate'''
        sub_coord = self.subCoords(coord, centroid)
        return map(lambda x: self.addCoords(centroid, self.mulCoords(x, sub_coord)),
                   self.cont_coefs)

    def __getShrinkage(self, coord, rest_coords):
        '''Return a shrinkage simplex'''
        return map(lambda x: self.addCoords(coord, self.mulCoords(self.shri_coef,
                                                                  self.subCoords(x, coord))),
                   rest_coords)
    
