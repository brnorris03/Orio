#
# Implementation of the modified Nelder-Mead Simplex algorithm
#
# The detailed algorithm is described in the following paper.
#   "Can Search Algorithms Save Large-scale Automatic Performance Tuning?"
#   by Prasanna Balaprakash, Stefan M. Wild, and Paul D. Hovland

import random, sys, time
import orio.main.tuner.search.search
from orio.main.util.globals import *

#-----------------------------------------------------

class MSimplex(orio.main.tuner.search.search.Search):
    '''
    The search engine that uses the Nelder-Mead Simplex algorithm, enhanced with a local search
    that finds the best neighboring coordinate.

    Below is a list of algorithm-specific arguments used to steer the search algorithm.
      search_distance            the distance number used in the local search to find the best
                                neighboring coordinate located within the specified distance
      reflection_coef           the reflection coefficient
      expansion_coef            the expansion coefficient
      contraction_coef          the contraction coefficient
      shrinkage_coef            the shrinkage coefficient
      size                      the size of the initial simplex
    '''

    # algorithm-specific argument names
    __X0 = 'x0'                           # default: all 0's (ie, [0,0,0])
    __SEARCH_DIST = 'search_distance'       # default: 1    #now, orio isn't parsing this at all. So it's always 1. See pparser.py
    __REFL_COEF = 'reflection_coef'       # default: [1.0]
    __EXP_COEF = 'expansion_coef'         # default: [2.0]
    __CONT_COEF = 'contraction_coef'      # default: [0.5]
    __SHRI_COEF = 'shrinkage_coef'        # default: 0.5
    __SIM_SIZE = 'size'                   # default: max dimension of the search space
    
    
    __CACHE_SIZE = 20                     # used for last_simplex_moves

    #-----------------------------------------------------

    def __init__(self, params):
        '''To instantiate a Modified Nelder-Mead simplex search engine'''
        
        orio.main.tuner.search.search.Search.__init__(self, params)

        if self.use_parallel_search:
            err('parallel search for msimplex is not supported yet.\n')
            
        # other private class variables
        self.__simplex_size = self.total_dims + 1

        # set all algorithm-specific arguments to their default values
        self.search_distance = 1
        self.sim_size = max(self.dim_uplimits)
        self.refl_coefs = [1.0]
        self.exp_coefs = [2.0]
        self.cont_coefs = [0.5]
        self.shri_coef = 0.5
        self.x0 = [0] * self.total_dims
        
        


        # read all algorithm-specific arguments
        self.__readAlgoArgs()
        
        # complain if more than 1 value is given for the reflection, expansion, or contraction coefficient.
        if len(self.refl_coefs)!=1 or len(self.exp_coefs)!=1 or len(self.cont_coefs)!=1:
            err('msimplex: reflection, expansion, or contraction coefficient can only be one value!!!!')
        

        # complain if both the search time limit and the total number of search runs are undefined
        if self.time_limit <= 0 and self.total_runs <= 0:
            err(('%s search requires either (both) the search time limit or (and) the ' +
                    'total number of search runs to be defined') % self.__class__.__name__)

    #-----------------------------------------------------
    # Method required by the search interface
    def searchBestCoord(self, startCoord=None):
        '''
        Search for the coordinate that yields the best performance parameters.
        
        
        '''
        # TODO: implement startCoord support

        info('\n----- begin msimplex search -----')

        # check for parallel search
        if self.use_parallel_search:
            err('orio.main.tuner.search.msimplex: msimplex search does not support parallel search')

        # check if the size of the search space is valid for this search
        self.__checkSearchSpace()
        
        
        if len(self.x0) != self.total_dims:
            err('orio.main.tuner.search.msimplex: initial coordiniate x0 has to match the total dimensions')


        # record the global best coordinate and its performance cost
        best_global_coord = None
        best_global_perf_cost = self.MAXFLOAT
        
        # record the number of runs
        self.runs = 0
        
        # start the timer
        self.start_time = time.time()

            
        # list of the last several moves (used for termination criteria)
        last_simplex_moves = []

        # randomly initialize a simplex in the search space
        simplex = self.__initSimplex()

        # get the performance cost of each coordinate in the simplex
        perf_costs = map(self.getPerfCost, simplex)
        
        # if repetition is more than 1, ignore the first measurement, and average the other ones
        perf_costs = map(lambda x: x[0] if len(x)==1 else sum(x[1:])/(len(x)-1), perf_costs)
        
        # flag to tell whether or not local min has reached
        self.localmin = False
        
        # a flag to break when search has run for long enough
        self.breakFlag = False

        while True:
            
            #record the previous best coordinate. We need to clear the visited neighbors cache if the best vertex has changed.
            old_best_global_coord = simplex[0]

            # sort the simplex coordinates in an increasing order of performance costs
            sorted_simplex_cost = zip(simplex, perf_costs)
            sorted_simplex_cost.sort(lambda x,y: cmp(x[1],y[1]))
 
            # unbox the coordinate-cost tuples
            simplex, perf_costs = zip(*sorted_simplex_cost)
            simplex = list(simplex)
            perf_costs = list(perf_costs)
            
            
            # record time elapsed vs best perf cost found so far in a format that could be read in by matlab/octave
            #progress = 'init' if best_global_coord == None else 'continue'
            #IOtime = Globals().stats.record(time.time()-self.start_time, perf_costs[0], simplex[0], progress)
            # don't include time on recording data in the tuning time
            #self.start_time += IOtime
                
            info('-> (run %s) simplex: %s' % (self.runs+1, simplex))

            # check if the time is up
            if self.time_limit > 0 and (time.time()-self.start_time) > self.time_limit:
                info('msimplex: time is up')
                break
            
            # stop when the number of runs has been reached
            if self.total_runs > 0 and self.runs >= self.total_runs:
                info('msimplex: total runs limit reached')
                break   
            
            # stop when local minimum is obtained
            if self.localmin:
                info('msimplex: local minimum reached')
                break
            
                
            # termination criteria: a loop is present
            #if str(simplex) in last_simplex_moves:
                #info('-> encountered a loop: %s' % simplex)
                #break

            # record the last several simplex moves (used for the termination criteria)
            #last_simplex_moves.append(str(simplex))
            #while len(last_simplex_moves) > self.__CACHE_SIZE:
                #last_simplex_moves.pop(0)
                
            
            
            # best coordinate
            best_coord = simplex[0]
            best_perf_cost = perf_costs[0]
            

            
            
            # replace simplex's best vertex with a better unvisited neighbor if simplex has been reduced to a point
            if simplex[1:] == simplex[:-1]:
                while not self.localmin:
                    neighbor = self.__chooseRandomNeighbor(best_coord, simplex, best_coord)

                    # break out the loop if the search method has run long enough                     
                    if self.breakFlag:
                        break
                     
                    cost = self.getPerfCost(neighbor)
                    cost = cost[0] if len(cost) == 1 else sum(cost[1:])/(len(cost)-1)
                    if cost < best_perf_cost:
                        simplex[0] = neighbor
                        best_coord = neighbor
                        perf_costs[0] = cost
                        best_perf_cost = cost
                        info('msimplex: replaces best vertex with neighbor %s after arriving at a 1 point simplex' % best_coord)
                        break
                
                # break out the loop if the search method has run long enough       
                if self.breakFlag:
                    break
                
                if self.localmin:
                    info('msimplex: local minimum reached at a 1 point simplex')
                    break
                    
            
            # re-init cache of visited neighbors (called used_neighbors) if the old best vertex differs from the new best vertex or this is the first iteration
            if best_global_coord == None or best_coord != old_best_global_coord:
                best_global_coord = 'notNone'
                #self.__initAvailableNeighbors(best_coord, simplex)
                self.used_neighbors = []

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
            refl_coords = map(self.__forceInBound, refl_coords)
            refl_perf_costs = map(self.getPerfCost, refl_coords)
            refl_perf_costs = map(lambda x: x[0] if len(x)==1 else sum(x[1:])/(len(x)-1), refl_perf_costs)              
                
            refl_perf_cost = min(refl_perf_costs)
            ipos = refl_perf_costs.index(refl_perf_cost)
            refl_coord = refl_coords[ipos]
            info('msimplex: reflection coord: %s' % (refl_coord))

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
                exp_coords = map(self.__forceInBound, exp_coords)
                exp_perf_costs = map(self.getPerfCost, exp_coords)
                exp_perf_costs = map(lambda x: x[0] if len(x)==1 else sum(x[1:])/(len(x)-1), exp_perf_costs)      
                    
                exp_perf_cost = min(exp_perf_costs)
                ipos = exp_perf_costs.index(exp_perf_cost)
                exp_coord = exp_coords[ipos]
                info('msimplex: expansion coord: %s' % (exp_coord))

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
                cont_coords = map(self.__forceInBound, cont_coords)
                cont_perf_costs = map(self.getPerfCost, cont_coords)
                cont_perf_costs = map(lambda x: x[0] if len(x)==1 else sum(x[1:])/(len(x)-1), cont_perf_costs)
                    
                cont_perf_cost = min(cont_perf_costs)
                ipos = cont_perf_costs.index(cont_perf_cost)
                cont_coord = cont_coords[ipos]
                info('msimplex: outer contraction coord: %s' % (cont_coord))
                
                if cont_coord == refl_coord:
                    #cont_coord = self.__chooseNeighbor(simplex, cont_coord)
                    cont_coord = self.__chooseRandomNeighbor(best_coord, simplex, cont_coord)
                    
                    # break out the loop if the search method has run long enough 
                    if self.breakFlag:
                        break
                    
                    temp = self.getPerfCost(cont_coord)
                    cont_perf_cost = temp[0] if len(temp)==1 else sum(temp[1:])/(len(temp)-1)
                    
                # if cost(contraction) < cost(reflection)
                if cont_perf_cost < refl_perf_cost:
                    next_coord = cont_coord
                    next_perf_cost = cont_perf_cost
                    info('--> outer contraction to %s' % next_coord )

            # if cost(reflection) >= cost(worst)
            else:
                
                # inner contraction
                
                cont_coords = self.__getContraction(worst_coord, centroid)
                cont_coords = map(self.__forceInBound, cont_coords)
                cont_perf_costs = map(self.getPerfCost, cont_coords)
                cont_perf_costs = map(lambda x: x[0] if len(x)==1 else sum(x[1:])/(len(x)-1), cont_perf_costs)
                    
                cont_perf_cost = min(cont_perf_costs)
                ipos = cont_perf_costs.index(cont_perf_cost)
                cont_coord = cont_coords[ipos]
                info('msimplex: inner contraction coord: %s' % (cont_coord))
                
                if cont_coord == worst_coord:
                    #cont_coord = self.__chooseNeighbor(simplex, cont_coord)
                    cont_coord = self.__chooseRandomNeighbor(best_coord, simplex, cont_coord)
                    
                    # break out the loop if the search method has run long enough 
                    if self.breakFlag:
                        break
                    
                    temp = self.getPerfCost(cont_coord)
                    cont_perf_cost = temp[0] if len(temp)==1 else sum(temp[1:])/(len(temp)-1)

                # if cost(contraction) < cost(worst)
                if cont_perf_cost < worst_perf_cost:
                    next_coord = cont_coord
                    next_perf_cost = cont_perf_cost
                    info('--> inner contraction to %s' % next_coord )




            # if shrinkage is needed
            if next_coord == None and next_perf_cost == None:

                # shrinkage
                info('msimplex: starts shrinkage')
                ssimplex = self.__getShrinkage(best_coord, simplex)
                ssimplex = map(self.__forceInBound, ssimplex)
                #ssimplex[1:] = map((lambda x,y: x if x!=y else self.__chooseNeighbor(simplex, x)), ssimplex[1:], simplex[1:])
                ssimplex[1:] = map((lambda x,y: x if x!=y else self.__chooseRandomNeighbor(best_coord, simplex, x)), ssimplex[1:], simplex[1:])
                
                # break out the loop if the search method has run long enough 
                if self.breakFlag:
                    break
                
                simplex = ssimplex
                perf_costs = map(self.getPerfCost, simplex)
                perf_costs = map(lambda x: x[0] if len(x)==1 else sum(x[1:])/(len(x)-1), perf_costs)
                    
                info('--> shrinkage on %s' % best_coord )
                    
            # replace the worst coordinate with the better coordinate
            else:
                simplex.pop()
                perf_costs.pop()
                simplex.append(next_coord)
                perf_costs.append(next_perf_cost)
                #if self.__dupCoord(simplex):
                    #info('msimplex: duplicate coordinates in simplex after the above operation')
                    
            
            # increment the number of runs
            self.runs += 1
            
            
            
            
                
        # get the best simplex coordinate and its performance cost
        best_global_coord = simplex[0]
        best_global_perf_cost = perf_costs[0]
        
        # record time elapsed vs best perf cost found so far in a format that could be read in by matlab/octave
        #Globals().stats.record(time.time()-self.start_time, perf_costs[0], simplex[0], 'done')

        info('-> best simplex coordinate: %s, cost: %e' %
                (best_global_coord, best_global_perf_cost))
            
            
        # compute the total search time
        search_time = time.time() - self.start_time
                                                                     
        info('----- end msimplex search -----')
 
        # return the best coordinate
        return best_global_coord, best_global_perf_cost, search_time, self.runs

    # Private methods
    #-----------------------------------------------------

    def __readAlgoArgs(self):
        '''To read all algorithm-specific arguments'''

        # check for algorithm-specific arguments
        for vname, rhs in self.search_opts.iteritems():
            
            # local search distance
            if vname == self.__SEARCH_DIST:
                if not isinstance(rhs, int) or rhs <= 0:
                    err('%s argument "%s" must be a positive integer'
                           % (self.__class__.__name__, vname))
                    
                self.search_distance = rhs
            
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
                    
            # edge length
            elif vname == self.__SIM_SIZE:
                if not isinstance(rhs, int) or rhs <= 0:
                    err('%s argument "%s" must be a positive integer'
                           % (self.__class__.__name__, vname))
                self.sim_size = rhs
                
                
            # reflection coefficient
            elif vname == self.__REFL_COEF:
                if isinstance(rhs, int) or isinstance(rhs, float):
                    rhs = [rhs]
                if isinstance(rhs, list):
                    for n in rhs:
                        if (not isinstance(n, int) and not isinstance(n, float)) or n <= 0:
                            err('%s argument "%s" must be number(s) greater than zero'
                                   % (self.__class__.__name__, vname))
                            
                    self.refl_coefs = rhs
                else:
                    err('%s argument "%s" must be number(s) greater than zero'
                           % (self.__class__.__name__, vname))
                    
                        
            # expansion coefficient
            elif vname == self.__EXP_COEF:
                if isinstance(rhs, int) or isinstance(rhs, float):
                    rhs = [rhs]
                if isinstance(rhs, list):
                    for n in rhs:
                        if (not isinstance(n, int) and not isinstance(n, float)) or n <= 1:
                            err('%s argument "%s" must be number(s) greater than one'
                                   % (self.__class__.__name__, vname))
                            
                    self.exp_coefs = rhs
                else:
                    err('%s argument "%s" must be number(s) greater than one'
                           % (self.__class__.__name__, vname))
                    
            
            # contraction coefficient
            elif vname == self.__CONT_COEF:
                if isinstance(rhs, int) or isinstance(rhs, float):
                    rhs = [rhs]
                if isinstance(rhs, list):
                    for n in rhs:
                        if (not isinstance(n, int) and not isinstance(n, float)) or n <= 0 or n >= 1:
                            err('%s argument "%s" must be number(s) between zero and one'
                                   % (self.__class__.__name__, vname))
                            
                    self.cont_coefs = rhs
                else:
                    err('%s argument "%s" must be number(s) between zero and one'
                           % (self.__class__.__name__, vname))
                    
            
            # shrinkage coefficient
            elif vname == self.__SHRI_COEF:
                if (not isinstance(rhs, int) and not isinstance(rhs, float)) or rhs <= 0 or rhs >= 1:
                    err('%s argument "%s" must be a single number between zero and one'
                           % (self.__class__.__name__, vname))
                    
                self.shri_coef = rhs
                
            # unrecognized algorithm-specific argument
            else:
                err('unrecognized %s algorithm-specific argument: "%s"' %
                       (self.__class__.__name__, vname))
    
    def __intersectCoords(self, coords1, coords2):
        '''return a list which is the intersection of coords1 and coords2, both lists of coordinates. A coordinate is a list of numbers
        this method is not used but just kept here for future reference'''
        coords1 = map(lambda x: str(x), coords1)
        coords2 = map(lambda x: str(x), coords2)
        coords1 = set(coords1)
        coords2 = set(coords2)
        inters = coords1.intersection(coords2)
        inters = list(inters)
        inters = map(lambda x: eval(x), inters)
        return inters
    
    
    def __removeCommonCoords(self, coords1, coords2):
        '''return a list which is coords1 with elements removed that are present in coords2
        this method is not used but just kept here for future reference'''
        coords1 = map(lambda x: str(x), coords1)
        coords2 = map(lambda x: str(x), coords2)
        coords1 = set(coords1)
        coords2 = set(coords2)
        result = coords1 - coords2
        result = list(result)
        result = map(lambda x: eval(x), result)
        return result
    
    def __initAvailableNeighbors(self, bestVertex, simplex):
        '''return a list of available neighbors given a best vertex. Not used any more. See __chooseRandomNeighbor'''
        neighbors = self.getNeighbors(bestVertex, self.search_distance)
        visited_neighbors = self.__intersectCoords(neighbors, simplex)
        self.avail_neighbors = self.__removeCommonCoords(neighbors, visited_neighbors)
        #debug('neighbors avail: %s' % (self.avail_neighbors))
        info('neighbors avail: %s' % (self.avail_neighbors))
    
    #-----------------------------------------------------

    def __chooseNeighbor(self, simplex, coord):
        '''return a random neighbor of coord, from the list of available neighbors generated initally by __initAvailableNeighbors. Not used any more. See __chooseRandomNeighbor'''
        while len(self.avail_neighbors)>0:
            ipos = random.randrange(0, len(self.avail_neighbors))
            neighbor = self.avail_neighbors.pop(ipos)
            valid_neighbor = self.__forceInBound(neighbor)
            if not valid_neighbor in simplex:
                info('neighbor chosen: %s' % (valid_neighbor))
                #debug('neighbors avail: %s' % (self.avail_neighbors))
                info('neighbors avail: %s' % (self.avail_neighbors))
                return valid_neighbor
            
        self.localmin = True
        return coord
        
        
    def __chooseRandomNeighbor(self, bestVertex, simplex, coord):
        '''return a random neighbor of bestVertex. If no such neighbor, return coord'''
        
        while len(self.used_neighbors)-3**self.total_dims < 0:
            
            #info('msimplex: size of used neighbors: %s' % (len(self.used_neighbors)))
            
            if self.time_limit > 0 and (time.time()-self.start_time) > self.time_limit:
                info('msimplex: time is up while choosing a random neighbor')
                self.breakFlag = True
                return coord
            
            if self.total_runs > 0 and self.runs >= self.total_runs:
                info('msimplex: total runs limit reached while choosing a random neighbor')
                self.breakFlag = True
                return coord
            
            
            
            lb = self.search_distance
            lb = -lb
            ub = self.search_distance
            
            neighbor = map(lambda x: x+random.randrange(lb, ub+1), bestVertex)
            if neighbor in self.used_neighbors:
                continue
            
            bounded_neighbor = self.__forceInBound(neighbor)
            
            if neighbor != bounded_neighbor:
                self.used_neighbors.append(neighbor)
            
            neighbor = bounded_neighbor
            
            if neighbor in self.used_neighbors:
                continue
            
            self.used_neighbors.append(neighbor)
            
            if not neighbor in simplex:
                info('neighbor chosen: %s' % (neighbor))
                return neighbor
        
        self.localmin = True
        return coord
            
                
        
        

    def __checkSearchSpace(self):
        '''Check the size of the search space if it is valid for this search method'''

        # is the search space size too small?
        # Nelder-Mead requires to initialize a simplex that has N+1 vertices, where N is the
        # number of dimensions
        if self.space_size < self.__simplex_size:
            err(('the search space is too small for %s algorithm. ' +
                    'please use the exhaustive search.') % self.__class__.__name__)

    def __forceInBound(self, coord):
        '''rounds coord and constrain it within the bound of search space, and return the new coord'''
        coord = map(lambda x: int(round(x)), coord)
        coord = map(lambda x: x if x>=0 else 0, coord)
        coord = map(lambda x,y: x if x<y else y-1, coord, self.dim_uplimits)
        return coord

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

    def __getCentroid(self, coords):
        '''Return a centroid coordinate'''
        total_coords = len(coords)
        centroid = coords[0]
        for c in coords[1:]:
            centroid = self.addCoords(centroid, c)
        centroid = map(lambda x: (1.0/total_coords)*x, centroid)
        return centroid

    def __getReflection(self, coord, centroid):
        '''Return a reflection coordinate'''
        sub_coord = self.subCoords(centroid, coord)
        return map(lambda x: self.addCoords(centroid, map(lambda y: x*y, sub_coord)),
                   self.refl_coefs)
    
    def __getExpansion(self, coord, centroid):
        '''Return an expansion coordinate'''
        sub_coord = self.subCoords(coord, centroid)
        return map(lambda x: self.addCoords(centroid, map(lambda y: x*y, sub_coord)),
                   self.exp_coefs)
    
    def __getContraction(self, coord, centroid):
        '''Return a contraction coordinate'''
        sub_coord = self.subCoords(coord, centroid)
        return map(lambda x: self.addCoords(centroid, map(lambda y: x*y, sub_coord)),
                   self.cont_coefs)

    def __getShrinkage(self, coord, rest_coords):
        '''Return a shrinkage simplex'''
        
        return map(lambda x: self.addCoords(coord, map(lambda y: self.shri_coef*y, self.subCoords(x, coord))),
                   rest_coords)
    
