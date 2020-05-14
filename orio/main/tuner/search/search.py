#
# The search engine used for search space exploration
#
import sys, math, time
from orio.main.util.globals import *

import orio.main.tuner.search.objective as objective

class Search:
    '''The search engine used to explore the search space '''

    MAXFLOAT = float('inf')

    #----------------------------------------------------------

    def __init__(self, params):
        '''To instantiate a search engine'''

        self.pb = objective.Function(params)

        #print params
        #print 'done'

        self.params=params
        debug('[Search] performance parameters: %s\n' % str(params), self)

        # the class variables that are essential to know when developing a new search engine subclass
        if 'search_time_limit' in params.keys(): self.time_limit = params['search_time_limit']
        else: self.time_limit = -1
        if 'search_total_runs' in params.keys(): self.total_runs = params['search_total_runs']
        else: self.total_runs = -1
        if 'search_resume' in params.keys(): self.resume = params['search_resume']
        else: self.resume = False
        if 'search_opts' in params.keys(): self.search_opts = params['search_opts']
        else: self.search_opts = {}



        if 'use_parallel_search' in params.keys(): self.use_parallel_search = params['use_parallel_search']
        else: self.use_parallel_search = False

        # the class variables that may be ignored when developing a new search engine subclass
        self.input_params = params.get('input_params')

        self.timing_code = ''

        self.verbose = Globals().verbose
    #----------------------------------------------------------

    def searchBestCoord(self):
        '''
        Explore the search space and return the coordinate that yields the best performance
        (i.e. minimum performance cost).

        This is the function that needs to be implemented in each new search engine subclass.
        '''
        raise NotImplementedError('%s: unimplemented abstract function "searchBestCoord"' %
                                  self.__class__.__name__)


    #----------------------------------------------------------

    def search(self, startCoord=None):
        '''Initiate the search process and return the best performance parameters'''

        # if the search space is empty
        if self.pb.total_dims == 0:
            return {}

        if self.resume:
            startCoord = self.search_opts.get('start_coord')
            if not isinstance(startCoord,list):
                err('%s argument "%s" must be a list of coordinate indices' % (self.__class__.__name__,'start_coord'))
            if not startCoord:
                startCoord = self.__findLastCoord()

        # find the coordinate resulting in the best performance
        best_coord,best_perf,search_time,runs = self.searchBestCoord(startCoord)
        corr_transfer = self.MAXFLOAT
        if isinstance(best_perf,tuple): #unpack optionally
            corr_transfer = best_perf[1]
            best_perf     = best_perf[0]

        # if no best coordinate can be found
        if best_coord == None:
            err ('the search cannot find a valid set of performance parameters. ' +
                 'the search time limit might be too short, or the performance parameter ' +
                 'constraints might prune out the entire search space.')
        else:
            info('----- begin summary -----')
            info(' best coordinate: %s=%s, cost=%e, transfer_time=%e, inputs=%s, search_time=%.2f, runs=%d' % \
                 (best_coord, self.pb.coordToPerfParams(best_coord), best_perf, corr_transfer, str(self.input_params), search_time, runs))
            info('----- end summary -----')


        if not Globals().extern:    
            # get the performance cost of the best parameters
            best_perf_cost = self.pb.getPerfCost(best_coord)
            # convert the coordinate to the corresponding performance parameters
            best_perf_params = self.pb.coordToPerfParams(best_coord)
        else:
            best_perf_cost=0
            best_perf_params=Globals().config


        # return the best performance parameters
        return (best_perf_params, best_perf_cost)

    #----------------------------------------------------------

    def factorial(self, n):
        '''Return the factorial value of the given number'''
        return reduce(lambda x,y: x*y, range(1, n+1), 1)

    def roundInt(self, i):
        '''Proper rounding for integer'''
        return int(round(i))

    def getRandomInt(self, lbound, ubound):
        '''To generate a random integer N such that lbound <= N <= ubound'''
        from random import randint

        if lbound > ubound:
            err('orio.main.tuner.search.search internal error: the lower bound of genRandomInt must not be ' +
                   'greater than the upper bound')
        return randint(lbound, ubound)

    def getRandomReal(self, lbound, ubound):
        '''To generate a random real number N such that lbound <= N < ubound'''
        from random import uniform

        if lbound > ubound:
            err('orio.main.tuner.search.search internal error: the lower bound of genRandomReal must not be ' +
                   'greater than the upper bound')
        return uniform(lbound, ubound)

    #----------------------------------------------------------

    def subCoords(self, coord1, coord2):
        '''coord1 - coord2'''
        return map(lambda x,y: x-y, coord1, coord2)

    def addCoords(self, coord1, coord2):
        '''coord1 + coord2'''
        return map(lambda x,y: x+y, coord1, coord2)

    def mulCoords(self, coef, coord):
        '''coef * coord'''
        return map(lambda x: self.roundInt(1.0*coef*x), coord)

    #----------------------------------------------------------

    def getCoordDistance(self, coord1, coord2):
        '''Return the distance between the given two coordinates'''

        d_sqr = 0
        for i in range(0, self.pb.total_dims):
            d_sqr += (coord2[i] - coord1[i])**2
        d = math.sqrt(d_sqr)
        return d

    #----------------------------------------------------------

    def getRandomCoord(self):
        '''Randomly pick a coordinate within the search space'''

        random_coord = []
        for i in range(0, self.pb.total_dims):
            iuplimit = self.pb.dim_uplimits[i]
            ipoint = self.getRandomInt(0, iuplimit-1)
            random_coord.append(ipoint)
        return random_coord


    def getInitCoord(self):
        '''Randomly pick a coordinate within the search space'''

        random_coord = []
        for i in range(0, self.pb.total_dims):
            #iuplimit = self.pb.dim_uplimits[i]
            #ipoint = self.getRandomInt(0, iuplimit-1)
            random_coord.append(0)
        return random_coord



    #----------------------------------------------------------

    def getNeighbors(self, coord, distance):
        '''Return all the neighboring coordinates (within the specified distance)'''

        # get all valid distances
        distances = [0] + range(1,distance+1,1) + range(-1,-distance-1,-1)

        # get all neighboring coordinates within the specified distance
        neigh_coords = [[]]
        for i in range(0, self.pb.total_dims):
            iuplimit = self.pb.dim_uplimits[i]
            cur_points = [coord[i]+d for d in distances]
            cur_points = filter(lambda x: 0 <= x < iuplimit, cur_points)
            n_neigh_coords = []
            for ncoord in neigh_coords:
                n_neigh_coords.extend([ncoord[:]+[p] for p in cur_points])
            neigh_coords = n_neigh_coords

        # remove the current coordinate from the neighboring coordinates list
        neigh_coords.remove(coord)

        # return all valid neighboring coordinates
        return neigh_coords

    #----------------------------------------------------------

    def searchBestNeighbor(self, coord, distance):
        '''
        Perform a local search by starting from the given coordinate then examining
        the performance costs of the neighboring coordinates (within the specified distance),
        then we perform this local search recursively once the neighbor with the best performance
        cost is found.
        '''

        # get all neighboring coordinates within the specified distance
        neigh_coords = self.getNeighbors(coord, distance)

        # record the best neighboring coordinate and its performance cost so far
        best_coord = coord
        best_perf_cost = self.pb.getPerfCost(coord)

        # examine all neighboring coordinates
        for n in neigh_coords:
            perf_cost = self.pb.getPerfCost(n)
            if perf_cost < best_perf_cost:
                best_coord = n
                best_perf_cost = perf_cost

        # recursively apply this local search, if new best neighboring coordinate is found
        if best_coord != coord:
            return self.searchBestNeighbor(best_coord, distance)

        # return the best neighboring coordinate and its performance cost
        return (best_coord, best_perf_cost)

    def __findLastCoord(self):
        from stat import S_ISREG, ST_MTIME, ST_MODE
        coord = None
        return coord
