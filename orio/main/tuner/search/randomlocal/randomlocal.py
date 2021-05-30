#
# Implementation of the random search algorithm
#

import sys, time
import math
import random
import orio.main.tuner.search.search
from orio.main.util.globals import *


# -----------------------------------------------------

class Randomlocal(orio.main.tuner.search.search.Search):
    '''
    The search engine that uses a random search approach, enhanced with a local search that finds
    the best neighboring coordinate.

    Below is a list of algorithm-specific arguments used to steer the search algorithm.
      local_distance            the distance number used in the local search to find the best
                                neighboring coordinate located within the specified distance
    '''

    # algorithm-specific argument names
    __LOCAL_DIST = 'local_distance'  # default: 0

    # --------------------------------------------------

    def __init__(self, params):
        '''To instantiate a random search engine'''

        random.seed(1)

        orio.main.tuner.search.search.Search.__init__(self, params)

        # set all algorithm-specific arguments to their default values
        self.local_distance = 0

        # read all algorithm-specific arguments
        self.__readAlgoArgs()

        # complain if both the search time limit and the total number of search runs are undefined
        if self.time_limit <= 0 and self.total_runs <= 0:
            err(('orio.main.tuner.search.randomlocal.randomlocal: %s search requires the search time limit (time_limit, seconds) and/or the ' +
                'total number of search runs (total_runs) to be defined') % self.__class__.__name__)

    # Method required by the search interface
    def searchBestCoord(self, startCoord=None):
        '''
        To explore the search space and retun the coordinate that yields the best performance
        (i.e. minimum performance cost).
        '''
        # TODO: implement startCoord support

        info('\n----- begin random search -----')

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
        sruns = 0
        fruns = 0
        # start the timer
        start_time = time.time()
        init = True
        coord_key = ''
        # execute the randomized search method
        while True:

            # randomly pick a set of coordinates to be empirically tested
            coords = []
            while len(coords) < coord_count:
                coord = self.__getNextCoord(coord_records, neigh_coords, init)
                coord_key = str(coord)
                init = False
                if coord:
                    coords.append(coord)
                else:
                    break

            # check if all coordinates in the search space have been explored
            if len(coords) == 0:
                break

            # determine the performance cost of all chosen coordinates
            # perf_costs = self.getPerfCosts(coords)

            perf_costs = {}
            transform_time = 0.0
            compile_time = 0.0
            mean_perf_cost = self.MAXFLOAT
            # determine the performance cost of all chosen coordinates
            # perf_costs = self.getPerfCosts(coords)
            # sys.exit()
            try:
                perf_costs = self.getPerfCosts(coords)
            except Exception as e:
                perf_costs[str(coords[0])] = [self.MAXFLOAT]
                info('FAILED: %s %s' % (e.__class__.__name__, e))
                fruns += 1
            # compare to the best result
            pcost_items = sorted(list(perf_costs.items()))
            for i, (coord_str, pcost) in enumerate(pcost_items):
                if type(pcost) == tuple:
                    (perf_cost, _) = pcost  # ignore transfer costs -- GPUs only
                else:
                    perf_cost = pcost
                coord_val = eval(coord_str)
                # info('%s %s' % (coord_val,perf_cost))
                perf_params = self.coordToPerfParams(coord_val)
                try:
                    floatNums = [float(x) for x in perf_cost]
                    mean_perf_cost = sum(floatNums) / len(perf_cost)
                except:
                    pass

                transform_time = self.getTransformTime(coord_key)
                compile_time = self.getCompileTime(coord_key)
                # info('(run %s) coordinate: %s, perf_params: %s, transform_time: %s, compile_time: %s, cost: %s' % (runs+i+1, coord_val, perf_params, transform_time, compile_time,perf_cost))
                if mean_perf_cost < best_perf_cost and mean_perf_cost > 0.0:
                    best_coord = coord_val
                    best_perf_cost = mean_perf_cost
                    info('>>>> best coordinate found: %s, cost: %e' % (coord_val, mean_perf_cost))

            # if a better coordinate is found, explore the neighboring coordinates
            if False and old_perf_cost != best_perf_cost:
                neigh_coords.extend(self.getNeighbors(best_coord, self.local_distance))
                old_perf_cost = best_perf_cost

            # increment the number of runs
            runs += 1  # len(mean_perf_cost)

            if not math.isinf(mean_perf_cost):
                sruns += 1
                pcosts = '[]'
                if perf_cost and len(perf_cost) > 1:
                    pcosts = '[' + ', '.join(["%2.4e" % x for x in perf_cost]) + ']'
                msgstr1 = '(run %d) | %s | sruns: %d, fruns: %d, coordinate: %s, perf_params: %s, ' % \
                          (runs + i, str(datetime.datetime.now()), sruns, fruns, str(coord_val), str(perf_params))
                msgstr2 = 'transform_time: %2.4e, compile_time: %2.4e, cost: %s' % \
                          (transform_time, compile_time, pcosts)
                info(msgstr1 + msgstr2)

            # check if the time is up
            # info('%s' % self.time_limit)
            if self.time_limit > 0 and (time.time() - start_time) > self.time_limit:
                break

            # check if the maximum limit of runs is reached
            # if self.total_runs > 0 and runs >= self.total_runs:
            if self.total_runs > 0 and sruns >= self.total_runs:
                break

        # compute the total search time
        search_time = time.time() - start_time

        info('----- end random search -----')
        info('----- begin random search summary -----')
        info(' total completed runs: %s' % runs)
        info(' total successful runs: %s' % sruns)
        info(' total failed runs: %s' % fruns)
        info('----- end random search summary -----')

        # return the best coordinate
        return best_coord, best_perf_cost, search_time, sruns

    # Private methods
    # --------------------------------------------------

    def __readAlgoArgs(self):
        '''To read all algorithm-specific arguments'''

        # check for algorithm-specific arguments
        for vname, rhs in self.search_opts.items():

            # local search distance
            if vname == self.__LOCAL_DIST:
                if not isinstance(rhs, int) or rhs < 0:
                    err('orio.main.tuner.search.randomlocal: %s argument "%s" must be a positive integer or zero'
                        % (self.__class__.__name__, vname))
                self.local_distance = rhs

            # unrecognized algorithm-specific argument
            else:
                err('orio.main.tuner.search.randomlocal: unrecognized %s algorithm-specific argument: "%s"' %
                    (self.__class__.__name__, vname))

    # --------------------------------------------------

    def __getNextCoord(self, coord_records, neigh_coords, init):
        '''Get the next coordinate to be empirically tested'''

        # info('neighcoords: %s' % neigh_coords)
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
        while init:
            coord = self.getInitCoord()
            if str(coord) not in coord_records:
                coord_records[str(coord)] = None
                return coord

        # randomly pick a coordinate that has never been explored before
        while True:
            coord = self.getRandomCoord()
            if str(coord) not in coord_records:
                coord_records[str(coord)] = None
                return coord

    # --------------------------------------------------
