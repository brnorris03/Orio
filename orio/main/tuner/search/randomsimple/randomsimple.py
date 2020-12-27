# Very basic random search

import sys, time
import math
import random
import orio.main.tuner.search.search
from orio.main.util.globals import *


class Randomsimple(orio.main.tuner.search.search.Search):
    def __init__(self, params):

        random.seed(1)
        orio.main.tuner.search.search.Search.__init__(self, params)

        self.__readAlgoArgs()

        if self.time_limit <= 0 and self.total_runs <= 0:
            err((
                            'orio.main.tuner.search.randomsimple.randomsimple: %s search requires the search time limit (time_limit, seconds) and/or the ' +
                            'total number of search runs (total_runs) to be defined') % self.__class__.__name__)

    def searchBestCoord(self, startCoord=None):

        info('\n----- begin random search -----')

        info( "Total runs: %d" % self.total_runs )
        info( "Time limit: %d" % self.time_limit )
        
        bestperfcost = self.MAXFLOAT
        bestcoord = None
        runs = 0

        # start the timer
        start_time = time.time()
        init = True

        visited = []

        while (self.time_limit < 0 or (time.time() - start_time) < self.time_limit) and (
                self.total_runs < 0 or runs < self.total_runs):
            # get a random point
            coord = self.getRandomCoord()

            if coord == None:
                debug( "No points left in the parameter space", obj=self, level=3 )
                break
            if not self.checkValidity( coord ) or coord in visited:
                debug( "invalid point", obj=self, level=3 )
                continue
            try:
                debug( "coord: %s run %s" % (coord, runs ), obj=self, level=3 )
                perf_costs = self.getPerfCost( coord )
                if bestperfcost > sum( perf_costs ):
                    info( "Point %s gives a better perf: %s -- %s" % (coord, sum( perf_costs ), bestperfcost ) )
                    bestperfcost = sum( perf_costs )
                    bestcoord = coord
            except Exception as e:
                info('FAILED: %s %s' % (e.__class__.__name__, e))
            runs += 1

            if not self.use_z3:
                visited.append( coord )
            else:
                point = self.coordToPerfParams( coord )
                self.z3solver.addPoint( point )

        search_time = time.time() - start_time
        return bestcoord, bestperfcost, search_time, runs

    def checkValidity(self, coord):
        perf_params = self.coordToPerfParams(coord)
        try:
            is_valid = eval(self.constraint, perf_params, dict(self.input_params))
        except Exception as e:
            err('failed to evaluate the constraint expression: "%s"\n%s %s' % (
            self.constraint, e.__class__.__name__, e))
            return False
        return is_valid

    def __readAlgoArgs(self):
        for vname, rhs in self.search_opts.items():
            if vname == 'total_runs':
                self.total_runs = rhs
            else:
                err('orio.main.tuner.search.randomsimple: unrecognized %s algorithm-specific argument: "%s"' %
                    (self.__class__.__name__, vname))

