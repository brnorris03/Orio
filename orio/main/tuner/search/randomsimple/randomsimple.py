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
#        self.have_z3 = False

        if self.time_limit <= 0 and self.total_runs <= 0:
            err(('orio.main.tuner.search.randomsimple.randomsimple: %s search requires either (both) the search time limit or (and) the ' +
                    'total number of search runs to be defined') % self.__class__.__name__)

    def searchBestCoord(self, startCoord=None):
        
        info('\n----- begin random search -----')
        print "Total runs:", self.total_runs
        print "Time limit:", self.time_limit
        
        bestperfcost = self.MAXFLOAT
        bestcoord = None
        runs = 0

        # start the timer
        start_time = time.time()
        init = True

        visited = []
        
        while ( self.time_limit < 0 or ( time.time() - start_time ) < self.time_limit ) and ( self.total_runs < 0 or runs < self.total_runs ):
            # get a random point
            coord = self.getRandomCoord()
            
            if not self.have_z3 and not self.checkValidity( coord ) or coord in visited:
                print "invalid point"
                continue
            try:
                print "coord:", coord, "run", runs
                perf_costs = self.getPerfCost( coord )
                if bestperfcost > sum( perf_costs ):
                    info( "Point %s gives a better perf: %s -- %s" % (coord, sum( perf_costs ), bestperfcost ) )
                    bestperfcost = sum( perf_costs )
                    bestcoord = coord
            except Exception, e:
                info('FAILED: %s %s' % (e.__class__.__name__, e))
            runs += 1
            if not self.have_z3:
                visited.append( coord )
            else:
                point = self.coordToPerfParams( coord )
                self.addPoint( point )

        search_time = time.time() - start_time
        return bestcoord, bestperfcost, search_time, runs

    def checkValidity( self, coord ):
        perf_params = self.coordToPerfParams(coord)
        try:
            is_valid = eval(self.constraint, perf_params, dict(self.input_params))
        except Exception, e:
            err('failed to evaluate the constraint expression: "%s"\n%s %s' % (self.constraint,e.__class__.__name__, e))
            return False
        return is_valid
    
    def __readAlgoArgs(self):
        for vname, rhs in self.search_opts.iteritems():
            if vname == 'total_runs':
                self.total_runs = rhs
            else:
                err('orio.main.tuner.search.randomsimple: unrecognized %s algorithm-specific argument: "%s"' %
                    (self.__class__.__name__, vname))
