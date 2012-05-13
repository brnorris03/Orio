#
# Implementation of the random search algorithm
#

import sys, time
import math
import random
import orio.main.tuner.search.search
from orio.main.util.globals import *

#-----------------------------------------------------

class Extern(orio.main.tuner.search.search.Search):
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
    
    def __init__(self, params):
        '''To instantiate a extern search engine'''

        random.seed(1)
        
        orio.main.tuner.search.search.Search.__init__(self, params)

        # set all algorithm-specific arguments to their default values
        self.local_distance = 0

        # read all algorithm-specific arguments
        self.__readAlgoArgs()
        
        # complain if both the search time limit and the total number of search runs are undefined
        if self.time_limit <= 0 and self.total_runs <= 0:
            err(('orio.main.tuner.search.randomsearch.randomsearch: %s search requires either (both) the search time limit or (and) the ' +
                    'total number of search runs to be defined') % self.__class__.__name__)
     
    # Method required by the search interface
    def searchBestCoord(self):
        '''
        To explore the search space and retun the coordinate that yields the best performance
        (i.e. minimum performance cost).
        '''

        info('\n----- begin extern eval  -----')

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
        sruns=0
        fruns=0
        # start the timer
        start_time = time.time()
        init = True
        param_config={}
        perf_costs={}
        # execute the randomized search method
        while True:
            config=Globals().config
            configfile=Globals().configfile
            params=self.params['axis_names']
            vals=self.params['axis_val_ranges']

            for i, p in enumerate(params):
                param_config[p]=vals[i][0]
            
            for token in config.split(','):
                param_val=token.split(':')
            if len(param_val)>1:
                param=param_val[0]
                val=param_val[1]
                if val=='False':
                    param_config[param]=False
                elif val=='True':
                    param_config[param]=True
                else:
                    param_config[param]=float(val) if '.' in val else int(val)
                        
            Globals().config=param_config

            if configfile != '':
                f = open(configfile, 'r') 
                for line in f:
                    param_val=''.join(line)
                    param_val=param_val.strip()
                    param_val=param_val.replace('\n','')
                    break
                f.close()

                param_val=param_val.split(' ')

                #print param_val

                for i, p in enumerate(params):
                    if i < len(param_val):
                        param_config[p]=vals[i][int(param_val[i])]

                #print param_config
                    
            
            
            n=len(param_config.keys())
            s = range(n)
            coord=[0*x for x in s]
            coord_key=str(coord)
            pf = param_config.copy()

            try:
                perf_costs=self.getPerfCostConfig(coord_key,param_config)
            except:
                perf_costs[coord_key]=[self.MAXFLOAT]
                info('FAILED')
                fruns +=1
                sys.stderr.write('FAILED\n')

            perf_cost,_=perf_costs.get(coord_key)


            try:
                floatNums = [float(x) for x in perf_cost]
                mean_perf_cost=sum(floatNums) / len(perf_cost)
            except:
                mean_perf_cost=perf_cost

                    
            transform_time=self.getTransformTime()
            compile_time=self.getCompileTime()
            
            best_coord = coord
            best_perf_cost = mean_perf_cost


            info('(run %s) sruns: %s, fruns: %s, coordinate: %s, perf_params: %s, transform_time: %s, compile_time: %s, cost: %s' % (runs, sruns, fruns, coord, pf, transform_time, compile_time,perf_cost))
            

            break
            

        search_time = time.time() - start_time
        
        info('----- end extern eval -----')

        count=0
        if (transform_time>0.0):
         count=1
        #count=1 
        #info('----- begin summary -----')
        #info(' best coordinate: %s, cost: %e' % (best_coord, best_perf_cost))
        #info(' total search time: %.2f seconds' % search_time)
        #info(' total completed runs: %s' % runs)
        #info(' total successful runs: %s' % sruns)
        #info(' total failed runs: %s' % fruns)
        #info('----- end summary -----')
        if configfile != '':
            res=('%1.8f' % (best_perf_cost))
            print res, count, perf_cost
            outstring=('%s %d %s\n'%(res, count, perf_cost))
            outfile=Globals().configfile+'.out'
            f = open(outfile, 'w')
            f.write(outstring)
            f.close
            #print res, count, perf_cost
            #sys.stderr.write(outfile+'\n')
            result=('%s %d\n'%(res, count))
            sys.stderr.write(result)
            #sys.stderr.write('%s\n'%pf)
        # return the best coordinate
        return best_coord, best_perf_cost, search_time, sruns
   
   # Private methods
   #--------------------------------------------------
    
    def __readAlgoArgs(self):
        '''To read all algorithm-specific arguments'''

        # check for algorithm-specific arguments
        for vname, rhs in self.search_opts.iteritems():

            # local search distance
            if vname == self.__LOCAL_DIST:
                if not isinstance(rhs, int) or rhs < 0:
                    err('orio.main.tuner.search.randomsearch: %s argument "%s" must be a positive integer or zero'
                           % (self.__class__.__name__, vname))
                self.local_distance = rhs

            # unrecognized algorithm-specific argument
            else:
                err('orio.main.tuner.search.randomsearch: unrecognized %s algorithm-specific argument: "%s"' %
                       (self.__class__.__name__, vname))

    #--------------------------------------------------

    def __getNextCoord(self, coord_records, neigh_coords,init):
        '''Get the next coordinate to be empirically tested'''

        #info('neighcoords: %s' % neigh_coords)
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
    
    #--------------------------------------------------
            
