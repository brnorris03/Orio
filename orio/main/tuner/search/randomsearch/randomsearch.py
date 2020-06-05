#
# Implementation of the random search algorithm
#

import sys, time
import math
import random
import orio.main.tuner.search.search
from orio.main.util.globals import *
import copy
import json

#-----------------------------------------------------

class Randomsearch(orio.main.tuner.search.search.Search):
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
        '''To instantiate a random search engine'''

        random.seed(1)

        orio.main.tuner.search.search.Search.__init__(self, params)

        # set all algorithm-specific arguments to their default values
        self.local_distance = 0
        #self.total_runs = 100   # set in Search superclass

        # read all algorithm-specific arguments
        self.__readAlgoArgs()


        self.init_samp = 2*self.total_runs   # BN: used to be hard-coded to 10,000
        # complain if both the search time limit and the total number of search runs are undefined
        if self.time_limit <= 0 and self.total_runs <= 0:
            err(('orio.main.tuner.search.randomsearch.randomsearch: %s search requires either (or both) ' +
                 'of the search parameters time limit in seconds (time_limit) or/and the ' +
                 'total number of search runs (total_runs) to be defined in the search {} section' +
                 'of the tuning spec.') % self.__class__.__name__)

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
        sruns=0
        fruns=0

        # start the timer
        start_time = time.time()
        init = True


        # randomly pick coordinates to be empirically tested
        coords = {}
        uneval_coords = []
        uneval_params = []

        #default code without transformation
        neigh_coords=[[0]*self.total_dims]

        while len(uneval_coords) < self.init_samp and len(uneval_coords) <= self.total_runs:
            coord = self.__getNextCoord(coord_records, neigh_coords, init)
            coord_key = str(coord)

            if not coord or len(coord) == 0:
                break

            # TODO: the valid point generation is extremely slow, reimplement
            if coord_key not in coords:
                coords[coord_key]=coord
                # if the given coordinate is out of the search space
                is_out = False
                for i in range(0, self.total_dims):
                    if coord[i] < 0 or coord[i] >= self.dim_uplimits[i]:
                        is_out = True
                        break
                if is_out:
                    continue
                # test if the performance parameters are valid
                perf_params = self.coordToPerfParams(coord)
                #print perf_params
                perf_params1=copy.copy(perf_params)

                try:
                    is_valid = eval(self.constraint, perf_params1, dict(self.input_params))
                except Exception, e:
                    err('failed to evaluate the constraint expression: "%s"\n%s %s' % (self.constraint,e.__class__.__name__, e))
                # if invalid performance parameters
                if not is_valid:
                    continue

                temp=[]
                for k in sorted(perf_params):
                    temp.append(perf_params[k])
                debug('sample-point:'+str(coord),obj=self,level=6)
                uneval_coords.append(coord)
                uneval_params.append(perf_params)


        info('Size of search space: ' + str(len(coords)))
        info('Unevaluated coordinates: ' + str(len(uneval_coords)))
        info('Unevaluated parameters: ' + str(len(uneval_params)))

        eval_coords = []
        eval_params = []
        eval_cost = []
        num_eval_best=0

        indices=random.sample(range(1,len(uneval_coords)),  self.total_dims)
        indices.insert(0,0)
        debug(msg='Current indices: ' + str(indices), obj=self, level=2)
        

        all_indices=set(range(len(uneval_coords)))
        init_indices=set(indices)
        remain_indices=list(all_indices.difference(init_indices))
        random.shuffle(remain_indices)
        indices.extend(remain_indices)

        perf_cost, mean_perf_cost = self.MAXFLOAT, self.MAXFLOAT
        for index in indices:
            coord=uneval_coords[index]
            coord_key = str(coord)
            params=uneval_params[index]
            eval_coords.append(coord)
            eval_params.append(params)

            debug(msg='Parameter values: ' + str(params), obj=self, level=2)
            runs += 1

            perf_costs={}
            try:
                perf_costs = self.getPerfCosts([coord])
            except Exception, e:
                perf_costs[str(coords)]=[self.MAXFLOAT]
                info('FAILED: %s %s' % (e.__class__.__name__, e))
                fruns +=1

            # compare to the best result
            pcost_items = perf_costs.items()
            pcost_items.sort(lambda x,y: cmp(eval(x[0]),eval(y[0])))
            for i, (coord_str, pcost) in enumerate(pcost_items):
                if type(pcost) == tuple: (perf_cost,_) = pcost    # ignore transfer costs -- GPUs only
                else: perf_cost = pcost

                try:
                    floatNums = [float(x) for x in perf_cost]
                    mean_perf_cost=sum(floatNums) / len(perf_cost)
                except:
                    mean_perf_cost=perf_cost


            transform_time=self.getTransformTime(coord_key)
            compile_time=self.getCompileTime(coord_key)


            res_obj={}
            res_obj['run']=runs
            res_obj['coordinate']=coord
            res_obj['perf_params']=params
            res_obj['transform_time']=transform_time
            res_obj['compile_time']=compile_time
            res_obj['cost']=perf_cost
            info('(run %s) | '%runs+json.dumps(res_obj))
            #info('run %s | coordinate: %s | perf_params: %s | transform_time: %s | compile_time: %s | cost: %s' % (runs, coord, params, transform_time, compile_time,perf_cost))

            eval_cost.append(mean_perf_cost)

            if mean_perf_cost < best_perf_cost and mean_perf_cost > 0.0:
                best_coord = coord
                best_perf_cost = mean_perf_cost
                info('>>>> best coordinate found: %s, cost: %e' % (coord, mean_perf_cost))
                num_eval_best=runs

            if not math.isinf(mean_perf_cost):
                sruns +=1

            # check if the time is up
            # info('%s' % self.time_limit)
            if self.time_limit > 0 and (time.time()-start_time) > self.time_limit:
                break

            # check if the maximum limit of runs is reached
            if self.total_runs > 0 and runs >= self.total_runs:
                break


        info('Best performance = ' + str(best_perf_cost))
        info('Best coordinate = ' + str(best_coord))
        end_time = time.time()
        search_time=start_time-end_time
        speedup=float(eval_cost[0])/float(best_perf_cost)



        # compute the total search time
        search_time = time.time() - start_time

        info('----- end random search -----')

        info('----- begin random search summary -----')
        info(' total completed runs: %s' % runs)
        info(' total successful runs: %s' % sruns)
        info(' total failed runs: %s' % fruns)
        info(' speedup: %s' % speedup)
        info(' found at: %s' % num_eval_best)
        info('----- end random search summary -----')



        # return the best coordinate
        return best_coord, best_perf_cost, search_time, sruns




    # old random search
    # ToDo: should be removed
    def searchBestCoordOld(self, startCoord=None):
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
        sruns=0
        fruns=0
        # start the timer
        start_time = time.time()
        init = True
        # execute the randomized search method
        while True:

            # randomly pick a set of coordinates to be empirically tested
            coords = []
            while len(coords) < coord_count:
                coord = self.__getNextCoord(coord_records, neigh_coords, init)
                init=False
                if coord:
                    coords.append(coord)
                else:
                    break

            # check if all coordinates in the search space have been explored
            if len(coords) == 0:
                break

            # determine the performance cost of all chosen coordinates
            #perf_costs = self.getPerfCosts(coords)

            perf_costs={}
            # determine the performance cost of all chosen coordinates
            #perf_costs = self.getPerfCosts(coords)
            #sys.exit()
            try:
                perf_costs = self.getPerfCosts(coords)
            except Exception, e:
                perf_costs[str(coords[0])]=[self.MAXFLOAT]
                info('FAILED: %s %s' % (e.__class__.__name__, e))
                fruns +=1

            # compare to the best result
            pcost_items = perf_costs.items()
            pcost_items.sort(lambda x,y: cmp(eval(x[0]),eval(y[0])))
            for i, (coord_str, pcost) in enumerate(pcost_items):
                if type(pcost) == tuple: (perf_cost,_) = pcost    # ignore transfer costs -- GPUs only
                else: perf_cost = pcost
                coord_val = eval(coord_str)
                #info('%s %s' % (coord_val,perf_cost))
                perf_params = self.coordToPerfParams(coord_val)
                try:
                    floatNums = [float(x) for x in perf_cost]
                    mean_perf_cost=sum(floatNums) / len(perf_cost)
                except:
                    mean_perf_cost=perf_cost

                transform_time=self.getTransformTime()
                compile_time=self.getCompileTime()
                #info('(run %s) coordinate: %s, perf_params: %s, transform_time: %s, compile_time: %s, cost: %s' % (runs+i+1, coord_val, perf_params, transform_time, compile_time,perf_cost))
                if mean_perf_cost < best_perf_cost and mean_perf_cost > 0.0:
                    best_coord = coord_val
                    best_perf_cost = mean_perf_cost
                    info('>>>> best coordinate found: %s, cost: %e' % (coord_val, mean_perf_cost))

            # if a better coordinate is found, explore the neighboring coordinates
            if False and old_perf_cost != best_perf_cost:
                neigh_coords.extend(self.getNeighbors(best_coord, self.local_distance))
                old_perf_cost = best_perf_cost



            # increment the number of runs
            runs += 1 #len(mean_perf_cost)


            if not math.isinf(mean_perf_cost):
                sruns +=1
                info('(run %s) sruns: %s, fruns: %s, coordinate: %s, perf_params: %s, transform_time: %s, compile_time: %s, cost: %s' % (runs+i, sruns, fruns, coord_val, perf_params, transform_time, compile_time,perf_cost))


            # check if the time is up
            # info('%s' % self.time_limit)
            if self.time_limit > 0 and (time.time()-start_time) > self.time_limit:
                break

            # check if the maximum limit of runs is reached
            #if self.total_runs > 0 and runs >= self.total_runs:
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
    #--------------------------------------------------

    def __readAlgoArgs(self):
        '''To read all algorithm-specific arguments'''

        # check for algorithm-specific arguments
        for vname, rhs in self.search_opts.iteritems():
            debug(msg=str(vname)+'=' +str(rhs), obj=self, level=3)
            # local search distance
            if vname == self.__LOCAL_DIST:
                if not isinstance(rhs, int) or rhs < 0:
                    err('orio.main.tuner.search.randomsearch: %s argument "%s" must be a positive integer or zero'
                           % (self.__class__.__name__, vname))
                self.local_distance = rhs

            # unrecognized algorithm-specific argument
            elif vname == 'total_runs':
                self.total_runs = rhs
            else:
                err('orio.main.tuner.search.randomsearch: unrecognized %s algorithm-specific argument: "%s"' %
                       (self.__class__.__name__, vname))

    #--------------------------------------------------

    def __getNextCoord(self, coord_records, neigh_coords,init):
        '''Get the next coordinate to be empirically tested'''


        #if len(coord_records) == 0:
        #  return [0] * self.total_dims

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
        while True:
            coord = self.getRandomCoord()
            if str(coord) not in coord_records:
                coord_records[str(coord)] = None
                return coord

    #--------------------------------------------------




    def __getNextCoordOld(self, coord_records, neigh_coords,init):
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
