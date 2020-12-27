#
# Implementation of the random search algorithm
#

import sys, time
import math
import random
import csv
import hashlib
import orio.main.tuner.search.search
from orio.main.util.globals import *

#-----------------------------------------------------

class CUDACFG(orio.main.tuner.search.search.Search):
    '''
    The search engine that uses a model-based search approach, enhanced with a local search that finds
    the best neighboring coordinate.

    Below is a list of algorithm-specific arguments used to steer the search algorithm.
      local_distance            the distance number used in the local search to find the best
                                neighboring coordinate located within the specified distance
    '''

    # algorithm-specific argument names
    __LOCAL_DIST = 'local_distance'       # default: 0
    __INSTMIX = 'instmix'                 # required
    
    #--------------------------------------------------
    
    def __init__(self, params):
        '''To instantiate a random search engine'''

        random.seed(1)
        
        orio.main.tuner.search.search.Search.__init__(self, params)

        # set all algorithm-specific arguments to their default values
        self.local_distance = 0

        # read all algorithm-specific arguments
        self.__readAlgoArgs()
        self.srun = 0
        
        # TODO: update to invoke the static analysis tool in a separate method, not read csv
        # Required (for now) option, reading file
        # Source file name is in self.odriver.srcname 
        self.instmixdata = []  # list of lists, first element contains column labels
        self.instmix = None # TODO 
        with open(self.instmix, 'rU') as csvfile:
            self.instmixdata = list(csv.reader(csvfile, delimiter=',', quotechar='"') )
        
        bcoord = self.__hashCoord('[1, 5, 1, 1, 0, 0]')
        #debug("Instruction mix data: %d" % self.instmixdata[0].index('coordinate_o'))
        # Hash coordinates
        ind = self.instmixdata[0].index('coordinate_o')
        l = len(self.instmixdata[0])-1
        self.instmixdata[0].insert(l,'kernel_type')
        self.instmixdata[0].insert(l+1,'hash_coordinate')
        self.allcoords = []
        self.coords = []
        times = []
        for row in self.instmixdata[1:]:
            if not row: continue
            row.insert(l,self.getIntensity(row[ind]))
            row.insert(l+1, self.__hashCoord(row[ind]))
            self.allcoords.append(eval(row[ind]))
            times.append(float(row[-1]))
            # TODO: this is only for debugging, will remove:
            if self.__hashCoord(row[ind]) == bcoord:
                info(("Best coordinate intensity: ", self.getIntensity(row), row))
            
        # Some sanity checks, will remove; Find best time
        info("Best known time: %f" % min(times))
       
        
        # complain if both the search time limit and the total number of search runs are undefined
        #if self.time_limit <= 0 and self.total_runs <= 0:
        #    err(('orio.main.tuner.search.cudacfg.cudacfg: %s search requires either (both) the search time limit or (and) the ' +
        #            'total number of search runs to be defined') % self.__class__.__name__)
        
        #         kernel_type = get_kernel_intensity(imix)
    
    def ind(self, name):
        if name in ['ldst','tex','surf','fp','int','simd','conv','ctrl','move','pred','misc']: name += '_s'
        return self.instmixdata[0].index(name)
    
    def val(self, row, label):
        return row[self.ind(label)]

    def getIntensity(self, d_instr_k):
        mem_int=d_instr_k[self.ind('ldst')] + d_instr_k[self.ind('tex')] + d_instr_k[self.ind('surf')]
        flops_int=d_instr_k[self.ind('fp')]+d_instr_k[self.ind('int')]+d_instr_k[self.ind('simd')]+d_instr_k[self.ind('conv')]
        ctrl_int=d_instr_k[self.ind('ctrl')]+d_instr_k[self.ind('move')]+d_instr_k[self.ind('pred')]
        total_int=mem_int + flops_int + ctrl_int + d_instr_k[self.ind('misc')]
        intensity='FLOPS'
        if (flops_int > mem_int) and (flops_int > ctrl_int):
            intensity='FLOPS'
        elif (mem_int > flops_int) and (mem_int > ctrl_int):
            intensity='MEMORY'
        else:
            intensity='CONTROL'
        return intensity
    
    def modelBased(self):
        '''This search algorithm uses existing data or models to evaluate objective function.
        '''
        return True 
    
    def getModelPerfCost(self, perf_params, coord):
        '''Use existing data or a model to return performance cost.'''
        # Instead of self.ptdriver.run(test_code, perf_params=perf_params,coord=coord_key)
        
        # Initialize the performance costs dictionary
        # (indexed by the string representation of the search coordinates)
        # e.g., {'[0,1]': 0.5, '[1,1]':0.2} key is coord, value is time
        
        #perf_costs = {str(coord): self.__lookupCost(coord) }
        self.srun +=1
        return ([self.__lookupCost(coord)],[0])
    
    def getModelBasedCoords(self):
        initialCoord = self.getRandomCoord(self.allcoords)
        prev = initialCoord
        self.coords = []
        tc = []
        tcindices = list(range(0,len(self.axis_val_ranges[0])))
        for row in self.instmixdata[1:]:
            tc = []
            bc = []
            #row = random.choice(self.instmixdata[1:])
            kernel_type = self.val(row,'kernel_type')
            coord = eval(self.val(row, 'coordinate_o'))
            if coord in self.coords: continue
            occupancy = float(self.val(row,'occupancy_mp_new').strip('%'))
            tmpbc = eval(self.val(row,'lmax_block_prev'))
            bci = len(bc)/2
            tmp = tcindices
            ind = len(tmp)/2
            
            debug("[orio.main.tuner.search.cudacfg] kernel type, Occupancy, lmax_bloc_prev = (%s,%s,%s)" %
                  (str(kernel_type), str(occupancy), str(tmp)),obj=self,level=4)
            if occupancy == 100.0:
                if kernel_type == "MEM": tc = tmp[ind:]
                else: tc = tmp[:max(ind,1)]
                if kernel_type in ['FLOPS','MEM']: bc = tmpbc[:max(bci,1)]
                else: bc = tmpbc[bci:] 
            elif occupancy > 60 and occupancy < 100:
                if kernel_type in ['MEM', 'CONTROL']: tc = tmp[:max(ind,1)]
                else: tc = tmp[ind:]
                if kernel_type in ['MEM', 'CONTROL']: bc = tmpbc[:max(bci,1)]
                else: bc = tmpbc[bci:] 
            if tc and coord[0] in tc and bc and coord[1] in bc: 
                # TODO change this to not rely on knowing that TC is the first parameter!!!
                self.coords.append(coord)
            
        info("Original search space size: %d. Reduced search space size: %d." %( len(self.allcoords),len(self.coords))) 
        random.shuffle(self.coords)
        return self.coords
    
    def getInitCoord(self):
        '''Randomly pick a coordinate within the search space'''

        random_coord = []
        if self.allcoords: 
            random_coord = eval(random.choice(self.coords))
        else:
            for i in range(0, self.total_dims):
                #iuplimit = self.dim_uplimits[i]
                #ipoint = self.getRandomInt(0, iuplimit-1)
                random_coord.append(0)
        debug("Starting search coordinate:", random_coord)
        return random_coord
    

    def getRandomCoord(self, coords=[]):
        '''Randomly pick a coordinate within the search space'''

        random_coord = []
        if coords:
            random_coord = random.choice(coords)
        elif self.allcoords: 
            random_coord = random.choice(self.coords)
        else:
            for i in range(0, self.total_dims):
                iuplimit = self.dim_uplimits[i]
                ipoint = self.getRandomInt(0, iuplimit-1)
                random_coord.append(ipoint)
        return random_coord
    
    # Method required by the search interface
    def searchBestCoord(self, startCoord=None):
        '''
        To explore the search space and retun the coordinate that yields the best performance
        (i.e. minimum performance cost).
        '''
        # TODO: implement startCoord support
        
        info('\n----- begin CUDACFG search -----')

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
        coord_key=''
        # execute the randomized search method
        while True:
        
            # Use a predefined list of coordinates, if given
            if self.allcoords: 
                coords = self.getModelBasedCoords()
            else:
                # randomly pick a set of coordinates to be empirically tested
                coords = []
                while len(coords) < coord_count:
                    coord = self.__getNextCoord(coord_records, neigh_coords, init)
                    coord_key = str(coord)
                    init=False
                    if coord:
                        coords.append(coord)
                    else:
                        break

            # check if all coordinates in the search space have been explored
            if len(coords) == 0:
                break

            perf_costs={}
            transform_time = 0.0
            compile_time = 0.0
            # determine the performance cost of all chosen coordinates
            #perf_costs = self.getPerfCosts(coords)
            #sys.exit()
            try:
                perf_costs = self.getPerfCosts(coords)
            except Exception as e:
                perf_costs[str(coords[0])]=[self.MAXFLOAT]
                info('FAILED: %s %s' % (e.__class__.__name__, e))
                fruns +=1
            # compare to the best result
            pcost_items = sorted(list(map(lambda x: eval(x), list(perf_costs.items()))))
            for i, (coord_str, pcost) in enumerate(pcost_items):
                if type(pcost) == tuple: (perf_cost,_) = pcost    # ignore transfer costs -- GPUs only
                else: perf_cost = pcost
                coord_val = eval(coord_str)
                #info('%s %s' % (coord_val,perf_cost))
                perf_params = self.coordToPerfParams(coord_val)
                if type(perf_cost) is list or type(perf_cost) is tuple:
                    try:
                        floatNums = [float(x) for x in perf_cost]
                        mean_perf_cost=sum(floatNums) / len(perf_cost)
                    except:
                        mean_perf_cost=perf_cost
                else:
                    info('Perf cost', perf_cost)
                    mean_perf_cost = perf_cost
                    
                transform_time=self.getTransformTime(coord_key)
                compile_time=self.getCompileTime(coord_key)    
                #info('(run %s) coordinate: %s, perf_params: %s, transform_time: %s, compile_time: %s, cost: %s' % (runs+i+1, coord_val, perf_params, transform_time, compile_time,perf_cost))
                if mean_perf_cost <= best_perf_cost and mean_perf_cost >= 0.0:
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
                pcosts = '[]'
                if perf_cost and len(perf_cost)>1:
                    pcosts = '[' + ', '.join(["%2.4e" % x for x in perf_cost]) + ']'
                msgstr1 = '(run %d) sruns: %d, fruns: %d, coordinate: %s, perf_params: %s, ' % \
                      (runs+i, sruns, fruns, str(coord_val), str(perf_params))
                msgstr2 =  'transform_time: %2.4e, compile_time: %2.4e, cost: %s' % \
                      (transform_time, compile_time, pcosts) 
                info(msgstr1 + msgstr2)
            
            
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
        info(' total successful runs: %s' % self.srun)
        info(' total failed runs: %s' % fruns)
        info('----- end random search summary -----')
        
        # return the best coordinate
        return best_coord, best_perf_cost, search_time, sruns
   
   # Private methods
   #--------------------------------------------------
    
    def __readAlgoArgs(self):
        '''To read all algorithm-specific arguments'''

        # check for algorithm-specific arguments
        for vname, rhs in self.search_opts.items():

            # local search distance
            if vname == self.__LOCAL_DIST:
                if not isinstance(rhs, int) or rhs < 0:
                    err('orio.main.tuner.search.cudacfg: %s argument "%s" must be a positive integer or zero'
                           % (self.__class__.__name__, vname))
                self.local_distance = rhs
            # CSV instruction mix file (TODO: revise)
            elif vname == self.__INSTMIX:
                self.instmix = rhs
            # unrecognized algorithm-specific argument
            else:
                err('orio.main.tuner.search.cudacfg: unrecognized %s algorithm-specific argument: "%s"' %
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
    
        # Pick a coordinate that has never been explored before based on performance model
        while True:
            # TODO: replace this with model-based selection of next coordinate
            coord = self.getRandomCoord()
            debug(coord, ", peformance cost: ", self.__lookupCost(coord))
            if str(coord) not in coord_records:
                coord_records[str(coord)] = None
                return coord
    
        

           
    #--------------------------------------------------
    def __hashCoord(self, coord):
        return int(hashlib.sha1(str(coord)).hexdigest(), 16) % (10 ** 8)
        
    def __lookupCost(self, coord):
        ind = self.instmixdata[0].index('coordinate_o')
        hcoord = self.__hashCoord(coord)
        for row in self.instmixdata[1:]:
            if not row: continue
            if not row or hcoord != row[-2]: continue
            return float(row[-1])

        