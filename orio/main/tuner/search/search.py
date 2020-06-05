#
# The search engine used for search space exploration
#
import sys, math, time
from orio.main.util.globals import *


class Search:
    '''The search engine used to explore the search space '''

    MAXFLOAT = float('inf')

    #----------------------------------------------------------
    
    def __init__(self, params):
        '''To instantiate a search engine'''

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

        if 'axis_names' in params.keys(): 
            self.total_dims = len(params['axis_names'])
        else: 
            err('the search space was not defined correctly, missing axis_names parameter')
        if 'axis_val_ranges' in params.keys(): 
            self.dim_uplimits = [len(r) for r in params['axis_val_ranges']]
        else: 
            err('the search space was not defined correctly, missing axis_val_ranges parameter')

        self.space_size = 0
        if self.total_dims > 0:
            self.space_size = reduce(lambda x,y: x*y, self.dim_uplimits, 1)

        #print self.dim_uplimits    
        #res='Space %d %d %1.3e' % (self.total_dims-num_bins,num_bins,self.space_size)        
        #info(res)
        #sys.exit()		

        #print str(params['ptdriver'].tinfo)
        
        if 'use_parallel_search' in params.keys(): self.use_parallel_search = params['use_parallel_search']
        else: self.use_parallel_search = False
        if 'ptdriver' in params.keys(): self.num_procs = params['ptdriver'].tinfo.num_procs
        else: self.num_procs = 1
        
        # the class variables that may be ignored when developing a new search engine subclass
        if 'cfrags' in params.keys(): self.cfrags = params['cfrags']
        else: self.cfrags = None
        if 'axis_names' in params.keys(): self.axis_names = params['axis_names']
        else: self.axis_names = None
        if 'axis_val_ranges' in params.keys(): self.axis_val_ranges = params['axis_val_ranges']
        else: self.axis_val_ranges = None
        if 'pparam_constraint' in params.keys(): self.constraint = params['pparam_constraint']
        else: self.constraint = 'None'
        if 'ptcodegen' in params.keys(): self.ptcodegen = params['ptcodegen']
        else: self.ptcodegen = None
        if 'ptdriver' in params.keys(): self.ptdriver = params['ptdriver']
        else: self.ptdriver = None
        if 'odriver' in params.keys(): self.odriver = params['odriver']
        else: self.odriver = None
        self.input_params = params.get('input_params')
        
        self.timing_code = ''
        
        self.verbose = Globals().verbose
        self.perf_cost_records = {}
        self.transform_time={}
        self.best_coord_info="None"

    #----------------------------------------------------------

    def searchBestCoord(self):
        '''
        Explore the search space and return the coordinate that yields the best performance
        (i.e. minimum performance cost).
        
        This is the function that needs to be implemented in each new search engine subclass.
        '''
        raise NotImplementedError('%s: unimplemented abstract function "searchBestCoord"' %
                                  self.__class__.__name__)
    
    def modelBased(self):
        '''
        Returns true if the search module uses model-based optimization and False if 
        it's purely based on empirical testing.
        By default, returns False, can be overridden by subclasses.
        '''
        return False
    
    #----------------------------------------------------------

    def search(self, startCoord=None):
        '''Initiate the search process and return the best performance parameters'''

        # if the search space is empty
        if self.total_dims == 0:
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
            self.best_coord_info = '%s=%s, cost=%e, transfer_time=%e, inputs=%s, search_space=%1.3e, search_time=%.2f, runs=%d' \
                                   % (best_coord, self.coordToPerfParams(best_coord), best_perf, corr_transfer, str(self.input_params), \
                                   self.space_size, search_time, runs)
            info('----- begin summary -----')
            info(' best coordinate: %s' % self.best_coord_info)
            info('----- end summary -----')

                
        if not Globals().extern:    
            # get the performance cost of the best parameters
            best_perf_cost = self.getPerfCost(best_coord)
            # convert the coordinate to the corresponding performance parameters
            best_perf_params = self.coordToPerfParams(best_coord)
        else:
            best_perf_cost=0
            best_perf_params=Globals().config
        

        # return the best performance parameters
        return (best_perf_params, best_perf_cost)

    #----------------------------------------------------------

    def coordToPerfParams(self, coord):
        '''To convert the given coordinate to the corresponding performance parameters'''

        # get the performance parameters that correspond the given coordinate
        perf_params = {}
        for i in range(0, self.total_dims):
            ipoint = coord[i]
            perf_params[self.axis_names[i]] = self.axis_val_ranges[i][ipoint]

        # return the obtained performance parameters
        return perf_params

    #----------------------------------------------------------

    def getPerfCost(self, coord):
        '''
        Empirically evaluate the performance cost of the code corresponding to the given coordinate
        '''

        perf_costs = self.getPerfCosts([coord])
        [(perf_cost,_),] = perf_costs.values()
        return perf_cost

    def getTransformTime(self, key):
        trans_time=0.0
        if key in self.transform_time:
            trans_time=self.transform_time[key]
        return trans_time
    
    def getCompileTime(self,key):
        compile_time = 0.0
        if key in self.ptdriver.compile_time:
            compile_time=self.ptdriver.compile_time[key]
        return compile_time
    
    #----------------------------------------------------------

    def getPerfCosts(self, coords):
        '''
        Empirically evaluate the performance costs of the codes corresponding the given coordinates
        @param coords:  all search space coordinates
        '''

        # initialize the performance costs mapping
        perf_costs = {}
        

        # filter out all invalid coordinates and previously evaluated coordinates
        uneval_coords = []
        for coord in coords:
            coord_key = str(coord)

            # if the given coordinate is out of the search space
            is_out = False
            for i in range(0, self.total_dims):
                if coord[i] < 0 or coord[i] >= self.dim_uplimits[i]:
                    is_out = True
                    break
            if is_out:
                perf_costs[coord_key] = ([self.MAXFLOAT],[self.MAXFLOAT])
                continue

            # if the given coordinate has been computed before
            if coord_key in self.perf_cost_records:
                perf_costs[coord_key] = self.perf_cost_records[coord_key]
                continue

            # get the performance parameters
            perf_params = self.coordToPerfParams(coord)

            
            # test if the performance parameters are valid
            try:
                is_valid = eval(self.constraint, perf_params, dict(self.input_params))
            except Exception, e:
                err('failed to evaluate the constraint expression: "%s"\n%s %s' % (self.constraint,e.__class__.__name__, e))

            # if invalid performance parameters
            if not is_valid:
                perf_costs[coord_key] = ([self.MAXFLOAT],[self.MAXFLOAT])
                continue

            # store all unevaluated coordinates
            uneval_coords.append(coord)

        # check the unevaluated coordinates
        if len(uneval_coords) == 0:
            return perf_costs

        #debug('search perf_params=' + str(perf_params))
        # execute the original code and obtain results for validation
        if Globals().validationMode and not Globals().executedOriginal:
            validation_map = {}
            transformed_code_seq = self.odriver.optimizeCodeFrags(self.cfrags, perf_params)
            transformed_code, _, externals = transformed_code_seq[0]
            validation_map['original'] = (transformed_code, externals)
            instrumented_code = self.ptcodegen.generate(validation_map)
            _ = self.ptdriver.run(instrumented_code)
            Globals().executedOriginal = True
        
        # get the transformed code for each corresponding coordinate for non-command-line parameters
        code_map = {}
        transformed_code_seq = []
        for coord in uneval_coords:
            if not Globals().disable_orio: always_print('.',end='')
            coord_key = str(coord)
            
            perf_params = self.coordToPerfParams(coord)
            
            if self.modelBased():
                self.transform_time[coord_key] = 0.0
                perf_costs[coord_key] = self.getModelPerfCost(perf_params, coord)
                # Do the code gen for later (static) analysis
                #self.odriver.optimizeCodeFrags(self.cfrags, perf_params)
                continue
            else: # Legacy, pure empirical
                start = time.time()
                #info('1. transformation time = %e'%time.time())
                try:
                    transformed_code_seq = self.odriver.optimizeCodeFrags(self.cfrags, perf_params)
                    elapsed = (time.time() - start)
                    #info('2. transformation time = %e'%time.time())
                    self.transform_time[coord_key]=elapsed
                except Exception:
                    err('[search] failed evaluation of coordinate: %s=%s.\tException: %s' %\
                        (str(coord), str(perf_params), str(sys.exc_info()[0])))
                    # Do not stop if a single test fails, continue with other transformations
                    #err('failed during evaluation of coordinate: %s=%s\n%s\nError:%s' \
                    #% (str(coord), str(perf_params), str(e.__class__), e.message), 
                    #code=0, doexit=False)
                    perf_costs[coord_key] = ([self.MAXFLOAT],[self.MAXFLOAT])
                    
                    elapsed = (time.time() - start)
                    #info('2. transformation time = %e'%time.time())
                    self.transform_time[coord_key]=elapsed
                    continue
            
            #info('transformation time = %e' % self.transform_time)
            if transformed_code_seq:
                if len(transformed_code_seq) != 1:
                    err('internal error: the optimized annotation code cannot contain multiple versions', doexit=True)
    
                transformed_code, _, externals = transformed_code_seq[0]
                code_map[coord_key] = (transformed_code, externals)
        if code_map == {}: # nothing to test
            return perf_costs
        #debug("search.py: about to test the following code segments (code_map):\n%s" % code_map, level=1)
        
        
        # Evaluate the performance costs for all coordinates
        new_perf_costs = None
        if self.modelBased():
            new_perf_costs = self.getModelPerfCosts(perf_params=perf_params,coord=coord_key)
        if not new_perf_costs:
            test_code = self.ptcodegen.generate(code_map)
            perf_params = self.coordToPerfParams(uneval_coords[0])
            new_perf_costs = self.ptdriver.run(test_code, perf_params=perf_params,coord=coord_key)
        #new_perf_costs = self.getPerfCostConfig(coord_key,perf_params)
        # remember the performance cost of previously evaluated coordinate
        self.perf_cost_records.update(new_perf_costs.items())
        # merge the newly obtained performance costs
        perf_costs.update(new_perf_costs.items())
        # also take the compile time
        

        #sys.exit()

        #return the performance cost

        return perf_costs
    
    #----------------------------------------------------------

    def getModelPerfCosts(self, perf_params, coord):
        '''
        Return performance costs based on a model or existing data, do not perform empirical tests.
                This is the function that needs to be implemented in each new search engine subclass
                that returns True in its implementation of the modelBased() method.
        '''
        raise NotImplementedError('%s: unimplemented abstract function "searchBestCoord"' %
                                  self.__class__.__name__)


    #----------------------------------------------------------

    def getPerfCostConfig(self, coord_key,param_config):
        '''
        Empirically evaluate the performance costs of the codes corresponding the given coordinates
        @param coords:  all search space coordinates
        '''

        is_valid=False
        perf_costs = []
        
        # test if the performance parameters are valid
        try:
            is_valid = eval(self.constraint, param_config)
        except Exception, e:
            err('failed to evaluate the constraint expression: "%s"\n%s %s' % (self.constraint,e.__class__.__name__, e))

            
        # if invalid performance parameters
        if not is_valid:
            perf_costs[coord_key] = [self.MAXFLOAT]
            return perf_costs


        #config=Globals().config
        params=self.params['axis_names']
        vals=self.params['axis_val_ranges']


        is_out=False
        for i, p in enumerate(params):
            min_val=min(vals[i])
            max_val=max(vals[i])
            #print p, min_val,max_val
            if param_config[p] < min_val or param_config[p] > max_val:
                is_out=True
                break
                
        if is_out:
            perf_costs[coord_key] = [self.MAXFLOAT]
            return perf_costs
            

        code_map = {}
        transformed_code_seq = self.odriver.optimizeCodeFrags(self.cfrags, param_config)

        
        if len(transformed_code_seq) != 1:
            err('internal error: the optimized annotation code cannot contain multiple versions', doexit=True)

        #coord_key='param'    
        transformed_code, _, externals = transformed_code_seq[0]
        code_map[coord_key] = (transformed_code, externals)
        #debug("search.py: about to test the following code segments (code_map):\n%s" % code_map, level=1)
        # evaluate the performance costs for all coordinates
        test_code = self.ptcodegen.generate(code_map)
        new_perf_costs = self.ptdriver.run(test_code)


        return new_perf_costs

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
        for i in range(0, self.total_dims):
            d_sqr += (coord2[i] - coord1[i])**2
        d = math.sqrt(d_sqr)
        return d

    #----------------------------------------------------------

    def getRandomCoord(self):
        '''Randomly pick a coordinate within the search space'''

        random_coord = []
        for i in range(0, self.total_dims):
            iuplimit = self.dim_uplimits[i]
            ipoint = self.getRandomInt(0, iuplimit-1)
            random_coord.append(ipoint)
        return random_coord
                                                                     

    def getInitCoord(self):
        '''Randomly pick a coordinate within the search space'''

        random_coord = []
        for i in range(0, self.total_dims):
            #iuplimit = self.dim_uplimits[i]
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
        for i in range(0, self.total_dims):
            iuplimit = self.dim_uplimits[i]
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
        best_perf_cost = self.getPerfCost(coord)

        # examine all neighboring coordinates
        for n in neigh_coords:
            perf_cost = self.getPerfCost(n)
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
