#
# The search engine used for search space exploration
#
import sys, math, time
from orio.main.util.globals import *

class Function:

    MAXFLOAT = float('inf')

    #----------------------------------------------------------

    def __init__(self, params):

        print "NOJ PARAMS",params
        # print 'done'

        self.params=params
        debug('[Search] performance parameters: %s\n' % str(params), self)

        if 'axis_names' in params.keys(): 
            self.total_dims = len(params['axis_names'])
        else: 
            err('the search space was not defined correctly, missing axis_names parameter')
        if 'axis_val_ranges' in params.keys(): 
            self.dim_uplimits = [len(r) for r in params['axis_val_ranges']]
        else: 
            err('the search space was not defined correctly, missing axis_val_ranges parameter')

        # self.space_size = 0
        # if self.pb.total_dims > 0:
        #     self.space_size = reduce(lambda x,y: x*y, self.pb.dim_uplimits, 1)
        #print self.pb.dim_uplimits    
        #res='Space %d %d %1.3e' % (self.pb.total_dims-num_bins,num_bins,self.pb.space_size)
        #info(res)
        #sys.exit()


        if 'axis_names' in params.keys(): self.axis_names = params['axis_names']
        else: self.axis_names = None

        if 'axis_val_ranges' in params.keys(): self.axis_val_ranges = params['axis_val_ranges']
        else: self.axis_val_ranges = None

        #print str(params['ptdriver'].tinfo)
        if 'ptdriver' in params.keys(): self.num_procs = params['ptdriver'].tinfo.num_procs
        else: self.num_procs = 1
        if 'ptdriver' in params.keys(): self.ptdriver = params['ptdriver']
        else: self.ptdriver = None
        if 'pparam_constraint' in params.keys(): self.constraint = params['pparam_constraint']
        else: self.constraint = 'None'
        if 'odriver' in params.keys(): self.odriver = params['odriver']
        else: self.odriver = None
        if 'ptcodegen' in params.keys(): self.ptcodegen = params['ptcodegen']
        else: self.ptcodegen = None
        if 'cfrags' in params.keys(): self.cfrags = params['cfrags']
        else: self.cfrags = None

        self.input_params = params.get('input_params')

        self.transform_time = {}
        self.perf_cost_records = {}

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

    #----------------------------------------------------------

    def getTransformTime(self, key):
        trans_time=0.0
        if key in self.transform_time:
            trans_time=self.transform_time[key]
        return trans_time

    #----------------------------------------------------------

    def getCompileTime(self,key):
        compile_time = 0.0
        if key in self.ptdriver.compile_time:
            compile_time=self.ptdriver.compile_time[key]
        return compile_time

    #----------------------------------------------------------

    def modelBased(self):
        '''
        Returns true if the search module uses model-based optimization and False if 
        it's purely based on empirical testing.
        By default, returns False, can be overridden by subclasses.
        '''
        return False

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

