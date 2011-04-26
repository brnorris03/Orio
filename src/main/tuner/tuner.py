#
# The tuner class to initiate the empirical performance tuning process
#

import re, sys

from orio.main.util.globals import *
import orio.main.dyn_loader, orio.main.tspec.tspec, orio.main.tuner.ptest_codegen, orio.main.tuner.ptest_driver


#--------------------------------------------------

# the name of the module containing various search algorithms
SEARCH_MOD_NAME = 'orio.main.tuner.search'

#--------------------------------------------------

class PerfTuner:
    '''The empirical performance tuner'''

    # regular expressions
    __vname_re = r'[A-Za-z_]\w*'
    __import_re = r'\s*import\s+spec\s+(' + __vname_re + r');\s*'

    #-------------------------------------------------
    
    def __init__(self, specs_map, odriver):
        '''To instantiate an empirical performance tuner object'''

        self.specs_map = specs_map
        self.odriver = odriver
        self.dloader = orio.main.dyn_loader.DynLoader()

        self.num_params=0
        self.num_configs=0
        self.num_bin=0
        self.num_int=0
        
    
    #-------------------------------------------------

    def tune(self, module_body_code, line_no, cfrags):
        '''
        Perform empirical performance tuning on the given annotated code. And return the best
        optimized code variant.
        '''
        
        # extract the tuning information specified from the given annotation
        tinfo = self.__extractTuningInfo(module_body_code, line_no)
        
        # determine if parallel search is required
        use_parallel_search = tinfo.batch_cmd != None

        # create a performance-testing code generator for each distinct problem size
        ptcodegens = []
        timing_code = ''
        for prob_size in self.__getProblemSizes(tinfo.iparam_params, tinfo.iparam_constraints):
            if self.odriver.lang == 'c':
                c = orio.main.tuner.ptest_codegen.PerfTestCodeGen(prob_size, tinfo.ivar_decls, tinfo.ivar_decl_file,
                                                                  tinfo.ivar_init_file, tinfo.ptest_skeleton_code_file,
                                                                  tinfo.random_seed, use_parallel_search)
            elif self.odriver.lang == 'fortran':
                c = orio.main.tuner.ptest_codegen.PerfTestCodeGenFortran(prob_size, tinfo.ivar_decls, tinfo.ivar_decl_file,
                                                                         tinfo.ivar_init_file, tinfo.ptest_skeleton_code_file,
                                                                         tinfo.random_seed, use_parallel_search)
            else:
                err('main.tuner.tuner:  unknown output language specified: %s' % self.odriver.lang)
      
            ptcodegens.append(c)

        # create the performance-testing driver
        ptdriver = orio.main.tuner.ptest_driver.PerfTestDriver(tinfo, use_parallel_search, 
                                                               self.odriver.lang, 
                                                               c.getTimerCode(use_parallel_search))

        # get the axis names and axis value ranges to represent the search space
        axis_names, axis_val_ranges = self.__buildCoordSystem(tinfo.pparam_params)

        # combine the performance parameter constraints
        pparam_constraint = 'True'
        for vname, rhs in tinfo.pparam_constraints:
            pparam_constraint += ' and (%s)' % rhs

        # dynamically load the search engine class
        
        #print tinfo.search_algo
        if Globals().extern:
            tinfo.search_algo='Extern'
            info('Running in %s mode' % tinfo.search_algo)
            info('Using parameters %s' % Globals().config)
            
        class_name = tinfo.search_algo
        mod_name = '.'.join([SEARCH_MOD_NAME, class_name.lower(), class_name.lower()])
        search_class = self.dloader.loadClass(mod_name, class_name)

        # convert the search time limit (from minutes to seconds) and get the total number of
        # search runs
        search_time_limit = 60 * tinfo.search_time_limit
        search_total_runs = tinfo.search_total_runs

        # get the search-algorithm-specific arguments
        search_opts = dict(tinfo.search_opts)
        
        # perform the performance tuning for each distinct problem size
        optimized_code_seq = []
        for ptcodegen in ptcodegens:

            if Globals().verbose:
                info('\n----- begin empirical tuning for problem size -----')
                iparams = ptcodegen.input_params[:]
                iparams.sort(lambda x,y: cmp(x[0],y[0]))
                for pname, pvalue in iparams:
                    info(' %s = %s' % (pname, pvalue))

            # create the search engine
            search_eng = search_class({'cfrags':cfrags, 
                                       'axis_names':axis_names, 
                                       'axis_val_ranges':axis_val_ranges, 
                                       'pparam_constraint':pparam_constraint,
                                       'search_time_limit':search_time_limit, 
                                       'search_total_runs':search_total_runs, 
                                       'search_opts':search_opts,
                                       'ptcodegen':ptcodegen, 
                                       'ptdriver':ptdriver, 'odriver':self.odriver,
                                       'use_parallel_search':use_parallel_search})

            # search for the best performance parameters
            best_perf_params, best_perf_cost = search_eng.search()

            # print the best performance parameters
            
            if Globals().verbose and not Globals().extern:
                info('----- the obtained best performance parameters -----')
                pparams = best_perf_params.items()
                pparams.sort(lambda x,y: cmp(x[0],y[0]))
                for pname, pvalue in pparams:
                    info(' %s = %s' % (pname, pvalue))
        
            # generate the optimized code using the obtained best performance parameters
            if Globals().extern:
                best_perf_params=Globals().config

            #print Globals().config    
            
            cur_optimized_code_seq = self.odriver.optimizeCodeFrags(cfrags, best_perf_params)

            # check the optimized code sequence
            if len(cur_optimized_code_seq) != 1:
                err('orio.main.tuner internal error: the empirically optimized code cannot contain multiple versions')
            
            # get the optimized code
            optimized_code, _ = cur_optimized_code_seq[0]

            # insert comments into the optimized code to include information about 
            # the best performance parameters and the input problem sizes
            iproblem_code = ''
            iparams = ptcodegen.input_params[:]
            iparams.sort(lambda x,y: cmp(x[0],y[0]))
            for pname, pvalue in iparams:
                if pname == '__builtins__':
                    continue
                iproblem_code += '  %s = %s \n' % (pname, pvalue)
            pparam_code = ''
            pparams = best_perf_params.items()
            pparams.sort(lambda x,y: cmp(x[0],y[0]))
            for pname, pvalue in pparams:
                if pname == '__builtins__':
                    continue
                pparam_code += '  %s = %s \n' % (pname, pvalue)
            info_code = '\n\n/**-- (Generated by Orio) \n'
            if not Globals().extern:
                info_code += 'Best performance cost: \n'
                info_code += '  %s \n' % best_perf_cost
            info_code += 'Tuned for specific problem sizes: \n'
            info_code += iproblem_code
            info_code += 'Best performance parameters: \n'
            info_code += pparam_code
            info_code += '--**/\n\n'
            optimized_code = info_code + optimized_code

            # store the optimized for this problem size
            optimized_code_seq.append((optimized_code, ptcodegen.input_params[:]))

        # return the optimized code
        return optimized_code_seq

    # Private methods
    #-------------------------------------------------

    def __extractTuningInfo(self, code, line_no):
        '''Extract tuning information from the given annotation code'''

        # parse the code
        match_obj = re.match(r'^' + self.__import_re + r'$', code)

        # if the code contains a single import statement
        if match_obj:

            # get the specification name and line number
            spec_name = match_obj.group(1)
            spec_name_line_no = line_no + code[:match_obj.start(1)].count('\n')
            
            # if the tuning info is not defined
            if spec_name not in self.specs_map:
                err('orio.main.tuner.tuner: %s: undefined specification: "%s"' % (spec_name_line_no, spec_name))

            # get the tuning information from the specifications map
            tinfo = self.specs_map[spec_name]

            # return the tuning information
            return tinfo

        # if the tuning specification is hardcoded into the given code
        else:

            # parse the specification code to get the tuning information
            tinfo = orio.main.tspec.tspec.TSpec().parseSpec(code, line_no)

            # return the tuning information
            return tinfo
        
    #-------------------------------------------------

    def __listAllCombinations(self, seqs):
        '''
        Enumerate all combinations of the given sequences.
          e.g. input: [['a','b'],[1,2]] --> [['a',1],['a',2],['b',1],['b',2]]
        '''
        
        # the base case
        if len(seqs) == 0:
            return []
        
        # the recursive step
        trailing_combs = self.__listAllCombinations(seqs[1:])
        if trailing_combs == []:
            trailing_combs = [[]]
        combs = []
        for i in seqs[0]:
            for c in trailing_combs:
                combs.append([i] + c)
                
        # return the combinations
        return combs
    
    #-------------------------------------------------

    def __getProblemSizes(self, iparam_params, iparam_constraints):
        '''Return all valid problem sizes'''

        # combine the input parameter constraints
        iparam_constraint = 'True'
        for vname, rhs in iparam_constraints:
            iparam_constraint += ' and (%s)' % rhs

        # compute all possible combinations of problem sizes
        prob_sizes = []
        pnames, pvalss = zip(*iparam_params)
        for pvals in self.__listAllCombinations(pvalss):
            prob_sizes.append(zip(pnames, pvals))

        # exclude all invalid problem sizes
        n_prob_sizes = []
        for p in prob_sizes:
            try:
                is_valid = eval(iparam_constraint, dict(p))
            except Exception, e:
                err('orio.main.tuner.tuner:%s: failed to evaluate the input parameter constraint expression\n --> %s: %s' %  (iparam_constraint,e.__class__.__name__, e))
            if is_valid:
                n_prob_sizes.append(p)
        prob_sizes = n_prob_sizes

        # check if the new problem sizes is empty
        if len(prob_sizes) == 0:
            err ('orio.main.tuner.tuner: no valid problem sizes exist. please check the input parameter ' +
                   'constraints')
        
        # return all possible combinations of problem sizes
        return prob_sizes

    #-------------------------------------------------

    def __buildCoordSystem(self, perf_params):
        '''Return information about the coordinate systems that represent the search space'''

        # get the axis names and axis value ranges
        axis_names = []
        axis_val_ranges = []
        for pname, prange in perf_params:
            axis_names.append(pname)
            
            # remove duplications and then perform sorting
            n_prange = []
            for r in prange:
                if r not in n_prange:
                    n_prange.append(r)
            prange = n_prange
            prange.sort()
            axis_val_ranges.append(prange)

        
        
        self.num_params=len(axis_names)
        self.num_configs=1
        self.num_bin=0
        self.num_int=self.num_params

        for vals in axis_val_ranges:
            #print len(vals)
            self.num_configs=self.num_configs*len(vals)
            if len(vals)==2:
                #print vals
                if False in vals or True in vals:
                    self.num_bin=self.num_bin+1

        self.num_int=self.num_int-self.num_bin

        info('Search_Space         = %1.3e' % self.num_configs)
        info('Number_of_Parameters = %02d' % self.num_params)
        info('Numeric_Parameters   = %02d' % self.num_int)
        info('Binary_Parameters    = %02d' % self.num_bin)
        
        

        return (axis_names, axis_val_ranges)
        

