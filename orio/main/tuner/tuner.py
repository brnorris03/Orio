#
# The tuner class to initiate the empirical performance tuning process
#

import re, sys, os

from orio.main.util.globals import *
import orio.main.dyn_loader, orio.main.tspec.tspec, orio.main.tuner.ptest_codegen, orio.main.tuner.ptest_driver


#--------------------------------------------------

# the name of the module containing various search algorithms
SEARCH_MOD_NAME = 'orio.main.tuner.search'

#--------------------------------------------------

class PerfTuner:
    '''
    The empirical performance tuner.
    This class is responsible for invoking the code generators of the annotation modules,
    compiling the resulting code, and interfacing with the search interface to run the 
    tests and collect the results.
    '''

    #-------------------------------------------------
    
    def __init__(self, odriver):
        '''To instantiate an empirical performance tuner object'''

        self.odriver = odriver
        self.dloader = orio.main.dyn_loader.DynLoader()

        self.num_params=0
        self.num_configs=0
        self.num_bin=0
        self.num_int=0
        
        self.tinfo = None

    
    #-------------------------------------------------

    def tune(self, module_body_code, line_no, cfrags):
        '''
        Perform empirical performance tuning on the given annotated code. And return the best
        optimized code variant.
        '''
        
        # extract the tuning information specified from the given annotation
        tinfo = self.__extractTuningInfo(module_body_code, line_no)
        self.tinfo = tinfo
        
        # determine if parallel search is required
        use_parallel_search = tinfo.batch_cmd != None

        # create a performance-testing code generator for each distinct problem size
        ptcodegens = []
        #timing_code = ''
        for prob_size in self.__getProblemSizes(tinfo.iparam_params, tinfo.iparam_constraints):
            if self.odriver.lang == 'c':
                c = orio.main.tuner.ptest_codegen.PerfTestCodeGen(prob_size, tinfo.ivar_decls, tinfo.ivar_decl_file,
                                                                  tinfo.ivar_init_file, tinfo.ptest_skeleton_code_file, self.odriver.lang,
                                                                  tinfo.random_seed, use_parallel_search, tinfo.validation_file)
            elif self.odriver.lang == 'cuda':
                c = orio.main.tuner.ptest_codegen.PerfTestCodeGenCUDA(prob_size, tinfo.ivar_decls, tinfo.ivar_decl_file,
                                                                  tinfo.ivar_init_file, tinfo.ptest_skeleton_code_file, self.odriver.lang,
                                                                  tinfo.random_seed, use_parallel_search)
            elif self.odriver.lang == 'opencl':
                c = orio.main.tuner.ptest_codegen.PerfTestCodeGenOpenCL(prob_size, tinfo.ivar_decls, tinfo.ivar_decl_file,
                                                                  tinfo.ivar_init_file, tinfo.ptest_skeleton_code_file, self.odriver.lang,
                                                                  tinfo.random_seed, use_parallel_search)
            elif self.odriver.lang == 'fortran':
                c = orio.main.tuner.ptest_codegen.PerfTestCodeGenFortran(prob_size, tinfo.ivar_decls, tinfo.ivar_decl_file,
                                                                         tinfo.ivar_init_file, tinfo.ptest_skeleton_code_file, self.odriver.lang,
                                                                         tinfo.random_seed, use_parallel_search)
            else:
                err('main.tuner.tuner:  unknown output language specified: %s' % self.odriver.lang)
      
            ptcodegens.append(c)

        # create the performance-testing driver
        ptdriver = orio.main.tuner.ptest_driver.PerfTestDriver(self.tinfo, use_parallel_search, 
                                                               self.odriver.lang, 
                                                               c.getTimerCode(use_parallel_search))

        # get the axis names and axis value ranges to represent the search space
        
        axis_names, axis_val_ranges = self.__buildCoordSystem(tinfo.pparam_params, tinfo.cmdline_params)

        info('%s' % axis_names)
        info('%s' % axis_val_ranges)
        

        # combine the performance parameter constraints
        pparam_constraint = 'True'
        for vname, rhs in tinfo.pparam_constraints:
            pparam_constraint += ' and (%s)' % rhs

        # dynamically load the search engine class and configure it
        
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
        search_use_z3 = tinfo.search_use_z3
        search_resume = tinfo.search_resume

        # get the search-algorithm-specific arguments
        search_opts = dict(tinfo.search_opts)
        
        # perform the performance tuning for each distinct problem size
        optimized_code_seq = []
        for ptcodegen in ptcodegens:
            if Globals().verbose:
                info('\n----- begin empirical tuning for problem size -----')
                # Sort y variable name... not sure it's really necessary
                iparams = sorted(ptcodegen.input_params[:])
                for pname, pvalue in iparams:
                    info(' %s = %s' % (pname, pvalue))
            iparams = sorted(ptcodegen.input_params[:])
            for pname, pvalue in iparams:
                Globals().metadata['size_' + pname] = pvalue

            debug(ptcodegen.input_params[:])
            # create the search engine
            search_eng = search_class({'cfrags':cfrags,                     # code versions
                                       'axis_names':axis_names,             # performance parameter names
                                       'axis_val_ranges':axis_val_ranges,   # performance parameter values
                                       'pparam_constraint':pparam_constraint,
                                       'search_time_limit':search_time_limit, 
                                       'search_total_runs':search_total_runs, 
                                       'search_resume':search_resume,
                                       'search_opts':search_opts,
                                       'ptcodegen':ptcodegen, 
                                       'ptdriver':ptdriver, 'odriver':self.odriver,
                                       'use_parallel_search':use_parallel_search,
                                       'input_params':ptcodegen.input_params[:]})

            
            # search for the best performance parameters
            best_perf_params, best_perf_cost = search_eng.search()

            # output the best performance parameters
            if Globals().verbose and not Globals().extern:
                info('----- the obtained best performance parameters -----')
                pparams = sorted(list(best_perf_params.items()))
                for pname, pvalue in pparams:
                    info(' %s = %s' % (pname, pvalue))
        
            # generate the optimized code using the obtained best performance parameters
            if Globals().extern:
                best_perf_params=Globals().config

            debug("[orio.main.tuner.tuner] Globals config: %s" % str(Globals().config), obj=self, level=6)
            
            cur_optimized_code_seq = self.odriver.optimizeCodeFrags(cfrags, best_perf_params)

            # check the optimized code sequence
            if len(cur_optimized_code_seq) != 1:
                err('orio.main.tuner internal error: the empirically optimized code cannot contain multiple versions')
            
            # get the optimized code
            optimized_code, _, externals = cur_optimized_code_seq[0]

            # insert comments into the optimized code to include information about 
            # the best performance parameters and the input problem sizes
            iproblem_code = ''
            iparams = sorted(ptcodegen.input_params[:])
            for pname, pvalue in iparams:
                if pname == '__builtins__':
                    continue
                iproblem_code += '  %s = %s \n' % (pname, pvalue)
            pparam_code = ''
            pparams = sorted(list(best_perf_params.items()))
            for pname, pvalue in pparams:
                if pname == '__builtins__':
                    continue
                pparam_code += '  %s = %s \n' % (pname, pvalue)
            info_code = '\n/**-- (Generated by Orio) \n'
            if not Globals().extern:
                info_code += 'Best performance cost: \n'
                info_code += '  %s \n' % best_perf_cost
            info_code += 'Tuned for specific problem sizes: \n'
            info_code += iproblem_code
            info_code += 'Best performance parameters: \n'
            info_code += pparam_code
            info_code += '--**/\n'
            optimized_code = info_code + optimized_code

            # store the optimized for this problem size
            optimized_code_seq.append((optimized_code, ptcodegen.input_params[:], externals))

        # return the optimized code
        return optimized_code_seq

    # Private methods
    #-------------------------------------------------

    def __extractTuningInfo(self, code, line_no):
        '''Extract tuning information from the given annotation code'''

        # parse the code
        match_obj = re.match(r'^\s*import\s+spec\s+([/A-Za-z_]+);\s*$', code)

        # if the code contains a single import statement
        if match_obj:

            # get the specification name
            spec_name = match_obj.group(1)
            spec_file = spec_name+'.spec'
            try:
                src_dir = '/'.join(list(Globals().src_filenames.keys())[0].split('/')[:-1])
                spec_file_path = os.getcwd() + '/' + src_dir + '/' + spec_file
                f = open(spec_file_path, 'r')
                tspec_code = f.read()
                f.close()
            except:
                err('%s: cannot open file for reading: %s' % (self.__class__, spec_file_path))

            tuning_spec_dict = orio.main.tspec.tspec.TSpec().parseProgram(tspec_code)

        # if the tuning specification is hardcoded into the given code
        elif code.lstrip().startswith('spec'):
            tuning_spec_dict = orio.main.tspec.tspec.TSpec().parseProgram(code)
        else:
            # parse the specification code to get the tuning information
            tuning_spec_dict = orio.main.tspec.tspec.TSpec().parseSpec(code, line_no)

        # return the tuning information
        return tuning_spec_dict
        
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
        pnames, pvalss = list(zip(*iparam_params))
        for pvals in self.__listAllCombinations(pvalss):
            prob_sizes.append(list(zip(pnames, pvals)))

        # exclude all invalid problem sizes
        n_prob_sizes = []
        for p in prob_sizes:
            try:
                is_valid = eval(iparam_constraint, dict(p))
            except Exception as e:
                err('orio.main.tuner.tuner:%s: failed to evaluate the input parameter constraint expression\n --> %s: %s' %  (iparam_constraint,e.__class__.__name__, e))
            if is_valid:
                n_prob_sizes.append(p)
        prob_sizes = n_prob_sizes

        # check if the new problem sizes is empty
        if len(prob_sizes) == 0:
            err('orio.main.tuner.tuner: no valid problem sizes exist. please check the input parameter ' +
                   'constraints')
        
        # return all possible combinations of problem sizes
        return prob_sizes

    #-------------------------------------------------

    def __buildCoordSystem(self, perf_params, cmdline_params):
        '''Return information about the coordinate systems that represent the search space'''

        debug("BUILDING COORD SYSTEM", obj=self,level=3)

        # get the axis names and axis value ranges
        axis_names = []
        axis_val_ranges = []
        for pname, prange in perf_params:
            axis_names.append(pname)
            # BN: why on earth would someone do this?????
            # axis_val_ranges.append(self.__sort(prange))
            axis_val_ranges.append(prange)

        for pname, prange in cmdline_params:
            axis_names.append('__cmdline_' + pname)
            axis_val_ranges.append(prange)

        self.num_params=len(axis_names)
        self.num_configs=1
        self.num_bin=0
        self.num_categorical = 0
        self.num_int=self.num_params

        ptype=[]
        for vals in axis_val_ranges:
            self.num_configs=self.num_configs*len(vals)
            ptype.append('I')
            if type(vals[0]) == bool:
                self.num_bin=self.num_bin+1
                ptype[len(ptype)-1]=('B')
            if type(vals[0]) == str:
                self.num_categorical = self.num_categorical+1

        self.num_int -= self.num_bin
        self.num_int -= self.num_categorical

        info('Search_Space           = %1.3e' % self.num_configs)
        info('Number_of_Parameters   = %02d' % self.num_params)
        info('Numeric_Parameters     = %02d' % self.num_int)
        info('Binary_Parameters      = %02d' % self.num_bin)
        info('Categorical_Parameters = %02d' % self.num_categorical)
        
        sys.stderr.write('%s\n'% Globals().configfile)

        return (axis_names, axis_val_ranges)

