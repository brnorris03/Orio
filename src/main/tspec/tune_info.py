#
# The class definitions for tuning information and its generator
#

import StringIO, sys, os, tokenize
from orio.main.util.globals import *

#--------------------------------------------------------------

class TuningInfo:
    '''
    Tuning information data structure created based on the information in the tuning
    specification.
    '''

    def __init__(self, build_info, pcount_info, search_info, pparam_info, iparam_info, 
                 ivar_info, ptest_code_info):
        '''To instantiate the tuning information'''

        # unpack all information
        build_cmd, batch_cmd, status_cmd, num_procs = build_info
        pcount_method, pcount_reps, random_seed = pcount_info
        search_algo, search_time_limit, search_total_runs, search_opts = search_info
        pparam_params, pparam_constraints = pparam_info
        iparam_params, iparam_constraints = iparam_info
        ivar_decls, ivar_decl_file, ivar_init_file = ivar_info
        ptest_skeleton_code_file, = ptest_code_info

        # build arguments
        self.build_cmd = build_cmd        # command for compiling the generated code
        self.batch_cmd = batch_cmd        # command for requesting a batch job
        self.status_cmd = status_cmd      # command for checking status of submitted batch
        self.num_procs = num_procs        # the number of processes used to run the test driver

        # performance counter arguments
        self.pcount_method = pcount_method             # default: 'basic timer' --> in microseconds
        self.pcount_reps = pcount_reps                 # default: 1
        self.random_seed = random_seed                 # default: None

        # search arguments
        self.search_algo = search_algo                 # default: 'Exhaustive'
        self.search_time_limit = search_time_limit     # default: -1
        self.search_total_runs = search_total_runs     # default: -1
        self.search_opts = search_opts                 # default: []

        # performance parameters
        self.pparam_params = pparam_params             # default: []
        self.pparam_constraints = pparam_constraints   # default: []

        # input parameters
        self.iparam_params = iparam_params             # default: []
        self.iparam_constraints = iparam_constraints   # default: []

        # input variables
        self.ivar_decls = ivar_decls               # user specified or None
        self.ivar_decl_file = ivar_decl_file       # user specified or None
        self.ivar_init_file = ivar_init_file       # user specified or None

        # performance-test code
        self.ptest_skeleton_code_file = ptest_skeleton_code_file    # default: None

    #-----------------------------------------------------------

    def __str__(self):
        '''Return a string representation for this instance'''
        return repr(self)

    def __repr__(self):
        '''Return a string representation for this instance (for debugging).'''
        s = ''
        s += '------------------\n'
        s += ' tuning info      \n'
        s += '------------------\n'
        s += ' build command: %s \n' % self.build_cmd
        s += ' batch command: %s \n' % self.batch_cmd
        s += ' status command: %s \n' % self.status_cmd
        s += ' num-processors: %s \n' % self.num_procs
        s += ' perf-counting method: %s \n' % self.pcount_method
        s += ' perf-counting repetitions: %s \n' % self.pcount_reps
        s += ' search algorithm: %s \n' % self.search_algo
        s += ' search time limit: %s \n' % self.search_time_limit
        s += ' search total runs: %s \n' % self.search_total_runs
        s += ' search options: \n'
        for id_name, rhs in self.search_opts:
            s += '    %s: %s \n' % (id_name, rhs)
        s += ' perf-parameter parameters: \n'
        for id_name, rhs in self.pparam_params:
            s += '    %s: %s \n' % (id_name, rhs)
        s += ' perf-parameter constraints: \n'
        for id_name, rhs in self.pparam_constraints:
            s += '    %s: %s \n' % (id_name, rhs)
        s += ' input-parameter parameters: \n'
        for id_name, rhs in self.iparam_params:
            s += '    %s: %s \n' % (id_name, rhs)
        s += ' input-parameter constraints: \n'
        for id_name, rhs in self.iparam_constraints:
            s += '    %s: %s \n' % (id_name, rhs)
        s += ' input-variable declarations: \n'
        for is_static, dtype, id_name, ddims, rhs in self.ivar_decls:
            modifier = 'dynamic'
            if is_static: modifier = 'static'
            s += ('   %s %s %s %s = %s \n' %
                      (modifier, dtype, id_name, ddims, rhs))
        s += ' input-variable declaration file: %s \n' % self.ivar_decl_file
        s += ' input-variable initialization file: %s \n' % self.ivar_init_file
        s += ' performance-test skeleton code file: %s \n' % self.ptest_skeleton_code_file
        return s

#--------------------------------------------------------------

class TuningInfoGen:
    '''A generator for tuning information'''

    def __init__(self):
        '''To instantiate a generator for tuning information'''
        pass

    #-----------------------------------------------------------

    def __extractVars(self, code):
        '''Return all variables that are present in the given code'''
                
        # tokenize the given expression code
        gtoks = tokenize.generate_tokens(StringIO.StringIO(code).readline)
        
        # iterate over each token and replace any matching token with its corresponding value
        vnames = []
        for toknum, tokval, _, _, _ in gtoks:
            if toknum == tokenize.NAME:
                vnames.append(tokval)
                
        # return all the found variable names
        return vnames
    
    #-----------------------------------------------------------

    def __genBuildInfo(self, stmt_seq, def_line_no):
        '''
        To generate information about the make command or compilers for compiling
        the performance testing code
        '''

        # all expected argument names
        BUILDCMD = 'build_command'
        BATCHCMD = 'batch_command'
        STATUSCMD = 'status_command'
        NUMPROCS = 'num_procs'
        
        # all expected build information
        build_cmd = None
        batch_cmd = None
        status_cmd = None
        num_procs = 1

        # iterate over each statement
        for stmt in stmt_seq:

            # get the statement keyword and its line number
            keyw = stmt[0]
            line_no = stmt[1]
            
            # skip all 'let' statements, and capture any unexpected statements
            if keyw == 'let':
                continue
            if keyw not in ('arg',):
                err('orio.main.tspec.tune_info: %s: unexpected statement type: "%s"' % (line_no, keyw))
                

            # unpack the statement
            _, _, (id_name, id_line_no), (rhs, rhs_line_no) = stmt
            
            # unknown argument name
            if id_name not in (BUILDCMD, BATCHCMD, STATUSCMD, NUMPROCS):
                err('orio.main.tspec.tune_info: %s: unknown build argument: "%s"' % (id_line_no, id_name))
                

            # evaluate the build command
            if id_name == BUILDCMD:
                if not isinstance(rhs, str):
                    err('orio.main.tspec.tune_info: %s: build command in build section must be a string' % rhs_line_no)
                    
                build_cmd = rhs

            # evaluate the batch command
            elif id_name == BATCHCMD:
                if not isinstance(rhs, str):
                    err('orio.main.tspec.tune_info: %s: batch command in build section must be a string' % rhs_line_no)
                    
                batch_cmd = rhs

            # evaluate the status command
            elif id_name == STATUSCMD:
                if not isinstance(rhs, str):
                    err('orio.main.tspec.tune_info: %s: status command in build section must be a string' % rhs_line_no)
                    
                status_cmd = rhs

            # evaluate the number of processors
            elif id_name == NUMPROCS:
                if not isinstance(rhs, int) or rhs <= 0:
                    warn('orio.main.tspec.tune_info:%s: number of processors in build section must be a positive integer' % rhs_line_no)
                    
                num_procs = rhs

        # return all build information
        return (build_cmd, batch_cmd, status_cmd, num_procs)

    #-----------------------------------------------------------

    def __genPerfCounterInfo(self, stmt_seq, def_line_no):
        '''To generate information about the performance counting techniques'''

        # all expected argument names
        METHOD = 'method'
        REPS = 'repetitions'
        RANDOM_SEED = 'random_seed'

        # all expected performance counting information
        pcount_method = None
        pcount_reps = None
        random_seed = None

        # iterate over each statement
        for stmt in stmt_seq:

            # get the statement keyword and its line number
            keyw = stmt[0]
            line_no = stmt[1]
            
            # skip all 'let' statements, and capture any unexpected statements
            if keyw == 'let':
                continue
            if keyw not in ('arg',):
                err('orio.main.tspec.tune_info: %s: unexpected statement type: "%s"' % (line_no, keyw))
                

            # unpack the statement
            _, _, (id_name, id_line_no), (rhs, rhs_line_no) = stmt
            
            # unknown argument name
            if id_name not in (METHOD, REPS, RANDOM_SEED):
                err('orio.main.tspec.tune_info: %s: unknown performance counter argument: "%s"' % (id_line_no, id_name))
                

            # evaluate build command
            if id_name == METHOD:
                if not isinstance(rhs, str):
                    err('orio.main.tspec.tune_info: %s: performance counting method must be a string' % rhs_line_no)
                    
                pcount_method = rhs

            # evaluate performance counting repetitions
            elif id_name == REPS:
                if not isinstance(rhs, int) or rhs <= 0:
                    warn('orio.main.tspec.tune_info: %s: performance counting repetitions must be a positive integer' % rhs_line_no)
                    
                pcount_reps = rhs
                
            # user-specified random seed (otherwhise non-repeatable value based on time is used)
            elif id_name == RANDOM_SEED:
                if not isinstance(rhs, int):
                    warn('orio.main.tspec.tune_info: %s: performance counting repetitions must be an integer' % rhs_line_no)
                random_seed = rhs
        
        # return all performance counting information
        return (pcount_method, pcount_reps, random_seed)

    #-----------------------------------------------------------

    def __genSearchInfo(self, stmt_seq, def_line_no):
        '''To generate information about the search technique used to explore the search space'''

        # all expected argument names
        ALGO = 'algorithm'
        TLIMIT = 'time_limit'
        TRUNS = 'total_runs'

        # all expected search information
        search_algo = None
        search_time_limit = None
        search_total_runs = None
        search_opts = []
        
        # iterate over each statement
        for stmt in stmt_seq:

            # get the statement keyword and its line number
            keyw = stmt[0]
            line_no = stmt[1]
            
            # skip all 'let' statements, and capture any unexpected statements
            if keyw == 'let':
                continue
            if keyw not in ('arg',):
                err('orio.main.tspec.tune_info: %s: unexpected statement type: "%s"' % (line_no, keyw))
                
            # unpack the statement
            _, _, (id_name, id_line_no), (rhs, rhs_line_no) = stmt

            # unknown argument name
            if id_name not in (ALGO, TLIMIT, TRUNS):
                if search_algo == None or not id_name.startswith(search_algo.lower() + '_'):
                    err('orio.main.tspec.tune_info: %s: unknown search argument: "%s"' % (id_line_no, id_name))                    

            # evaluate build command
            if id_name == ALGO:
                if not isinstance(rhs, str):
                    err('orio.main.tspec.tune_info: %s: search algorithm must be a string' % rhs_line_no)
                    
                search_algo = rhs

            # evaluate search time limit
            elif id_name == TLIMIT:
                if (not isinstance(rhs, int) and not isinstance(rhs, float)) or rhs <= 0:
                    err('orio.main.tspec.tune_info: %s: search time limit must be a positive number' % rhs_line_no)
                    
                search_time_limit = rhs

            # evaluate the total number of search runs
            elif id_name == TRUNS:
                if not isinstance(rhs, int) or rhs <= 0:
                    warn('orio.main.tspec.tune_info: %s: total number of search runs must be a positive number'  % rhs_line_no)
                    
                search_total_runs = rhs

            # evaluate all other algorithm-specific arguments
            elif search_algo != None and id_name.startswith(search_algo.lower() + '_'):
                id_name_orig = id_name
                id_name = id_name[len(search_algo)+1:]
                if id_name == '':
                    warn('orio.main.tspec.tune_info: %s: invalid algorithm-specific argument name: "%s"'  % (id_line_no, id_name_orig))
                    
                search_opts.append((id_name, rhs))

        # return all search information
        return (search_algo, search_time_limit, search_total_runs, search_opts)

    #-----------------------------------------------------------

    def __genPerfParamsInfo(self, stmt_seq, def_line_no):
        '''To generate information about the performance parameters used in the code transformation'''

        # all expected performance parameters
        pparam_params = []
        pparam_constraints = []

        # iterate over each statement
        for stmt in stmt_seq:

            # get the statement keyword and its line number
            keyw = stmt[0]
            line_no = stmt[1]
            
            # skip all 'let' statements, and capture any unexpected statements
            if keyw == 'let':
                continue
            if keyw not in ('param', 'constraint'):
                err('orio.main.tspec.tune_info: %s: unexpected statement type: "%s"' % (line_no, keyw))
                
            # evaluate parameter
            if keyw == 'param':
                _, _, (id_name, id_line_no), is_range, (rhs, rhs_line_no) = stmt
                if not is_range:
                    rhs = [rhs]
                pparam_params.append((id_name, rhs))

            # evaluate constraints
            elif keyw == 'constraint':
                _, _, (id_name, id_line_no), (rhs, rhs_line_no) = stmt
                pparam_constraints.append((id_name, rhs))

        # return all performance parameter information
        return (pparam_params, pparam_constraints)

    #-----------------------------------------------------------

    def __genInputParamsInfo(self, stmt_seq, def_line_no):
        '''To generate information about the input parameters used in the input variables'''

        # all expected input parameters
        iparam_params = []
        iparam_constraints = []

        # iterate over each statement
        for stmt in stmt_seq:

            # get the statement keyword and its line number
            keyw = stmt[0]
            line_no = stmt[1]
            
            # skip all 'let' statements, and capture any unexpected statements
            if keyw == 'let':
                continue
            if keyw not in ('param', 'constraint'):
                err('orio.main.tspec.tune_info: %s: unexpected statement type: "%s"' % (line_no, keyw))
                
            # evaluate parameter
            if keyw == 'param':
                _, _, (id_name, id_line_no), is_range, (rhs, rhs_line_no) = stmt
                if not is_range:
                    rhs = [rhs]
                iparam_params.append((id_name, rhs))

            # evaluate constraints
            elif keyw == 'constraint':
                _, _, (id_name, id_line_no), (rhs, rhs_line_no) = stmt
                iparam_constraints.append((id_name, rhs))

        # return all input parameter information
        return (iparam_params, iparam_constraints)

    #-----------------------------------------------------------

    def __genInputVarsInfo(self, stmt_seq, def_line_no):
        '''To generate information about the input variables'''

        # all expected argument names
        DECL_FILE = 'decl_file'
        INIT_FILE = 'init_file'

        # all input variable information
        ivar_decls = []
        ivar_decl_file = None
        ivar_init_file = None

        # iterate over each statement
        for stmt in stmt_seq:

            # get the statement keyword and its line number
            keyw = stmt[0]
            line_no = stmt[1]
            
            # skip all 'let' statements, and capture any unexpected statements
            if keyw == 'let':
                continue
            if keyw not in ('arg', 'decl'):
                err('orio.main.tspec.tune_info: %s: unexpected statement type: "%s"' % (line_no, keyw))
                
            # evaluate arguments
            if keyw == 'arg':
                
                # unpack the statement
                _, _, (id_name, id_line_no), (rhs, rhs_line_no) = stmt
                
                # declaration code
                if id_name == DECL_FILE:
                    if not isinstance(rhs, str):
                        err('orio.main.tspec.tune_info: %s: declaration file must be a string' % rhs_line_no)
                        
                    if not os.path.exists(rhs):
                        err('orio.main.tspec.tune_info: %s: cannot find the declaration file: "%s"' % (rhs_line_no, rhs))
                        
                    ivar_decl_file = rhs

                # initialization code
                elif id_name == INIT_FILE:
                    if not isinstance(rhs, str):
                        err('orio.main.tspec.tune_info: %s: initialization file must be a string' % rhs_line_no)
                        
                    if not os.path.exists(rhs):
                        warn('orio.main.tspec.tune_info: %s: cannot find the initialization file: "%s"' % (rhs_line_no, rhs))
                        
                    ivar_init_file = rhs

                # unknown argument name
                else:
                    err('orio.main.tspec.tune_info: %s: unknown input variable argument: "%s"' % (id_line_no, id_name))
                    
            # evaluate declarations
            elif keyw == 'decl':
                _, _, (id_name, id_line_no), type_seq, dim_exp_seq, (rhs, rhs_line_no) = stmt
                type_seq = [t for t,_ in type_seq]
                is_array = len(dim_exp_seq) > 0
                is_static = 'static' in type_seq
                is_dynamic = 'dynamic' in type_seq
                # TODO handle structs

                if is_static and is_dynamic:
                    err('orio.main.tspec.tune_info: %s: a declared variable cannot be both static and dynamic' % line_no)
                    
                if (not is_array) and (is_static or is_dynamic):
                    err('orio.main.tspec.tune_info: %s: static and dynamic types are only for arrays' % line_no)
                    
                if is_array and (not is_static) and (not is_dynamic):
                    err('orio.main.tspec.tune_info: %s: missing static/dynamic type for arrays variable' % line_no)
                    
                if not is_array:
                    is_static = None
                if 'static' in type_seq:
                    type_seq.remove('static')
                if 'dynamic' in type_seq:
                    type_seq.remove('dynamic')
                if len(type_seq) == 0:
                    err('orio.main.tspec.tune_info: %s: missing type name' % line_no)
                    
                if len(type_seq) > 1:
                    err('orio.main.tspec.tune_info: %s: unrecognized type name: "%s"' % (line_no, type_seq[0]))
               
                dtype = type_seq[-1]
                ddims = [d for d,_ in dim_exp_seq]
                ivar_decls.append((is_static, dtype, id_name, ddims, rhs))

        # check how the user wants to declare and initialize the input variables
        if len(ivar_decls) > 0:
            
            # invalid options
            if ivar_decl_file:
                warn(('orio.main.tspec.tune_info:  since input variables are declared in the tuning specification, ' +
                       'the declaration file "%s" is not needed') % ivar_decl_file)
                
            
            # both declarations and initializations are generated by the driver
            if ivar_init_file == None:
                for _,_,iname,_,rhs in ivar_decls:
                    if rhs == None:
                        warn(('orio.main.tspec.tune_info:  missing an initial value in the input variable ' +
                                'declaration of "%s" in the tuning specification') % iname)
                        

            # declarations are generated by the driver; initializations are provided by the user
            else:   
                for _,_,iname,_,rhs in ivar_decls:
                    if rhs != None:
                        warn (('orio.main.tspec.tune_info: since initializations of the input variables are provided ' +
                                'by the user in file "%s", input variable "%s" must be ' +
                                'declared without an initial value') % (ivar_init_file, iname))
                        

        else:
            
            # missing declarations and/or initializations
            if ivar_decl_file == None or ivar_init_file == None:
                warn('orio.main.tspec.tune_info:  missing declarations and/or initializations of the input ' +
                       'variables in the tuning specification.')
                
            
            # both declaration and initialization are provided by the user
            else:
                pass
            
        # return all input variables information
        return (ivar_decls, ivar_decl_file, ivar_init_file)

    #-----------------------------------------------------------

    def __genPerfTestCodeInfo(self, stmt_seq, def_line_no):
        '''To generate information about the performance counting techniques'''

        # all expected argument names
        SKELETON_CODE_FILE = 'skeleton_code_file'

        # all expected performance counting information
        ptest_skeleton_code_file = None

        # iterate over each statement
        for stmt in stmt_seq:

            # get the statement keyword and its line number
            keyw = stmt[0]
            line_no = stmt[1]
            
            # skip all 'let' statements, and capture any unexpected statements
            if keyw == 'let':
                continue
            if keyw not in ('arg',):
                err('orio.main.tspec.tune_info: %s: unexpected statement type: "%s"' % (line_no, keyw))
                
            # unpack the statement
            _, _, (id_name, id_line_no), (rhs, rhs_line_no) = stmt
            
            # unknown argument name
            if id_name not in (SKELETON_CODE_FILE,):
                err('orio.main.tspec.tune_info: %s: unknown performance counter argument: "%s"' % (id_line_no, id_name))
                
            # evaluate skeleton code file
            if id_name == SKELETON_CODE_FILE:
                if not isinstance(rhs, str):
                    warn(('orio.main.tspec.tune_info: %s: filename of the performance-test skeleton code file must be ' +
                            'a string') % rhs_line_no)
                    
                if not os.path.exists(rhs):
                    warn(('orio.main.tspec.tune_info: %s: cannot find the file specified in performance-test ' +
                            'skeleton code file: "%s"') % (rhs_line_no, rhs))
                    
                ptest_skeleton_code_file = rhs

        # return all performance-test code information
        return (ptest_skeleton_code_file,)

    #-----------------------------------------------------------

    def generate(self, stmt_seq):
        '''To generate tuning information from the given sequence of statements'''

        # all expected definition names
        BUILD = 'build'
        PERF_COUNTER = 'performance_counter'
        SEARCH = 'search'
        PERF_PARAMS = 'performance_params'
        INPUT_PARAMS = 'input_params'
        INPUT_VARS = 'input_vars'
        PTEST_CODE = 'performance_test_code'

        # all expected definition information
        build_info = None
        pcount_info = ('basic timer', 1)
        search_info = ('Exhaustive', -1, -1, [])
        pparam_info = ([], [])
        iparam_info = ([], [])
        ivar_info = None
        ptest_code_info = (None, )

        # iterate over each statement
        for stmt in stmt_seq:

            # get the statement keyword and its line number
            keyw = stmt[0]
            line_no = stmt[1]

            # skip all 'let' statements, and capture any unrecognized statements
            if keyw == 'let':
                continue
            if keyw != 'def':
                warn('orio.main.tspec.tune_info: unrecognized keyword: "%s"' % keyw)
                

            # unpack the statement
            _, _, (dname, dname_line_no), body_stmt_seq = stmt

            # unknown definition name
            if dname not in (BUILD, PERF_COUNTER, SEARCH, PERF_PARAMS, INPUT_PARAMS, 
                             INPUT_VARS, PTEST_CODE):
                err('orio.main.tspec.tune_info: %s: unknown definition name: "%s"' % (dname_line_no, dname))
                
            
            # build definition
            if dname == BUILD:
                (build_cmd, batch_cmd, status_cmd,
                 num_procs) = self.__genBuildInfo(body_stmt_seq, line_no)
                if build_cmd == None:
                    err('orio.main.tspec.tune_info: %s: missing build command in the build section' % line_no)
                    
                if ((batch_cmd != None and status_cmd == None) or
                    (batch_cmd == None and status_cmd != None)):
                    warn(('orio.main.tspec.tune_info: %s: both batch and status commands in build section ' +
                            'must not be empty') % line_no)
                    
                if batch_cmd == None and num_procs > 1:
                    warn(('orio.main.tspec.tune_info: %s: number of processors in build section must be greater than ' +
                            'one for non-batch (or non-parallel) search') % line_no)
                    
                build_info = (build_cmd, batch_cmd, status_cmd, num_procs)

            # performance counter definition
            elif dname == PERF_COUNTER:
                pcount_method, pcount_reps, random_seed = self.__genPerfCounterInfo(body_stmt_seq, line_no)
                default_p_method, default_p_reps = pcount_info
                if pcount_method == None:
                    pcount_method = default_p_method
                if pcount_reps == None:
                    pcount_reps = default_p_reps
                pcount_info = (pcount_method, pcount_reps, random_seed)
                
            # search definition
            elif dname == SEARCH:
                (search_algo, search_time_limit,
                 search_total_runs, search_opts) = self.__genSearchInfo(body_stmt_seq, line_no)
                default_s_algo, default_s_tlimit, default_s_truns, _ = search_info
                if search_algo == None:
                    search_algo = default_s_algo
                if search_time_limit == None:
                    search_time_limit = default_s_tlimit
                if search_total_runs == None:
                    search_total_runs = default_s_truns
                search_info = (search_algo, search_time_limit, search_total_runs, search_opts)
            
            # performance parameters definition
            elif dname == PERF_PARAMS:
                pparam_params, pparam_constraints = self.__genPerfParamsInfo(body_stmt_seq, line_no)
                if len(pparam_constraints) > 0 and len(pparam_params) == 0:
                    err('orio.main.tspec.tune_info: %s: constraints require parameters definitions' % dname_line_no)
                    
                pparam_info = (pparam_params, pparam_constraints)
                
            # input parameters definition
            elif dname == INPUT_PARAMS:
                iparam_params, iparam_constraints = self.__genInputParamsInfo(body_stmt_seq, line_no)
                if len(iparam_constraints) > 0 and len(iparam_params) == 0:
                    err('orio.main.tspec.tune_info: %s: constraints require parameters definitions' % dname_line_no)
                    
                iparam_info = (iparam_params, iparam_constraints)

            # input variables definition
            elif dname == INPUT_VARS:
                (ivar_decls, ivar_decl_file,
                 ivar_init_file) = self.__genInputVarsInfo(body_stmt_seq, line_no)
                ivar_info = (ivar_decls, ivar_decl_file, ivar_init_file)

            # performance-test code definition
            elif dname == PTEST_CODE:
                (ptest_skeleton_code_file, ) = self.__genPerfTestCodeInfo(body_stmt_seq, line_no)
                ptest_code_info = (ptest_skeleton_code_file,)

        # check if the build definition is missing
        if build_info == None:
            err('orio.main.tspec.tune_info:  missing build definition in the tuning specification')
            
        # check if the input variables definition is missing
        if ivar_info == None:
            err('orio.main.tspec.tune_info:  missing input variables definition in the tuning specification')
            
        # return the tuning information
        return TuningInfo(build_info, pcount_info, search_info, pparam_info, iparam_info,
                          ivar_info, ptest_code_info)


