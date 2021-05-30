#
# The class definitions for tuning information and its generator
#

import io, sys, os, tokenize
from orio.main.util.globals import *


# --------------------------------------------------------------

class TuningInfo:
    '''
    Tuning information data structure created based on the information in the tuning
    specification.
    '''

    def __init__(self, build_info, pcount_info, power_info, search_info, pparam_info,
                 cmdline_info, iparam_info,
                 ivar_info, ptest_code_info, validation_info, other_info):
        '''
        Tuning parameters specified by the user in the tuning spec.
        '''

        # unpack all information

        pcount_method, pcount_reps, random_seed, timing_array_size = pcount_info
        power_method, power_reps, random_seed, power_array_size = power_info
        search_algo, search_time_limit, search_total_runs, search_use_z3, search_resume, search_opts = search_info
        pparam_params, pparam_constraints = pparam_info
        cmdline_params, cmdline_constraints = cmdline_info
        iparam_params, iparam_constraints = iparam_info
        ivar_decls, ivar_decl_file, ivar_init_file = ivar_info
        ptest_skeleton_code_file, = ptest_code_info
        validation_file, expected_output = validation_info
        if other_info and len(other_info) > 1:
            device_spec_file, _ = other_info
        else:
            device_spec_file = other_info

        # build arguments
        self.pre_build_cmd = build_info.get('prebuild_cmd')  # command to run before invoking build_cmd
        self.build_cmd = build_info.get('build_cmd')  # command for compiling the generated code
        self.post_build_cmd = build_info.get(
            'postbuild_cmd')  # command to run after building (before running) code, it is applied to the executable
        self.cc = build_info.get('cc')  # C compiler needed for timer function
        self.fc = build_info.get('fc')
        self.libs = build_info.get('libs')  # extra libraries for linking
        self.batch_cmd = build_info.get('batch_cmd')  # command for requesting a batch job
        self.status_cmd = build_info.get('status_cmd')  # command for checking status of submitted batch
        self.num_procs = build_info.get('num_procs')  # the number of processes used to run the test driver
        self.timer_file = build_info.get('timer_file')  # user-specified implementation of the getClock() function
        self.post_run_cmd = build_info.get(
            'postrun_cmd')  # command to run after executing timing test (will be passed the executable name and coordinate string as an argument)

        # performance counter arguments
        self.pcount_method = pcount_method  # default: 'basic timer' --> in microseconds
        self.pcount_reps = pcount_reps  # default: 5
        # self.pcount_subreps = pcount_subreps           # mandatory subrepetitions (to enable timing of very small computations), default: 10
        self.random_seed = random_seed  # default: None
        self.timing_array_size = timing_array_size  # default an odd number >= pcount_reps

        self.power_method = power_method
        self.power_reps = power_reps
        self.power_array_size = power_array_size

        # search arguments
        self.search_algo = search_algo  # default: 'Exhaustive'
        self.search_time_limit = search_time_limit  # default: -1
        self.search_total_runs = search_total_runs  # default: -1
        self.search_use_z3 = search_use_z3  # default: False
        self.search_resume = search_resume  # default: False
        self.search_opts = search_opts  # default: []

        # performance parameters
        self.pparam_params = pparam_params  # default: []
        self.pparam_constraints = pparam_constraints  # default: []

        # command-line parameters for each code version
        self.cmdline_params = cmdline_params  # default: []
        self.cmdline_constraints = cmdline_constraints  # default: []

        # input parameters
        self.iparam_params = iparam_params  # default: []
        self.iparam_constraints = iparam_constraints  # default: []

        # input variables
        self.ivar_decls = ivar_decls  # user specified or None
        self.ivar_decl_file = ivar_decl_file  # user specified or None
        self.ivar_init_file = ivar_init_file  # user specified or None

        # performance-test code
        self.ptest_skeleton_code_file = ptest_skeleton_code_file  # default: None

        # validation info
        self.validation_file = validation_file
        self.expected_output = expected_output

        # device spec file
        self.device_spec_file = device_spec_file

    # -----------------------------------------------------------

    def __str__(self):
        '''Return a string representation for this instance'''
        return repr(self)

    def __repr__(self):
        '''Return a string representation for this instance (for debugging).'''
        s = ''
        s += '------------------\n'
        s += ' tuning info      \n'
        s += '------------------\n'
        s += ' pre-build command: %s \n' % self.pre_build_cmd
        s += ' build command: %s \n' % self.build_cmd
        s += ' post-build command: %s\n' % self.post_build_cmd
        s += ' post-run command: %s\n' % self.post_run_cmd
        s += ' C compiler (CC): %s \n' % self.cc
        s += ' libraries to link: %s \n' % self.libs
        s += ' batch command: %s \n' % self.batch_cmd
        s += ' status command: %s \n' % self.status_cmd
        s += ' num-processors: %s \n' % self.num_procs
        s += ' perf-counting method: %s \n' % self.pcount_method
        s += ' perf-counting repetitions: %s \n' % self.pcount_reps
        s += ' number of timing results to store: %s \n ' % self.timing_array_size
        s += ' power measurement method: %s \n' % self.power_method
        s += ' power measurement repetitions: %s \n' % self.power_reps
        s += ' number of power measurements to store: %s \n ' % self.power_array_size
        s += ' timer routine file: %s \n' % self.timer_file
        s += ' search algorithm: %s \n' % self.search_algo
        s += ' search time limit (seconds): %s \n' % self.search_time_limit
        s += ' search total runs: %s \n' % self.search_total_runs
        s += ' search use z3 [True/False]: %s \n' % self.search_use_z3
        s += ' search resume [True/False]: %s\n' % self.search_resume
        s += ' search options: \n'
        for id_name, rhs in self.search_opts:
            s += '    %s: %s \n' % (id_name, rhs)

        s += ' perf-parameter parameters: \n'
        for id_name, rhs in self.pparam_params:
            s += '    %s: %s \n' % (id_name, rhs)
        s += ' perf-parameter constraints: \n'
        for id_name, rhs in self.pparam_constraints:
            s += '    %s: %s \n' % (id_name, rhs)

        s += ' command-line-parameters: \n'
        for id_name, rhs in self.cmdline_params:
            s += '    %s: %s \n' % (id_name, rhs)
        s += ' command-line-constraints: \n'
        for id_name, rhs in self.cmdline_constraints:
            s += '    %s: %s \n' % (id_name, rhs)

        s += ' input-parameter parameters: \n'
        for id_name, rhs in self.iparam_params:
            s += '    %s: %s \n' % (id_name, rhs)
        s += ' input-parameter constraints: \n'
        for id_name, rhs in self.iparam_constraints:
            s += '    %s: %s \n' % (id_name, rhs)
        s += ' input-variable declarations: \n'
        for is_static, is_managed, dtype, id_name, ddims, rhs in self.ivar_decls:
            modifier = 'dynamic'
            if is_static: modifier = 'static'
            if is_managed: modifier = 'managed'
            s += ('   %s %s %s %s = %s \n' %
                  (modifier, dtype, id_name, ddims, rhs))
        s += ' input-variable declaration file: %s \n' % self.ivar_decl_file
        s += ' input-variable initialization file: %s \n' % self.ivar_init_file
        s += ' performance-test skeleton code file: %s \n' % self.ptest_skeleton_code_file
        s += ' validation file: %s \n' % self.validation_file
        s += ' expected output: %s \n' % self.expected_output
        return s


# --------------------------------------------------------------

class TuningInfoGen:
    '''A generator for tuning information'''

    def __init__(self):
        '''To instantiate a generator for tuning information'''
        pass

    # -----------------------------------------------------------

    def __extractVars(self, code):
        '''Return all variables that are present in the given code'''

        # tokenize the given expression code
        gtoks = tokenize.generate_tokens(io.StringIO(code).readline)

        # iterate over each token and replace any matching token with its corresponding value
        vnames = []
        for toknum, tokval, _, _, _ in gtoks:
            if toknum == tokenize.NAME:
                vnames.append(tokval)

        # return all the found variable names
        return vnames

    # -----------------------------------------------------------

    def __genBuildInfo(self, stmt_seq, def_line_no):
        '''
        To generate information about the make command or compilers for compiling
        the performance testing code
        '''

        # all expected argument names
        PREBUILDCMD = 'prebuild_command'
        BUILDCMD = 'build_command'
        POSTBUILDCMD = 'postbuild_command'
        POSTRUNCMD = 'postrun_command'
        LIBS = 'libs'  # TODO: remove (replaced by FLIBS, CLIBS, CXXLIBS)
        CC = 'CC'
        CXX = 'CXX'
        FC = 'FC'
        CFLAGS = 'CFLAGS'
        CXXFLAGS = 'CXXFLAGS'
        FFLAGS = 'FCFLAGS'
        CLIBS = 'CLIBS'
        CXXLIBS = 'CXXLIBS'
        FLIBS = 'FCLIBS'
        BATCHCMD = 'batch_command'
        STATUSCMD = 'status_command'
        NUMPROCS = 'num_procs'
        TIMER_FILE = 'timer_file'

        # all expected build information
        prebuild_cmd = None
        build_cmd = None
        postbuild_cmd = None
        postrun_cmd = None
        cc = 'gcc'  # default C compiler, needed for timer routine
        fc = 'gfortran'
        libs = ''
        batch_cmd = None
        status_cmd = None
        num_procs = 1
        timer_file = None

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
            if id_name not in (
            BUILDCMD, PREBUILDCMD, POSTBUILDCMD, POSTRUNCMD, BATCHCMD, STATUSCMD, NUMPROCS, LIBS, CC, TIMER_FILE):
                err('orio.main.tspec.tune_info: %s: unknown build argument: "%s"' % (id_line_no, id_name))

            # evaluate the pre-build command
            if id_name == PREBUILDCMD:
                if not isinstance(rhs, str):
                    err(
                        'orio.main.tspec.tune_info: %s: prebuild_command in build section must be a string' % rhs_line_no)

                prebuild_cmd = rhs

            # evaluate the build command
            if id_name == BUILDCMD:
                if not isinstance(rhs, str):
                    err('orio.main.tspec.tune_info: %s: build command in build section must be a string' % rhs_line_no)

                build_cmd = rhs

            # evaluate the pre-build command
            if id_name == POSTBUILDCMD:
                if not isinstance(rhs, str):
                    err(
                        'orio.main.tspec.tune_info: %s: postbuild_command in build section must be a string' % rhs_line_no)

                postbuild_cmd = rhs

            if id_name == POSTRUNCMD:
                if not isinstance(rhs, str):
                    err(
                        'orio.main.tspec.tune_info: %s: postrun_command in build section must be a string' % rhs_line_no)

                postrun_cmd = rhs

            # Need C compiler for timing routine (even when generating Fortran)
            elif id_name == CC:
                if not isinstance(rhs, str):
                    err(
                        'orio.main.tspec.tune_info: %s: C compiler in build section (CC variable) must be a string' % rhs_line_no)

                cc = rhs

            elif id_name == FC:
                if not isinstance(rhs, str):
                    err(
                        'orio.main.tspec.tune_info: %s: Fortran compiler in build section (FC variable) must be a string' % rhs_line_no)

                fc = rhs

            # evaluate the libs 
            elif id_name == LIBS:
                if not isinstance(rhs, str):
                    err(
                        'orio.main.tspec.tune_info: %s: link-time libraries in build section must be a string' % rhs_line_no)

                libs = rhs

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
                    warn(
                        'orio.main.tspec.tune_info:%s: number of processors in build section must be a positive integer' % rhs_line_no)

                num_procs = rhs

            # User-specified timer file
            elif id_name == TIMER_FILE:
                if not isinstance(rhs, str) or not os.path.exists(rhs):
                    err('orio.main.tspec.tune_info: %s: specified timer file (%s) not found' % rhs)

                timer_file = rhs

        # return all build information
        return (
        prebuild_cmd, build_cmd, postbuild_cmd, postrun_cmd, batch_cmd, status_cmd, num_procs, libs, cc, fc, timer_file)

    # -----------------------------------------------------------

    def __genPerfCounterInfo(self, stmt_seq, def_line_no):
        '''To generate information about the performance counting techniques'''

        # all expected argument names
        METHOD = 'method'
        REPS = 'repetitions'
        RANDOM_SEED = 'random_seed'
        TIMING_ARRAY_SIZE = 'timing_array_size'

        # all expected performance counting information
        pcount_method = None
        pcount_reps = None
        random_seed = None
        timing_array_size = None

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
            if id_name not in (METHOD, REPS, RANDOM_SEED, TIMING_ARRAY_SIZE):
                err('orio.main.tspec.tune_info: %s: unknown performance counter argument: "%s"' % (id_line_no, id_name))

            # evaluate build command
            if id_name == METHOD:
                if not isinstance(rhs, str):
                    err('orio.main.tspec.tune_info: %s: performance counting method must be a string' % rhs_line_no)

                pcount_method = rhs

            # evaluate performance counting repetitions
            elif id_name == REPS:
                if not isinstance(rhs, int) or rhs <= 0:
                    warn(
                        'orio.main.tspec.tune_info: %s: performance counting repetitions must be a positive integer' % rhs_line_no)

                pcount_reps = rhs

            elif id_name == TIMING_ARRAY_SIZE:
                if not isinstance(rhs, int) or rhs <= 0:
                    warn(
                        'orio.main.tspec.tune_info: %s: performance counting array size must be a positive integer' % rhs_line_no)

                timing_array_size = rhs

            # user-specified random seed (otherwhise non-repeatable value based on time is used)
            elif id_name == RANDOM_SEED:
                if not isinstance(rhs, int):
                    warn(
                        'orio.main.tspec.tune_info: %s: performance counting random seed must be an integer' % rhs_line_no)
                random_seed = rhs

        # return all performance counting information
        return (pcount_method, pcount_reps, random_seed, timing_array_size)

    # -----------------------------------------------------------

    def __genPowerInfo(self, stmt_seq, def_line_no):
        '''To generate information about the performance counting techniques'''

        # all expected argument names
        METHOD = 'method'
        REPS = 'repetitions'
        RANDOM_SEED = 'random_seed'
        POWER_ARRAY_SIZE = 'power_array_size'

        # all expected performance counting information
        power_method = None
        power_reps = None
        random_seed = None
        power_array_size = None

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
            if id_name not in (METHOD, REPS, RANDOM_SEED, POWER_ARRAY_SIZE):
                err('orio.main.tspec.tune_info: %s: unknown power measurement argument: "%s"' % (id_line_no, id_name))

            # evaluate build command
            if id_name == METHOD:
                if not isinstance(rhs, str):
                    err('orio.main.tspec.tune_info: %s: power measurement method must be a string' % rhs_line_no)

                power_method = rhs

            # evaluate performance counting repetitions
            elif id_name == REPS:
                if not isinstance(rhs, int) or rhs <= 0:
                    warn(
                        'orio.main.tspec.tune_info: %s: power measurement repetitions must be a positive integer' % rhs_line_no)

                power_reps = rhs

            elif id_name == POWER_ARRAY_SIZE:
                if not isinstance(rhs, int) or rhs <= 0:
                    warn(
                        'orio.main.tspec.tune_info: %s: power measurement array size must be a positive integer' % rhs_line_no)

                power_array_size = rhs

            # user-specified random seed (otherwhise non-repeatable value based on time is used)
            elif id_name == RANDOM_SEED:
                if not isinstance(rhs, int):
                    warn(
                        'orio.main.tspec.tune_info: %s: power measurement random seed must be an integer' % rhs_line_no)
                random_seed = rhs

        # return all performance counting information
        return (power_method, power_reps, random_seed, power_array_size)

    # -----------------------------------------------------------

    def __genSearchInfo(self, stmt_seq, def_line_no):
        """Generate information about the search technique used to explore the search space"""

        # all expected argument names
        ALGO = 'algorithm'
        TLIMIT = 'time_limit'
        TRUNS = 'total_runs'
        USE_Z3 = 'use_z3'
        RESUME = 'resume'

        # all expected search information
        search_algo = None
        search_time_limit = None
        search_total_runs = None
        search_resume = False
        search_use_z3 = False
        search_opts = []

        cmdline_params = Globals().cmdline.get('search')
        if cmdline_params:  # Handle the command-line --search option
            parts = cmdline_params.split(';')
            stmt_seq = [('arg', 0, (ALGO,0), (parts[0],0))]
            for p in parts[1:]:
                if p.find('=') < 0:
                    err('orio.main.tspec.tune_info: --search command-line argument contains an invalid option: %s' % p)
                lhs,rhs=p.strip().split('=')
                stmt_seq.append(('arg', 0, (lhs,0), (eval(rhs),0)))
             
        # iterate over each statement
        for stmt in stmt_seq:

            # get the tuning spec statement keyword and its line number
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
            if id_name not in (ALGO, TLIMIT, TRUNS, RESUME, USE_Z3):
                if search_algo == None or not id_name.startswith(search_algo.lower() + '_'):
                    err('orio.main.tspec.tune_info: %s: unknown search argument: "%s"' % (id_line_no, id_name))

                    # evaluate search algorithm options
            if id_name == ALGO:
                if not isinstance(rhs, str):
                    err('orio.main.tspec.tune_info: %s: search algorithm must be a string' % rhs_line_no)

                search_algo = rhs

            # evaluate search time limit
            elif id_name == TLIMIT:
                if (not isinstance(rhs, int) and not isinstance(rhs, float)) or rhs <= 0:
                    err(
                        'orio.main.tspec.tune_info: %s: search time limit (seconds) must be a positive number' % rhs_line_no)

                search_time_limit = rhs

            # evaluate the total number of search runs
            elif id_name == TRUNS:
                if not isinstance(rhs, int) or rhs <= 0:
                    warn(
                        'orio.main.tspec.tune_info: %s: total number of search runs must be a positive number' % rhs_line_no)

                search_total_runs = rhs

                # evaluate use_z3 argument
            elif id_name == USE_Z3:
                if not isinstance(rhs, bool):
                    err(
                        'orio.main.tspec.tune_info: %s: search parameter uze_z3 must be either True or False.' % rhs_line_no)

                search_use_z3 = rhs

            # evaluate all other algorithm-specific arguments
            elif search_algo != None and id_name.startswith(search_algo.lower() + '_'):
                id_name_orig = id_name
                id_name = id_name[len(search_algo) + 1:]
                if id_name == '':
                    warn('orio.main.tspec.tune_info: %s: invalid algorithm-specific argument name: "%s"' % (
                    id_line_no, id_name_orig))

                search_opts.append((id_name, rhs))

            # Resume search if possible starting with the last coordinate
            elif id_name == RESUME:
                if not isinstance(rhs, bool):
                    warn('orio.main.tspec.tune_info: %s: search resume option must be True or False' % rhs_line_no)
                    search_resume = False
                else:
                    search_resume = rhs

        # return all search information
        return (search_algo, search_time_limit, search_total_runs, search_use_z3, search_resume, search_opts)

    # -----------------------------------------------------------

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
            if keyw not in ('param', 'option', 'constraint'):
                err('orio.main.tspec.tune_info: %s: unexpected statement type: "%s"' % (line_no, keyw))

            # evaluate parameter
            if keyw in ('param', 'option'):
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

    # -----------------------------------------------------------

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

    # -----------------------------------------------------------

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
                        err('orio.main.tspec.tune_info: %s: cannot find the declaration file: "%s"' % (
                        rhs_line_no, rhs))

                    ivar_decl_file = rhs

                # initialization code
                elif id_name == INIT_FILE:
                    if not isinstance(rhs, str):
                        err('orio.main.tspec.tune_info: %s: initialization file must be a string' % rhs_line_no)

                    if not os.path.exists(rhs):
                        warn('orio.main.tspec.tune_info: %s: cannot find the initialization file: "%s"' % (
                        rhs_line_no, rhs))

                    ivar_init_file = rhs

                # unknown argument name
                else:
                    err('orio.main.tspec.tune_info: %s: unknown input variable argument: "%s"' % (id_line_no, id_name))

            # evaluate declarations
            elif keyw == 'decl':
                _, _, (id_name, id_line_no), type_seq, dim_exp_seq, (rhs, rhs_line_no) = stmt
                type_seq = [t for t, _ in type_seq]
                is_array = len(dim_exp_seq) > 0
                is_static = 'static' in type_seq
                is_dynamic = 'dynamic' in type_seq
                is_managed = 'managed' in type_seq
                # TODO handle structs

                if is_static and is_dynamic:
                    err(
                        'orio.main.tspec.tune_info: %s: a declared variable cannot be both static and dynamic' % line_no)

                if is_static and is_managed:
                    err(
                        'orio.main.tspec.tune_info: %s: a declared variable cannot be both static and managed' % line_no)

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
                if 'managed' in type_seq:
                    type_seq.remove('managed')
                if len(type_seq) == 0:
                    err('orio.main.tspec.tune_info: %s: missing type name' % line_no)

                if len(type_seq) > 2:
                    err('orio.main.tspec.tune_info: %s: unrecognized type name: "%s"' % (line_no, type_seq[0]))

                if len(type_seq) == 1:
                    dtype = type_seq[-1]
                else:
                    dtype = type_seq[-2] + ' ' + type_seq[-1]
                ddims = [d for d, _ in dim_exp_seq]
                ivar_decls.append((is_static, is_managed, dtype, id_name, ddims, rhs))

        # check how the user wants to declare and initialize the input variables
        if len(ivar_decls) > 0:

            # invalid options
            if ivar_decl_file:
                warn(('orio.main.tspec.tune_info:  since input variables are declared in the tuning specification, ' +
                      'the declaration file "%s" is not needed') % ivar_decl_file)

            # both declarations and initializations are generated by the driver
            if ivar_init_file == None:
                for _, _, _, iname, _, rhs in ivar_decls:
                    if rhs == None:
                        warn(('orio.main.tspec.tune_info:  missing an initial value in the input variable ' +
                              'declaration of "%s" in the tuning specification') % iname)


            # declarations are generated by the driver; initializations are provided by the user
            else:
                for _, _, _, iname, _, rhs in ivar_decls:
                    if rhs != None:
                        warn(('orio.main.tspec.tune_info: since initializations of the input variables are provided ' +
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

    # -----------------------------------------------------------

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
                    warn(
                        ('orio.main.tspec.tune_info: %s: filename of the performance-test skeleton code file must be ' +
                         'a string') % rhs_line_no)

                if not os.path.exists(rhs):
                    warn(('orio.main.tspec.tune_info: %s: cannot find the file specified in performance-test ' +
                          'skeleton code file: "%s"') % (rhs_line_no, rhs))

                ptest_skeleton_code_file = rhs

        # return all performance-test code information
        return (ptest_skeleton_code_file,)

    # -----------------------------------------------------------

    def __genValidationInfo(self, stmt_seq, def_line_no):
        '''To generate information about the input variables'''

        VALIDATION_FILE = 'validation_file'
        EXPECTED_OUTPUT = 'expected_output'

        validation_file = None
        expected_output = None

        # iterate over each statement
        for stmt in stmt_seq:
            # get the statement keyword and its line number
            keyw = stmt[0]
            line_no = stmt[1]

            # capture any unexpected statements
            if keyw not in ('arg'):
                err('orio.main.tspec.tune_info: %s: unexpected statement type: "%s"' % (line_no, keyw))

            # evaluate arguments
            if keyw == 'arg':

                # unpack the statement
                _, _, (id_name, id_line_no), (rhs, rhs_line_no) = stmt

                # declaration code
                if id_name == VALIDATION_FILE:
                    if not isinstance(rhs, str):
                        err('orio.main.tspec.tune_info: %s: validation file must be a string' % rhs_line_no)

                    if not os.path.exists(rhs):
                        err('orio.main.tspec.tune_info: %s: cannot find the validation file: "%s"' % (rhs_line_no, rhs))

                    validation_file = rhs

                # expected output
                elif id_name == EXPECTED_OUTPUT:
                    expected_output = rhs

                # unknown argument name
                else:
                    err('orio.main.tspec.tune_info: %s: unknown validation argument: "%s"' % (id_line_no, id_name))

        if validation_file == None:
            err('orio.main.tspec.tune_info: missing validation file.')

        return (validation_file, expected_output)

    # -----------------------------------------------------------

    def __genOtherInfo(self, stmt_seq, def_line_no):
        '''Generate other miscellaneous info such as GPU device identification, etc. '''

        DEVICE = 'device_spec_file'

        device_spec_file = None

        # iterate over each statement
        for stmt in stmt_seq:
            # get the statement keyword and its line number
            keyw = stmt[0]
            line_no = stmt[1]

            # capture any unexpected statements
            if keyw not in ('arg'):
                err('orio.main.tspec.tune_info: %s: unexpected statement type in [Other] section: "%s"' % (
                line_no, keyw))

            # evaluate arguments
            if keyw == 'arg':

                # unpack the statement
                _, _, (id_name, id_line_no), (rhs, rhs_line_no) = stmt

                # declaration code
                if id_name == DEVICE:
                    if not isinstance(rhs, str):
                        err('orio.main.tspec.tune_info: %s: device specification must be a path (string)' % rhs_line_no)

                    if not os.path.exists(rhs):
                        err('orio.main.tspec.tune_info: %s: cannot find the device specification file: "%s"' % (
                        rhs_line_no, rhs))

                    device_spec_file = rhs

                # unknown argument name
                else:
                    err('orio.main.tspec.tune_info: %s: unknown argument in [Other] section: "%s"' % (
                    id_line_no, id_name))

        return (device_spec_file, None)

    # -----------------------------------------------------------

    def generate(self, stmt_seq):
        '''To generate tuning information from the given sequence of statements'''

        # all expected definition names
        BUILD = 'build'
        PERF_COUNTER = 'performance_counter'
        POWER = 'power'
        SEARCH = 'search'
        PERF_PARAMS = 'performance_params'
        CMDLINE_PARAMS = 'cmdline_params'
        INPUT_PARAMS = 'input_params'
        INPUT_VARS = 'input_vars'
        PTEST_CODE = 'performance_test_code'
        VALIDATION = 'validation'
        OTHER = 'other'

        # all expected definition information
        build_info = {'build_cmd': 'gcc -O3', 'libs': ''}
        pcount_info = ('basic timer', 5, None, None)
        power_info = ('none', 5, None, None)
        search_info = ('Exhaustive', -1, -1, False, False, [])
        pparam_info = ([], [])
        cmdline_info = ([], [])
        iparam_info = ([], [])
        ivar_info = None
        ptest_code_info = (None,)
        validation_info = (None, None)
        other_info = None

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
            if dname not in (BUILD, PERF_COUNTER, POWER, SEARCH, PERF_PARAMS, CMDLINE_PARAMS, INPUT_PARAMS,
                             INPUT_VARS, PTEST_CODE, VALIDATION, OTHER):
                err('orio.main.tspec.tune_info: %s: unknown definition name: "%s"' % (dname_line_no, dname))

            # build definition
            if dname == BUILD:
                (prebuild_cmd, build_cmd, postbuild_cmd, postrun_cmd, batch_cmd, status_cmd,
                 num_procs, libs, cc, fc, timer_file) = self.__genBuildInfo(body_stmt_seq, line_no)
                if build_cmd == None:
                    err('orio.main.tspec.tune_info: %s: missing build command in the build section' % line_no)

                if ((batch_cmd != None and status_cmd == None) or
                        (batch_cmd == None and status_cmd != None)):
                    warn(('orio.main.tspec.tune_info: %s: both batch and status commands in build section ' +
                          'must not be empty') % line_no)

                if batch_cmd == None and num_procs > 1:
                    warn(('orio.main.tspec.tune_info: %s: number of processors in build section must be greater than ' +
                          'one for non-batch (or non-parallel) search') % line_no)

                build_info = {'prebuild_cmd': prebuild_cmd,
                              'build_cmd': build_cmd,
                              'postbuild_cmd': postbuild_cmd,
                              'postrun_cmd': postrun_cmd,
                              'batch_cmd': batch_cmd,
                              'status_cmd': status_cmd,
                              'num_procs': num_procs,
                              'libs': libs,
                              'cc': cc,
                              'fc': fc,
                              'timer_file': timer_file}

            # performance counter definition
            elif dname == PERF_COUNTER:
                pcount_method, pcount_reps, random_seed, timing_array_size = self.__genPerfCounterInfo(body_stmt_seq,
                                                                                                       line_no)
                default_p_method, default_p_reps, _, _ = pcount_info
                if pcount_method == None:
                    pcount_method = default_p_method
                if pcount_reps == None:
                    pcount_reps = default_p_reps
                if not timing_array_size:
                    timing_array_size = pcount_reps + (pcount_reps + 1) % 2
                pcount_info = (pcount_method, pcount_reps, random_seed, timing_array_size)

            # Power/energy measurement
            elif dname == POWER:
                power_method, power_reps, random_seed, power_array_size = self.__genPowerInfo(body_stmt_seq, line_no)
                default_p_method, default_p_reps, _, _ = power_info
                if power_method == None:
                    power_method = default_p_method
                if power_reps == None:
                    power_reps = default_p_reps
                if not power_array_size:
                    power_array_size = power_reps + (power_reps + 1) % 2
                power_info = (power_method, power_reps, random_seed, power_array_size)


            # search definition
            elif dname == SEARCH:
                (search_algo, search_time_limit,
                 search_total_runs, search_use_z3, search_resume,
                 search_opts) = self.__genSearchInfo(body_stmt_seq, line_no)
                default_s_algo, default_s_tlimit, default_s_truns, search_use_z3, default_s_resume, _ = search_info
                if search_algo == None:
                    search_algo = default_s_algo
                if search_time_limit == None:
                    search_time_limit = default_s_tlimit
                if search_total_runs == None:
                    search_total_runs = default_s_truns
                if search_resume == None:
                    search_resume = False
                search_info = (search_algo, search_time_limit, search_total_runs, search_use_z3,
                               search_resume, search_opts)

            # performance parameters definition
            elif dname == PERF_PARAMS:
                pparam_params, pparam_constraints = self.__genPerfParamsInfo(body_stmt_seq, line_no)
                if len(pparam_constraints) > 0 and len(pparam_params) == 0:
                    err('orio.main.tspec.tune_info: %s: constraints require parameters definitions' % dname_line_no)

                pparam_info = (pparam_params, pparam_constraints)
                debug("tune_info TuningInfo pparam_params" + str(pparam_params))

            elif dname == CMDLINE_PARAMS:
                cmdline_params, cmdline_constraints = self.__genPerfParamsInfo(body_stmt_seq, line_no)
                if len(cmdline_constraints) > 0 and len(cmdline_params) == 0:
                    err(
                        'orio.main.tspec.tune_info: %s: command line constraints require command line parameters definitions' % dname_line_no)

                cmdline_info = (cmdline_params, cmdline_constraints)
                debug("tune_info TuningInfo cmdline_params" + str(cmdline_params))


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
                (ptest_skeleton_code_file,) = self.__genPerfTestCodeInfo(body_stmt_seq, line_no)
                ptest_code_info = (ptest_skeleton_code_file,)

            # performance-test code definition
            elif dname == VALIDATION:
                validation_info = self.__genValidationInfo(body_stmt_seq, line_no)

            # Other misc. parameters
            elif dname == OTHER:
                other_info = self.__genOtherInfo(body_stmt_seq, line_no)

        # check if the input variables definition is missing
        if ivar_info == None:
            err('orio.main.tspec.tune_info:  missing input variables definition in the tuning specification')

        # return the tuning information
        return TuningInfo(build_info, pcount_info, power_info, search_info, pparam_info, cmdline_info,
                          iparam_info, ivar_info, ptest_code_info, validation_info, other_info)
