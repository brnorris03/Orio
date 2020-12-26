#
# To compile and execute the performance-testing code to get the performance cost
#

import os, time, re, datetime, uuid

from orio.main.util.globals import *
import subprocess as sp

# -----------------------------------------------------

perftest_counter = 0


class PerfTestDriver:
    '''
    The performance-testing driver used to compile and execute the testing code
    to get the performance cost
    '''

    # the file names of the testing code (i.e. source and executable)
    __PTEST_FNAME = '_orio_perftest'

    # types of performance-counting methods
    __PCOUNT_BASIC = 'basic timer'  # in microseconds (not accurate, large overhead)
    __PCOUNT_BGP = 'bgp counter'  # in clock cycles (accurate, low overhead)
    __POWER_WATTPROF = 'wattprof'

    # -----------------------------------------------------

    def __init__(self, tinfo, use_parallel_search, language="c", timing_code=''):
        '''To instantiate the performance-testing driver'''

        self.tinfo = tinfo
        self.use_parallel_search = use_parallel_search
        self.compile_time = {}
        self.extra_compiler_opts = ''

        global perftest_counter
        perftest_counter += 1

        if not (language == 'c' or language == 'fortran' or language == 'cuda' or language == 'opencl'):
            err('orio.main.tuner.ptest_driver: unknown output language specified: %s' % language)
        self.language = language

        self.__PTEST_FNAME = Globals().out_prefix + self.__PTEST_FNAME

        if language == 'c' or language == 'opencl':
            self.ext = '.c'
        elif language == 'cuda':
            self.ext = '.cu'
        else:
            self.ext = '.F90'

        self.src_name = self.__PTEST_FNAME + self.ext
        self.src_name2 = self.__PTEST_FNAME + '_preprocessed' + self.ext
        self.original_src_name = self.__PTEST_FNAME + '_original' + self.ext
        self.exe_name = self.__PTEST_FNAME + '.exe'
        self.original_exe_name = self.__PTEST_FNAME + '_original.exe'
        if self.language == 'cuda' or self.language == 'opencl':
            self.obj_name = self.__PTEST_FNAME + '.o'
            self.original_obj_name = self.__PTEST_FNAME + '_original.o'

        if not self.tinfo.timer_file:
            if self.language == 'c':
                self.timer_file = 'timer_cpu.c'
            elif self.language == 'fortran':
                self.timer_file = 'timer_cpu.F90'
            else:
                self.timer_file = None
        else:
            self.timer_file = self.tinfo.timer_file

        self.timer_code = timing_code
        if language == 'fortran':  self.timer_code = ''  # timer routine is embedded in F90 driver

        if self.tinfo.pcount_method not in (self.__PCOUNT_BASIC, self.__PCOUNT_BGP):
            err('orio.main.tuner.ptest_driver:  unknown performance-counting method: "%s"' % self.tinfo.pcount_method)

        if self.tinfo.power_method not in (self.__POWER_WATTPROF, "none"):
            err('orio.main.tuner.ptest_driver:  unknown power measurement method: "%s"' % self.tinfo.pcount_method)

        # get all extra options
        self.extra_compiler_opts = ''
        if self.tinfo.pcount_method == self.__PCOUNT_BGP:
            self.extra_compiler_opts += ' -DBGP_COUNTER'
        self.extra_compiler_opts += ' -DORIO_REPS=%s' % self.tinfo.pcount_reps
        # self.extra_compiler_opts += ' -DORIO_TIMES_ARRAY_SIZE=%s' % self.tinfo.timing_array_size

        # for efficiency
        self.first = True

        # Keep track of failed and succcessful runs
        self.failedRuns = 0
        self.successfulRuns = 0

        # For processing output
        self.resultre = re.compile(r'\w*({.*})')

        pass

    # -----------------------------------------------------

    def __write(self, test_code, perf_params=None):
        '''Write the test code into a file'''

        global perftest_counter
        global last_counter
        suffix = str(perftest_counter)
        last_counter = perftest_counter
        perftest_counter += 1
        self.src_name = self.__PTEST_FNAME + suffix + self.ext
        self.obj_name = self.__PTEST_FNAME + suffix + '.o'
        self.exe_name = self.__PTEST_FNAME + suffix + '.exe'
        paraminfo = '/*\n'
        if perf_params is not None:
            for pname, pval in list(perf_params.items()):
                paraminfo += '%s:%s\n' % (pname, pval)
        paraminfo += '*/'

        try:
            f = open(self.src_name, 'w')
            f.write(paraminfo)
            f.write(test_code)
            f.close()
        except:
            err('orio.main.tuner.ptest_driver: cannot open file for writing: %s' % self.src_name)

        if self.first:
            if self.language != 'cuda' and self.language != 'opencl' and not self.tinfo.timer_file and not (
            os.path.exists(self.timer_file)):
                # Generate the timing routine file
                try:
                    f = open(self.timer_file, 'w')
                    f.write(self.timer_code)
                    f.close()
                except:
                    err('orio.main.tuner.ptest_driver:  cannot open file for writing: timer_cpu.c')

            if not os.path.exists(self.original_src_name):
                try:
                    f = open(self.original_src_name, 'w')
                    f.write(test_code)
                    f.close()
                except:
                    err('orio.main.tuner.ptest_driver:  cannot open file for writing: %s' % self.original_src_name)

        if Globals().meta is not None:
            cmd = ''
            if Globals().out_filename is not None:
                try:
                    cmd = Globals().out_filename
                    cmd = cmd.replace("%iter", str(last_counter))

                except Exception as e:
                    err('orio.main.tuner.ptest_driver: failed to execute the outfile rename: "%s"\n --> %s: %s' \
                        % (Globals().out_filename, e.__class__.__name__, e), doexit=False)

            if perf_params is not None:
                Globals().metadata.update({'LastCounter': last_counter})
                for pname, pval in list(perf_params.items()):
                    a_dict = {pname: pval}
                    Globals().metadata.update(a_dict)
                    Globals().metadata.update({pname: pval})
            ## get filename
            for key in Globals().src_filenames:
                Globals().metadata.update({'SourceName': key})
                Globals().metadata.update({'LastCounter': last_counter})
            try:
                if cmd and not os.path.exists(cmd):
                    os.makedirs(cmd)
            except Exception as e:
                warn('orio.main.tuner.ptest_driver: failed to execute meta update: "%s"\n --> %s: %s' \
                     % (Globals().meta, e.__class__.__name__, e), doexit=False)

        return

    # -----------------------------------------------------

    def __preprocess(self):
        ''' 
        Apply PBound to generated code. 
        Some of the build options (include paths, etc.) are used. 
        '''
        if self.tinfo.pre_build_cmd:
            # apply the pre-build command if defined

            cmd = ('%s %s -o %s %s' % (self.tinfo.pre_build_cmd, self.extra_compiler_opts,
                                       self.src_name2, self.src_name))
            # TODO: log all commands
            status = os.system(cmd)
            if status:
                err('orio.main.tuner.ptest_driver:  failed to apply the pre-build command: "%s"' % cmd)

        else:
            self.src_name2 = self.src_name
        return

    # -----------------------------------------------------

    def __build(self, perf_params=None, coord=None):
        '''Compile the testing code'''

        # compile the timing code (if needed)
        if self.timer_file:
            timer_objfile = self.timer_file[:self.timer_file.rfind('.')] + '.o'
        else:
            timer_objfile = None

        if self.use_parallel_search: timer_objfile = ''

        # build_cmd
        cflags_tag = '@CFLAGS'
        build_cmd = self.tinfo.build_cmd
        if perf_params is not None:
            match_obj = re.search(cflags_tag, build_cmd)
            if match_obj:
                build_cmd = re.sub(cflags_tag, perf_params.get('CFLAGS', ''), build_cmd)
            while True:
                match_obj = None
                match_obj = re.search('@(?P<alphanum>\w*)@', build_cmd)
                if match_obj is None:
                    break
                else:
                    param_val = match_obj.group('alphanum')
                    build_cmd = re.sub(match_obj.group(), str(perf_params.get(param_val, '')), build_cmd)

        if timer_objfile and not os.path.exists(timer_objfile):
            # TODO: Too crude, need to make sure object is newer than source
            cmd = ('%s -O0 -c -o %s %s' % (build_cmd, timer_objfile, self.timer_file))
            info(' compiling timer:\n\t' + cmd)
            status = os.system(cmd)
            if status or not os.path.exists(timer_objfile):
                err('orio.main.tuner.ptest_driver:  failed to compile the timer code: "%s"' % cmd)

        # compile the original code if needed
        if self.first:
            if self.language == 'cuda':
                cmd = ('%s %s -DORIGINAL -o %s -c %s' % (
                build_cmd, self.extra_compiler_opts, self.original_obj_name, self.src_name2))
                info(' compiling the original code:\n\t' + cmd)
                status = os.system(cmd)
                if status:
                    err('orio.main.tuner.ptest_driver: failed to compile the original version of cuda code: "%s"' % cmd)
                cmd = ('%s %s -DORIGINAL -o %s %s' % (
                build_cmd, self.extra_compiler_opts, self.original_exe_name, self.original_obj_name))
            else:
                if timer_objfile and os.path.exists(timer_objfile):
                    cmd = ('%s %s -DORIGINAL -o %s %s %s %s' % (build_cmd, self.extra_compiler_opts,
                                                                self.original_exe_name, self.src_name2,
                                                                timer_objfile, self.tinfo.libs))
                else:
                    cmd = ('%s %s -DORIGINAL -o %s %s %s' % (build_cmd, self.extra_compiler_opts,
                                                             self.original_exe_name, self.src_name2,
                                                             self.tinfo.libs))

            info(' building the original code:\n\t' + cmd)
            status = os.system(cmd)
            if status:
                err('orio.main.tuner.ptest_driver:  failed to compile the original version of the code: "%s"' % cmd)

        # compile the test code
        if self.language == 'cuda':
            cmd = ('%s %s -o %s -c %s' % (build_cmd, self.extra_compiler_opts, self.obj_name, self.src_name))
            info(' compiling test:\n\t' + cmd)
            status = os.system(cmd)
            if status:
                err('orio.main.tuner.ptest_driver: failed to compile the test cuda code: "%s"' % cmd)
            cmd = ('%s %s -o %s %s' % (build_cmd, self.extra_compiler_opts, self.exe_name, self.obj_name))
        elif self.language == 'opencl':
            cmd = ('%s %s -o %s %s %s' % (build_cmd, self.extra_compiler_opts,
                                          self.exe_name, self.src_name2,
                                          self.tinfo.libs))
        else:
            cmd = ('%s %s -o %s %s %s %s' % (build_cmd, self.extra_compiler_opts,
                                             self.exe_name, self.src_name2,
                                             timer_objfile, self.tinfo.libs))
        info(' building test:\n\t' + cmd)

        start = time.time()
        # TODO: log all commands
        status = os.system(cmd)
        elapsed = time.time() - start
        if coord is not None:
            self.compile_time[coord] = elapsed

        if status:
            warn('orio.main.tuner.ptest_driver:  failed to compile the testing code: "%s", skipping test' % cmd)

        if self.tinfo.post_build_cmd:
            # Run the postbuild command
            cmd = ('%s %s' % (self.tinfo.post_build_cmd, self.exe_name))
            # TODO: log all commands
            status = os.system(cmd)
            if status:
                err('orio.main.tuner.ptest_driver:  failed to apply the post-build command: "%s"' % cmd)
        return status

    # -----------------------------------------------------

    def __execute(self, perf_params, coord):
        '''Execute the test to get the performance costs. 
        @param perf_params: a dictionary of current parameter name-value pairs
                            corresponding to a single coordinate in the search space.
        '''
        global last_counter

        Globals().metadata['src_filenames'] = ",".join(Globals().src_filenames)

        # check if the executable exists
        if not os.path.exists(self.exe_name):
            err('orio.main.tuner.ptest_driver:  the executable of the test code does not exist')

        # initialize the performance costs dictionary
        # (indexed by the string representation of the search coordinates)
        # e.g., {'[0,1]':0.2, '[1,1]':0.3}
        perf_costs = {}
        output = None

        # Extract command-line arguments if any
        cmdlineargs = ' '
        for pname, pval in list(perf_params.items()):
            if pname.startswith('__cmdline_'):
                cmdlineargs += pname.replace('__cmdline_', '').strip('"') + ' ' + str(pval) + ' '

        # execute the search process in parallel
        if self.use_parallel_search:
            cmd = '%s %s %s' % (self.tinfo.batch_cmd, self.exe_name, cmdlineargs)
            info(' running test:\n\t' + cmd)
            # TODO: redo this to take output file name
            try:
                f = os.popen(cmd)
                output = f.read()
                f.close()
                # TODO: very bad assumption that the last number out is the batch job name
                jobid = output.strip().split('\n')[-1]
                status_cmd = '%s %s | grep %s | wc -l' % (self.tinfo.status_cmd, jobid, jobid)
                status = '1'
                while status == '1':
                    time.sleep(3)
                    f = os.popen(status_cmd)
                    status = f.read().strip()
                    f.close()
                # TODO: generate an output file, instead of reading the batch-generated file
                outfile = '%s.output' % jobid
                while not os.path.exists(outfile):
                    time.sleep(3)
                f = open(outfile)
                output = f.read()
                f.close()
                if output: perf_costs = eval(output)
            except Exception as e:
                err('orio.main.tuner.ptest_driver: failed to execute the test code: "%s"\n --> %s: %s' % (
                cmd, e.__class__.__name__, e))

        # execute the search sequentially
        else:
            cmd = '%s ./%s %s' % (Globals().pre_cmd, self.exe_name, cmdlineargs)
            info(' running test:\n\t' + cmd)
            try:
                # process = sp.Popen(cmd, 'w', shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE)
                # retCode = process.stdin.close()
                # out = process.stdout.readlines()
                f = os.popen(cmd)
                out = f.readlines()
                f.close()
            except Exception as e:
                self.failedruns += 1
                err('orio.main.tuner.ptest_driver: failed to execute the test code: "%s"\n --> %s: %s' \
                    % (cmd, e.__class__.__name__, e), doexit=False)

            if self.tinfo.post_run_cmd:
                # Run the post-run command from the build section of the tuning spec (not command-line option)
                cmd = ('%s %s "%s"' % (self.tinfo.post_run_cmd, self.exe_name, coord))
                info(' running postrun_command:\n\t' + cmd)
                status = os.system(cmd)
                if status:
                    err('orio.main.tuner.ptest_driver:  failed to run the postrun_command: "%s"' % cmd)

            if Globals().post_cmd is not None:
                try:
                    cmd = Globals().post_cmd
                    uniq = "profile-" + datetime.datetime.now().strftime("%y-%m-%d:%H:%S") + '-' + uuid.uuid4().hex
                    cmd = cmd.replace("%unique", uniq)
                    cmd = cmd.replace("%iter", str(last_counter))
                    cmd = cmd.replace("%exe", self.exe_name)
                    status = os.system(cmd)
                    if status:
                        err(
                            'orio.main.tuner.ptest_driver: failed to execute the post-command: "%s"' % Globals().post_cmd,
                            doexit=False)
                except Exception as e:
                    err('orio.main.tuner.ptest_driver: failed to execute the post-command: "%s"\n --> %s: %s' \
                        % (Globals().post_cmd, e.__class__.__name__, e), doexit=False)

            # Parse the output to get the times (and in some cases, e.g., for GPU code, the data transfer times)
            try:
                if out:
                    # info('out:\n %s' % out)
                    perf_costs = {}
                    perf_costs_reps = []
                    transfers = []
                    for line in out:
                        # info('the line:\n%s' % line)
                        # Output lines have the form {'[coordinate]' : time} or {'[coordinate]' : (time, transfer_time)}
                        # where [coordinate] is a list of indices, e.g., [2,4,1,0,0]
                        if line.strip().startswith('{'):
                            output = line.strip()
                            rep = eval(str(output))
                            key = list(rep.keys())[0]  # the coordinate, e.g., [2,4,1,0,0]
                            if isinstance(rep[key], tuple):  # cases where we have (time, transfer_time) values
                                perf_costs_reps.append(rep[key][0])
                                transfers.append(rep[key][1])
                            else:  # cases where we have just time values
                                perf_costs_reps.append(rep[key])
                                transfers.append(float('inf'))
                            perf_costs[key] = (perf_costs_reps, transfers)
                        else:
                            # warn(errmsg="Error processing test result: %s" % line)
                            parts = line.strip().split('@')
                            rep = eval(str(parts[1]))
                            key = list(rep.keys())[0]  # the coordinate, e.g., [2,4,1,0,0]
                            perf_costs_reps.append(float('inf'))  # time
                            transfers.append(float('inf'))  # transfer time
                            perf_costs[key] = (perf_costs_reps, transfers)
                # if output: perf_costs = eval(str(output))
                self.successfulRuns += 1
            except Exception as e:
                self.failedRuns += 1
                err(
                    'orio.main.tuner.ptest_driver: failed to process test result, command was "%s", output: "%s\n --> %s: %s' %
                    (cmd + cmdlineargs, perf_costs, e.__class__.__name__, str(e)), doexit=False)

        # exit()
        # check if the performance cost is already acquired
        # info(str(perf_costs))
        if not perf_costs:
            err('orio.main.tuner.ptest_driver:  performance testing failed: "%s"' % cmd, doexit=False)
            infpair = (float('inf'), float('inf'))
            for k in list(perf_costs.keys()): perf_costs[k] = infpair

        # compare original and transformed codes' results
        if Globals().validationMode and Globals().executedOriginal:
            cmd = 'diff ./newexec.out ./origexec.out'
            info(' running diff:\n\t' + cmd)
            status = os.system(cmd)
            if status:
                infpair = (float('inf'), float('inf'))
                for k in list(perf_costs.keys()): perf_costs[k] = infpair
                err('orio.main.tuner.ptest_driver: results of transformed code differ from the original: "%s"' % status,
                    doexit=False)

        # return the performance costs dictionary
        return perf_costs

    # -----------------------------------------------------

    def __cleanup(self):
        '''Delete all the generated files'''

        if Globals().keep_temps:
            if self.first: self.first = False
            return

        if self.first:
            for fname in [self.original_src_name, self.original_exe_name, self.timer_file]:
                try:
                    if fname and os.path.exists(fname):
                        os.unlink(fname)
                except:
                    err('orio.main.tuner.ptest_driver: cannot delete file: %s' % fname)
            self.first = False

        for fname in [self.exe_name, self.src_name]:
            try:
                if fname and os.path.exists(fname):
                    os.unlink(fname)
            except:
                err('orio.main.tuner.ptest_driver: cannot delete file: %s' % fname)

    # -----------------------------------------------------

    def run(self, test_code, perf_params=None, coord=None):
        '''To compile and to execute the given testing code to get the performance cost
        @param test_code: the code for testing multiple coordinates in the search space
        @param perf_params: the performance parameters
        @param coord: current coordinate in the parameter space
        @return: a dictionary of the times corresponding to each coordinate in the search space
        '''
        # write the testing code
        self.__write(test_code, perf_params=perf_params)

        # preprocess source code, e.g., run pbound if enabled
        self.__preprocess()

        # compile the testing code
        if self.__build(perf_params=perf_params, coord=coord):
            return {}

        # execute the testing code to get performance costs
        perf_costs = self.__execute(perf_params, coord=coord)

        # delete all generated and used files
        self.__cleanup()

        # return the performance costs
        return perf_costs
