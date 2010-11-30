#
# To compile and execute the performance-testing code to get the performance cost
#

import os, sys, time, random, re

from orio.main.util.globals import *

#-----------------------------------------------------

# define an integer counter used for file naming 
counter = 0

#-----------------------------------------------------

class PerfTestDriver:
    '''
    The performance-testing driver used to compile and execute the testing code
    to get the performance cost
    '''

    # the file names of the testing code (i.e. source and executable)
    __PTEST_FNAME = '_orio_perftest'

    # types of performance-counting methods
    __PCOUNT_BASIC = 'basic timer'   # in microseconds (not accurate, large overhead)
    __PCOUNT_BGP = 'bgp counter'     # in clock cycles (accurate, low overhead)

    #-----------------------------------------------------
    
    def __init__(self, build_cmd, batch_cmd, status_cmd, num_procs, pcount_method, pcount_reps,
                 use_parallel_search, language="c"):
        '''To instantiate the performance-testing driver'''

        self.build_cmd = build_cmd
        self.batch_cmd = batch_cmd
        self.status_cmd = status_cmd
        self.num_procs = num_procs
        self.pcount_method = pcount_method
        self.pcount_reps = pcount_reps
        self.use_parallel_search = use_parallel_search

        global counter
        counter += 1
        if language == 'c': 
            self.src_name = self.__PTEST_FNAME + str(counter) + '.c'
        else:
            self.src_name = self.__PTEST_FNAME + str(counter) + '.F90'
        self.exe_name = self.__PTEST_FNAME + str(counter) + '.exe'

        if self.pcount_method not in (self.__PCOUNT_BASIC, self.__PCOUNT_BGP):
            err('orio.main.tuner.ptest_driver:  unknown performance-counting method: "%s"' % self.pcount_method)

    #-----------------------------------------------------

    def __write(self, test_code):
        '''Write the testing code into a file'''

        try:
            f = open(self.src_name, 'w')
            f.write(test_code)
            f.close()
        except:
            err('orio.main.tuner.ptest_driver:  cannot open file for writing: %s' % self.src_name)

    #-----------------------------------------------------

    def __build(self):
        '''Compile the testing code'''
        
        # get all extra options
        extra_compiler_opts = ''
        if self.pcount_method == self.__PCOUNT_BGP:
            extra_compiler_opts += ' -DBGP_COUNTER'
        extra_compiler_opts += ' -DREPS=%s' % self.pcount_reps
            
        # compile the testing code
        cmd = ('%s %s -o %s %s' % (self.build_cmd, extra_compiler_opts,
                                   self.exe_name, self.src_name))
        info(' compiling:\n\t' + cmd)
        status = os.system(cmd)
        if status:
            err('orio.main.tuner.ptest_driver:  failed to compile the testing code: "%s"' % cmd)

    #-----------------------------------------------------

    def __execute(self):
        '''Execute the executable to get the performance costs'''

        # check if the executable exists
        if not os.path.exists(self.exe_name):
            err('orio.main.tuner.ptest_driver:  the executable of the testing code does not exist')

        # initialize the performance costs dictionary
        # (indexed by the string representation of the search coordinates)
        # e.g., {'[0,1]':0.2, '[1,1]':0.3}
        perf_costs = {}

        # execute the search process in parallel
        if self.use_parallel_search:
            cmd = '%s %s' % (self.batch_cmd, self.exe_name)
            info(' running:\n\t' + cmd)
            # TODO: redo this to take output file name
            try:
                f = os.popen(cmd)    
                output = f.read()
                f.close()
                # TODO: very bad assumption that the last number out is the batch job name
                jobid = output.strip().split('\n')[-1]
                status_cmd = '%s %s | grep %s | wc -l' % (self.status_cmd, jobid, jobid)
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
            except Exception, e:
                err('orio.main.tuner.ptest_driver: failed to execute the testing code: "%s"\n --> %s: %s' % (cmd,e.__class__.__name__, e))
                
        # execute the search sequentially
        else:
            cmd = '%s ./%s' % (Globals().pre_cmd,self.exe_name)
            info(' running:\n\t' + cmd)
            try:
                f = os.popen(cmd)
                out = f.readlines()
                f.close()
            except Exception, e:
                err('orio.main.tuner.ptest_driver: failed to execute the testing code: "%s"\n --> %s: %s' % (cmd,e.__class__.__name__, e))
                
            try:
                if out: 
                    for line in out: 
                        if line.strip().startswith('{'): 
                            output = line.strip()
                            break
                if output: perf_costs = eval(str(output))
            except Exception, e:
                err('orio.main.tuner.ptest_driver: failed to process test result, command was "%s", output: "%s\n --> %s: %s' %
                      (cmd,perf_costs,e.__class__.__name__,e))


        # check if the performance cost is already acquired
        if not perf_costs:
            err('orio.main.tuner.ptest_driver:  performance testing failed: "%s"' % cmd)

        # return the performance costs dictionary
        return perf_costs
            
    #-----------------------------------------------------

    def __cleanup(self):
        '''Delete all the generated files'''

        if Globals().keep_temps: return
        for fname in [self.exe_name, self.src_name]:
            try:
                if os.path.exists(fname):
                    os.unlink(fname)
            except:
                err('orio.main.tuner.ptest_driver:  cannot delete file: %s' % fname)

    #-----------------------------------------------------

    def run(self, test_code):
        '''To compile and to execute the given testing code to get the performance cost
        @param test_code: the code for testing multiple coordinates in the search space
        @return: a dictionary of the times corresponding to each coordinate in the search space
        '''

        # write the testing code
        self.__write(test_code)

        # compile the testing code
        self.__build()

        # execute the testing code to get performance costs
        perf_costs = self.__execute()

        # delete all generated and used files
        self.__cleanup()

        # return the performance costs
        return perf_costs


