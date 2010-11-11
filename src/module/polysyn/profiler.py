#
# A profiler used to perform profiling on the pluto code, in order to identify the hotspot,
# to which where syntactic transformations can be applied.
#

import os, re, sys
from orio.main.util.globals import *

#---------------------------------------------------------

class Profiler:
    '''The profiler used to identify the hotspot'''

    def __init__(self, verbose, profiling_code, compile_cmd, compile_opts):
        '''To instantiate a profiler instance'''

        self.verbose = verbose
        self.profiling_code = profiling_code
        self.compile_cmd = compile_cmd
        self.compile_opts = compile_opts

    #---------------------------------------------------------

    def __getProfileCode(self, pluto_code):
        '''To generate the profiling code, and the starting line number of the profiled code'''

        # regular expression
        polysyn_re = r'/\*@\s*profiled\s+code\s*@\*/'
        math_re = r'#include\s*<math\.h>'
        
        # get the profiling code
        try:
            f = open(self.profiling_code, 'r')
            profiling_code = f.read()
            f.close()
        except:
            err('orio.module.polysyn.profiler:  cannot open file for reading: %s' % self.profiling_code)
        
        # find the tag used to indicate the location of the code to be profiled
        match_obj = re.search(polysyn_re, profiling_code)

        # if no desired tag is found
        if match_obj == None:
            err('orio.module.polysyn.profiler:  missing profiling tag in the profiling code: "%s"' % self.profiling_code)

        # get the starting line number of the code to be profiled
        start_pos = match_obj.start()
        start_line_no = profiling_code[:start_pos].count('\n') + 1

        # insert the pluto code into the profiling code
        pcode, num = re.subn(math_re, '/* #include <math.h> */', profiling_code)
        profile_code, num = re.subn(polysyn_re, pluto_code, pcode)
        profile_code = '#include <math.h>\n' + profile_code


        # check the number of performed replacements
        if num > 1:
            err('orio.module.polysyn.profiler:  there are more than one profiling tags in the profiling code')

        # return the profiling code, and the starting line number of the profiled code
        return (profile_code, start_line_no)

    #---------------------------------------------------------

    def __getHotspotsInfo(self, profile_code, start_line_no):
        '''
        To identify hotspots using gprof.
        Each hotspot information consists of the timing percentage, followed by the code line number.
        '''

        # delete "gmon.out" if already exists
        if os.path.exists('gmon.out'):
            try:
                os.unlink('gmon.out')
            except:
                err('orio.module.polysyn.profiler:  failed to delete file: %s' % 'gmon.out')

        # used file names
        src_fname = '_polysyn_profiling.c'
        exe_fname = '_polysyn_profiling.exe'

        # write the profiling code
        try:
            f = open(src_fname, 'w')
            f.write(profile_code)
            f.close()
        except:
            err('orio.module.polysyn.profiler:  cannot open file for writing: %s' % src_fname)            

        # save the number of OpenMP threads and set it to one (i.e. single core)
        orig_num_threads = os.getenv('OMP_NUM_THREADS')
        os.putenv('OMP_NUM_THREADS', '1')
        
        # compile the profiling code
        compile_cmd = self.compile_cmd if self.compile_cmd else 'gcc'
        compile_opts = self.compile_opts if self.compile_opts else ''
        cmd = ('%s %s -g -pg -o %s %s -lm' %
               (compile_cmd, compile_opts, exe_fname, src_fname))
        info('orio.module.polysyn.profiler compiling:\n\t' + cmd)
        try:
            os.system(cmd)
        except:
            err('orio.module.polysyn.profiler:  failed to compile the profiling code: "%s"' % cmd)

        # execute the executable to generate "gmon.out" file that contains the profiling information
        exec_cmd = './%s' % exe_fname
        info('orio.module.polysyn.profiler executing:\n\t' + exec_cmd)
        try:
            status = os.system(exec_cmd)
        except:
            err('orio.module.polysyn.profiler:  failed to execute the profiling code: "%s"' % exec_cmd)

        # run gprof to extract the profiling information
        gprof_cmd = 'gprof -lbQ %s' % exe_fname
        info('orio.module.polysyn.profiler executing profiler:\n\t' + gprof_cmd)
        try:
            f = os.popen(gprof_cmd)
            profile_info = f.read()
            f.close()
        except Exception, e:
            err('orio.module.polysyn.profiler:  failed to execute GProf: "%s"\n --> %s: %s' % (gprof_cmd,e.__class__.__name__, e))

        # delete unneeded files
        if not Globals().keep_temps:
            removed_files = [src_fname, exe_fname, 'gmon.out']
            for f in removed_files:
                try:
                    os.unlink(f)
                except:
                    err('orio.module.polysyn.profiler:  failed to delete file: %s' % f)
        
        # set back the number of OpenMP threads to the original one
        if orig_num_threads == None:
            os.unsetenv('OMP_NUM_THREADS')
        else:
            os.putenv('OMP_NUM_THREADS', str(orig_num_threads))

        # extract hotspots information from the profiling information
        tpercent_re = r'\d+((\.)?\d*)?'
        linenum_re = r':(\d+)'
        hotspots_info = []
        for line in profile_info.split('\n'):
            line = line.strip()
            m = re.match(tpercent_re, line)
            if m:
                time_percent = eval(m.group())
                m = re.search(linenum_re, line)
                linenum = eval(m.group(1))
                if linenum >= start_line_no:
                    linenum = linenum - start_line_no + 1
                    hotspots_info.append((time_percent, linenum))

        # return the hotspots information
        return hotspots_info

    #---------------------------------------------------------

    def getHotspotsInfo(self, pluto_code):
        '''To identify hotspots information by using gprof'''

        # get the profiling code, and the starting line number of the profiled code
        profile_code, start_line_no = self.__getProfileCode(pluto_code)

        # get the hotspots information
        hotspots_info = self.__getHotspotsInfo(profile_code, start_line_no)

        # return the hotspots information
        return hotspots_info
