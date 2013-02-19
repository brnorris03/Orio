# This file is part of Contractor
# Original author: Boyana Norris, norris@mcs.anl.gov
# (c) 2010 UChicago Argonne, LLC
# For copying information, see the file LICENSE

import logging, os, sys, traceback
from matplotlib_logger import MatplotlibLogger


class Globals:
    '''
    A singleton class in which to stash various useful global variables for bocca.
    Do not instantiate this class directly, rather use the Globals helper function,
    e.g., myglobals = Globals().
    '''
    
    class __impl:
        """ Implementation of the singleton interface """

        def __init__(self,cmdline={}):
            
            self.loggers = {}
            self.language = 'c'         # default language is C
            self.error_pre = "\x1B[00;31m"
            self.error_post = "\x1B[00m"
            if 'dry_run' in cmdline.keys():
                self.dry_run = cmdline['dry_run']
            else:
                self.dry_run = False     # When True, don't execute anything, just print commands
            
            if 'shell' in cmdline.keys():
                self.shell = cmdline['shell']
            else:
                self.shell = '/bin/sh'   # Shell used by Orio 
            if 'verbose' in cmdline.keys():
                self.verbose = cmdline['verbose']
            else:
                self.verbose = False

            if 'extern' in cmdline.keys():
                self.extern = cmdline['extern']
            else:
                self.extern = False

            if 'config' in cmdline.keys():
                self.config = cmdline['config']
            else:
                self.config = ''

            if 'configfile' in cmdline.keys():
                self.configfile = cmdline['configfile']
                #f = open(self.configfile, 'r')
                #k=f.read()
                #sys.stderr.write(k)
                #f.close()

            else:
                self.configfile = ''

                
                
            if 'out_prefix' in cmdline.keys():
                self.out_prefix = cmdline['out_prefix']
            else:
                self.out_prefix = '_'         # prefix for output file name
                
            if 'src_filenames' in cmdline.keys():
                self.src_filenames = cmdline['src_filenames']
            else:
                self.src_filenames = {}       # dictionary; keys: input source files; vals: names of output files
            if 'out_filename' in cmdline.keys():
                self.out_filename = cmdline['out_filename']
            else:
                self.out_filename = None      # output file name
            if 'spec_filename' in cmdline.keys():
                self.spec_filename = cmdline['spec_filename']
            else:
                self.spec_filename = None     # the name of the tuning specification file
            if 'erase_annot' in cmdline.keys():
                self.erase_annot = cmdline['erase_annot']
            else:
                self.erase_annot = False      # do we need to remove annotations from the output?
            if 'keep_temps' in cmdline.keys():
                self.keep_temps = cmdline['keep_temps']
            else:
                self.keep_temps = False       # keep intermediate generated files
            if 'rename_objects' in cmdline.keys():
                self.rename_objects = cmdline['rename_objects']
            else:
                self.rename_objects = False   # rename compiler files to match original source name
            if 'external_command' in cmdline.keys():
                self.external_command = cmdline['external_command']
            else:
                self.external_command = ''    # command line being wrapped (not processed, just passed along)
            if 'disable_orio' in cmdline.keys():
                self.disable_orio = cmdline['disable_orio']
            else:
                self.disable_orio = False     # True when orio is wrapping something other than compilation, e.g., linking
            if 'pre_cmd' in cmdline.keys():
                self.pre_cmd = cmdline['pre_cmd']
            else:
                self.pre_cmd = ''             # Command string with which to prefix the execution of the Orio-built code
    
            
            # Configure logging
            if 'logging' in cmdline.keys():
                self.logging = cmdline['logging']
            else:
                self.logging = True
            if 'logger' in cmdline.keys():
                thelogger = logging.getLogger(cmdline['logger'])
            else:
                thelogger = logging.getLogger("Orio")
            if 'logfile' in cmdline.keys():
                self.logfile = cmdline['logfile']
            else:
                self.logfile = 'tuning' + str(os.getpid()) + '.log'        
            thelogger.addHandler(logging.FileHandler(filename=self.logfile))
            # Because commands are output with extra formatting, for now do not use the logger for stderr output
            #streamhandler = logging.StreamHandler()
            #streamhandler.setLevel(logging.INFO)
            #self.logger.addHandler(streamhandler)
            self.loggers['TuningLog'] = thelogger
          
            self.loggers['Matplotlib'] = MatplotlibLogger().getLogger()
      
    
            # Enable debugging
            self.debug_level = 5
            if 'ORIO_DEBUG' in os.environ.keys() and os.environ['ORIO_DEBUG'] == '1': 
                self.debug = True
                self.loggers['TuningLog'].setLevel(logging.DEBUG)
                if 'ORIO_DEBUG_LEVEL' in os.environ.keys():
                    self.debug_level = os.environ['ORIO_DEBUG_LEVEL']
            else: 
                self.debug = False
                self.loggers['TuningLog'].setLevel(logging.INFO)
            
            # counters
            self.counter = 0
            
            # function definitions of a C compilation unit
            # these definitions are generated during CUDA C Loop transformations
            # TODO: refactor this after getting a global AST view
            self.cunit_declarations = [] 
            
            # Enable validation of transformed vs. original code execution results 
            if 'validate' in cmdline.keys():
                self.validationMode = cmdline['validate']
            else:
                self.validationMode = False
            self.executedOriginal = False
            
            pass

        def addLogger(self, thelogger):
            self.loggers.add(thelogger)

        def test(self):
            """ Test method, return singleton id """
            return id(self)
    


    __single = None  # Used for ensuring singleton instance
    def __init__(self,cmdline={}):
        if Globals.__single is None:
                # Create instance
                Globals.__single = Globals.__impl(cmdline)
        
    def __call__(self):
        return self

    def __getattr__(self,attr):
        """ Delegate access to implementation """
        return getattr(self.__single, attr)

    def __setattr__(self, attr, value):
        """ Delegate access to implementation """
        return setattr(self.__single, attr, value)

    def getcounter(self):
        """ Increments the global counter and returns the new value """
        self.counter += 1
        return self.counter

# ---------------------------------------------------------------------------------
""" 
Various error-handling related miscellanea
"""

def err(errmsg='',errcode=1, doexit=True):
    sys.stderr.write(Globals().error_pre + 'ERROR: ' + errmsg + Globals().error_post + '\n')
    Globals().loggers['TuningLog'].error(errmsg + '\n' + '\n'.join(traceback.format_stack()))
    if Globals().debug:
        traceback.print_stack()
    if doexit: sys.exit(errcode)

def warn(msg='',end = '\n', pre='', post=''):
    sys.stderr.write(pre + 'WARNING: ' +  msg + post + end)
    Globals().loggers['TuningLog'].warning(msg)

def info(msg='', end='\n', pre='', post='', logging=True):
    if Globals().verbose:
        sys.stdout.write(pre + msg + post + end)
    
    if Globals().logging:
        Globals().loggers['TuningLog'].info(msg)

def debug(msg='', end='\n', pre='', post='', logging=True, level=5):
    if Globals().debug and level <= Globals().debug_level:
        sys.stdout.write(pre + 'DEBUG:' + str(msg) + post + end)
        if logging:
            Globals().loggers['TuningLog'].debug(msg)
        
def exit(msg='',code=1, doexit=True,pre='',post=''): 
    if msg != '': 
        sys.stderr.write(pre + 'ERROR: ' + msg + post + '\n')  
    Globals().loggers['TuningLog'].error(msg)
    if doexit: sys.exit(code)
