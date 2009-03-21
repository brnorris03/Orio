#
# Contain a parser to extract the command line options, and a class definition for
# command line options
#

import getopt, os, sys

#----------------------------------------------

# the usage message
USAGE_MSG = '''
description: compile shell for Orio

usage: %s [options] <ifile> 
  <ifile>   input file containing the annotated code

options:
  -e, --erase-annot              remove annotations from the output
  -h, --help                     display this message
  -k, --keep-temps               do not remove intermediate generated files
  -o <file>, --output=<file>     place the output in <file> (only valid when processing 
                                 single files)
  -p, --output-prefix=<string>   generate output filename from input filename by prepending 
                                 the specified string (default is '_', e.g., f.c becomes _f.c).
  -r, --rename-objects           after compiling the Orio-generated source, rename the object 
                                 files to be the same as those that would result from compiling
                                 the original source code
  -s <file>, --spec=<file>       read tuning specifications from <file>
  -v, --verbose                  verbosely show details of the results of the running program

environment variables: 
  ORIO_FLAGS                     the string value is used to augment the list of Orio command-lin
                                 options
''' % os.path.basename(sys.argv[0])

#----------------------------------------------

class CmdLineOpts:
    '''The command line options'''

    def __init__(self, src_filenames, spec_filename, verbose, erase_annot, 
                keep_temps, rename_objects, ext_cline, disable_orio):
        '''To instantiate an object that represents the command line options'''

        self.src_filenames = src_filenames     # dictionary; keys: input source files; vals: names of output files
        self.spec_filename = spec_filename     # the name of the tuning specification file
        self.verbose = verbose                 # show details of the results of the running program
        self.erase_annot = erase_annot         # do we need to remove annotations from the output?
        self.keep_temps = keep_temps           # keep intermediate generated files
        self.rename_objects = rename_objects   # rename compiler files to match original source name
        self.external_command = ext_cline      # command line being wrapped (not processed, just passed along)
        self.disable_orio = disable_orio       # True when orio is wrapping something other than compilation, e.g., linking
        pass

#----------------------------------------------

class CmdParser:
    '''Parser for command line options'''
    
    def __init__(self):
        '''To instantiate the command line option parser'''
        pass

    #----------------------------------------------

    def parse(self, argv):
        '''To extract the command line options'''

        # Preprocess the command line to accomodate cases when Orio is used 
		# as a preprocessor to the compiler, e.g., orcc <orio_opts> compiler <compiler_tops> source.c
        orioargv = []
        otherargv = []
        srcfiles = {}
        index = 1
        wrapper = False
        for arg in argv[1:]:
            if not wrapper  and arg.startswith('-'): 
                orioargv.append(arg)
                continue
            argisinput = False
            if not arg.startswith('-'):
                # Look for the source(s)
                if arg.count('.') > 0:
                    suffix = arg[arg.rfind('.')+1:]
                    if suffix.lower() in ['c','cpp','cxx','h','hpp','hxx','f','f90','f95','f03']:
                        srcfiles[arg] = '_' + arg
                        argisinput = True
            if not argisinput:
                if not wrapper: wrapper = True
                if wrapper: otherargv.append(arg)
            index += 1

        # fix non-Orio command line options as much as possible since the shell eats quotes and such
        externalargs=[]
        index = 0
        while index < len(otherargv):
            arg = otherargv[index]
            if arg.count('=') > 0:
                key,val=arg.split('=')
                index += 1
                if val[0] == val[-1] == '"':
                    val = "'" + val + "'"
                else:
                    val = "'\"" + val
                    if index < len(otherargv): arg = otherargv[index]
                    while index < len(otherargv) and not arg.startswith('-'): 
                        val += ' ' + arg
                        arg = otherargv[index]
                        index += 1
                    val += "\"'"
                externalargs.append(key + '=' + val)
            else:
                externalargs.append(arg)
                index += 1
        #print >>sys.stderr,'new args:',externalargs

        # check the ORIO_FLAGS env. variable for more options
        if 'ORIO_FLAGS' in os.environ.keys():
            orioargv.extend(os.environ['ORIO_FLAGS'].split())

        # variables to represents the command line options
        out_prefix = '_'
        out_filename = None
        rename_objects = False
        spec_filename = None
        verbose = False
        erase_annot = False      
        keep_temps = False
        disable_orio = False

        # get all options
        try:
            opts, args = getopt.getopt(orioargv,
                                       'ehko:p:rs:v',
                                       ['erase-annot', 'help', 'keep-temps',' output=', 
                                       'output-prefix=', 'rename-objects', 'spec=', 'verbose'])
        except Exception, e:
            print 'Orio error: %s' % e
            print USAGE_MSG
            sys.exit(1)

        # evaluate all options
        for opt, arg in opts:
            if opt in ('-e', '--erase-annot'):
                erase_annot = True
            elif opt in ('-h', '--help'):
                print USAGE_MSG
                sys.exit(1)
            elif opt in ('-k', '--keep-temps'):
                keep_temps = True
            elif opt in ('-o', '--output'):
                out_filename = arg
            elif opt in ('-p', '--output-prefix'):
                out_prefix = arg
            elif opt in ('-r', '--rename-objects'):
                rename_objects = True
            elif opt in ('-s', '--spec'):
                spec_filename = arg
            elif opt in ('-v', '--verbose'):
                verbose = True

        # check on the arguments
        if len(srcfiles) < 1:
            if otherargv: 
                disable_orio = True
            else:
                print 'Orio error: missing file arguments'
                print USAGE_MSG
                sys.exit(1)

        for src_filename in srcfiles:
            # check if the source file is readable
            try:
                f = open(src_filename, 'r')
                f.close()
            except:
                print 'Orio error: cannot open source file for reading: %s' % src_filename
                sys.exit(1)

        # check if the tuning specification file is readable
        if spec_filename:
            try:
                f = open(spec_filename, 'r')
                f.close()
            except:
                print 'Orio error: cannot open file for reading: %s' % spec_filename
                sys.exit(1)

        # create the output filenames
        if len(srcfiles) == 1 and out_filename: srcfiles[srcfiles.keys()[0]] = out_filename
        else:
            for src_filename in srcfiles.keys():
                dirs, fname = os.path.split(src_filename)
                srcfiles[src_filename] = os.path.join(dirs, out_prefix + fname)

        # create an object for the command line options
        cline_opts = CmdLineOpts(srcfiles, spec_filename, verbose, erase_annot, keep_temps, 
                                 rename_objects, externalargs, disable_orio)

        # return the command line option object
        return cline_opts

