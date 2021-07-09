#
# Contain a parser to extract the command line options, and a class definition for
# command line options
#

import getopt, os, sys

#----------------------------------------------

# the usage message
USAGE_MSG = '''
Description: compile shell for Orio

Usage: %s [options] <ifile> 
  <ifile>   input file containing the annotated code

Options:
  -c, --pre-command=<string>     Command string with which to prefix the execution of the 
                                 Orio-built code, e.g., tau_exec
  --post-command=<string>        Command string to run after each execution of Orio-built code,
                                 e.g., taudb_loadtrial
  -d <level>, --debug=<level>    Enable debugging output [default off], level is an int between 
                                 1 and 6 indicating the level of verbosity.
  -e, --erase-annot              remove annotations from the output
  -h, --help                     display this message
  -o <file>, --output=<file>     place the output in <file> (only valid when processing 
                                 single files)
  -p, --output-prefix=<string>   generate output filename from input filename by prepending 
                                 the specified string (default is '_', e.g., f.c becomes _f.c).
  -r, --rename-objects           after compiling the Orio-generated source, rename the object 
                                 files to be the same as those that would result from compiling
                                 the original source code
  -s <file>, --spec=<file>       read tuning specifications from <file>
  --search="searchalg;opt1=val1;..." Search algorithm and its options; overrides tuning spec search
                                 section entries, e.g., --search="Mlsearch;total_runs=100"
  --stop-on-error                exit with an error code when first exception occurs
  -x, --external                 run orio in external mode
  --config=<p1:v1,p2:v2,..>      configurations for external mode
  --configfile=filename          configuration filename 
  -v, --verbose                  verbosely show details of the results of the running program
  --validate                     validate by comparing output of original and transformed codes
  --meta                         export metadata as json
  --marker-loops                 developer-only option, can interfere with performance

environment variables: 
  ORIO_FLAGS                     the string value is used to augment the list of Orio command-lin
                                 options
  ORIO_DEBUG                     when set, print debugging information (orio.main.y for developer use)
  ORIO_DEBUG_LEVEL               integer value between 1 and 6 indicating the level of debugging output
                                 verbosity
                                 
For more details, please refer to the documentation at https://trac.mcs.anl.gov/projects/performance/wiki/OrioUserGuide
''' % os.path.basename(sys.argv[0])

#----------------------------------------------

class CmdParser:
    '''Parser for command line options'''
    
    def __init__(self):
        '''To instantiate the command line option parser'''
        pass

    #----------------------------------------------

    def parse(self, argv):
        '''To extract the command line options'''

        cmdline = {}
        # Preprocess the command line to accomodate cases when Orio is used 
        # as a preprocessor to the compiler, e.g., orcc <orio_opts> compiler <compiler_tops> source.c
        orioargv = []
        otherargv = []
        srcfiles = {}
        index = 1
        orioarg = False
        wrapper = False
        for arg in argv[1:]:
            if not wrapper and arg.startswith('-'):
                orioargv.append(arg)
                if arg in ['-c','-o','-p','-s']: # switch with an argument
                    orioarg = True
                continue
            argisinput = False
            if not arg.startswith('-'):
                if orioarg: 
                    orioargv.append(arg)
                    orioarg = False
                    continue
                # Look for the source(s)
                if arg.count('.') > 0:
                    suffix = arg[arg.rfind('.')+1:]
                    if suffix.lower() in ['c','cc','cpp','cxx','h','hpp','hxx','f','f90','f95','f03']:
                        srcfiles[arg] = '_' + arg
                        argisinput = True
            if not argisinput:
                if not wrapper: wrapper = True
                if wrapper: otherargv.append(arg)
            index += 1
        if wrapper:
            cmdline['external_command'] = otherargv

        # fix non-Orio command line options as much as possible (esp. -D) since the shell eats quotes and such
        externalargs=[]
        index = 0
        while index < len(otherargv):
            arg = otherargv[index]
            if arg.count('=') > 0 and arg.startswith('-D'):
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

        # check the ORIO_FLAGS env. variable for more options
        if 'ORIO_FLAGS' in list(os.environ.keys()):
            orioargv.extend(os.environ['ORIO_FLAGS'].split())

        # get all options
        try:
            opts, args = getopt.getopt(orioargv,
                                       'c:d:ehko:p:rs:vx',
                                       ['pre-command=','debug=','config=','configfile=', 'erase-annot', 'help', 'keep-temps',' output=',
                                        'output-prefix=', 'rename-objects',  'spec=', 'verbose', 'extern',
                                        'stop-on-error', 'search=',
                                        'validate', 'post-command=', 'meta', 'marker-loops',
                                        'logdir='])
        except Exception as e:
            sys.stderr.write('Orio command-line error: %s' % e)
            sys.stderr.write(USAGE_MSG + '\n')
            sys.exit(1)

        #evaluate all options
        for opt, arg in opts:
            if opt in ('-c', '--pre-command'):
                cmdline['pre_cmd'] = arg
            elif opt in ('-d', '--debug'):
                cmdline['debug'] = arg
            elif opt in ('--post-command'):
                cmdline['post_cmd'] = arg
            elif opt in ('-e', '--erase-annot'):
                cmdline['erase_annot'] = True
            elif opt in ('-h', '--help'):
                sys.stdout.write(USAGE_MSG +'\n')
                sys.exit(0)
            elif opt in ('-k', '--keep-temps'):
                cmdline['keep_temps'] = True
            elif opt in ('-o', '--output'):
                cmdline['out_filename'] = arg
            elif opt in ('-p', '--output-prefix'):
                cmdline['out_prefix'] = arg
            elif opt in ('-r', '--rename-objects'):
                cmdline['rename_objects'] = True
            elif opt in ('-s', '--spec'):
                cmdline['spec_filename'] = arg
            elif opt in('--search'):
                cmdline['search'] = arg     # --search="searchalg;opt1=val1;..." 
            elif opt in ('-v', '--verbose'):
                cmdline['verbose'] = True
            elif opt in ('-x','--extern'):
                cmdline['extern'] = True
            elif opt in ('--config'):
                cmdline['config'] = arg
            elif opt in ('--configfile'):
                cmdline['configfile'] = arg
            elif opt in ('--validate'):
                cmdline['validate'] = True
            elif opt in ('--meta'):
                cmdline['meta'] = True
            elif opt in ('--stop-on-error'):
                cmdline['stop-on-error'] = True
            elif opt in ('--marker-loops'):
                cmdline['marker-loops'] = True  # generate fake loops for Meliora
            elif opt in ('--logdir'):
                cmdline['logdir'] = arg   # tuning logs directory
                
        # check on the arguments
        if len(srcfiles) < 1:
            if otherargv: 
                cmdline['disable_orio'] = True
            else:
                sys.stderr.write('Orio command-line error: missing file arguments')
                sys.stderr.write(USAGE_MSG + '\n')
                sys.exit(1)

        for src_filename in srcfiles: # check if the source files are readable
            try:
                f = open(src_filename, 'r')
                f.close()
            except:
                sys.stderr.write('orio.main.cmd_line_opts: cannot open source file for reading: %s' % src_filename)
                sys.exit(1)

        if 'spec_filename' in list(cmdline.keys()): spec_filename = cmdline['spec_filename']
        else: spec_filename = None
        # check if the tuning specification file is readable
        if spec_filename:
            try:
                f = open(spec_filename, 'r')
                f.close()
            except:
                sys.stderr.write('orio.main.cmd_line_opts: cannot open file for reading: %s' % spec_filename)
                sys.exit(1)

        # create the output filenames
        if len(list(srcfiles.keys())) == 1 and 'out_filename' in list(cmdline.keys()): 
            srcfiles[list(srcfiles.keys())[0]] = cmdline['out_filename']
        else:
            for src_filename in list(srcfiles.keys()):
                dirs, fname = os.path.split(src_filename)
                if 'out_prefix' in list(cmdline.keys()): out_prefix=cmdline['out_prefix']
                else: out_prefix = '_'
                srcfiles[src_filename] = os.path.join(dirs, out_prefix + fname)

        cmdline['src_filenames'] = srcfiles
        return cmdline

