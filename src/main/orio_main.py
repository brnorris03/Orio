#
# The orio.main.file of the Orio tool
#

import os, sys
from orio.main.util.globals import *

#----------------------------------------------

# source language types
C_CPP = 1
FORTRAN = 2
CUDA = 3

#----------------------------------------------

def start(argv, lang):
    '''The orio.main.starting procedure'''
#    import orio.main.util.globals as gl

    retcode = 0 
    # check for Fortran source, which is not supported yet now
    if lang == FORTRAN:
        language = 'fortran'
        sys.stderr.write('WARNING: Fortran support is limited\n')
    elif lang == C_CPP:
        language = 'c'
    elif lang == CUDA:
        language = 'cuda'
    else:
        sys.stderr.write('orio.main.main:  Language not supported at this time.')
        sys.exit(1)

    # import other required Python packages
    import ann_parser, cmd_line_opts, opt_driver, tspec.tspec

    # process the command line
    cmdline = cmd_line_opts.CmdParser().parse(argv)
    

    g = Globals(cmdline)
    Globals().language = language

    #print 'Globals'
    #print g.out_prefix
    
    # Simply pass through command  (Orio won't do anything)
    if g.disable_orio and g.external_command:
        cmd = ' '.join(g.external_command)
        info(cmd)
        retcode = os.system(cmd)
        sys.exit(retcode)

    if not g.disable_orio: info('\n====== START ORIO ======')

    for srcfile, out_filename in g.src_filenames.items():
        annotations_found = False

        if not g.disable_orio:
            # read source code
            info('\n----- begin reading the source file: %s -----' % srcfile)
            try:
                f = open(srcfile, 'r')
                src_code = f.read()
                f.close()
            except:
                err('orio.main.main: cannot open file for reading: %s' % srcfile)
            info('----- finished reading the source file -----')
    
            # obtain the mapping for performance tuning specifications
            specs_map = {}
            if g.spec_filename:
                info('\n----- begin reading the tuning specification file: %s -----' % g.spec_filename)
                try:
                    f = open(g.spec_filename, 'r')
                    tspec_prog = f.read()
                    f.close()
                except:
                    err('orio.main.main: cannot open file for reading: %s' % g.spec_filename)
                specs_map = tspec.tspec.TSpec().parseProgram(tspec_prog)
                info('----- finish reading the tuning specification -----')
    
            # parse the source code and return a sequence of code fragments
            info('\n----- begin parsing annotations -----')
            # for efficiency (e.g., do as little as possible when there are no annotations):
            if ann_parser.AnnParser.leaderAnnRE().search(src_code): 
                cfrags = ann_parser.AnnParser().parse(src_code)
                annotations_found = True
            else:
                info('----- did not find any Orio annotations -----')
            info('----- finished parsing annotations -----')
    
            # perform optimizations based on information specified in the annotations
            if annotations_found:
                info('\n----- begin optimizations -----')
                odriver = opt_driver.OptDriver(specs_map, language=language)
                optimized_code_seq = odriver.optimizeCodeFrags(cfrags, {}, True)
                info('----- finish optimizations -----')
        
                # remove all annotations from output
                if g.erase_annot:
                    info('\n----- begin removing annotations from output-----')
                    optimized_code_seq = [[ann_parser.AnnParser().removeAnns(c), i] \
                                          for c, i in optimized_code_seq]
                    info('----- finished removing annotations from output-----')
    
                # write output
                info('\n----- begin writing the output file(s) -----')
                for optimized_code, input_params in optimized_code_seq:
                    if len(optimized_code_seq) > 1:
                        path_name, ext = os.path.splitext(out_filename)
                        suffix = ''
                        for pname, pval in input_params:
                            suffix += '_%s_%s' % (pname, pval)
                        if g.out_filename: out_filename = g.out_filename
                        else: out_filename = ('%s%s' % (path_name, suffix)) + ext
                    info('--> writing output to: %s' % out_filename)
                    try:
                        f = open(out_filename, 'w')
                        f.write(optimized_code)
                        f.close()
                    except:
                        err('orio.main.main:  cannot open file for writing: %s' % g.out_filename)
                info('----- finished writing the output file(s) -----')

        # ----- end of "if not g.disable_orio:" -----

        # if orio was invoked as a compiler wrapper, perform the original command
        if g.external_command:
            if not annotations_found: fname = srcfile
            else: fname = out_filename
            cmd = ' '.join(g.external_command + [fname])
            info('[orio] %s'% cmd)
            retcode = os.system(cmd)
            if retcode != 0: err('orio.main.main: external command returned with error %s: %s' %(retcode, cmd),doexit=False)

    if not g.disable_orio and g.rename_objects:
        for srcfile, genfile in g.src_filenames.items():
            genparts = genfile.split('.')
            genobjfile = '.'.join(genparts[:-1])  + '.o'    # the Orio-generated object
            srcparts = srcfile.split('.')
            objfile = '.'.join(srcparts[:-1]) + '.o'     # the object corresponding to the input filename
            if os.path.exists(genobjfile): 
                info('----- Renaming %s to %s -----' % (genobjfile, objfile))
                os.system('mv %s %s' % (genobjfile,objfile))


    if not g.disable_orio: info('\n====== END ORIO ======')
    sys.exit(retcode)



