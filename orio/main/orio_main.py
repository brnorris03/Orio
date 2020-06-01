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
OPENCL = 4
C_CPP_ANNOTATE = 5


def start(argv, lang):
    '''The orio.main.starting procedure'''

    retcode = 0 
    # check for Fortran source, which is not supported yet now
    if lang == FORTRAN:
        language = 'fortran'
        sys.stderr.write('WARNING: Fortran support is limited\n')
    elif lang == C_CPP:
        language = 'c'
    elif lang == CUDA:
        language = 'cuda'
    elif lang == OPENCL:
        language = 'opencl'
    elif lang == C_CPP_ANNOTATE:
        language = 'c_annotate'
    else:
        sys.stderr.write('orio.main.main:  Language not supported at this time.')
        sys.exit(1)

    # import other required Python packages
    import pragma_preprocessor, ann_parser, cmd_line_opts, opt_driver
    
    # process the command line
    cmdline = cmd_line_opts.CmdParser().parse(argv)
    

    g = Globals(cmdline)
    Globals().language = language

    # Simply pass through command  (Orio won't do anything)
    if g.disable_orio and g.external_command:
        cmd = ' '.join(g.external_command)
        #info(cmd)
        retcode = os.system(cmd)
        if retcode != 0: sys.exit(1)

    if not g.disable_orio: always_print('\n====== START ORIO: %s ======' % timestamp(), end='')
    final_output_file = None
    annotated_files = 0 # for multi-file tuning
    for srcfile, out_filename in g.src_filenames.items():
        annotations_found = False

        debug('Processing %s,%s' % (srcfile,out_filename))
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
    
            # parse the source file and build a symbol table
            #stbuilder = st_builder.STBuilder(srcfile)
            #symtab = stbuilder.build_st()
            Globals().setFuncDec(src_code)
            # obtain the mapping for performance tuning specifications
            tspec_prog = ''
            if g.spec_filename:
                info('\n----- begin reading the tuning specification file: %s -----' % g.spec_filename)
                try:
                    f = open(g.spec_filename, 'r')
                    tspec_prog = f.read()
                    f.close()
                except Exception, e:
                    err('orio.main.main: Exception %s. Cannot open file for reading: %s' % \
                        (e,g.spec_filename))
                else:
                #tuning_spec_dict = tspec.tspec.TSpec().parseProgram(tspec_prog)
                    info('----- finished reading the tuning specification -----')
                
            # Just add the tuning spec to the file being parsed
            if tspec_prog:
                src_code = '/*@ begin PerfTuning (' + tspec_prog + ')\n@*/\n' + src_code + '\n/*@ end @*/\n'
                
            # parse the source code and return a sequence of code fragments
            info('\n----- begin parsing annotations -----')

            # Search for pragma orio entries (currently only loops supported)
            if pragma_preprocessor.PragmaPreprocessor.leaderPragmaRE().search(src_code):
                src_code = pragma_preprocessor.PragmaPreprocessor().preprocess(src_code)

            # for efficiency (e.g., do as little as possible when there are no annotations):
            if ann_parser.AnnParser.leaderAnnRE().search(src_code): 
                cfrags = ann_parser.AnnParser().parse(src_code)     # list of CodeFragment objects
                annotations_found = True
                annotated_files += 1
            else:
                info('----- did not find any Orio annotations -----')
            info('----- finished parsing annotations -----')
            
            # perform optimizations based on information specified in the annotations
            if annotations_found:
                info('\n----- begin optimizations -----')
                odriver = opt_driver.OptDriver(src=srcfile, language=language)
                optimized_code_seq = odriver.optimizeCodeFrags(cfrags, True)
                info('----- finish optimizations -----')
        
                # remove all annotations from output
                if g.erase_annot:
                    info('\n----- begin removing annotations from output-----')
                    optimized_code_seq = [[ann_parser.AnnParser().removeAnns(c), i, e] \
                                          for c, i, e in optimized_code_seq]
                    info('----- finished removing annotations from output-----')
    
                # write output
                info('\n----- begin writing the output file(s) -----')

                optimized_code, _, externals = optimized_code_seq[0]
                if g.out_filename: out_filename = g.out_filename
                g.tempfilename = out_filename
                info('--> writing output to: %s' % out_filename)
                try:
                    f = open(out_filename, 'w')
                    f.write(externals)
                    f.write(optimized_code)
                    f.close()
                except:
                    err('orio.main.main:  cannot open file for writing: %s' % out_filename)
                info('----- finished writing the output file(s) -----')
                final_output_file = out_filename
        # ----- end of "if not g.disable_orio:" -----

        # if orio was invoked as a compiler wrapper, perform the original command
        if g.external_command:
            if not annotations_found: fname = srcfile
            else: fname = out_filename
            cmd = ' '.join(g.external_command + [fname])
            info('[orio] %s'% cmd)
            retcode = os.system(cmd)
            #if retcode != 0: err('orio.main.main: external command returned with error %s: %s' %(retcode, cmd),doexit=False)
            if retcode != 0: retcode = 1

        # if need to rename the object file to match the name of the original source file
        if not g.disable_orio and g.rename_objects:
            genparts = out_filename.split('.')
            genobjfile = '.'.join(genparts[:-1])  + '.o' # the Orio-generated object
            srcparts = srcfile.split('.')
            objfile = '.'.join(srcparts[:-1]) + '.o'     # the object corresponding to the input filename
            if os.path.exists(genobjfile): 
                info('----- Renaming %s to %s -----' % (genobjfile, objfile))
                os.system('mv %s %s' % (genobjfile,objfile))
    # ----- end of "for srcfile, out_filename in g.src_filenames.items():" -----

    if not g.disable_orio: always_print('\n====== END ORIO: %s [output: %s, log: %s] ======' % (timestamp(),final_output_file,g.logfile))
    # remove tuning logs for source files without any annotations
    if not g.disable_orio and annotated_files == 0: os.remove(g.logfile)
    sys.exit(retcode)


