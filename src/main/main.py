#
# The main file of the Orio tool
#

import os, sys

#----------------------------------------------

# source language types
C_CPP = 1
FORTRAN = 2

#----------------------------------------------

def start(argv, lang):
    '''The main starting procedure'''

    # check for Fortran source, which is not supported yet now
    if lang == FORTRAN:
        language = 'fortran'
        print 'warning: Fortran support is limited'
    elif lang == C_CPP:
        language = 'c'
    else:
        print 'error: Language not supported at this time.'
        sys.exit(1)

    # import other required Python packages
    import ann_parser, cmd_line_opts, opt_driver, tspec.tspec

    # get the command line
    cline_opts = cmd_line_opts.CmdParser().parse(argv)

    # need to be verbose?
    verbose = cline_opts.verbose

    if not cline_opts.disable_orio:
        if verbose: print '\n====== START ORIO ======'

        for srcfile, out_filename in cline_opts.src_filenames.items():
            # read source code
            if verbose: print '\n----- begin reading the source file: %s -----' % srcfile
            try:
                f = open(srcfile, 'r')
                src_code = f.read()
                f.close()
            except:
                print 'error: cannot open file for reading: %s' % srcfile
                sys.exit(1)
            if verbose: print '----- finish reading the source file -----'
    
            # obtain the mapping for performance tuning specifications
            specs_map = {}
            if cline_opts.spec_filename:
                if verbose: print ('\n----- begin reading the tuning specification file: %s -----' %
                                   cline_opts.spec_filename)
                try:
                    f = open(cline_opts.spec_filename, 'r')
                    tspec_prog = f.read()
                    f.close()
                except:
                    print 'error: cannot open file for reading: %s' % cline_opts.spec_filename
                    sys.exit(1)
                specs_map = tspec.tspec.TSpec().parseProgram(tspec_prog)
                if verbose: print '----- finish reading the tuning specification -----'
    
            # parse the source code and return a sequence of code fragments
            if verbose: print '\n----- begin parsing annotations -----'
            cfrags = ann_parser.AnnParser(verbose).parse(src_code)
            if verbose: print '----- finish parsing annotations -----'
    
            # perform optimizations based on information specified in the annotations
            if verbose: print '\n----- begin optimizations -----'
            odriver = opt_driver.OptDriver(specs_map, cline_opts, language=language)
            optimized_code_seq = odriver.optimizeCodeFrags(cfrags, {}, True)
            if verbose: print '----- finish optimizations -----'
        
            # remove all annotations from output
            if cline_opts.erase_annot:
                if verbose: print '\n----- begin removing annotations from output-----'
                optimized_code_seq = [[ann_parser.AnnParser().removeAnns(c), i] \
                                      for c, i in optimized_code_seq]
                if verbose: print '----- finish removing annotations from output-----'
    
            # write output
            if verbose: print '\n----- begin writing the output file(s) -----'
            for optimized_code, input_params in optimized_code_seq:
                if len(optimized_code_seq) > 1:
                    path_name, ext = os.path.splitext(out_filename)
                    suffix = ''
                    for pname, pval in input_params:
                        suffix += '_%s_%s' % (pname, pval)
                    out_filename = ('%s%s' % (path_name, suffix)) + ext
                if verbose: print '--> writing output to: %s' % out_filename
                try:
                    f = open(out_filename, 'w')
                    f.write(optimized_code)
                    f.close()
                except:
                    print 'error: cannot open file for writing: %s' % cline_opts.out_filename
                    sys.exit(1)
            if verbose: print '----- finish writing the output file(s) -----'

        if verbose: print '\n====== END ORIO ======'

        # ----- end of "if not cline_opts.disable_orio:" -----

    # if orio was invoked as a compiler wrapper, perform the original command
    if cline_opts.external_command:
        cmd = ' '.join(cline_opts.external_command)
        if verbose: print '----- invoking external (wrapped) command: -----' 
        print cmd
        os.system(cmd)

    if not cline_opts.disable_orio and cline_opts.rename_objects:
        for srcfile, genfile in cline_opts.src_filenames.items():
            genparts = genfile.split('.')
            genobjfile = '.'.join(genparts[:-1])  + '.o'    # the Orio-generated object
            srcparts = srcfile.split('.')
            objfile = '.'.join(srcparts[:-1]) + '.o'     # the object corresponding to the input filename
            if verbose: print '----- Renaming', genobjfile, 'to', objfile, '-----'
            if os.path.exists(genobjfile): os.system('mv %s %s' % (genobjfile,objfile))



