#
# The main file (and class) for optimization using Pluto code
#

import sys, re, os, glob
import ann_parser, module.module

#-----------------------------------------

class Pluto(module.module.Module):
    '''The class definition for Pluto transformation module'''
    
    def __init__(self, perf_params, module_body_code, annot_body_code, cmd_line_opts,
                 line_no, indent_size):
        '''To instantiate a Pluto transformation module'''
        
        module.module.Module.__init__(self, perf_params, module_body_code, annot_body_code,
                                      cmd_line_opts, line_no, indent_size)
        
    #---------------------------------------------------------------------
    
    def transform(self):
        '''To transform code using Pluto'''

        # get optimization parameters
        var_val_pairs = ann_parser.AnnParser(self.perf_params).parse(self.module_body_code)

        # get all needed performance parameters
        table = dict(var_val_pairs)
        if 'tile_sizes' not in table:
            print 'error:Pluto: missing "tile_sizes" parameter' 
            sys.exit(1)
        if 'tile_level' not in table:
            print 'error:Pluto: missing "tile_level" parameter' 
            sys.exit(1)
        if 'unroll_factor' not in table:
            print 'error:Pluto: missing "unroll_factor" parameter' 
            sys.exit(1)
        if 'vectorize' not in table:
            print 'error:Pluto: missing "vectorize" parameter' 
            sys.exit(1)
        tile_sizes = table['tile_sizes']
        tile_level = table['tile_level']
        unroll_factor = table['unroll_factor']
        vectorize = table['vectorize']

        # sanity check of the obtained performance parameters
        for t in tile_sizes:
            if not isinstance(t, int) or t<=0:
                print 'error:Pluto: tile size must be a positive integer' 
                sys.exit(1)
        if not isinstance(tile_level, int) or not (0<=tile_level<=2) :
            print 'error:Pluto: number of tile levels must be either 0, 1, or 2' 
            sys.exit(1)
        if not isinstance(unroll_factor, int) or unroll_factor<1:
            print 'error:Pluto: invalid unroll factor: %s'  % unroll_factor
            sys.exit(1)
        if not isinstance(vectorize, int) or not (vectorize==0 or vectorize==1):
            print 'error:Pluto: vectorize value must be either 0 or 1'
            sys.exit(1)
        
        # initialize the code to be transformed
        code = self.annot_body_code
        
        # the used tags
        pluto_open_tag_re = r'/\*\s*pluto\s+start.*?\*/'
        pluto_close_tag_re = r'/\*\s*pluto\s+end.*?\*/'

        # find the opening and closing tags of the pluto code  
        open_m = re.search(pluto_open_tag_re, code)
        close_m = re.search(pluto_close_tag_re, code)
        if (not open_m) or (not close_m):
            print ('error:polysyn: cannot find the opening and closing tags for the PLuTo code')
            sys.exit(1)
            
        # check if Pluto has been correctly installed  
        if os.popen('polycc').read() == '':
            print 'error: Pluto is not installed. Cannot use "polycc" command.'
            sys.exit(1)

        # write the tile.sizes file to set the used tile sizes
        ts_fname = 'tile.sizes'
        content = ''
        for t in tile_sizes:
            content += '%s\n' % t
        try:
            f = open(ts_fname, 'w')
            f.write(content)
            f.close()
        except:
            print 'error: cannot write to file: %s' % ts_fname
            sys.exit(1)

        # set command line arguments
        use_tiling = False
        cmd_args = ''
        if tile_level == 1:
            cmd_args += ' --tile'
            use_tiling = True
        if tile_level == 2:
            cmd_args += ' --tile --l2tile'
            use_tiling = True
        if unroll_factor > 1:
            cmd_args += ' --unroll --ufactor=%s' % unroll_factor
        if vectorize:
            cmd_args += ' --prevector'

        # write code to be optimized into a file
        fname = '_orio_pluto_.c'
        try:
            f = open(fname, 'w')
            f.write(code)
            f.close()
        except:
            print 'error: cannot open file for writing: %s' % fname
            sys.exit(1)

        # run Pluto
        cmd = 'polycc %s %s' % (fname, cmd_args)
        print '  running Pluto with command: %s' % cmd
        try:
            os.system(cmd)
        except:
            print 'error: failed to run command: %s' % cmd
            sys.exit(1)
        
        # delete unneeded files
        path_name, ext = os.path.splitext(fname)
        removed_fnames = [fname] + glob.glob(path_name + '.kernel.*')
        if use_tiling:
            removed_fnames += [ts_fname]
        for f in removed_fnames:
            try:
                os.unlink(f)
            except:
                print 'error: failed to remove file: %s' % f
                sys.exit(1)

        # get the Pluto-generated code
        plutogen_fnames = glob.glob(path_name + '.*' + ext)
        if len(plutogen_fnames) != 1:
            print 'error: failed to generate Pluto-transformed code'
            sys.exit(1)
        plutogen_fname = plutogen_fnames[0]
        try:
            f = open(plutogen_fname, 'r')
            pluto_code = f.read()
            f.close()
        except:
            print 'error: cannot open file for writing: %s' % fname
            sys.exit(1)

        # delete the Pluto-generated file
        try:
            os.unlink(plutogen_fname)
        except:
            print 'error: failed to remove file: %s' % plutogen_fname
            sys.exit(1)

        # return the Pluto-generated code
        return pluto_code

