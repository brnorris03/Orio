#
# The orio.main.file (and class) for optimization using Pluto code
#

import sys, re, os, glob
import ann_parser, orio.module.module
from orio.main.util.globals import *

#-----------------------------------------

class Pluto(orio.module.module.Module):
    '''Class definition for Pluto transformation module.'''
    
    def __init__(self, perf_params, module_body_code, annot_body_code,
                 line_no, indent_size, language='C'):
        '''Instantiate a Pluto transformation module.'''
        
        orio.module.module.Module.__init__(self, perf_params, module_body_code, annot_body_code,
                                      line_no, indent_size, language)
        
    #---------------------------------------------------------------------
    
    def transform(self):
        '''To transform code using Pluto'''

        # get optimization parameters
        var_val_pairs = ann_parser.AnnParser(self.perf_params).parse(self.module_body_code)

        # get all needed performance parameters
        table = dict(var_val_pairs)
        if 'tile_sizes' not in table:
            err('orio.module.pluto.pluto: Pluto: missing "tile_sizes" parameter' )
        if 'tile_level' not in table:
            err('orio.module.pluto.pluto: Pluto: missing "tile_level" parameter' )
        if 'unroll_factor' not in table:
            err('orio.module.pluto.pluto: Pluto: missing "unroll_factor" parameter' )
        if 'vectorize' not in table:
            err('orio.module.pluto.pluto: Pluto: missing "vectorize" parameter' )
        tile_sizes = table['tile_sizes']
        tile_level = table['tile_level']
        unroll_factor = table['unroll_factor']
        vectorize = table['vectorize']

        # sanity check of the obtained performance parameters
        for t in tile_sizes:
            if not isinstance(t, int) or t<=0:
                err('orio.module.pluto.pluto: Pluto: tile size must be a positive integer' )
        if not isinstance(tile_level, int) or not (0<=tile_level<=2) :
            err('orio.module.pluto.pluto: Pluto: number of tile levels must be either 0, 1, or 2' )
        if not isinstance(unroll_factor, int) or unroll_factor<1:
            err('orio.module.pluto.pluto: Pluto: invalid unroll factor: %s'  % unroll_factor)
        if not isinstance(vectorize, int) or not (vectorize==0 or vectorize==1):
            err('orio.module.pluto.pluto: Pluto: vectorize value must be either 0 or 1')
        
        # initialize the code to be transformed
        code = self.annot_body_code
        
        # the used tags
        pluto_open_tag_re = r'/\*\s*pluto\s+start.*?\*/'
        pluto_close_tag_re = r'/\*\s*pluto\s+end.*?\*/'

        # find the opening and closing tags of the pluto code  
        open_m = re.search(pluto_open_tag_re, code)
        close_m = re.search(pluto_close_tag_re, code)
        if (not open_m) or (not close_m):
            err('orio.module.pluto.pluto: cannot find the opening and closing tags for the Pluto code')
            
        # check if Pluto has been correctly installed  
        if os.popen('polycc').read() == '':
            err('orio.module.pluto.pluto:  Pluto is not installed. Cannot use "polycc" command.')

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
            err('orio.module.pluto.pluto:  cannot write to file: %s' % ts_fname)

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
            err('orio.module.pluto.pluto:  cannot open file for writing: %s' % fname)

        # run Pluto
        cmd = 'polycc %s %s' % (fname, cmd_args)
        info('orio.module.pluto.pluto: running Pluto with command: %s' % cmd,level=1)
        try:
            os.system(cmd)
        except:
            err('orio.module.pluto.pluto:  failed to run command: %s' % cmd)
        
        # delete unneeded files
        path_name, ext = os.path.splitext(fname)
        removed_fnames = [fname] + glob.glob(path_name + '.kernel.*')
        if use_tiling:
            removed_fnames += [ts_fname]
        for f in removed_fnames:
            try:
                os.unlink(f)
            except:
                err('orio.module.pluto.pluto:  failed to remove file: %s' % f)

        # get the Pluto-generated code
        plutogen_fnames = glob.glob(path_name + '.*' + ext)
        if len(plutogen_fnames) != 1:
            err('orio.module.pluto.pluto:  failed to generate Pluto-transformed code')
        plutogen_fname = plutogen_fnames[0]
        try:
            f = open(plutogen_fname, 'r')
            pluto_code = f.read()
            f.close()
        except:
            err('orio.module.pluto.pluto:  cannot open file for writing: %s' % fname)

        # delete the Pluto-generated file
        try:
            os.unlink(plutogen_fname)
        except:
            err('orio.module.pluto.pluto:  failed to remove file: %s' % plutogen_fname)

        # remove some macro definitions
        line_pos = pluto_code.index('\n')
        pluto_code = '\n' + pluto_code[line_pos+1:]

        # return the Pluto-generated code
        return pluto_code

