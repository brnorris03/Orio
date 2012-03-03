#
# A class used to perform polyhedral-based transformation by using the available tool,
# called Pluto. The used polyhedral transformation is loop tiling and automatic parallelization
# for multicore platforms.
#

import glob, re, os, sys
from orio.main.util.globals import *

#---------------------------------------------------------

class PolyTransformation:
    '''The polyhedral transformation'''

    def __init__(self, verbose, parallel, tiles):
        '''To instantiate a polyhedral transformation instance'''

        self.verbose = verbose
        self.tiles = tiles
        self.parallel = parallel
        
    #---------------------------------------------------------
    
    def __plutoTransform(self, code):
        '''Use Pluto to perform polyhedral transformations'''

        # check if Pluto has been correctly installed
        if os.popen('polycc').read() == '':
            err('orio.module.polysyn.poly_transformation:  Pluto is not installed. Cannot use "polycc" command.')

        # check loop tiling
        use_tiling = True
        if len(self.tiles) == 0:
            use_tiling = False

        # write the tile sizes into "tile.sizes" file
        if use_tiling:
            ts_fname = 'tile.sizes'
            content = ''
            for t in self.tiles:
                content += '%s\n' % t
            try:
                f = open(ts_fname, 'w')
                f.write(content)
                f.close()
            except:
                err('orio.module.polysyn.poly_transformation:  cannot write to file: %s' % ts_fname)
                
        # write the annotation body code into a file
        fname = '_orio_polysyn.c'
        try:
            f = open(fname, 'w')
            f.write(code)
            f.close()
        except:
            err('orio.module.polysyn.poly_transformation:  cannot open file for writing: %s' % fname)

        # create the Pluto command
        cmd = 'polycc %s --noprevector' % fname
        if self.parallel:
            cmd += ' --parallel'
        if use_tiling:
            cmd += ' --tile --l2tile'

        # execute Pluto
        info('orio.module.polysyn.poly_transformation running command:\n\t%s\n' % cmd)
        try:
            os.system(cmd)
        except:
            err('orio.module.polysyn.poly_transformation:  failed to run command: %s' % cmd)
   
        # delete unneeded files
        path_name, ext = os.path.splitext(fname)
        removed_fnames = [fname] + glob.glob(path_name + '.kernel.*')
        if use_tiling:
            removed_fnames += [ts_fname]
        for f in removed_fnames:
            try:
                os.unlink(f)
            except:
                err('orio.module.polysyn.poly_transformation:  failed to remove file: %s' % f)

        # get the Pluto-generated code
        plutogen_fname = path_name + '.tiled' + ext
        if not os.path.exists(plutogen_fname):
            err('orio.module.polysyn.poly_transformation:  failed to generate Pluto-transformed code')
        try:
            f = open(plutogen_fname, 'r')
            pluto_code = f.read()
            f.close()
        except:
            err('orio.module.polysyn.poly_transformation:  cannot open file for writing: %s' % fname)
            
        # delete the Pluto-generated file
        try:
            os.unlink(plutogen_fname)
        except:
            err('orio.module.polysyn.poly_transformation:  failed to remove file: %s' % plutogen_fname)

        # return the Pluto-generated code
        return pluto_code

    #---------------------------------------------------------
    
    def transform(self, code):
        '''To perform loop tiling and parallelization using Pluto'''
        
        # use Pluto to perform polyhedral transformations
        pluto_code = self.__plutoTransform(code)
        
        # return the Pluto-generated code
        return pluto_code
    
    
