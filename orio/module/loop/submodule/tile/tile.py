#
# Loop tiling transformation
#

import sys
import orio.module.loop.submodule.submodule, transformation
from orio.main.util.globals import *

#---------------------------------------------------------------------

class Tile(orio.module.loop.submodule.submodule.SubModule):
    '''The loop tiling transformation submodule.'''
    
    def __init__(self, perf_params = None, transf_args = None, stmt = None, language='C'):
        '''To instantiate a loop tiling transformation submodule.'''
        
        orio.module.loop.submodule.submodule.SubModule.__init__(self, perf_params, transf_args, stmt)
        
    #-----------------------------------------------------------------
    
    def readTransfArgs(self, perf_params, transf_args):
        '''Process the given transformation arguments'''

        # all expected argument names
        TSIZE = 'tsize'
        TINDEX = 'tindex'
        
        # all expected transformation arguments
        tsize = None
        tindex = None

        # iterate over all transformation arguments
        for aname, rhs, line_no in transf_args:

            # evaluate the RHS expression
            try:
                rhs = eval(rhs, perf_params)
            except Exception, e:
                err('orio.module.loop.submodule.tile.tile: %s: failed to evaluate the argument expression: %s\n --> %s: %s' 
                    % (line_no, rhs,e.__class__.__name__, e))

            # tile size
            if aname == TSIZE:
                tsize = (rhs, line_no)
                
            # tile loop index name
            elif aname == TINDEX:
                tindex = (rhs, line_no)

            # unknown argument name
            else:
                err('orio.module.loop.submodule.tile.tile: %s: unrecognized transformation argument: "%s"' % (line_no, aname))

        # check for undefined transformation arguments
        if tsize == None:
            err('orio.module.loop.submodule.tile.tile: %s: missing tile size argument' % self.__class__.__name__)
        if tindex == None:
            err('orio.module.loop.submodule.tile.tile: %s: missing tile loop index name argument' % self.__class__.__name__)

        # check semantics of the transformation arguments
        tsize, tindex = self.checkTransfArgs(tsize, tindex)
        
        # return information about the transformation arguments
        return (tsize, tindex)

    #-----------------------------------------------------------------

    def checkTransfArgs(self, tsize, tindex):
        '''Check the semantics of the given transformation arguments'''
    
        # evaluate the tile size
        rhs, line_no = tsize
        if not isinstance(rhs, int) or rhs <= 0:
            err('orio.module.loop.submodule.tile.tile: %s: tile size must be a positive integer: %s' % (line_no, rhs))
        tsize = rhs
            
        # evaluate the tile loop index name
        rhs, line_no = tindex
        if not isinstance(rhs, str):
            err('orio.module.loop.submodule.tile.tile: %s: tile loop index name must be a string: %s' % (line_no, rhs))
        tindex = rhs

        # return information about the transformation arguments
        return (tsize, tindex)
                
    #-----------------------------------------------------------------

    def tile(self, tsize, tindex, stmt):
        '''To apply loop tiling transformation'''
        
        # perform the loop tiling transformation
        t = transformation.Transformation(tsize, tindex, stmt)
        transformed_stmt = t.transform()

        try:
            if not transformed_stmt.label and stmt.label:
                transformed_stmt.label = stmt.label
        except Exception,e:
            err('orio.module.loop.submodule.tile.tile: error assigning label\n --> %s : %s' % (e.__class__.__name__, e.message))

        # return the transformed statement
        debug("SUCCESS tile:", obj=self)
        return transformed_stmt
    
    #-----------------------------------------------------------------

    def transform(self):
        '''To perform code transformations'''

        # read all transformation arguments
        tsize, tindex = self.readTransfArgs(self.perf_params, self.transf_args)

        # perform the loop tiling transformation
        transformed_stmt = self.tile(tsize, tindex, self.stmt)

        if not transformed_stmt.meta.get('id') and self.stmt.meta.get('id'):
            transformed_stmt.meta['id'] = 'loop_' + self.stmt.meta['id']

        # return the transformed statement
        return transformed_stmt



    
