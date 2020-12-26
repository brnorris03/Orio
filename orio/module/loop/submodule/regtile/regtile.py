#
# Loop transformation submodule.that implements a general approach of register tiling (also
# called loop unrolling-and-jamming), which can handle loop transformation in non-rectangular
# iteration spaces.
#

import sys
import orio.module.loop.submodule.submodule
from orio.module.loop.submodule.regtile import transformation
from orio.main.util.globals import *

#---------------------------------------------------------------------

class RegTile(orio.module.loop.submodule.submodule.SubModule):
    '''The register tiling transformation submodule.'''
    
    def __init__(self, perf_params = None, transf_args = None, stmt = None, language='C'):
        '''Instantiate a register tiling transformation submodule.'''
        
        orio.module.loop.submodule.submodule.SubModule.__init__(self, perf_params, transf_args, stmt, language)

    #-----------------------------------------------------------------
    
    def readTransfArgs(self, perf_params, transf_args):
        '''Process the given transformation arguments'''

        # all expected argument names
        LOOPS = 'loops'
        UFACTORS = 'ufactors'

        # all expected transformation arguments
        loops = None
        ufactors = None

        # iterate over all transformation arguments
        for aname, rhs, line_no in transf_args:

            # evaluate the RHS expression
            try:
                rhs = eval(rhs, perf_params)
            except Exception as e:
                err('orio.module.loop.submodule.regtile.regtile: %s: failed to evaluate the argument expression: %s\n --> %s: %s' 
                    % (line_no, rhs,e.__class__.__name__, e))
                
            # unroll factors
            if aname == LOOPS:
                loops = (rhs, line_no)
    
            # unroll factors
            elif aname == UFACTORS:
                ufactors = (rhs, line_no)
    
            # unknown argument name
            else:
                err('orio.module.loop.submodule.regtile.regtile: %s: unrecognized transformation argument: "%s"' % (line_no, aname))

        # check for undefined transformation arguments
        if loops == None:
            err('orio.module.loop.submodule.regtile.regtile: %s: missing loops argument' % self.__class__.__name__)
        if ufactors == None:
            err('orio.module.loop.submodule.regtile.regtile: %s: missing unroll factors argument' % self.__class__.__name__)

        # check semantics of the transformation arguments
        loops, ufactors = self.checkTransfArgs(loops, ufactors)
                
        # return information about the transformation arguments
        return (loops, ufactors)

    #-----------------------------------------------------------------

    def checkTransfArgs(self, loops, ufactors):
        '''Check the semantics of the given transformation arguments'''
        
        # evaluate the unroll factors
        rhs, line_no = loops
        if not isinstance(rhs, list) and not isinstance(rhs, tuple):
            err('orio.module.loop.submodule.regtile.regtile: %s: loops value must be a list/tuple: %s' % (line_no, rhs))
        for e in rhs:
            if not isinstance(e, str):
                err('orio.module.loop.submodule.regtile.regtile: %s: loops element must be a string, found: %s' % (line_no, e))
        for e in rhs:
            if rhs.count(e) > 1:
                err('orio.module.loop.submodule.regtile.regtile: %s: loops value contains duplication: "%s"' % (line_no, e))
        loops = rhs

        # evaluate the unroll factors
        rhs, line_no = ufactors
        if not isinstance(rhs, list) and not isinstance(rhs, tuple):
            err('orio.module.loop.submodule.regtile.regtile: %s: unroll factors value must be a list/tuple: %s' % (line_no, rhs))
        for e in rhs:
            if not isinstance(e, int) or e <= 0:
                err('orio.module.loop.submodule.regtile.regtile: %s: unroll factor must be a positive integer, found: %s' % (line_no, e))
        ufactors = rhs

        # compare the loops and unroll factors
        if len(loops) != len(ufactors):
            err('orio.module.loop.submodule.regtile.regtile: %s: mismatch on the number of loops and unroll factors' % line_no)

        # return information about the transformation arguments
        return (loops, ufactors)

    #-----------------------------------------------------------------

    def tileForRegs(self, loops, ufactors, stmt):
        '''To apply register tiling transformation'''

        # perform the register tiling transformation
        t = transformation.Transformation(loops, ufactors, stmt)
        transformed_stmt = t.transform()

        # return the transformed statement
        return transformed_stmt

    #-----------------------------------------------------------------

    def transform(self):
        '''To perform code transformations'''

        # read all transformation arguments
        loops, ufactors = self.readTransfArgs(self.perf_params, self.transf_args)

        # perform the register tiling transformation
        transformed_stmt = self.tileForRegs(loops, ufactors, self.stmt)

        if not transformed_stmt.meta.get('id') and self.stmt.meta.get('id'):
            transformed_stmt.meta['id'] = 'loop_' + self.stmt.meta['id']

        # return the transformed statement
        return transformed_stmt



    
