#
# Loop transformation submodule.that implements unroll and jam
#

import sys
import orio.module.loop.submodule.submodule, transformation
from orio.main.util.globals import *

#---------------------------------------------------------------------

class UnrollJam(orio.module.loop.submodule.submodule.SubModule):
    '''The unroll-and-jam transformation submodule.'''
    
    def __init__(self, perf_params = None, transf_args = None, stmt = None, language='C'):
        '''To instantiate an unroll-and-jam transformation submodule.'''
        
        orio.module.loop.submodule.submodule.SubModule.__init__(self, perf_params, transf_args, stmt, language)

    #-----------------------------------------------------------------
    
    def readTransfArgs(self, perf_params, transf_args):
        '''Process the given transformation arguments'''

        # all expected argument names
        UFACTOR = 'ufactor'
        PARALLEL = 'parallelize'

        # all expected transformation arguments
        ufactor = None
        parallelize = (False, None)

        # iterate over all transformation arguments
        for aname, rhs, line_no in transf_args:

            # evaluate the RHS expression
            try:
                rhs = eval(rhs, perf_params)
            except Exception, e:
                err('orio.module.loop.submodule.unrolljam.unrolljam: %s: failed to evaluate the argument expression: %s\n --> %s: %s' % (line_no, rhs,e.__class__.__name__, e))
                
            # unroll factor
            if aname == UFACTOR:
                ufactor = (rhs, line_no)
    
            # need to parallelize the loop
            elif aname == PARALLEL:
                parallelize = (rhs, line_no)
    
            # unknown argument name
            else:
                err('orio.module.loop.submodule.unrolljam.unrolljam: %s: unrecognized transformation argument: "%s"' % (line_no, aname))

        # check for undefined transformation arguments
        if ufactor == None:
            err('orio.module.loop.submodule.unrolljam.unrolljam: %s: missing unroll factor argument' % self.__class__.__name__)

        # check semantics of the transformation arguments
        ufactor, parallelize = self.checkTransfArgs(ufactor, parallelize)

        # return information about the transformation arguments
        return (ufactor, parallelize)

    #-----------------------------------------------------------------

    def checkTransfArgs(self, ufactor, parallelize):
        '''Check the semantics of the given transformation arguments'''
        
        # evaluate the unroll factor
        rhs, line_no = ufactor
        if not isinstance(rhs, int) or rhs <= 0:
            err('orio.module.loop.submodule.unrolljam.unrolljam: %s: unroll factor must be a positive integer: %s' % (line_no, rhs))
        ufactor = rhs

        # evaluate the parallelization indicator
        rhs, line_no = parallelize
        if not isinstance(rhs, bool):
            err('orio.module.loop.submodule.unrolljam.unrolljam: %s: loop parallelization value must be a boolean: %s' % (line_no, rhs))
        parallelize = rhs

        # return information about the transformation arguments
        return (ufactor, parallelize)

    #-----------------------------------------------------------------

    def unrollAndJam(self, ufactor, do_jamming, stmt, parallelize):
        '''To apply unroll-and-jam transformation'''
        
        debug('orio.module.loop.submodule.unrolljam.UnrollJam: starting unrollAndJam')

        # perform the unroll-and-jam transformation
        t = transformation.Transformation(ufactor, do_jamming, stmt, parallelize)
        transformed_stmt = t.transform()

        # return the transformed statement
        return transformed_stmt

    #-----------------------------------------------------------------

    def transform(self):
        '''To perform code transformations'''

        # read all transformation arguments
        ufactor, parallelize = self.readTransfArgs(self.perf_params, self.transf_args)

        # perform the unroll-and-jam transformation
        transformed_stmt = self.unrollAndJam(ufactor, True, self.stmt, parallelize)

        if not transformed_stmt.meta.get('id') and self.stmt.meta.get('id'):
            transformed_stmt.meta['id'] = 'loop_' + self.stmt.meta['id']

        # return the transformed statement
        return transformed_stmt



    
