#
# Loop transformation submodule.that implements loop permutation/interchange
#

import sys
import orio.module.loop.submodule.submodule
from orio.module.loop.submodule.permut import transformation
from orio.main.util.globals import *

#---------------------------------------------------------------------

class Permut(orio.module.loop.submodule.submodule.SubModule):
    '''The loop permutation transformation submodule.'''
    
    def __init__(self, perf_params = None, transf_args = None, stmt = None, language='C'):
        '''To instantiate a loop permutation transformation submodule.'''
        
        orio.module.loop.submodule.submodule.SubModule.__init__(self, perf_params, transf_args, stmt, language)

    #-----------------------------------------------------------------

    def readTransfArgs(self, perf_params, transf_args):
        '''Process the given transformation arguments'''

        # all expected argument names
        SEQ = 'seq'

        # all expected transformation arguments
        seq = None

        # iterate over all transformation arguments
        for aname, rhs, line_no in transf_args:

            # evaluate the RHS expression
            try:
                rhs = eval(rhs, perf_params)
            except Exception as e:
                err('orio.module.loop.submodule.permut.permut: %s: failed to evaluate the argument expression: %s\n --> %s: %s' % (line_no, rhs,e.__class__.__name__, e))
                
            # permutation sequence
            if aname == SEQ:
                seq = (rhs, line_no)
                
            # unknown argument name
            else:
                err('orio.module.loop.submodule.permut.permut: %s: unrecognized transformation argument: "%s"' % (line_no, aname))

        # check for undefined transformation arguments
        if seq == None:
            err('orio.module.loop.submodule.permut.permut: %s: missing permutation sequence argument' % self.__class__.__name__)

        # check semantics of the transformation arguments
        seq, = self.checkTransfArgs(seq)
        
        # return information about the transformation arguments
        return (seq, )

    #-----------------------------------------------------------------

    def checkTransfArgs(self, seq):
        '''Check the semantics of the given transformation arguments'''

        # evaluate the permutation sequence
        rhs, line_no = seq
        if not isinstance(rhs, list) and not isinstance(rhs, tuple):
            err('orio.module.loop.submodule.permut.permut:%s: permutation sequence must be a list/tuple of loop index names: %s' %
                   (line_no, rhs))
        inames = {}
        for i in rhs:
            if isinstance(i, str):
                pass
            elif isinstance(i, list) and len(i) == 1 and isinstance(i[0], str):
                i = i[0]
            else:
                err('orio.module.loop.submodule.permut.permut:%s: invalid element of the permutation sequence: %s' %
                       (line_no, i))
            if i in inames:
                err('orio.module.loop.submodule.permut.permut:%s: permutation sequence contains repeated loop index: %s' %
                       (line_no, i))
            inames[i] = None
        seq = rhs
        
        # return information about the transformation arguments
        return (seq, )

    #-----------------------------------------------------------------

    def permute(self, seq, stmt):
        '''To apply loop permutation/interchange transformation'''

        # perform the loop permutation transformation
        t = transformation.Transformation(seq, stmt)
        transformed_stmt = t.transform()

        # return the transformed statement
        return transformed_stmt

    #-----------------------------------------------------------------

    def transform(self):
        '''To perform code transformations'''

        # read all transformation arguments
        seq, = self.readTransfArgs(self.perf_params, self.transf_args)

        # perform the loop permutation transformation
        transformed_stmt = self.permute(seq, self.stmt)

        if not transformed_stmt.meta.get('id') and self.stmt.meta.get('id'):
            transformed_stmt.meta['id'] = 'loop_' + self.stmt.meta['id']
        
        # return the transformed statement
        return transformed_stmt



    
