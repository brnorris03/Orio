#
# Loop transformation submodule.that enables pragma directive insertions.
#

import sys
import orio.module.loop.submodule.submodule, transformation
from orio.main.util.globals import *

#---------------------------------------------------------------------

class Pragma(orio.module.loop.submodule.submodule.SubModule):
    '''The pragma directive insertion submodule.'''
    
    def __init__(self, perf_params = None, transf_args = None, stmt = None, language='C'):
        '''To instantiate a pragma insertion submodule.'''

        orio.module.loop.submodule.submodule.SubModule.__init__(self, perf_params, transf_args, stmt, language)

    #-----------------------------------------------------------------

    def readTransfArgs(self, perf_params, transf_args):
        '''Process the given transformation arguments'''

        # all expected argument names
        PRAGMAS = 'pragma_str'

        # all expected transformation arguments
        pragmas = []

        # iterate over all transformation arguments
        for aname, rhs, line_no in transf_args:

            # evaluate the RHS expression
            try:
                rhs = eval(rhs, perf_params)
            except Exception, e:
                err('orio.module.loop.submodule.pragma.pragma: %s: failed to evaluate the argument expression: %s\n --> %s: %s' % (line_no, rhs,e.__class__.__name__, e))

            # pragma directives
            if aname == PRAGMAS:
                pragmas = (rhs, line_no)

            # unknown argument name
            else:
                err('orio.module.loop.submodule.pragma.pragma: %s: unrecognized transformation argument: "%s"' % (line_no, aname))

        # check semantics of the transformation arguments
        pragmas, = self.checkTransfArgs(pragmas)
        
        # return information about the transformation arguments
        return (pragmas, )

    #-----------------------------------------------------------------

    def checkTransfArgs(self, pragmas):
        '''Check the semantics of the given transformation arguments'''

        # evaluate the pragma directives
        rhs, line_no = pragmas
        if isinstance(rhs, str):
            pragmas = [rhs]
        else:
            if ((not isinstance(rhs, list) and not isinstance(rhs, tuple)) or
                not reduce(lambda x,y: x and y, map(lambda x: isinstance(x, str), rhs), True)):
                err('orio.module.loop.submodule.pragma.pragma:%s: pragma directives must be a list/tuple of strings: %s' %
                       (line_no, rhs))
            pragmas = rhs

        # return information about the transformation arguments
        return (pragmas, )

    #-----------------------------------------------------------------

    def insertPragmas(self, pragmas, stmt):
        '''To apply pragma directive insertion'''

        # perform the pragma directive insertion
        t = transformation.Transformation(pragmas, stmt)
        transformed_stmt = t.transform()

        # return the transformed statement
        return transformed_stmt

    #-----------------------------------------------------------------

    def transform(self):
        '''To perform code transformations'''

        # read all transformation arguments
        pragmas, = self.readTransfArgs(self.perf_params, self.transf_args)

        # perform the pragma directive insertion 
        transformed_stmt = self.insertPragmas(pragmas, self.stmt)

        # return the transformed statement
        return transformed_stmt



    
