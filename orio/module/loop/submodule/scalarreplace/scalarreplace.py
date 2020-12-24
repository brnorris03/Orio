#
# Scalar replacement transformation
#

import sys
from orio.main.util.globals import *
import orio.module.loop.submodule.submodule, transformation

#---------------------------------------------------------------------

class ScalarReplace(orio.module.loop.submodule.submodule.SubModule):
    '''The scalar replacement transformation submodule.'''
    
    def __init__(self, perf_params = None, transf_args = None, stmt = None, language='C'):
        '''Instantiate a scalar replacement transformation submodule.'''
        
        orio.module.loop.submodule.submodule.SubModule.__init__(self, perf_params, transf_args, stmt, language)
    
    #-----------------------------------------------------------------
    
    def readTransfArgs(self, perf_params, transf_args):
        '''Process the given transformation arguments'''

        # all expected argument names
        DTYPE = 'dtype'
        PREFIX = 'prefix'

        # all expected transformation arguments
        dtype = None
        prefix = None
        
        # iterate over all transformation arguments
        for aname, rhs, line_no in transf_args:

            # evaluate the RHS expression
            try:
                rhs = eval(rhs, perf_params)
            except Exception, e:
                err('orio.module.loop.submodule.scalarreplace.scalarreplace: %s: failed to evaluate the argument expression: %s\n --> %s: %s' 
                    % (line_no, rhs,e.__class__.__name__, e))
                
            # data type
            if aname in DTYPE:
                dtype = (rhs, line_no)

            # prefix name for scalars
            elif aname in PREFIX:
                prefix = (rhs, line_no)

            # unknown argument name
            else:
                err('orio.module.loop.submodule.scalarreplace.scalarreplace: %s: unrecognized transformation argument: "%s"' % (line_no, aname))

        # check semantics of the transformation arguments
        dtype, prefix = self.checkTransfArgs(dtype, prefix) 

        # return information about the transformation arguments
        return (dtype, prefix)

    #-----------------------------------------------------------------

    def checkTransfArgs(self, dtype, prefix):
        '''Check the semantics of the given transformation arguments'''
                
        # evaluate the data type
        if dtype != None:
            rhs, line_no = dtype
            if dtype != None and not isinstance(rhs, str):
                err('orio.module.loop.submodule.scalarreplace.scalarreplace: %s: data type argument must be a string: %s' % (line_no, rhs))
            dtype = rhs
        
        # evaluate the prefix name for scalars variables
        if prefix != None:
            rhs, line_no = prefix
            if rhs != None and not isinstance(rhs, str):
                err('orio.module.loop.submodule.scalarreplace.scalarreplace: %s: the prefix name of scalars must be a string: %s' % (line_no, rhs))
            prefix = rhs
            
        # return information about the transformation arguments
        return (dtype, prefix)

    #-----------------------------------------------------------------

    def replaceScalars(self, dtype, prefix, stmt):
        '''To apply scalar replacement transformation'''
        
        # perform the scalar replacement transformation
        t = transformation.Transformation(dtype, prefix, stmt)
        transformed_stmt = t.transform()
        
        # return the transformed statement
        return transformed_stmt
    
    #-----------------------------------------------------------------

    def transform(self):
        '''To perform code transformations'''

        # read all transformation arguments
        dtype, prefix = self.readTransfArgs(self.perf_params, self.transf_args)

        # perform the bound replacement transformation
        transformed_stmt = self.replaceScalars(dtype, prefix, self.stmt)

        if not transformed_stmt.meta.get('id') and self.stmt.meta.get('id'):
            transformed_stmt.meta['id'] = 'loop_' + self.stmt.meta['id']

        # return the transformed statement
        return transformed_stmt



    
