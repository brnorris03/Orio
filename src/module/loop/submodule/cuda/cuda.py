#
# Loop transformation submodule.that implements CUDA kernel generation
#

import sys
import orio.module.loop.submodule.submodule

#---------------------------------------------------------------------

class CUDA(orio.module.loop.submodule.submodule.SubModule):
    '''The unrolling transformation submodule.'''
    
    def __init__(self, perf_params = None, transf_args = None, stmt = None, language='cuda'):
        '''To instantiate an unrolling transformation submodule.'''
        
        orio.module.loop.submodule.submodule.SubModule.__init__(self, perf_params, transf_args, stmt, language)

        # TODO: future transformations here, e.g., 
        #self.cudastream_smod = orio.module.loop.submodule.cudastream.cudastream.CudaStream()
        
    #-----------------------------------------------------------------

    def readTransfArgs(self, perf_params, transf_args):
        '''Process the given transformation arguments'''
        # TODO (e.g., see unrolljam transformation in the CPU submodules)
        return 

    #-----------------------------------------------------------------

    def cuda(self, ufactor, stmt, parallelize):
        '''Apply CUDA transformations'''
        # TODO
        return 
    
    #-----------------------------------------------------------------

    def transform(self):
        '''The implementation of the abstract transform method for CUDA'''

        # TODO
        # 1. Read all transformation arguments
        
        # 2. Perform the transformation of the statement
        
        # 3. Return the transformed statement
                                                      
        return
