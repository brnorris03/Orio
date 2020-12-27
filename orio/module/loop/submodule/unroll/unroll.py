#
# Loop transformation submodule.that implements a pure loop unrolling
#

import sys
import orio.module.loop.submodule.submodule
import orio.module.loop.submodule.unrolljam.unrolljam

#---------------------------------------------------------------------

class Unroll(orio.module.loop.submodule.submodule.SubModule):
    '''The unrolling transformation submodule.'''
    
    def __init__(self, perf_params = None, transf_args = None, stmt = None, language='C'):
        '''To instantiate an unrolling transformation submodule.'''
        
        orio.module.loop.submodule.submodule.SubModule.__init__(self, perf_params, transf_args, stmt, language)

        self.ujam_smod = orio.module.loop.submodule.unrolljam.unrolljam.UnrollJam()
        
    #-----------------------------------------------------------------

    def readTransfArgs(self, perf_params, transf_args):
        '''Process the given transformation arguments'''
        return self.ujam_smod.readTransfArgs(perf_params, transf_args)

    #-----------------------------------------------------------------

    def unroll(self, ufactor, stmt, parallelize):
        '''To apply unroll-and-jam transformation'''
        return self.ujam_smod.unrollAndJam(ufactor, False, stmt, parallelize)
    
    #-----------------------------------------------------------------

    def transform(self):
        '''To perform code transformations'''

        # read all transformation arguments
        ufactor, parallelize = self.readTransfArgs(self.perf_params, self.transf_args)
        
        # perform the unroll-and-jam transformation
        transformed_stmt = self.unroll(ufactor, self.stmt, parallelize)

        if not transformed_stmt.meta.get('id') and self.stmt.meta.get('id'):
            transformed_stmt.meta['id'] = 'loop_' + self.stmt.meta['id']

        # return the transformed statement
        return transformed_stmt
                                                      
