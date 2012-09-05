from orio.module.loops.submodule.submodule import SubModule
import orio.module.loops.submodule.pack.transformation as transformation
#import orio.main.util.globals as g

#----------------------------------------------------------------------------------------------------------------------
class Pack(SubModule):


    def __init__(self, perf_params=None, transf_args=None, stmt=None, language='C', tinfo=None):
        super(Pack, self).__init__(perf_params, transf_args, stmt, language, tinfo)

    #--------------------------------------------------------------------------

    def readTransfArgs(self, perf_params, transf_args):
        '''Process the given transformation arguments'''

        # expected argument names

        # all expected transformation arguments
        args = []

        # check semantics of the transformation arguments
        args = self.checkTransfArgs(args)
        
        # return information about the transformation arguments
        return args

    #--------------------------------------------------------------------------

    def checkTransfArgs(self, args):
        '''Check the semantics of the given transformation arguments'''

        # evaluate the pragma directives
        _ = args
        checked = []

        # return information about the transformation arguments
        return checked

    #--------------------------------------------------------------------------

    def transform(self):
        '''Performs code transformations'''

        # read all transformation arguments
        args = self.readTransfArgs(self.perf_params, self.transf_args)

        # perform the transformation 
        t = transformation.Transformation(args, self.stmt)
        transformed_stmt = t.transform()

        # return the transformed statement
        return transformed_stmt
#----------------------------------------------------------------------------------------------------------------------


