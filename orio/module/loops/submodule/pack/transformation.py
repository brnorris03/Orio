#import orio.module.loops.ast as ast

#----------------------------------------------------------------------------------------------------------------------
class Transformation:

    def __init__(self, args, stmt):
        self.stmt = stmt
        self.args = args

    #--------------------------------------------------------------------------

    def transform(self):

        if len(self.args) == 0:
            return self.stmt

        #
        transformed_stmt = self.stmt

        # return the transformed statement
        return transformed_stmt
#----------------------------------------------------------------------------------------------------------------------


