from orio.module.loops.submodule.submodule import SubModule
import orio.module.loops.submodule.pragma.transformation as transformation
import orio.main.util.globals as g

#----------------------------------------------------------------------------------------------------------------------
class Pragma(SubModule):
    '''The pragma directive insertion submodule.'''

    def __init__(self, perf_params=None, transf_args=None, stmt=None, language='C', tinfo=None):
        super(Pragma, self).__init__(perf_params, transf_args, stmt, language, tinfo)

    #--------------------------------------------------------------------------

    def readTransfArgs(self, perf_params, transf_args):
        '''Process the given transformation arguments'''

        # expected argument names
        PRAGMAS = 'pragma_str'

        # all expected transformation arguments
        pragmas = []

        # iterate over all transformation arguments
        for aname, rhs, line_no in transf_args:

            # evaluate the RHS expression
            try:
                rhs = eval(str(rhs), perf_params)
            except Exception, e:
                g.err(__name__+': at line %s, failed to evaluate the argument expression: %s\n --> %s: %s'
                      % (line_no, rhs, e.__class__.__name__, e))

            # pragma directives
            if aname == PRAGMAS:
                pragmas = (rhs, line_no)

            # unknown argument name
            else:
                g.err(__name__+': %s: unrecognized transformation argument: "%s"' % (line_no, aname))

        # check semantics of the transformation arguments
        pragmas = self.checkTransfArgs(pragmas)
        
        # return information about the transformation arguments
        return pragmas

    #--------------------------------------------------------------------------

    def checkTransfArgs(self, pragmas):
        '''Check the semantics of the given transformation arguments'''

        # evaluate the pragma directives
        rhs, line_no = pragmas
        if isinstance(rhs, str):
            pragmas = [rhs]
        else:
            if ((not isinstance(rhs, list) and not isinstance(rhs, tuple)) or
                not reduce(lambda x,y: x and y, map(lambda x: isinstance(x, str), rhs), True)):
                g.err(__name__+':%s: pragma directives must be a list/tuple of strings: %s'
                      % (line_no, rhs))
            pragmas = rhs

        # return information about the transformation arguments
        return pragmas

    #--------------------------------------------------------------------------

    def transform(self):
        '''Performs code transformations'''

        # read all transformation arguments
        pragmas = self.readTransfArgs(self.perf_params, self.transf_args)

        # perform the transformation 
        t = transformation.Transformation(pragmas, self.stmt)
        transformed_stmt = t.transform()

        # return the transformed statement
        return transformed_stmt
#----------------------------------------------------------------------------------------------------------------------



