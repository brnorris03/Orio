from orio.module.loops.submodule.submodule import SubModule
import orio.module.loops.submodule.pack.transformation as transformation
import orio.main.util.globals as g
from functools import reduce

#----------------------------------------------------------------------------------------------------------------------
class Pack(SubModule):


    def __init__(self, perf_params=None, transf_args=None, stmt=None, language='C', tinfo=None):
        super(Pack, self).__init__(perf_params, transf_args, stmt, language, tinfo)

    #--------------------------------------------------------------------------

    def readTransfArgs(self, perf_params, transf_args):
        '''Process the given transformation arguments'''

        # expected argument names
        PREFETCH = 'prefetch'
        DISTANCE = 'prefetch_distance'

        # default argument values
        prefetches = []
        dist = 0

        # iterate over all transformation arguments
        for aname, rhs, line_no in transf_args:

            # evaluate the RHS expression
            try:
                rhs = eval(str(rhs), perf_params)
            except Exception as e:
                g.err(__name__+': at line %s, failed to evaluate the argument expression: %s\n --> %s: %s'
                      % (line_no, rhs, e.__class__.__name__, e))

            if aname == PREFETCH:
                prefetches += list(rhs)

            elif aname == DISTANCE:
                if not isinstance(rhs, int):
                    g.err(__name__+': %s: %s must be a positive integer: %s\n' % (line_no, aname, rhs))
                else:
                    dist = rhs

            # unknown argument name
            else:
                g.err(__name__+': %s: unrecognized transformation argument: "%s"' % (line_no, aname))

        # check semantics of the transformation arguments
        #argss = self.checkTransfArgs(args)
        
        # return information about the transformation arguments
        return {PREFETCH:prefetches,DISTANCE:dist}

    #--------------------------------------------------------------------------

    def checkTransfArgs(self, args):
        '''Check the semantics of the given transformation arguments'''

        # evaluate the arguments
        checked = []
        for rhs, line_no in args:
            if isinstance(rhs, str):
                checked = [rhs]
            else:
                if ((not isinstance(rhs, list) and not isinstance(rhs, tuple)) or
                    not reduce(lambda x,y: x and y, [isinstance(x, str) for x in rhs], True)):
                    g.err(__name__+':%s: pragma directives must be a list/tuple of strings: %s'
                          % (line_no, rhs))
                checked = rhs

        # return information about the transformation arguments
        return checked

    #--------------------------------------------------------------------------

    def transform(self):
        '''Performs code transformations'''

        # read all transformation arguments
        args = self.readTransfArgs(self.perf_params, self.transf_args)

        # perform the transformation 
        t = transformation.Transformation(self.stmt, args)
        transformed_stmt = t.transform()

        # return the transformed statement
        return transformed_stmt
#----------------------------------------------------------------------------------------------------------------------


