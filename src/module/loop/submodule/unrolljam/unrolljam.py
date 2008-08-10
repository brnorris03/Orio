#
# Loop transformation submodule that implements unroll and jam
#

import sys
import module.loop.submodule.submodule, transformator

#---------------------------------------------------------------------

class UnrollJam(module.loop.submodule.submodule.SubModule):
    '''The unroll-and-jam transformation submodule'''
    
    def __init__(self, perf_params = None, transf_args = None, stmt = None):
        '''To instantiate an unroll-and-jam transformation submodule'''
        
        module.loop.submodule.submodule.SubModule.__init__(self, perf_params, transf_args, stmt)

    #-----------------------------------------------------------------
    
    def readTransfArgs(self, perf_params, transf_args):
        '''Process the given transformation arguments'''

        # all expected argument names
        UFACTOR = 'ufactor'
        INIT_CLOOP = 'init_cleanup_loop'

        # all expected transformation arguments
        ufactor = None
        init_cleanup_loop = (False, None)

        # iterate over all transformation arguments
        for aname, rhs, line_no in transf_args:

            # evaluate the RHS expression
            try:
                rhs = eval(rhs, perf_params)
            except Exception, e:
                print 'error:%s: failed to evaluate the argument expression: %s' % (line_no, rhs)
                print ' --> %s: %s' % (e.__class__.__name__, e)
                sys.exit(1)
                
            # unroll factor
            if aname == UFACTOR:
                ufactor = (rhs, line_no)
    
            # need to initialize the cleanup loop
            elif aname == INIT_CLOOP:
                init_cleanup_loop = (rhs, line_no)
    
            # unknown argument name
            else:
                print 'error:%s: unrecognized transformation argument: "%s"' % (line_no, aname)
                sys.exit(1)

        # check for undefined transformation arguments
        if ufactor == None:
            print 'error:%s: missing unroll factor argument' % self.__class__.__name__
            sys.exit(1)

        # check semantics of the transformation arguments
        ufactor, init_cleanup_loop = self.checkTransfArgs(ufactor, init_cleanup_loop)

        # return information about the transformation arguments
        return (ufactor, init_cleanup_loop)

    #-----------------------------------------------------------------

    def checkTransfArgs(self, ufactor, init_cleanup_loop):
        '''Check the semantics of the given transformation arguments'''
        
        # evaluate the unroll factor
        rhs, line_no = ufactor
        if not isinstance(rhs, int) or rhs <= 0:
            print 'error:%s: unroll factor must be a positive integer: %s' % (line_no, rhs)
            sys.exit(1)
        ufactor = rhs

        # evaluate the initialization of cleanup loop
        rhs, line_no = init_cleanup_loop
        if (not isinstance(rhs, bool)) and (not isinstance(rhs, int)):
            print 'error:%s: invalid type of "init_cleanup_loop" value : %s' % (line_no, rhs)
            sys.exit(1)
        init_cleanup_loop = rhs

        # return information about the transformation arguments
        return (ufactor, init_cleanup_loop)

    #-----------------------------------------------------------------

    def unrollAndJam(self, ufactor, do_jamming, stmt, init_cleanup_loop):
        '''To apply unroll-and-jam transformation'''

        # perform the unroll-and-jam transformation
        t = transformator.Transformator(ufactor, do_jamming, stmt, init_cleanup_loop)
        transformed_stmt = t.transform()

        # return the transformed statement
        return transformed_stmt

    #-----------------------------------------------------------------

    def transform(self):
        '''To perform code transformations'''

        # read all transformation arguments
        ufactor, init_cleanup_loop = self.readTransfArgs(self.perf_params, self.transf_args)

        # perform the unroll-and-jam transformation
        transformed_stmt = self.unrollAndJam(ufactor, True, self.stmt, init_cleanup_loop)
        
        # return the transformed statement
        return transformed_stmt



    
