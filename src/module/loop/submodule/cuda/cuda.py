#
# Loop transformation submodule.that implements CUDA kernel generation
#

import orio.module.loop.submodule.submodule
import orio.main.util.globals as g
import transformation

#---------------------------------------------------------------------

class CUDA(orio.module.loop.submodule.submodule.SubModule):
    '''The cuda transformation submodule.'''
    
    def __init__(self, perf_params = None, transf_args = None, stmt = None, language='cuda'):
        '''To instantiate the transformation submodule.'''
        
        orio.module.loop.submodule.submodule.SubModule.__init__(self, perf_params, transf_args, stmt, language)

        # TODO: future transformations here, e.g., 
        #self.cudastream_smod = orio.module.loop.submodule.cudastream.cudastream.CudaStream()
        
    #-----------------------------------------------------------------

    def readTransfArgs(self, perf_params, transf_args):
        '''Process the given transformation arguments'''

        # all expected argument names
        THREADCOUNT = 'threadCount'
        MAXBLOCKS = 'maxBlocks'

        # all expected transformation arguments
        threadCount = None
        maxBlocks = None

        # iterate over all transformation arguments
        for aname, rhs, line_no in transf_args:

            # evaluate the RHS expression
            try:
                rhs = eval(rhs, perf_params)
            except Exception, e:
                g.err('orio.module.loop.submodule.cuda.cuda: %s: failed to evaluate the argument expression: %s\n --> %s: %s' % (line_no, rhs,e.__class__.__name__, e))
                
            # thread count
            if aname == THREADCOUNT:
                threadCount = (rhs, line_no)
    
            # max number of blocks
            elif aname == MAXBLOCKS:
                maxBlocks = (rhs, line_no)
    
            # unknown argument name
            else:
                g.err('orio.module.loop.submodule.cuda.cuda: %s: unrecognized transformation argument: "%s"' % (line_no, aname))

        # check for initialization of mandatory transformation arguments
        if threadCount == None:
            g.err('orio.module.loop.submodule.cuda.cuda: %s: missing threadCount argument' % self.__class__.__name__)
        elif maxBlocks == None:
            g.err('orio.module.loop.submodule.cuda.cuda: %s: missing maxBlocks argument' % self.__class__.__name__)

        # check semantics of the transformation arguments
        threadCount, maxBlocks = self.checkTransfArgs(threadCount, maxBlocks)

        # return information about the transformation arguments
        return (threadCount, maxBlocks)

    #-----------------------------------------------------------------

    def checkTransfArgs(self, threadCount, maxBlocks):
        '''Check the semantics of the given transformation arguments'''
        
        # sanity check
        rhs, line_no = threadCount
        if not isinstance(rhs, int) or rhs <= 0:
            g.err('orio.module.loop.submodule.cuda.cuda: %s: threadCount must be a positive integer: %s' % (line_no, rhs))
        threadCount = rhs

        # sanity check
        rhs, line_no = maxBlocks
        if not isinstance(rhs, int) or rhs <= 0:
            g.err('orio.module.loop.submodule.cuda.cuda: %s: maxBlocks must be a positive integer: %s' % (line_no, rhs))
        maxBlocks = rhs

        # return information about the transformation arguments
        return (threadCount, maxBlocks)

    #-----------------------------------------------------------------

    def cudify(self, stmt, threadCount, blockCount):
        '''Apply CUDA transformations'''
        
        g.debug('orio.module.loop.submodule.cuda.CUDA: starting CUDA transformations')

        # perform transformation
        t = transformation.Transformation(stmt, threadCount, blockCount)
        transformed_stmt = t.transform()

        # return the transformed statement
        return transformed_stmt

    #-----------------------------------------------------------------

    def transform(self):
        '''The implementation of the abstract transform method for CUDA'''

        # read all transformation arguments
        threadCount, blockCount = self.readTransfArgs(self.perf_params, self.transf_args)
        
        # perform the transformation of the statement
        transformed_stmt = self.cudify(self.stmt, threadCount, blockCount)
        
        return transformed_stmt
