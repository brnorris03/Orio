#
# Loop transformation submodule.that implements CUDA kernel generation
#

import sys
import orio.module.loop.submodule.submodule, orio.module.loop.ast_lib.forloop_lib
from orio.main.util.globals import *
from orio.module.loop.ast import *

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
                err('orio.module.loop.submodule.cuda.cuda: %s: failed to evaluate the argument expression: %s\n --> %s: %s' % (line_no, rhs,e.__class__.__name__, e))
                
            # thread count
            if aname == THREADCOUNT:
                threadCount = (rhs, line_no)
    
            # max number of blocks
            elif aname == MAXBLOCKS:
                maxBlocks = (rhs, line_no)
    
            # unknown argument name
            else:
                err('orio.module.loop.submodule.cuda.cuda: %s: unrecognized transformation argument: "%s"' % (line_no, aname))

        # check for initialization of mandatory transformation arguments
        if threadCount == None:
            err('orio.module.loop.submodule.cuda.cuda: %s: missing threadCount argument' % self.__class__.__name__)
        elif maxBlocks == None:
            err('orio.module.loop.submodule.cuda.cuda: %s: missing maxBlocks argument' % self.__class__.__name__)

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
            err('orio.module.loop.submodule.cuda.cuda: %s: threadCount must be a positive integer: %s' % (line_no, rhs))
        threadCount = rhs

        # sanity check
        rhs, line_no = maxBlocks
        if not isinstance(rhs, int) or rhs <= 0:
            err('orio.module.loop.submodule.cuda.cuda: %s: maxBlocks must be a positive integer: %s' % (line_no, rhs))
        maxBlocks = rhs

        # return information about the transformation arguments
        return (threadCount, maxBlocks)

    #-----------------------------------------------------------------

    def cuda(self, stmt, threadCount, blockCount):
        '''Apply CUDA transformations'''
        
        # get rid of compound statement that contains only a single statement
        while isinstance(stmt.stmt, orio.module.loop.ast.CompStmt) and len(stmt.stmt.stmts) == 1:
            stmt.stmt = stmt.stmt.stmts[0]
        
        # extract for-loop structure
        for_loop_info = orio.module.loop.ast_lib.forloop_lib.ForLoopLib().extractForLoopInfo(stmt)
        index_id, lbound_exp, ubound_exp, stride_exp, loop_body = for_loop_info
        

        # generate the transformed statement
        arg_prefix = 'orcuda_arg_'
        args = [
                # TODO: parse loop body, tie every id to its type, declare every id as a pointer-based arg
                FieldDecl('double*', arg_prefix+str(Globals().getcounter())),
                FieldDecl('double*', arg_prefix+str(Globals().getcounter()))
                ]
        tid = 'tid'
        decl_tid = VarDecl('int', [tid])
        assign_tid = AssignStmt(tid,
                                BinOpExp(BinOpExp(IdentExp('blockIdx.x'), # constant
                                                  IdentExp('blockDim.x'), # constant
                                                  BinOpExp.MUL),
                                         IdentExp('threadIdx.x'),         # constant
                                         BinOpExp.ADD)
                                )
        if_stmt = IfStmt(BinOpExp(IdentExp(tid), ubound_exp, BinOpExp.LE),
                         # TODO: traverse the loop and replace all indices with tid
                         loop_body
                         )
        # TODO: pull this FunDecl out of the enclosing FunDecl and make it a sibling, instead of a child
        dev_kernel = FunDecl('orcuda_kern_'+str(Globals().getcounter()),
                              'void',
                              ['__global__'],
                              args,
                              CompStmt([decl_tid,assign_tid,if_stmt]))
        
        transformed_stmt = dev_kernel

        return transformed_stmt
    
    #-----------------------------------------------------------------

    def transform(self):
        '''The implementation of the abstract transform method for CUDA'''

        # read all transformation arguments
        threadCount, blockCount = self.readTransfArgs(self.perf_params, self.transf_args)
        
        # perform the transformation of the statement
        transformed_stmt = self.cuda(self.stmt, threadCount, blockCount)
        
        return transformed_stmt
