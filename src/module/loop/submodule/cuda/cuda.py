#
# Loop transformation submodule.that implements CUDA kernel generation
#

import sys
import orio.module.loop.submodule.submodule, orio.module.loop.ast_lib.forloop_lib, orio.module.loop.ast_lib.common_lib
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
        index_id, _, ubound_exp, _, loop_body = orio.module.loop.ast_lib.forloop_lib.ForLoopLib().extractForLoopInfo(stmt)

        loop_lib = orio.module.loop.ast_lib.common_lib.CommonLib()
        
        # collect all identifiers from the loop's upper bound expression
        collectIdents = lambda n: [n.name] if isinstance(n, IdentExp) else []
        ubound_ids = loop_lib.collectNode(collectIdents, ubound_exp)
        
        # create decls for ubound_exp id's, assuming all ids are int's
        ubound_id_decls = [FieldDecl('int*', x) for x in ubound_ids]

        # add dereferences to all id's in the ubound_exp
        addDerefs = lambda n: ParenthExp(UnaryExp(n, UnaryExp.DEREF)) if isinstance(n, IdentExp) else n
        loop_lib.rewriteNode(addDerefs, ubound_exp)
        
        # collect all identifiers from the loop body
        loop_body_ids = loop_lib.collectNode(collectIdents, loop_body)
        lbi = filter(lambda x: x != index_id.name, loop_body_ids)
        
        # create decls for loop body id's
        lbi_decls = [FieldDecl('double*', x) for x in set(lbi)]
        
        # add dereferences to all non-array id's in the loop body
        collectArrayIdents = lambda n: [n.exp.name] if (isinstance(n, ArrayRefExp) and isinstance(n.exp, IdentExp)) else []
        array_ids = loop_lib.collectNode(collectArrayIdents, loop_body)
        non_array_ids = list(set(lbi).difference(set(array_ids)))
        print non_array_ids
        addDerefs2 = lambda n: ParenthExp(UnaryExp(n, UnaryExp.DEREF)) if (isinstance(n, IdentExp) and n.name in non_array_ids) else n
        loop_body2 = loop_lib.rewriteNode(addDerefs2, loop_body)

        # replace all array indices with thread id
        tid = 'tid'
        rewriteToTid = lambda x: IdentExp(tid) if isinstance(x, IdentExp) else x
        rewriteArrayIndices = lambda n: ArrayRefExp(n.exp, loop_lib.rewriteNode(rewriteToTid, n.sub_exp)) if isinstance(n, ArrayRefExp) else n
        loop_body3 = loop_lib.rewriteNode(rewriteArrayIndices, loop_body2)

        # generate the transformed statement
        # temp_id = FieldDecl('double*', 'orcuda_arg_'+str(Globals().getcounter())),
        decl_tid = VarDecl('int', [tid])
        assign_tid = AssignStmt(tid,
                                BinOpExp(BinOpExp(IdentExp('blockIdx.x'), # constant
                                                  IdentExp('blockDim.x'), # constant
                                                  BinOpExp.MUL),
                                         IdentExp('threadIdx.x'),         # constant
                                         BinOpExp.ADD)
                                )
        if_stmt = IfStmt(BinOpExp(IdentExp(tid), ubound_exp, BinOpExp.LE), loop_body3)

        dev_kernel = FunDecl('orcuda_kern_'+str(Globals().getcounter()),
                              'void',
                              ['__global__'],
                              ubound_id_decls + lbi_decls,
                              CompStmt([decl_tid,assign_tid,if_stmt]))
        
        # TODO: refactor, make this more graceful
        Globals().cunit_declarations += codegen.CodeGen('cuda').generator.generate(dev_kernel, '', '  ')

        transformed_stmt = Comment('placeholder for cuda resource marshaling code')

        return transformed_stmt
    
    #-----------------------------------------------------------------

    def transform(self):
        '''The implementation of the abstract transform method for CUDA'''

        # read all transformation arguments
        threadCount, blockCount = self.readTransfArgs(self.perf_params, self.transf_args)
        
        # perform the transformation of the statement
        transformed_stmt = self.cuda(self.stmt, threadCount, blockCount)
        
        return transformed_stmt
