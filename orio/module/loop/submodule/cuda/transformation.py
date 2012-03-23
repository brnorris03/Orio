#
# Contains the CUDA transformation module
#

import orio.module.loop.ast_lib.forloop_lib, orio.module.loop.ast_lib.common_lib
import orio.main.util.globals as g
import orio.module.loop.ast as ast
from orio.module.loop.ast import *

#----------------------------------------------------------------------------------------------------------------------

class Transformation(object):
    '''Code transformation'''

    def __init__(self, stmt, devProps, targs):
        '''Instantiate a code transformation object'''
        self.stmt         = stmt
        self.devProps     = devProps
        self.threadCount  = targs['threadCount']
        self.cacheBlocks  = targs['cacheBlocks']
        self.pinHostMem   = targs['pinHostMem']
        self.streamCount  = targs['streamCount']
        self.domain       = targs['domain']
        self.dataOnDevice = targs['dataOnDevice']
        
        if self.streamCount > 1 and (self.devProps['deviceOverlap'] == 0 or self.devProps['asyncEngineCount'] == 0):
            g.err('orio.module.loop.submodule.cuda.transformation: ' +
                  'device cannot handle overlaps or concurrent data transfers; so no speedup from streams')
        
        # control flag to perform device-side timing
        self.doDeviceTiming = False
        
    def transform(self):
        '''Transform the enclosed for-loop'''
        # get rid of compound statement that contains only a single statement
        while isinstance(self.stmt.stmt, ast.CompStmt) and len(self.stmt.stmt.stmts) == 1:
            self.stmt.stmt = self.stmt.stmt.stmts[0]
        
        # extract for-loop structure
        index_id, _, ubound_exp, _, loop_body = orio.module.loop.ast_lib.forloop_lib.ForLoopLib().extractForLoopInfo(self.stmt)

        # abbreviations
        loop_lib = orio.module.loop.ast_lib.common_lib.CommonLib()
        int0 = ast.NumLitExp(0, ast.NumLitExp.INT)
        int1 = ast.NumLitExp(1, ast.NumLitExp.INT)
        int2 = ast.NumLitExp(2, ast.NumLitExp.INT)
        int3 = ast.NumLitExp(3, ast.NumLitExp.INT)
        prefix = 'orcu_'

        nthreadsIdent = ast.IdentExp('nthreads')
        nstreamsIdent = ast.IdentExp('nstreams')

        maxThreadsPerBlock       = self.devProps['maxThreadsPerBlock']
        maxThreadsPerBlockNumLit = ast.NumLitExp(str(maxThreadsPerBlock), ast.NumLitExp.INT)

        #--------------------------------------------------------------------------------------------------------------
        # analysis
        # collect all identifiers from the loop's upper bound expression
        collectIdents = lambda n: [n.name] if isinstance(n, ast.IdentExp) else []
        
        # collect all LHS identifiers within the loop body
        def collectLhsIds(n):
            if isinstance(n, ast.BinOpExp) and n.op_type == ast.BinOpExp.EQ_ASGN:
                if isinstance(n.lhs, ast.IdentExp):
                    return [n.lhs.name]
                elif isinstance(n.lhs, ast.ArrayRefExp) and isinstance(n.lhs.exp, ast.IdentExp):
                    return [n.lhs.exp.name]
                else: return []
            else: return []
        def collectRhsIds(n):
            if isinstance(n, ast.BinOpExp) and n.op_type == ast.BinOpExp.EQ_ASGN:
                return loop_lib.collectNode(collectIdents, n.rhs)
            else: return []
        lhs_ids = loop_lib.collectNode(collectLhsIds, loop_body)
        rhs_ids = loop_lib.collectNode(collectRhsIds, loop_body)

        # collect all array and non-array idents in the loop body
        collectArrayIdents = lambda n: [n.exp.name] if (isinstance(n, ast.ArrayRefExp) and isinstance(n.exp, ast.IdentExp)) else []
        array_ids = set(loop_lib.collectNode(collectArrayIdents, loop_body))
        lhs_array_ids = list(set(lhs_ids).intersection(array_ids))
        rhs_array_ids = list(set(rhs_ids).intersection(array_ids))
        isReduction = len(lhs_array_ids) == 0

        #--------------------------------------------------------------------------------------------------------------
        # in validation mode, output original code's results and (later on) compare against transformed code's results
        if g.Globals().validationMode and not g.Globals().executedOriginal:
          original = self.stmt.replicate()
          results  = []
          printFpIdent  = ast.IdentExp('fp')
          results += [
            ast.VarDeclInit('FILE*', printFpIdent,
                            ast.FunCallExp(ast.IdentExp('fopen'), [ast.StringLitExp('origexec.out'), ast.StringLitExp('w')]))
          ]
          results += [original]
          for var in lhs_ids:
            if var in array_ids:
              bodyStmts = [original.stmt]
              bodyStmts += [ast.ExpStmt(ast.FunCallExp(ast.IdentExp('fprintf'),
                [printFpIdent, ast.StringLitExp("\'"+var+"[%d]\',%f; "), index_id, ast.ArrayRefExp(ast.IdentExp(var), index_id)])
              )]
              original.stmt = ast.CompStmt(bodyStmts)
            else:
              results += [ast.ExpStmt(
                ast.FunCallExp(ast.IdentExp('fprintf'), [printFpIdent, ast.StringLitExp("\'"+var+"\',%f"), ast.IdentExp(var)])
              )]
          results += [ast.ExpStmt(ast.FunCallExp(ast.IdentExp('fclose'), [printFpIdent]))]
      
          return ast.CompStmt(results)
        
        #--------------------------------------------------------------------------------------------------------------
        # begin rewrite the loop body
        # create decls for ubound_exp id's, assuming all ids are int's
        ubound_ids = loop_lib.collectNode(collectIdents, ubound_exp)
        kernelParams = [ast.FieldDecl('int', x) for x in ubound_ids]
        host_arraysize = ubound_ids[0]
        hostVecLenIdent = ast.IdentExp(host_arraysize)
        scSizeIdent = ast.IdentExp('scSize')

        # collect all identifiers from the loop body
        loop_body_ids = loop_lib.collectNode(collectIdents, loop_body)
        lbi = set(filter(lambda x: x != index_id.name, loop_body_ids))
        
        # create decls for loop body id's
        if isReduction:
            lbi = lbi.difference(set(lhs_ids))
        kernelParams += [ast.FieldDecl('double*', x) for x in lbi]
        scalar_ids = list(lbi.difference(array_ids))
        
        kernel_temps = []
        if isReduction:
            for var in lhs_ids:
                tempIdent = ast.IdentExp(prefix + 'var' + str(g.Globals().getcounter()))
                kernel_temps += [tempIdent]
                rrLhs = lambda n: tempIdent if (isinstance(n, ast.IdentExp) and n.name == var) else n
                loop_body = loop_lib.rewriteNode(rrLhs, loop_body)

        # add dereferences to all non-array id's in the loop body
        addDerefs2 = lambda n: ast.ParenthExp(ast.UnaryExp(n, ast.UnaryExp.DEREF)) if (isinstance(n, ast.IdentExp) and n.name in scalar_ids) else n
        loop_body2 = loop_lib.rewriteNode(addDerefs2, loop_body)

        collectLhsExprs = lambda n: [n.lhs] if isinstance(n, ast.BinOpExp) and n.op_type == ast.BinOpExp.EQ_ASGN else []
        loop_lhs_exprs = loop_lib.collectNode(collectLhsExprs, loop_body2)

        # replace all array indices with thread id
        tid = 'tid'
        tidIdent = ast.IdentExp(tid)
        rewriteToTid = lambda x: tidIdent if isinstance(x, ast.IdentExp) and x.name == str(index_id) else x
        rewriteArrayIndices = lambda n: ast.ArrayRefExp(n.exp, loop_lib.rewriteNode(rewriteToTid, n.sub_exp)) if isinstance(n, ast.ArrayRefExp) else n
        loop_body3 = loop_lib.rewriteNode(rewriteArrayIndices, loop_body2)
        # end rewrite the loop body
        #--------------------------------------------------------------------------------------------------------------

        #--------------------------------------------------------------------------------------------------------------
        domainStmts   = []
        domainArgs    = []
        domainParams  = []
        if self.domain != None:
          domainStmts = [Comment('stencil domain parameters')]
          gridn = 'gn'
          gridm = 'gm'
          #gridp = 'gp'
          dof   = 'dof'
          nos   = 'nos'
          sidx  = 'sidx'
          gridmIdent = IdentExp(gridm)
          gridnIdent = IdentExp(gridn)
          #gridpIdent = IdentExp(gridp)
          dofIdent   = IdentExp(dof)
          nosIdent   = IdentExp(nos)
          sidxIdent  = IdentExp(sidx)
          int4       = NumLitExp(4, NumLitExp.INT)
          int5       = NumLitExp(5, NumLitExp.INT)
          int6       = NumLitExp(6, NumLitExp.INT)
          int7       = NumLitExp(7, NumLitExp.INT)
          domainStmts += [
            VarDeclInit('int', gridmIdent, FunCallExp(IdentExp('round'), [
              FunCallExp(IdentExp('pow'), [hostVecLenIdent, CastExpr('double', BinOpExp(int1, int3, BinOpExp.DIV))])
            ])),
            VarDeclInit('int', gridnIdent, gridmIdent),
            #VarDeclInit('int', gridpIdent, gridmIdent),
            VarDeclInit('int', dofIdent, int1), # 1 dof
            VarDeclInit('int', nosIdent, int7), # 7 stencil points
            VarDecl('int', [sidx + '[' + nos + ']']),
            ExpStmt(BinOpExp(ArrayRefExp(sidxIdent, int0), int0, BinOpExp.EQ_ASGN)),
            ExpStmt(BinOpExp(ArrayRefExp(sidxIdent, int1), dofIdent, BinOpExp.EQ_ASGN)),
            ExpStmt(BinOpExp(ArrayRefExp(sidxIdent, int2), BinOpExp(gridmIdent, dofIdent, BinOpExp.MUL), BinOpExp.EQ_ASGN)),
            ExpStmt(BinOpExp(ArrayRefExp(sidxIdent, int3), BinOpExp(gridmIdent, BinOpExp(gridnIdent, dofIdent, BinOpExp.MUL), BinOpExp.MUL), BinOpExp.EQ_ASGN)),
            ExpStmt(BinOpExp(ArrayRefExp(sidxIdent, int4), UnaryExp(dofIdent, UnaryExp.MINUS), BinOpExp.EQ_ASGN)),
            ExpStmt(BinOpExp(ArrayRefExp(sidxIdent, int5), BinOpExp(UnaryExp(gridmIdent, UnaryExp.MINUS), dofIdent, BinOpExp.MUL), BinOpExp.EQ_ASGN)),
            ExpStmt(BinOpExp(ArrayRefExp(sidxIdent, int6), BinOpExp(UnaryExp(gridmIdent, UnaryExp.MINUS),
                                                                    BinOpExp(gridnIdent, dofIdent, BinOpExp.MUL), BinOpExp.MUL), BinOpExp.EQ_ASGN))
          ]
          for var in lhs_array_ids:
            domainStmts += [
              ExpStmt(FunCallExp(IdentExp('cudaMemset'), [IdentExp(var), int0, scSizeIdent]))
            ]
          domainStmts += [
            ExpStmt(BinOpExp(ast.IdentExp('dimGrid.x'), hostVecLenIdent, BinOpExp.EQ_ASGN)),
            ExpStmt(BinOpExp(ast.IdentExp('dimBlock.x'), nosIdent, BinOpExp.EQ_ASGN))
            ]
          domainArgs = [sidxIdent]
          domainParams = [ast.FieldDecl('int*', sidx)]
          kernelParams += domainParams
        #--------------------------------------------------------------------------------------------------------------

        #--------------------------------------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------------------------------------
        # begin generate the kernel
        idx        = prefix + 'i'
        size       = prefix + 'n'
        idxIdent   = ast.IdentExp(idx)
        sizeIdent  = ast.IdentExp(size)
        blockIdx   = ast.IdentExp('blockIdx.x')
        blockSize  = ast.IdentExp('blockDim.x')
        threadIdx  = ast.IdentExp('threadIdx.x')
        tidVarDecl = ast.VarDeclInit('int', tidIdent, ast.BinOpExp(ast.BinOpExp(blockIdx, blockSize, ast.BinOpExp.MUL), threadIdx, ast.BinOpExp.ADD))
        kernelStmts   = [tidVarDecl]
        redKernStmts  = [tidVarDecl]
        redkernParams = []
        cacheReads    = []
        cacheWrites   = []
        if self.cacheBlocks:
            for var in array_ids:
                sharedVar = 'shared_' + var
                kernelStmts += [
                    # __shared__ double shared_var[threadCount];
                    ast.VarDecl('__shared__ double', [sharedVar + '[' + str(self.threadCount) + ']'])
                ]
                sharedVarExp = ast.ArrayRefExp(ast.IdentExp(sharedVar), threadIdx)
                varExp       = ast.ArrayRefExp(ast.IdentExp(var), tidIdent)
                
                # cache reads
                if var in rhs_array_ids:
                    cacheReads += [
                        # shared_var[threadIdx.x]=var[tid];
                        ast.ExpStmt(ast.BinOpExp(sharedVarExp, varExp, ast.BinOpExp.EQ_ASGN))
                    ]
                # var[tid] -> shared_var[threadIdx.x]
                rrToShared = lambda n: sharedVarExp \
                                if isinstance(n, ast.ArrayRefExp) and \
                                   isinstance(n.exp, ast.IdentExp) and \
                                   n.exp.name == var \
                                else n
                rrRhsExprs = lambda n: ast.BinOpExp(n.lhs, loop_lib.rewriteNode(rrToShared, n.rhs), n.op_type) \
                                if isinstance(n, ast.BinOpExp) and \
                                   n.op_type == ast.BinOpExp.EQ_ASGN \
                                else n
                loop_body3 = loop_lib.rewriteNode(rrRhsExprs, loop_body3)

                # cache writes also
                if var in lhs_array_ids:
                    rrLhsExprs = lambda n: ast.BinOpExp(loop_lib.rewriteNode(rrToShared, n.lhs), n.rhs, n.op_type) \
                                    if isinstance(n, ast.BinOpExp) and \
                                       n.op_type == ast.BinOpExp.EQ_ASGN \
                                    else n
                    loop_body3 = loop_lib.rewriteNode(rrLhsExprs, loop_body3)
                    cacheWrites += [ast.ExpStmt(ast.BinOpExp(varExp, sharedVarExp, ast.BinOpExp.EQ_ASGN))]

        if isReduction:
            for temp in kernel_temps:
                kernelStmts += [ast.VarDeclInit('double', temp, int0)]

        #--------------------------------------------------
        if self.domain == None:
          # the rewritten loop body statement
          kernelStmts += [
            IfStmt(BinOpExp(tidIdent, ubound_exp, BinOpExp.LE),
                   CompStmt(cacheReads + [loop_body3] + cacheWrites))
          ]
        else:
          kernelStmts  = [VarDeclInit('int', tidIdent, BinOpExp(blockIdx, ArrayRefExp(sidxIdent, threadIdx), BinOpExp.ADD))]
          kernelStmts += [
            IfStmt(BinOpExp(BinOpExp(tidIdent, int0, BinOpExp.GE),
                            BinOpExp(tidIdent, hostVecLenIdent, BinOpExp.LT),
                            BinOpExp.LAND),
                   CompStmt([loop_body3]))
          ]

        #--------------------------------------------------
        # begin reduction statements
        reducts      = 'reducts'
        reductsIdent = ast.IdentExp(reducts)
        blkdata      = prefix + 'vec'+str(g.Globals().getcounter())
        blkdataIdent = ast.IdentExp(blkdata)
        if isReduction:
            kernelStmts += [ast.Comment('reduce single-thread results within a block')]
            # declare the array shared by threads within a block
            kernelStmts += [ast.VarDecl('__shared__ double', [blkdata+'['+str(self.threadCount)+']'])]
            # store the lhs/computed values into the shared array
            kernelStmts += [ast.ExpStmt(ast.BinOpExp(ast.ArrayRefExp(blkdataIdent, threadIdx),
                                                     loop_lhs_exprs[0],
                                                     ast.BinOpExp.EQ_ASGN))]
            # sync threads prior to reduction
            kernelStmts += [ast.ExpStmt(ast.FunCallExp(ast.IdentExp('__syncthreads'),[]))];
            # at each step, divide the array into two halves and sum two corresponding elements
            kernelStmts += [ast.VarDecl('int', [idx])]
            kernelStmts += [
                ast.ForStmt(ast.BinOpExp(idxIdent, ast.BinOpExp(ast.IdentExp('blockDim.x'), int2, ast.BinOpExp.DIV), ast.BinOpExp.EQ_ASGN),
                            ast.BinOpExp(idxIdent, int0, ast.BinOpExp.GT),
                            ast.BinOpExp(idxIdent, int1, ast.BinOpExp.ASGN_SHR),
                            ast.CompStmt([ast.IfStmt(ast.BinOpExp(threadIdx, idxIdent, ast.BinOpExp.LT),
                                                     ast.ExpStmt(ast.BinOpExp(ast.ArrayRefExp(blkdataIdent, threadIdx),
                                                                              ast.ArrayRefExp(blkdataIdent,
                                                                                              ast.BinOpExp(threadIdx, idxIdent, ast.BinOpExp.ADD)),
                                                                              ast.BinOpExp.ASGN_ADD))
                                                     ),
                                          ast.ExpStmt(ast.FunCallExp(ast.IdentExp('__syncthreads'),[]))]))
            ]
            # the first thread within a block stores the results for the entire block
            kernelParams += [ast.FieldDecl('double*', reducts)]
            kernelStmts += [
                ast.IfStmt(ast.BinOpExp(threadIdx, int0, ast.BinOpExp.EQ),
                           ast.ExpStmt(ast.BinOpExp(ast.ArrayRefExp(reductsIdent, blockIdx),
                                                    ast.ArrayRefExp(blkdataIdent, int0),
                                                    ast.BinOpExp.EQ_ASGN)))
            ]
        # end reduction statements
        #--------------------------------------------------
        
        #--------------------------------------------------
        # begin multi-stage reduction kernel
        blkdata      = prefix + 'vec'+str(g.Globals().getcounter())
        blkdataIdent = ast.IdentExp(blkdata)
        if isReduction:
          redkernParams += [ast.FieldDecl('int', size), ast.FieldDecl('double*', reducts)]
          redKernStmts += [ast.VarDecl('__shared__ double', [blkdata+'['+str(maxThreadsPerBlock)+']'])]
          redKernStmts += [ast.IfStmt(ast.BinOpExp(tidIdent, sizeIdent, ast.BinOpExp.LT),
                                      ast.ExpStmt(ast.BinOpExp(ast.ArrayRefExp(blkdataIdent, threadIdx),
                                                               ast.ArrayRefExp(reductsIdent, tidIdent),
                                                               ast.BinOpExp.EQ_ASGN)),
                                      ast.ExpStmt(ast.BinOpExp(ast.ArrayRefExp(blkdataIdent, threadIdx),
                                                               int0,
                                                               ast.BinOpExp.EQ_ASGN)))]
          redKernStmts += [ast.ExpStmt(ast.FunCallExp(ast.IdentExp('__syncthreads'),[]))]
          redKernStmts += [ast.VarDecl('int', [idx])]
          redKernStmts += [ast.ForStmt(
            ast.BinOpExp(idxIdent, ast.BinOpExp(ast.IdentExp('blockDim.x'), int2, ast.BinOpExp.DIV), ast.BinOpExp.EQ_ASGN),
            ast.BinOpExp(idxIdent, int0, ast.BinOpExp.GT),
            ast.BinOpExp(idxIdent, int1, ast.BinOpExp.ASGN_SHR),
            ast.CompStmt([ast.IfStmt(ast.BinOpExp(threadIdx, idxIdent, ast.BinOpExp.LT),
                                     ast.ExpStmt(ast.BinOpExp(ast.ArrayRefExp(blkdataIdent, threadIdx),
                                                              ast.ArrayRefExp(blkdataIdent,
                                                                              ast.BinOpExp(threadIdx, idxIdent, ast.BinOpExp.ADD)),
                                                              ast.BinOpExp.ASGN_ADD))
                                     ),
                          ast.ExpStmt(ast.FunCallExp(ast.IdentExp('__syncthreads'),[]))])
          )]
          redKernStmts += [
            ast.IfStmt(ast.BinOpExp(threadIdx, int0, ast.BinOpExp.EQ),
                       ast.ExpStmt(ast.BinOpExp(ast.ArrayRefExp(reductsIdent, blockIdx),
                                                ast.ArrayRefExp(blkdataIdent, int0),
                                                ast.BinOpExp.EQ_ASGN)))
          ]
        # end multi-stage reduction kernel
        #--------------------------------------------------

        dev_kernel_name = prefix + 'kernel'+str(g.Globals().getcounter())
        dev_kernel = ast.FunDecl(dev_kernel_name, 'void', ['__global__'], kernelParams, ast.CompStmt(kernelStmts))
        
        dev_redkern_name = prefix + 'blksum'+str(g.Globals().getcounter())
        dev_redkern = ast.FunDecl(dev_redkern_name, 'void', ['__global__'], redkernParams, ast.CompStmt(redKernStmts))
        
        # after getting interprocedural AST, make this a sub to that AST
        g.Globals().cunit_declarations += orio.module.loop.codegen.CodeGen('cuda').generator.generate(dev_kernel, '', '  ')
        if isReduction:
          g.Globals().cunit_declarations += '\n' + orio.module.loop.codegen.CodeGen('cuda').generator.generate(dev_redkern, '', '  ')
        # end generate the kernel
        #--------------------------------------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------------------------------------

        
        #--------------------------------------------------------------------------------------------------------------
        # begin marshal resources
        # declare device variables
        dev = 'dev_'
        if self.dataOnDevice:
          dev = ''
        dev_lbi = map(lambda x: dev+x, list(lbi))
        dev_block_r = dev + reducts
        host_ids = []
        local_ids = []
        if isReduction:
          dev_lbi   += [dev_block_r]
          local_ids += [dev_block_r]
        hostDecls  = [ast.Comment('declare variables')]
        if self.dataOnDevice:
          hostDecls += [ast.VarDecl('double', map(lambda x: '*'+x, local_ids))]
        else:
          hostDecls += [ast.VarDecl('double', map(lambda x: '*'+x, dev_lbi + host_ids))]
        hostDecls += [ast.VarDeclInit('int', nthreadsIdent, ast.NumLitExp(self.threadCount, ast.NumLitExp.INT))]
        if self.streamCount > 1:
          hostDecls += [ast.VarDeclInit('int', nstreamsIdent, ast.NumLitExp(self.streamCount, ast.NumLitExp.INT))]
        
        # calculate device dimensions
        gridxIdent  = ast.IdentExp('dimGrid.x')
        deviceDims  = [ast.Comment('calculate device dimensions')]
        deviceDims += [ast.VarDecl('dim3', ['dimGrid', 'dimBlock'])]
        deviceDims += [
            ast.ExpStmt(ast.BinOpExp(gridxIdent,
                                     ast.FunCallExp(ast.IdentExp('ceil'),
                                                    [ast.BinOpExp(ast.CastExpr('float', hostVecLenIdent),
                                                                  ast.CastExpr('float', nthreadsIdent),
                                                                  ast.BinOpExp.DIV)
                                                    ]),
                                     ast.BinOpExp.EQ_ASGN))]
        # initialize block size
        deviceDims += [ast.ExpStmt(ast.BinOpExp(ast.IdentExp('dimBlock.x'), nthreadsIdent, ast.BinOpExp.EQ_ASGN))]

        # -------------------------------------------------
        streamDecls = []
        mallocs   = [ast.Comment('allocate device memory')]
        h2dcopys  = [ast.Comment('copy data from host to device')]
        h2dasyncs = []
        sizeofDblCall = ast.FunCallExp(ast.IdentExp('sizeof'), [ast.IdentExp('double')])
        mallocs += [ast.VarDeclInit('int', scSizeIdent, ast.BinOpExp(hostVecLenIdent, sizeofDblCall, ast.BinOpExp.MUL))]
        
        # -------------------------------------------------
        # if streaming, divide vectors into chunks and asynchronously overlap copy-copy and copy-exec ops
        soffset           = prefix + 'soff'
        boffset           = prefix + 'boff'
        soffsetIdent      = ast.IdentExp(soffset)
        boffsetIdent      = ast.IdentExp(boffset)
        chunklenIdent     = ast.IdentExp('chunklen')
        chunkremIdent     = ast.IdentExp('chunkrem')
        blks4chunkIdent   = ast.IdentExp('blks4chunk')
        blks4chunksIdent  = ast.IdentExp('blks4chunks')
        calc_offset       = []
        calc_boffset      = []
        hostDecls        += [ast.VarDecl('int', [idx])]
        if isReduction:
          hostDecls      += [ast.VarDecl('int', [size])]
        if self.streamCount > 1:
          streamDecls += [ast.Comment('create streams')]
          streamDecls += [ast.VarDecl('int', [soffset])]
          if isReduction:
            streamDecls += [ast.VarDecl('int', [boffset])]
          streamDecls   += [ast.VarDecl('cudaStream_t', ['stream[nstreams+1]'])]
          streamDecls   += [
            ast.ForStmt(ast.BinOpExp(idxIdent, int0, ast.BinOpExp.EQ_ASGN),
                        ast.BinOpExp(idxIdent, nstreamsIdent, ast.BinOpExp.LE),
                        ast.UnaryExp(idxIdent, ast.UnaryExp.POST_INC),
                        ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaStreamCreate'),
                                                   [ast.UnaryExp(ast.IdentExp('stream[' + idx + ']'),
                                                                 ast.UnaryExp.ADDRESSOF)])))
          ]
          streamDecls   += [
            ast.VarDeclInit('int', chunklenIdent, ast.BinOpExp(hostVecLenIdent, nstreamsIdent, ast.BinOpExp.DIV)),
            ast.VarDeclInit('int', chunkremIdent, ast.BinOpExp(hostVecLenIdent, nstreamsIdent, ast.BinOpExp.MOD)),
          ]
          calc_offset = [
            ast.ExpStmt(ast.BinOpExp(soffsetIdent,
                                     ast.BinOpExp(idxIdent, chunklenIdent, ast.BinOpExp.MUL),
                                     ast.BinOpExp.EQ_ASGN))
          ]
          calc_boffset = [
            ast.ExpStmt(ast.BinOpExp(boffsetIdent,
                                     ast.BinOpExp(idxIdent, blks4chunkIdent, ast.BinOpExp.MUL),
                                     ast.BinOpExp.EQ_ASGN))
          ]
        # -------------------------------------------------
        dev_scalar_ids = map(lambda x: (x,dev+x), scalar_ids)
        for sid,dsid in dev_scalar_ids:
          # malloc scalars
          mallocs += [
            ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaMalloc'),
                                       [ast.CastExpr('void**', ast.UnaryExp(ast.IdentExp(dsid), ast.UnaryExp.ADDRESSOF)),
                                        sizeofDblCall
                                        ]))]
          # memcopy scalars
          h2dcopys += [
            ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaMemcpy'),
                                       [ast.IdentExp(dsid), ast.UnaryExp(ast.IdentExp(sid), ast.UnaryExp.ADDRESSOF),
                                        sizeofDblCall,
                                        ast.IdentExp('cudaMemcpyHostToDevice')
                                        ]))]
        # -------------------------------------------------
        dev_array_ids = map(lambda x: (x,dev+x), array_ids)
        registeredHostIds = []
        for aid,daid in dev_array_ids:
          # malloc arrays
          mallocs += [
            ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaMalloc'),
                                       [ast.CastExpr('void**', ast.UnaryExp(ast.IdentExp(daid), ast.UnaryExp.ADDRESSOF)),
                                        scSizeIdent
                                        ]))]
          # memcopy device to host
          if aid in rhs_array_ids:
            if self.streamCount > 1:
              # pin host memory while streaming
              mallocs += [
                ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaHostRegister'),
                                           [ast.IdentExp(aid), hostVecLenIdent,
                                            ast.IdentExp('cudaHostRegisterPortable')
                                            ]))
              ]
              registeredHostIds += [aid] # remember to unregister at the end
              h2dasyncs += [
                ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaMemcpyAsync'),
                                           [ast.BinOpExp(ast.IdentExp(daid), soffsetIdent, ast.BinOpExp.ADD),
                                            ast.BinOpExp(ast.IdentExp(aid),  soffsetIdent, ast.BinOpExp.ADD),
                                            ast.BinOpExp(chunklenIdent, sizeofDblCall, ast.BinOpExp.MUL),
                                            ast.IdentExp('cudaMemcpyHostToDevice'),
                                            ast.ArrayRefExp(ast.IdentExp('stream'), idxIdent)
                                            ]))]
            else:
              h2dcopys += [
                ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaMemcpy'),
                                           [ast.IdentExp(daid), ast.IdentExp(aid), scSizeIdent,
                                            ast.IdentExp('cudaMemcpyHostToDevice')
                                            ]))]
        # for-loop over streams to do async copies
        if self.streamCount > 1:
          h2dcopys += [
            ast.ForStmt(ast.BinOpExp(idxIdent, int0, ast.BinOpExp.EQ_ASGN),
                        ast.BinOpExp(idxIdent, nstreamsIdent, ast.BinOpExp.LT),
                        ast.UnaryExp(idxIdent, ast.UnaryExp.POST_INC),
                        ast.CompStmt(calc_offset + h2dasyncs)),
            # copy the remainder in the last/reserve stream
            ast.IfStmt(ast.BinOpExp(chunkremIdent, int0, ast.BinOpExp.NE),
                       ast.CompStmt(calc_offset +
                         [ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaMemcpyAsync'),
                           [ast.BinOpExp(ast.IdentExp(daid), soffsetIdent, ast.BinOpExp.ADD),
                            ast.BinOpExp(ast.IdentExp(aid),  soffsetIdent, ast.BinOpExp.ADD),
                            ast.BinOpExp(chunkremIdent, sizeofDblCall, ast.BinOpExp.MUL),
                            ast.IdentExp('cudaMemcpyHostToDevice'),
                            ast.ArrayRefExp(ast.IdentExp('stream'), idxIdent)
                            ]))
                       ]))
          ]
        # -------------------------------------------------
        # malloc block-level result var
        if isReduction:
            mallocs += [
                ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaMalloc'),
                                           [ast.CastExpr('void**', ast.UnaryExp(ast.IdentExp(dev_block_r), ast.UnaryExp.ADDRESSOF)),
                                            ast.BinOpExp(ast.ParenthExp(ast.BinOpExp(gridxIdent, int1, ast.BinOpExp.ADD)),
                                                         sizeofDblCall,
                                                         ast.BinOpExp.MUL)
                                            ]))]
            for var in host_ids:
                if self.pinHostMem:
                    mallocs += [
                        ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaHostAlloc'),
                                                   [ast.CastExpr('void**', ast.UnaryExp(ast.IdentExp(var), ast.UnaryExp.ADDRESSOF)),
                                                    scSizeIdent,
                                                    ast.IdentExp('cudaHostAllocDefault')
                                                    ]))]
                else:
                    mallocs += [
                        ast.AssignStmt(var,
                                       ast.CastExpr('double*',
                                                    ast.FunCallExp(ast.IdentExp('malloc'),
                                                   [ast.BinOpExp(gridxIdent, sizeofDblCall, ast.BinOpExp.MUL)])))]
                    if self.streamCount > 1:
                        mallocs += [
                            ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaHostRegister'),
                                                       [ast.IdentExp(var),
                                                        gridxIdent,
                                                        ast.IdentExp('cudaHostRegisterPortable')
                                                        ]))
                        ]
                        registeredHostIds += [var]
        # -------------------------------------------------
        # invoke device kernel function:
        # -- kernelFun<<<numOfBlocks,numOfThreads>>>(dev_vars, ..., dev_result);
        kernell_calls = [Comment('invoke device kernel')]
        if self.streamCount == 1:
          args = map(lambda x: IdentExp(x), [host_arraysize] + dev_lbi)
          kernell_call = ExpStmt(FunCallExp(IdentExp(dev_kernel_name+'<<<dimGrid,dimBlock>>>'), args + domainArgs))
          kernell_calls += [kernell_call]
        else:
          args    = [chunklenIdent]
          argsrem = [chunkremIdent]
          boffsets = []
          # adjust array args using offsets
          dev_array_idss = map(lambda x: dev+x, array_ids)
          for arg in dev_lbi:
              if arg in dev_array_idss:
                  args    += [ast.BinOpExp(ast.IdentExp(arg), soffsetIdent, ast.BinOpExp.ADD)]
                  argsrem += [ast.BinOpExp(ast.IdentExp(arg), soffsetIdent, ast.BinOpExp.ADD)]
              elif arg == dev_block_r:
                  args    += [ast.BinOpExp(ast.IdentExp(arg), boffsetIdent, ast.BinOpExp.ADD)]
                  argsrem += [ast.BinOpExp(ast.IdentExp(arg), boffsetIdent, ast.BinOpExp.ADD)]
              else:
                  args    += [ast.IdentExp(arg)]
                  argsrem += [ast.IdentExp(arg)]
          kernell_call    = ast.ExpStmt(ast.FunCallExp(ast.IdentExp(dev_kernel_name+'<<<blks4chunk,dimBlock,0,stream['+idx+']>>>'), args))
          kernell_callrem = ast.ExpStmt(ast.FunCallExp(ast.IdentExp(dev_kernel_name+'<<<blks4chunk,dimBlock,0,stream['+idx+']>>>'), argsrem))
          kernell_calls += [
            # calc blocks per stream
            ast.VarDeclInit('int', blks4chunkIdent, ast.BinOpExp(gridxIdent, nstreamsIdent, ast.BinOpExp.DIV)),
            ast.IfStmt(ast.BinOpExp(ast.BinOpExp(gridxIdent, nstreamsIdent, ast.BinOpExp.MOD),
                                    int0,
                                    ast.BinOpExp.NE),
                       ast.ExpStmt(ast.UnaryExp(blks4chunkIdent, ast.UnaryExp.POST_INC)))
          ]
          # calc total number of blocks to reduce
          if isReduction:
            kernell_calls += [ast.VarDeclInit('int', blks4chunksIdent, ast.BinOpExp(blks4chunkIdent, nstreamsIdent, ast.BinOpExp.MUL))]
            boffsets = calc_boffset
          # kernel invocations
          kernell_calls += [
            ast.ForStmt(
              ast.BinOpExp(idxIdent, int0, ast.BinOpExp.EQ_ASGN),
              ast.BinOpExp(idxIdent, nstreamsIdent, ast.BinOpExp.LT),
              ast.UnaryExp(idxIdent, ast.UnaryExp.POST_INC),
              ast.CompStmt(calc_offset + boffsets + [kernell_call])),
            # kernel invocation on the last chunk
            ast.IfStmt(ast.BinOpExp(chunkremIdent, int0, ast.BinOpExp.NE),
                       ast.CompStmt(calc_offset + boffsets + [kernell_callrem] +
                                    ([ast.ExpStmt(ast.UnaryExp(blks4chunksIdent, ast.UnaryExp.POST_INC))] if isReduction else []))
                       )
          ]
        
        # -------------------------------------------------
        # iteratively keep block-summing, until nothing more to sum: aka multi-staged reduction
        stageBlocks       = prefix + 'blks'
        stageThreads      = prefix + 'trds'
        stageBlocksIdent  = ast.IdentExp(stageBlocks)
        stageThreadsIdent = ast.IdentExp(stageThreads)
        stageReds         = []
        bodyStmts         = []
        maxThreadsPerBlockM1 = ast.NumLitExp(str(maxThreadsPerBlock - 1), ast.NumLitExp.INT)
        if isReduction:
          if self.streamCount > 1:
            stageReds += [ast.VarDeclInit('int', stageBlocksIdent,  blks4chunksIdent)]
          else:
            stageReds += [ast.VarDeclInit('int', stageBlocksIdent,  gridxIdent)]
          stageReds += [ast.VarDecl('int', [stageThreads])]
          bodyStmts += [ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaDeviceSynchronize'), []))]
          bodyStmts += [ast.ExpStmt(ast.BinOpExp(sizeIdent, stageBlocksIdent, ast.BinOpExp.EQ_ASGN))]
          bodyStmts += [
            ast.ExpStmt(ast.BinOpExp(stageBlocksIdent,
                                     ast.BinOpExp(ast.ParenthExp(ast.BinOpExp(stageBlocksIdent, maxThreadsPerBlockM1, ast.BinOpExp.ADD)),
                                                  maxThreadsPerBlockNumLit,
                                                  ast.BinOpExp.DIV),
                                     ast.BinOpExp.EQ_ASGN))]
          bodyStmts += [ast.IfStmt(ast.BinOpExp(sizeIdent, maxThreadsPerBlockNumLit, ast.BinOpExp.LT),
                                   ast.CompStmt([
                                     ast.ExpStmt(ast.BinOpExp(stageThreadsIdent, int1, ast.BinOpExp.EQ_ASGN)),
                                     ast.WhileStmt(ast.BinOpExp(stageThreadsIdent, sizeIdent, ast.BinOpExp.LT),
                                                   ast.ExpStmt(ast.BinOpExp(stageThreadsIdent, int1, ast.BinOpExp.ASGN_SHL)))]),
                                   ast.ExpStmt(ast.BinOpExp(stageThreadsIdent, maxThreadsPerBlockNumLit, ast.BinOpExp.EQ_ASGN)))
          ]
          bodyStmts += [ast.ExpStmt(ast.FunCallExp(ast.IdentExp(dev_redkern_name+'<<<'+stageBlocks+','+stageThreads+'>>>'),
                                                   [sizeIdent, ast.IdentExp(dev_block_r)]))]
          stageReds += [ast.WhileStmt(ast.BinOpExp(stageBlocksIdent, int1, ast.BinOpExp.GT), ast.CompStmt(bodyStmts))]
        
        # -------------------------------------------------
        # copy data from devices to host
        d2hcopys = [ast.Comment('copy data from device to host')]
        d2hasyncs    = []
        d2hasyncsrem = []
        for var in lhs_ids:
            res_scalar_ids = filter(lambda x: x[1] == (dev+var), dev_scalar_ids)
            for rsid,drsid in res_scalar_ids:
                d2hcopys += [
                    ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaMemcpy'),
                                                   [ast.IdentExp(rsid), ast.IdentExp(drsid),
                                                    sizeofDblCall,
                                                    ast.IdentExp('cudaMemcpyDeviceToHost')
                                                    ]))]
            res_array_ids  = filter(lambda x: x[1] == (dev+var), dev_array_ids)
            for raid,draid in res_array_ids:
                if self.streamCount > 1:
                    d2hasyncs += [
                        ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaMemcpyAsync'),
                                                   [ast.BinOpExp(ast.IdentExp(raid),  soffsetIdent, ast.BinOpExp.ADD),
                                                    ast.BinOpExp(ast.IdentExp(draid), soffsetIdent, ast.BinOpExp.ADD),
                                                    ast.BinOpExp(chunklenIdent, sizeofDblCall, ast.BinOpExp.MUL),
                                                    ast.IdentExp('cudaMemcpyDeviceToHost'),
                                                    ast.ArrayRefExp(ast.IdentExp('stream'), idxIdent)
                                                    ]))]
                    d2hasyncsrem += [
                        ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaMemcpyAsync'),
                                                   [ast.BinOpExp(ast.IdentExp(raid),  soffsetIdent, ast.BinOpExp.ADD),
                                                    ast.BinOpExp(ast.IdentExp(draid), soffsetIdent, ast.BinOpExp.ADD),
                                                    ast.BinOpExp(chunkremIdent, sizeofDblCall, ast.BinOpExp.MUL),
                                                    ast.IdentExp('cudaMemcpyDeviceToHost'),
                                                    ast.ArrayRefExp(ast.IdentExp('stream'), idxIdent)
                                                    ]))]
                else:
                    d2hcopys += [
                        ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaMemcpy'),
                                                   [ast.IdentExp(raid),
                                                    ast.IdentExp(draid),
                                                    scSizeIdent,
                                                    ast.IdentExp('cudaMemcpyDeviceToHost')
                                                    ]))]
        # -------------------------------------------------
        # memcpy block-level result var
        if isReduction:
          d2hcopys += [
                  ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaMemcpy'),
                                                 [ast.UnaryExp(ast.IdentExp(lhs_ids[0]),ast.UnaryExp.ADDRESSOF),
                                                  ast.IdentExp(dev_block_r),
                                                  sizeofDblCall,
                                                  ast.IdentExp('cudaMemcpyDeviceToHost')
                                                  ]))]
        # -------------------------------------------------
        if self.streamCount > 1 and not isReduction:
          d2hcopys += [ast.ForStmt(ast.BinOpExp(idxIdent, int0, ast.BinOpExp.EQ_ASGN),
                                   ast.BinOpExp(idxIdent, nstreamsIdent, ast.BinOpExp.LT),
                                   ast.UnaryExp(idxIdent, ast.UnaryExp.POST_INC),
                                   ast.CompStmt(calc_offset + d2hasyncs))]
          d2hcopys += [
               ast.IfStmt(ast.BinOpExp(chunkremIdent, int0, ast.BinOpExp.NE),
                          ast.CompStmt(calc_offset + d2hasyncsrem))]
          # synchronize
          d2hcopys   += [ast.ForStmt(
                             ast.BinOpExp(idxIdent, int0, ast.BinOpExp.EQ_ASGN),
                             ast.BinOpExp(idxIdent, nstreamsIdent, ast.BinOpExp.LE),
                             ast.UnaryExp(idxIdent, ast.UnaryExp.POST_INC),
                             ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaStreamSynchronize'),
                                                        [ast.ArrayRefExp(ast.IdentExp('stream'), idxIdent)])))]
        # -------------------------------------------------
        # free allocated memory and resources
        free_vars = [ast.Comment('free allocated memory')]
        freeStreams = []
        if self.streamCount > 1:
            # unregister pinned memory
            for var in registeredHostIds:
                free_vars += [ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaHostUnregister'), [ast.IdentExp(var)]))]
            # destroy streams
            freeStreams += [ast.ForStmt(
                               ast.BinOpExp(idxIdent, int0, ast.BinOpExp.EQ_ASGN),
                               ast.BinOpExp(idxIdent, nstreamsIdent, ast.BinOpExp.LE),
                               ast.UnaryExp(idxIdent, ast.UnaryExp.POST_INC),
                               ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaStreamDestroy'),
                                                          [ast.ArrayRefExp(ast.IdentExp('stream'), idxIdent)])))]
        for var in dev_lbi:
            free_vars += [ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaFree'), [ast.IdentExp(var)]))]
        for var in host_ids:
            if self.pinHostMem:
                free_vars += [ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaFreeHost'), [ast.IdentExp(var)]))]
            else:
                free_vars += [ast.ExpStmt(ast.FunCallExp(ast.IdentExp('free'), [ast.IdentExp(var)]))]

        # end marshal resources
        #--------------------------------------------------------------------------------------------------------------
        
        
        #--------------------------------------------------------------------------------------------------------------
        # cuda timing calls
        timerDecls = []
        timerStart = []
        timerStop  = []
        if self.doDeviceTiming:
            timerDecls += [
                ast.VarDecl('cudaEvent_t', ['start', 'stop']),
                ast.VarDecl('float', [prefix + 'elapsed']),
                ast.VarDecl('FILE*', [prefix + 'fp'])
            ]
            timerStart += [
                ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaEventCreate'), [ast.UnaryExp(ast.IdentExp('start'), ast.UnaryExp.ADDRESSOF)])),
                ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaEventCreate'), [ast.UnaryExp(ast.IdentExp('stop'),  ast.UnaryExp.ADDRESSOF)])),
                ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaEventRecord'), [ast.IdentExp('start'), int0]))
            ]
            timerStop  += [
                ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaEventRecord'), [ast.IdentExp('stop'), int0])),
                ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaEventSynchronize'), [ast.IdentExp('stop')])),
                ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaEventElapsedTime'),
                                           [ast.UnaryExp(ast.IdentExp(prefix + 'elapsed'), ast.UnaryExp.ADDRESSOF),
                                            ast.IdentExp('start'), ast.IdentExp('stop')])),
                ast.AssignStmt(prefix + 'fp',
                            ast.FunCallExp(ast.IdentExp('fopen'), [ast.StringLitExp(prefix + 'time.out'), ast.StringLitExp('a')])),
                ast.ExpStmt(ast.FunCallExp(ast.IdentExp('fprintf'),
                                           [ast.IdentExp(prefix + 'fp'),
                                            ast.StringLitExp('Kernel_time@rep[%d]:%fms. '),
                                            ast.IdentExp('orio_i'),
                                            ast.IdentExp(prefix + 'elapsed')])),
                ast.ExpStmt(ast.FunCallExp(ast.IdentExp('fclose'), [ast.IdentExp(prefix + 'fp')])),
                ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaEventDestroy'), [ast.IdentExp('start')])),
                ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaEventDestroy'), [ast.IdentExp('stop')]))
            ]

        #--------------------------------------------------------------------------------------------------------------
        # in validation mode, compare original and transformed codes' results
        results  = []
        if g.Globals().validationMode:
          printFpIdent = ast.IdentExp('fp')
          results += [
            ast.VarDeclInit('FILE*', printFpIdent,
                            ast.FunCallExp(ast.IdentExp('fopen'), [ast.StringLitExp('newexec.out'), ast.StringLitExp('w')]))
          ]
          for var in lhs_ids:
            if var in array_ids:
              results += [ast.ForStmt(
                ast.BinOpExp(idxIdent, int0, ast.BinOpExp.EQ_ASGN),
                ast.BinOpExp(idxIdent, ubound_exp, ast.BinOpExp.LE),
                ast.UnaryExp(idxIdent, ast.UnaryExp.POST_INC),
                ast.ExpStmt(ast.FunCallExp(ast.IdentExp('fprintf'),
                                           [printFpIdent, ast.StringLitExp("\'"+var+"[%d]\',%f; "), idxIdent, ast.ArrayRefExp(ast.IdentExp(var), idxIdent)]))
              )]
            else:
              results += [ast.ExpStmt(
                ast.FunCallExp(ast.IdentExp('fprintf'), [printFpIdent, ast.StringLitExp("\'"+var+"\',%f"), ast.IdentExp(var)])
              )]
          results += [ast.ExpStmt(ast.FunCallExp(ast.IdentExp('fclose'), [printFpIdent]))]
          
        #--------------------------------------------------------------------------------------------------------------
        # no CPU-GPU data transfers
        if (self.dataOnDevice):
          mallocs   = []
          h2dcopys  = []
          d2hcopys  = []
          free_vars = []
        # add up all components
        transformed_stmt = ast.CompStmt(
            hostDecls + deviceDims + streamDecls + mallocs + h2dcopys
          + timerDecls + timerStart
          + domainStmts
          + kernell_calls
          + timerStop
          + stageReds
          + d2hcopys
          + free_vars
          + freeStreams
          + results
        )
        return transformed_stmt


