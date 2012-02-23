#
# Contain the CUDA transformation module
#

import orio.module.loop.ast_lib.forloop_lib, orio.module.loop.ast_lib.common_lib
import orio.main.util.globals as g
import orio.module.loop.ast as ast

#----------------------------------------------------------------------------------------------------------------------

class Transformation:
    '''Code transformation'''

    def __init__(self, stmt, devProps, targs):
        '''Instantiate a code transformation object'''
        self.stmt        = stmt
        self.devProps    = devProps
        self.threadCount, self.cacheBlocks, self.pinHostMem, self.streamCount = targs
        
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

        loop_lib = orio.module.loop.ast_lib.common_lib.CommonLib()
        tcount = str(self.threadCount)
        int0 = ast.NumLitExp(0,ast.NumLitExp.INT)

        #--------------------------------------------------------------------------------------------------------------
        # begin rewrite the loop body
        # collect all identifiers from the loop's upper bound expression
        collectIdents = lambda n: [n.name] if isinstance(n, ast.IdentExp) else []
        ubound_ids = loop_lib.collectNode(collectIdents, ubound_exp)
        
        # create decls for ubound_exp id's, assuming all ids are int's
        kernelParams = [ast.FieldDecl('int', x) for x in ubound_ids]

        # collect all identifiers from the loop body
        loop_body_ids = loop_lib.collectNode(collectIdents, loop_body)
        lbi = set(filter(lambda x: x != index_id.name, loop_body_ids))
        
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

        # create decls for loop body id's
        if isReduction:
            lbi = lbi.difference(set(lhs_ids))
        kernelParams += [ast.FieldDecl('double*', x) for x in lbi]
        scalar_ids = list(lbi.difference(array_ids))
        
        kernel_temps = []
        if isReduction:
            for var in lhs_ids:
                temp = 'orcu_var' + str(g.Globals().getcounter())
                kernel_temps += [temp]
                rrLhs = lambda n: ast.IdentExp(temp) if (isinstance(n, ast.IdentExp) and n.name == var) else n
                loop_body = loop_lib.rewriteNode(rrLhs, loop_body)

        # add dereferences to all non-array id's in the loop body
        addDerefs2 = lambda n: ast.ParenthExp(ast.UnaryExp(n, ast.UnaryExp.DEREF)) if (isinstance(n, ast.IdentExp) and n.name in scalar_ids) else n
        loop_body2 = loop_lib.rewriteNode(addDerefs2, loop_body)

        collectLhsExprs = lambda n: [n.lhs] if isinstance(n, ast.BinOpExp) and n.op_type == ast.BinOpExp.EQ_ASGN else []
        loop_lhs_exprs = loop_lib.collectNode(collectLhsExprs, loop_body2)

        # replace all array indices with thread id
        tid = 'tid'
        rewriteToTid = lambda x: ast.IdentExp(tid) if isinstance(x, ast.IdentExp) else x
        rewriteArrayIndices = lambda n: ast.ArrayRefExp(n.exp, loop_lib.rewriteNode(rewriteToTid, n.sub_exp)) if isinstance(n, ast.ArrayRefExp) else n
        loop_body3 = loop_lib.rewriteNode(rewriteArrayIndices, loop_body2)
        # end rewrite the loop body
        #--------------------------------------------------------------------------------------------------------------


        #--------------------------------------------------------------------------------------------------------------
        # begin generate the kernel
        kernelStmts = []
        blockIdx  = ast.IdentExp('blockIdx.x')
        blockSize = ast.IdentExp('blockDim.x')
        threadIdx = ast.IdentExp('threadIdx.x')
        kernelStmts += [
            ast.VarDeclInit('int', tid,
                            ast.BinOpExp(ast.BinOpExp(blockIdx, blockSize, ast.BinOpExp.MUL), threadIdx, ast.BinOpExp.ADD))
        ]
        cacheReads  = []
        cacheWrites = []
        if self.cacheBlocks:
            for var in array_ids:
                sharedVar = 'shared_' + var
                kernelStmts += [
                    # __shared__ double shared_var[threadCount];
                    ast.VarDecl('__shared__ double', [sharedVar + '[' + tcount + ']'])
                ]
                sharedVarExp = ast.ArrayRefExp(ast.IdentExp(sharedVar), threadIdx)
                varExp       = ast.ArrayRefExp(ast.IdentExp(var), ast.IdentExp(tid))
                
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

        kernelStmts += [
            ast.IfStmt(ast.BinOpExp(ast.IdentExp(tid), ubound_exp, ast.BinOpExp.LE),
                       ast.CompStmt(cacheReads + [loop_body3] + cacheWrites))
        ]
        
        # begin reduction statements
        block_r = 'block_r'
        if isReduction:
            kernelStmts += [ast.Comment('reduce single-thread results within a block')]
            # declare the array shared by threads within a block
            kernelStmts += [ast.VarDecl('__shared__ double', ['cache['+tcount+']'])]
            # store the lhs/computed values into the shared array
            kernelStmts += [ast.AssignStmt('cache[threadIdx.x]',loop_lhs_exprs[0])]
            # sync threads prior to reduction
            kernelStmts += [ast.ExpStmt(ast.FunCallExp(ast.IdentExp('__syncthreads'),[]))];
            # at each step, divide the array into two halves and sum two corresponding elements
            # int i = blockDim.x/2;
            idx = 'orcu_i'
            idxId = ast.IdentExp(idx)
            int2 = ast.NumLitExp(2,ast.NumLitExp.INT)
            kernelStmts += [ast.VarDeclInit('int', idx, ast.BinOpExp(ast.IdentExp('blockDim.x'), int2, ast.BinOpExp.DIV))]
            #while(i!=0){
            #  if(threadIdx.x<i)
            #    cache[threadIdx.x]+=cache[threadIdx.x+i];
            #  __syncthreads();
            # i/=2;
            #}
            kernelStmts += [ast.WhileStmt(ast.BinOpExp(idxId, int0, ast.BinOpExp.NE),
                                      ast.CompStmt([ast.IfStmt(ast.BinOpExp(threadIdx, idxId, ast.BinOpExp.LT),
                                                               ast.ExpStmt(ast.BinOpExp(ast.ArrayRefExp(ast.IdentExp('cache'), threadIdx),
                                                                                        ast.ArrayRefExp(ast.IdentExp('cache'),
                                                                                                        ast.BinOpExp(threadIdx,
                                                                                                                     idxId,
                                                                                                                     ast.BinOpExp.ADD)),
                                                                                        ast.BinOpExp.ASGN_ADD))
                                                               ),
                                                    ast.ExpStmt(ast.FunCallExp(ast.IdentExp('__syncthreads'),[])),
                                                    ast.AssignStmt(idx,ast.BinOpExp(idxId, int2, ast.BinOpExp.DIV))
                                                    ])
                                      )]
            # the first thread within a block stores the results for the entire block
            kernelParams += [ast.FieldDecl('double*', block_r)]
            # if(threadIdx.x==0) block_r[blockIdx.x]=cache[0];
            kernelStmts += [
                ast.IfStmt(ast.BinOpExp(threadIdx, int0, ast.BinOpExp.EQ),
                           ast.AssignStmt('block_r[blockIdx.x]',ast.ArrayRefExp(ast.IdentExp('cache'), int0)))
            ]
        # end reduction statements

        dev_kernel_name = 'orcu_kernel'+str(g.Globals().getcounter())
        dev_kernel = ast.FunDecl(dev_kernel_name, 'void', ['__global__'], kernelParams, ast.CompStmt(kernelStmts))
        
        # after getting interprocedural AST, make this a sub to that AST
        g.Globals().cunit_declarations += orio.module.loop.codegen.CodeGen('cuda').generator.generate(dev_kernel, '', '  ')
        # end generate the kernel
        #--------------------------------------------------------------------------------------------------------------
        
        
        #--------------------------------------------------------------------------------------------------------------
        # begin marshal resources
        # declare device variables
        dev = 'dev_'
        dev_lbi = map(lambda x: dev+x, list(lbi))
        dev_block_r = dev + block_r
        host_ids = []
        if isReduction:
            dev_lbi  += [dev_block_r]
            host_ids += [block_r]
        hostDecls  = [ast.Comment('declare variables')]
        hostDecls += [ast.VarDecl('double', map(lambda x: '*'+x, dev_lbi + host_ids))]
        
        # calculate device dimensions
        hostDecls += [ast.VarDecl('dim3', ['dimGrid', 'dimBlock'])]
        gridxIdent = ast.IdentExp('dimGrid.x')
        host_arraysize = ubound_ids[0]
        streamCountNumLit = ast.NumLitExp(self.streamCount, ast.NumLitExp.INT)
        # initialize grid size
        deviceDims  = [ast.Comment('calculate device dimensions')]
        deviceDims += [
            ast.ExpStmt(ast.BinOpExp(gridxIdent,
                                     ast.FunCallExp(ast.IdentExp('ceil'),
                                                    [ast.BinOpExp(ast.CastExpr('float', ast.IdentExp(host_arraysize)),
                                                                  ast.CastExpr('float', ast.IdentExp(tcount)),
                                                                  ast.BinOpExp.DIV)
                                                    ]),
                                     ast.BinOpExp.EQ_ASGN))]
        # initialize block size
        deviceDims += [ast.ExpStmt(ast.BinOpExp(ast.IdentExp('dimBlock.x'), ast.IdentExp(tcount), ast.BinOpExp.EQ_ASGN))]

        # -------------------------------------------------
        # allocate device memory
        mallocs  = [ast.Comment('allocate device memory')]
        # copy data from host to device
        h2dcopys = [ast.Comment('copy data from host to device')]
        # asynchronous copies
        h2dasyncs   = []
        dblIdent    = ast.IdentExp('double')
        sizeofIdent = ast.IdentExp('sizeof')
        scaled_host_arraysize = 'scSize'
        scSizeIdent = ast.IdentExp(scaled_host_arraysize)
        mallocs += [
            ast.VarDeclInit('int', scaled_host_arraysize,
                            ast.BinOpExp(ast.IdentExp(host_arraysize),
                                         ast.FunCallExp(sizeofIdent,[dblIdent]),
                                         ast.BinOpExp.MUL))
        ]
        
        # -------------------------------------------------
        # if streaming, divide vectors into chunks and asynchronously overlap copy-copy and copy-exec ops
        sidx           = 'orcu_i'
        sidxIdent      = ast.IdentExp(sidx)
        soffset        = 'orcu_soff'
        soffsetIdent   = ast.IdentExp(soffset)
        boffset        = 'orcu_boff'
        boffsetIdent   = ast.IdentExp(boffset)
        calc_offset    = []
        calc_boffset   = []
        hostDecls     += [ast.VarDecl('int', [sidx])]
        if self.streamCount > 1:
            # declare and create streams
            hostDecls += [ast.VarDecl('cudaStream_t', ['stream[' + str(self.streamCount) + ']'])]
            hostDecls += [ast.VarDecl('int', [soffset, boffset])]
            mallocs   += [
              ast.ForStmt(ast.BinOpExp(sidxIdent, int0, ast.BinOpExp.EQ_ASGN),
                          ast.BinOpExp(sidxIdent, streamCountNumLit, ast.BinOpExp.LT),
                          ast.UnaryExp(sidxIdent, ast.UnaryExp.POST_INC),
                          ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaStreamCreate'),
                                                     [ast.UnaryExp(ast.IdentExp('stream[' + sidx + ']'),
                                                                   ast.UnaryExp.ADDRESSOF)])))
            ]
            calc_offset = [
              ast.ExpStmt(ast.BinOpExp(soffsetIdent,
                                       ast.BinOpExp(sidxIdent,
                                                    ast.ParenthExp(ast.BinOpExp(scSizeIdent,
                                                                                streamCountNumLit,
                                                                                ast.BinOpExp.DIV)),
                                                    ast.BinOpExp.MUL),
                                       ast.BinOpExp.EQ_ASGN))
            ]
            calc_boffset = [
              ast.ExpStmt(ast.BinOpExp(boffsetIdent,
                                       ast.BinOpExp(sidxIdent,
                                                    ast.ParenthExp(ast.BinOpExp(gridxIdent,
                                                                                streamCountNumLit,
                                                                                ast.BinOpExp.DIV)),
                                                    ast.BinOpExp.MUL),
                                       ast.BinOpExp.EQ_ASGN))
            ]
        # -------------------------------------------------
        dev_scalar_ids = map(lambda x: (x,dev+x), scalar_ids)
        for sid,dsid in dev_scalar_ids:
            # malloc scalars in the form of:
            # -- cudaMalloc((void**)&dev_alpha,sizeof(double));
            mallocs += [
                ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaMalloc'),
                                           [ast.CastExpr('void**', ast.UnaryExp(ast.IdentExp(dsid), ast.UnaryExp.ADDRESSOF)),
                                            ast.FunCallExp(sizeofIdent, [dblIdent])
                                            ]))]
            # memcopy scalars in the form of:
            # -- cudaMemcpy(dev_alpha,&host_alpha,sizeof(double),cudaMemcpyHostToDevice);
            h2dcopys += [
                ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaMemcpy'),
                                           [ast.IdentExp(dsid), ast.UnaryExp(ast.IdentExp(sid), ast.UnaryExp.ADDRESSOF),
                                            ast.FunCallExp(sizeofIdent, [dblIdent]),
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
                                                   [ast.IdentExp(aid),
                                                    ast.IdentExp(host_arraysize),
                                                    ast.IdentExp('cudaHostRegisterPortable')
                                                    ]))
                    ]
                    registeredHostIds += [aid] # remember to unregister at the end
                    h2dasyncs += [
                        ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaMemcpyAsync'),
                                                   [ast.BinOpExp(ast.IdentExp(daid), soffsetIdent, ast.BinOpExp.ADD),
                                                    ast.BinOpExp(ast.IdentExp(aid),  soffsetIdent, ast.BinOpExp.ADD),
                                                    ast.BinOpExp(scSizeIdent, streamCountNumLit, ast.BinOpExp.DIV),
                                                    ast.IdentExp('cudaMemcpyHostToDevice'),
                                                    ast.ArrayRefExp(ast.IdentExp('stream'), sidxIdent)
                                                    ]))]
                else:
                    h2dcopys += [
                        ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaMemcpy'),
                                                   [ast.IdentExp(daid),
                                                    ast.IdentExp(aid),
                                                    scSizeIdent,
                                                    ast.IdentExp('cudaMemcpyHostToDevice')
                                                    ]))]
        # for-loop over streams
        if self.streamCount > 1:
            h2dcopys += [
              ast.ForStmt(ast.BinOpExp(sidxIdent, int0, ast.BinOpExp.EQ_ASGN),
                          ast.BinOpExp(sidxIdent, streamCountNumLit, ast.BinOpExp.LT),
                          ast.UnaryExp(sidxIdent, ast.UnaryExp.POST_INC),
                          ast.CompStmt(calc_offset + h2dasyncs))]
        # -------------------------------------------------
        # malloc block-level result var
        if isReduction:
            mallocs += [
                ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaMalloc'),
                                           [ast.CastExpr('void**', ast.UnaryExp(ast.IdentExp(dev_block_r), ast.UnaryExp.ADDRESSOF)),
                                            ast.BinOpExp(gridxIdent,
                                                         ast.FunCallExp(sizeofIdent,[dblIdent]),
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
                                                   [ast.BinOpExp(gridxIdent,
                                                                 ast.FunCallExp(sizeofIdent,[dblIdent]),
                                                                 ast.BinOpExp.MUL)
                                                    ])))]
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
        kernell_calls = [ast.Comment('invoke device kernel')]
        if self.streamCount == 1:
            args = map(lambda x: ast.IdentExp(x), [host_arraysize] + dev_lbi)
            kernell_call = ast.ExpStmt(ast.FunCallExp(ast.IdentExp(dev_kernel_name+'<<<dimGrid,dimBlock>>>'), args))
            kernell_calls += [kernell_call]
        else:
            dev_array_idss = map(lambda x: dev+x, array_ids)
            args = [ast.IdentExp(host_arraysize)]
            # adjust array args using offsets
            for arg in dev_lbi:
                if arg in dev_array_idss:
                    args += [ast.BinOpExp(ast.IdentExp(arg), soffsetIdent, ast.BinOpExp.ADD)]
                elif arg == dev_block_r:
                    args += [ast.BinOpExp(ast.IdentExp(arg), boffsetIdent, ast.BinOpExp.ADD)]
                else:
                    args += [ast.IdentExp(arg)]
            kernell_call = ast.ExpStmt(ast.FunCallExp(ast.IdentExp(dev_kernel_name+'<<<dimGrid,dimBlock,0,stream['+sidx+']>>>'), args))
            kernell_calls += [ast.ForStmt(
                               ast.BinOpExp(sidxIdent, int0, ast.BinOpExp.EQ_ASGN),
                               ast.BinOpExp(sidxIdent, streamCountNumLit, ast.BinOpExp.LT),
                               ast.UnaryExp(sidxIdent, ast.UnaryExp.POST_INC),
                               ast.CompStmt(calc_offset+calc_boffset+[kernell_call]))]
        
        # -------------------------------------------------
        # copy data from devices to host
        d2hcopys = [ast.Comment('copy data from device to host')]
        d2hasyncs = []
        for var in lhs_ids:
            res_scalar_ids = filter(lambda x: x[1] == (dev+var), dev_scalar_ids)
            for rsid,drsid in res_scalar_ids:
                d2hcopys += [
                    ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaMemcpy'),
                                                   [ast.IdentExp(rsid), ast.IdentExp(drsid),
                                                    ast.FunCallExp(sizeofIdent,[dblIdent]),
                                                    ast.IdentExp('cudaMemcpyDeviceToHost')
                                                    ]))]
            res_array_ids  = filter(lambda x: x[1] == (dev+var), dev_array_ids)
            for raid,draid in res_array_ids:
                if self.streamCount > 1:
                    d2hasyncs += [
                        ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaMemcpyAsync'),
                                                   [ast.BinOpExp(ast.IdentExp(raid),  soffsetIdent, ast.BinOpExp.ADD),
                                                    ast.BinOpExp(ast.IdentExp(draid), soffsetIdent, ast.BinOpExp.ADD),
                                                    ast.BinOpExp(scSizeIdent, streamCountNumLit, ast.BinOpExp.DIV),
                                                    ast.IdentExp('cudaMemcpyDeviceToHost'),
                                                    ast.ArrayRefExp(ast.IdentExp('stream'), sidxIdent)
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
            if self.streamCount > 1:
                d2hasyncs += [
                    ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaMemcpyAsync'),
                                               [ast.BinOpExp(ast.IdentExp(block_r),     boffsetIdent, ast.BinOpExp.ADD),
                                                ast.BinOpExp(ast.IdentExp(dev_block_r), boffsetIdent, ast.BinOpExp.ADD),
                                                ast.BinOpExp(scSizeIdent, streamCountNumLit, ast.BinOpExp.DIV),
                                                ast.IdentExp('cudaMemcpyDeviceToHost'),
                                                ast.ArrayRefExp(ast.IdentExp('stream'), sidxIdent)
                                                ]))]
            else:
                d2hcopys += [
                        ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaMemcpy'),
                                                       [ast.IdentExp(block_r), ast.IdentExp(dev_block_r),
                                                        ast.BinOpExp(gridxIdent,
                                                                     ast.FunCallExp(sizeofIdent,[dblIdent]),
                                                                     ast.BinOpExp.MUL),
                                                        ast.IdentExp('cudaMemcpyDeviceToHost')
                                                        ]))]
        # -------------------------------------------------
        if self.streamCount > 1:
            d2hcopys   += [ast.ForStmt(
                               ast.BinOpExp(sidxIdent, int0, ast.BinOpExp.EQ_ASGN),
                               ast.BinOpExp(sidxIdent, streamCountNumLit, ast.BinOpExp.LT),
                               ast.UnaryExp(sidxIdent, ast.UnaryExp.POST_INC),
                               ast.CompStmt(calc_boffset + d2hasyncs))]
            # synchronize
            d2hcopys   += [ast.ForStmt(
                               ast.BinOpExp(sidxIdent, int0, ast.BinOpExp.EQ_ASGN),
                               ast.BinOpExp(sidxIdent, streamCountNumLit, ast.BinOpExp.LT),
                               ast.UnaryExp(sidxIdent, ast.UnaryExp.POST_INC),
                               ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaStreamSynchronize'),
                                                          [ast.ArrayRefExp(ast.IdentExp('stream'), sidxIdent)])))]
        # -------------------------------------------------
        # reduce block-level results
        pp = []
        if isReduction:
            pp += [ast.Comment('post-processing on the host')]
            pp += [ast.ForStmt(ast.BinOpExp(sidxIdent, int0, ast.BinOpExp.EQ_ASGN),
                               ast.BinOpExp(sidxIdent, gridxIdent, ast.BinOpExp.LT),
                               ast.UnaryExp(sidxIdent, ast.UnaryExp.POST_INC),
                               ast.ExpStmt(ast.BinOpExp(ast.IdentExp(lhs_ids[0]),
                                                        ast.ArrayRefExp(ast.IdentExp(block_r), sidxIdent),
                                                        ast.BinOpExp.ASGN_ADD)))]
        # -------------------------------------------------
        # free allocated memory and resources
        free_vars = [ast.Comment('free allocated memory')]
        if self.streamCount > 1:
            # unregister pinned memory
            for var in registeredHostIds:
                free_vars += [ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaHostUnregister'), [ast.IdentExp(var)]))]
            # destroy streams
            free_vars += [ast.ForStmt(
                               ast.BinOpExp(sidxIdent, int0, ast.BinOpExp.EQ_ASGN),
                               ast.BinOpExp(sidxIdent, streamCountNumLit, ast.BinOpExp.LT),
                               ast.UnaryExp(sidxIdent, ast.UnaryExp.POST_INC),
                               ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaStreamDestroy'),
                                                          [ast.ArrayRefExp(ast.IdentExp('stream'), sidxIdent)])))]
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
                ast.VarDecl('float', ['orcuda_elapsed']),
                ast.VarDecl('FILE*', ['orcuda_fp'])
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
                                           [ast.UnaryExp(ast.IdentExp('orcuda_elapsed'), ast.UnaryExp.ADDRESSOF),
                                            ast.IdentExp('start'), ast.IdentExp('stop')])),
                ast.AssignStmt('orcuda_fp',
                            ast.FunCallExp(ast.IdentExp('fopen'), [ast.StringLitExp('orcuda_time.out'), ast.StringLitExp('a')])),
                ast.ExpStmt(ast.FunCallExp(ast.IdentExp('fprintf'),
                                           [ast.IdentExp('orcuda_fp'),
                                            ast.StringLitExp('Kernel_time@rep[%d]:%fms. '),
                                            ast.IdentExp('orio_i'),
                                            ast.IdentExp('orcuda_elapsed')])),
                ast.ExpStmt(ast.FunCallExp(ast.IdentExp('fclose'), [ast.IdentExp('orcuda_fp')])),
                ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaEventDestroy'), [ast.IdentExp('start')])),
                ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaEventDestroy'), [ast.IdentExp('stop')]))
            ]
        #--------------------------------------------------------------------------------------------------------------
        
        transformed_stmt = ast.CompStmt(
            hostDecls + deviceDims + mallocs + h2dcopys
            +
            timerDecls + timerStart
            +
            kernell_calls
            +
            timerStop
            +
            d2hcopys + pp + free_vars
        )
        return transformed_stmt
    

