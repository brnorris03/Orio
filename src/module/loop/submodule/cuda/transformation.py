#
# Contain the CUDA transformation module
#

import orio.module.loop.ast_lib.forloop_lib, orio.module.loop.ast_lib.common_lib
import orio.main.util.globals as g
import orio.module.loop.ast as ast

#----------------------------------------------------------------------------------------------------------------------

class Transformation:
    '''Code transformation'''

    def __init__(self, stmt, threadCount, blockCount, devProps):
        '''Instantiate a code transformation object'''
        self.stmt        = stmt
        self.threadCount = threadCount
        self.blockCount  = blockCount
        self.devProps    = devProps

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

        # begin rewrite the loop body        
        # collect all identifiers from the loop's upper bound expression
        collectIdents = lambda n: [n.name] if isinstance(n, ast.IdentExp) else []
        ubound_ids = loop_lib.collectNode(collectIdents, ubound_exp)
        
        # create decls for ubound_exp id's, assuming all ids are int's
        ubound_id_decls = [ast.FieldDecl('int*', x) for x in ubound_ids]

        # add dereferences to all id's in the ubound_exp
        addDerefs = lambda n: ast.ParenthExp(ast.UnaryExp(n, ast.UnaryExp.DEREF)) if isinstance(n, ast.IdentExp) else n
        loop_lib.rewriteNode(addDerefs, ubound_exp)
        
        # collect all identifiers from the loop body
        loop_body_ids = loop_lib.collectNode(collectIdents, loop_body)
        lbi = set(filter(lambda x: x != index_id.name, loop_body_ids))
        
        # create decls for loop body id's
        lbi_decls = [ast.FieldDecl('double*', x) for x in lbi]
        
        # collect all LHS identifiers within the loop body
        def collectLhsIds(n):
            if isinstance(n, ast.BinOpExp) and n.op_type == ast.BinOpExp.EQ_ASGN:
                if isinstance(n.lhs, ast.IdentExp):
                    return [n.lhs.name]
                elif isinstance(n.lhs, ast.ArrayRefExp) and isinstance(n.lhs.exp, ast.IdentExp):
                    return [n.lhs.exp.name]
                else: return []
            else: return []
        loop_lhs_ids = loop_lib.collectNode(collectLhsIds, loop_body)

        # collect all array and non-array idents in the loop body
        collectArrayIdents = lambda n: [n.exp.name] if (isinstance(n, ast.ArrayRefExp) and isinstance(n.exp, ast.IdentExp)) else []
        array_ids = loop_lib.collectNode(collectArrayIdents, loop_body)
        non_array_ids = list(lbi.difference(set(array_ids)))
        loop_lhs_array_ids = list(set(loop_lhs_ids).intersection(set(array_ids)))
        isReduction = len(loop_lhs_array_ids) == 0

        # add dereferences to all non-array id's in the loop body
        addDerefs2 = lambda n: ast.ParenthExp(ast.UnaryExp(n, ast.UnaryExp.DEREF)) if (isinstance(n, ast.IdentExp) and n.name in non_array_ids) else n
        loop_body2 = loop_lib.rewriteNode(addDerefs2, loop_body)

        collectLhsExprs = lambda n: [n.lhs] if isinstance(n, ast.BinOpExp) and n.op_type == ast.BinOpExp.EQ_ASGN else []
        loop_lhs_exprs = loop_lib.collectNode(collectLhsExprs, loop_body2)

        # replace all array indices with thread id
        tid = 'tid'
        rewriteToTid = lambda x: ast.IdentExp(tid) if isinstance(x, ast.IdentExp) else x
        rewriteArrayIndices = lambda n: ast.ArrayRefExp(n.exp, loop_lib.rewriteNode(rewriteToTid, n.sub_exp)) if isinstance(n, ast.ArrayRefExp) else n
        loop_body3 = loop_lib.rewriteNode(rewriteArrayIndices, loop_body2)
        # end rewrite the loop body


        # begin generate the kernel
        decl_tid = ast.VarDecl('int', [tid])
        assign_tid = ast.AssignStmt(tid,
                                    ast.BinOpExp(ast.BinOpExp(ast.IdentExp('blockIdx.x'),
                                                 ast.IdentExp('blockDim.x'),
                                                 ast.BinOpExp.MUL),
                                         ast.IdentExp('threadIdx.x'),
                                         ast.BinOpExp.ADD)
                                    )
        if_stmt = ast.IfStmt(ast.BinOpExp(ast.IdentExp(tid), ubound_exp, ast.BinOpExp.LE), loop_body3)
        
        # begin reduction statements
        reducts = []
        block_reduct_decl = []
        block_r = 'block_r'
        if isReduction:
            reducts += [ast.Comment('reduce single-thread results within a block')]
            # declare the array shared by threads within a block
            reducts += [ast.VarDecl('__shared__ double', ['cache['+tcount+']'])]
            # store the lhs/computed values into the shared array
            reducts += [ast.AssignStmt('cache[threadIdx.x]',loop_lhs_exprs[0])]
            # sync threads prior to reduction
            reducts += [ast.ExpStmt(ast.FunCallExp(ast.IdentExp('__syncthreads'),[]))];
            # at each step, divide the array into two halves and sum two corresponding elements
            # int i = blockDim.x/2;
            idx = 'i'
            idxId = ast.IdentExp(idx)
            int2 = ast.NumLitExp(2,ast.NumLitExp.INT)
            reducts += [ast.VarDecl('int', [idx])]
            reducts += [ast.AssignStmt(idx, ast.BinOpExp(ast.IdentExp('blockDim.x'), int2, ast.BinOpExp.DIV))]
            #while(i!=0){
            #  if(threadIdx.x<i)
            #    cache[threadIdx.x]+=cache[threadIdx.x+i];
            #  __syncthreads();
            # i/=2;
            #}
            reducts += [ast.WhileStmt(ast.BinOpExp(idxId, int0, ast.BinOpExp.NE),
                                      ast.CompStmt([ast.IfStmt(ast.BinOpExp(ast.IdentExp('threadIdx.x'), idxId, ast.BinOpExp.LT),
                                                               ast.AssignStmt('cache[threadIdx.x]',
                                                                              ast.BinOpExp(ast.ArrayRefExp(ast.IdentExp('cache'),
                                                                                                           ast.IdentExp('threadIdx.x')),
                                                                                           ast.ArrayRefExp(ast.IdentExp('cache'),
                                                                                                           ast.BinOpExp(ast.IdentExp('threadIdx.x'),
                                                                                                                        idxId,
                                                                                                                        ast.BinOpExp.ADD)),
                                                                                           ast.BinOpExp.ADD))
                                                               ),
                                                    ast.ExpStmt(ast.FunCallExp(ast.IdentExp('__syncthreads'),[])),
                                                    ast.AssignStmt(idx,ast.BinOpExp(idxId, int2, ast.BinOpExp.DIV))
                                                    ])
                                      )]
            # the first thread within a block stores the results for the entire block
            block_reduct_decl += [ast.FieldDecl('double*', block_r)]
            #if(threadIdx.x==0)  block_r[blockIdx.x]=cache[0];
            reducts += [ast.IfStmt(ast.BinOpExp(ast.IdentExp('threadIdx.x'),
                                                ast.NumLitExp(0, ast.NumLitExp.INT),
                                                ast.BinOpExp.EQ),
                                   ast.AssignStmt('block_r[blockIdx.x]',ast.ArrayRefExp(ast.IdentExp('cache'),
                                                                                        int0)))
                        ]
        # end reduction statements

        dev_kernel_name = 'orcuda_kern_'+str(g.Globals().getcounter())
        dev_kernel = ast.FunDecl(dev_kernel_name,
                                 'void',
                                 ['__global__'],
                                 ubound_id_decls + lbi_decls + block_reduct_decl,
                                 ast.CompStmt([decl_tid,assign_tid,if_stmt] + reducts))
        
        
        # TODO: refactor, make this more graceful
        g.Globals().cunit_declarations += orio.module.loop.codegen.CodeGen('cuda').generator.generate(dev_kernel, '', '  ')
        # end generate the kernel
        
        
        # begin marshal resources
        # declare device variables
        dev = 'dev_'
        dev_lbi = map(lambda x: dev+x, list(lbi))
        dev_block_r = dev + block_r
        host_ids = []
        if isReduction:
            dev_lbi += [dev_block_r]
            host_ids += [block_r]
        dev_ubounds = map(lambda x: dev+x, ubound_ids)
        dev_double_decls = ast.VarDecl('double', map(lambda x: '*'+x, dev_lbi + host_ids))
        dev_int_decls = ast.VarDecl('int*', dev_ubounds)
        
        # calculate device dimensions
        dev_dim_decls = ast.VarDecl('dim3', ['dimGrid', 'dimBlock'])
        gridx = 'dimGrid.x'
        blocx = 'dimBlock.x'
        host_arraysize = ubound_ids[0]
        dev_arraysize = dev + host_arraysize
        # initialize grid size
        init_gsize = ast.AssignStmt(gridx,
                                    ast.FunCallExp(ast.IdentExp('ceil'),
                                                   [ast.BinOpExp(ast.CastExpr('float', ast.IdentExp(host_arraysize)),
                                                                 ast.CastExpr('float', ast.IdentExp(tcount)),
                                                                 ast.BinOpExp.DIV)
                                                    ]))
        # initialize block size
        init_bsize = ast.AssignStmt(blocx, ast.IdentExp(tcount))

        # allocate device memory
        # copy data from host to device
        # -- cudaMalloc((void**)&dev_arraysize,sizeof(int));
        malloc_ubound = ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaMalloc'),
                                                   [ast.CastExpr('void**', ast.UnaryExp(ast.IdentExp(dev_arraysize), ast.UnaryExp.ADDRESSOF)),
                                                    ast.FunCallExp(ast.IdentExp('sizeof'), [ast.IdentExp('int')])
                                                    ]))
        # -- cudaMemcpy(dev_arraysize,&host_arraysize,sizeof(int),cudaMemcpyHostToDevice);
        memcpy_ubound = ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaMemcpy'),
                                                   [ast.IdentExp(dev_arraysize),
                                                    ast.UnaryExp(ast.IdentExp(host_arraysize), ast.UnaryExp.ADDRESSOF),
                                                    ast.FunCallExp(ast.IdentExp('sizeof'), [ast.IdentExp('int')]),
                                                    ast.IdentExp('cudaMemcpyHostToDevice')
                                                    ]))
        dev_scalar_ids = map(lambda x: (x,dev+x), non_array_ids)
        malloc_scalars = []
        memcopy_scalars = []
        for sid,dsid in dev_scalar_ids:
            # malloc scalars in the form of:
            # -- cudaMalloc((void**)&dev_alpha,sizeof(double));
            malloc_scalars += [
                        ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaMalloc'),
                                                   [ast.CastExpr('void**', ast.UnaryExp(ast.IdentExp(dsid), ast.UnaryExp.ADDRESSOF)),
                                                    ast.FunCallExp(ast.IdentExp('sizeof'), [ast.IdentExp('double')])
                                                    ]))]
            # memcopy scalars in the form of:
            # -- cudaMemcpy(dev_alpha,&host_alpha,sizeof(double),cudaMemcpyHostToDevice);
            memcopy_scalars += [
                        ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaMemcpy'),
                                                   [ast.IdentExp(dsid),
                                                    ast.UnaryExp(ast.IdentExp(sid), ast.UnaryExp.ADDRESSOF),
                                                    ast.FunCallExp(ast.IdentExp('sizeof'), [ast.IdentExp('double')]),
                                                    ast.IdentExp('cudaMemcpyHostToDevice')
                                                    ]))]

        dev_array_ids = map(lambda x: (x,dev+x), set(array_ids))
        malloc_arrays = []
        memcpy_arrays = []
        for aid,daid in dev_array_ids:
            # malloc arrays in the form of:
            # -- cudaMalloc((void**)&dev_X,host_arraysize*sizeof(double));
            malloc_arrays += [
                        ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaMalloc'),
                                                   [ast.CastExpr('void**', ast.UnaryExp(ast.IdentExp(daid), ast.UnaryExp.ADDRESSOF)),
                                                    ast.BinOpExp(ast.IdentExp(host_arraysize),
                                                                 ast.FunCallExp(ast.IdentExp('sizeof'),[ast.IdentExp('double')]),
                                                                 ast.BinOpExp.MUL)
                                                    ]))]
            # memcopy in the form of:
            # -- cudaMemcpy(dev_X,host_X,host_arraysize*sizeof(double),cudaMemcpyHostToDevice);
            memcpy_arrays += [
                        ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaMemcpy'),
                                                   [ast.IdentExp(daid),
                                                    ast.IdentExp(aid),
                                                    ast.BinOpExp(ast.IdentExp(host_arraysize),
                                                                 ast.FunCallExp(ast.IdentExp('sizeof'),[ast.IdentExp('double')]),
                                                                 ast.BinOpExp.MUL),
                                                    ast.IdentExp('cudaMemcpyHostToDevice')
                                                    ]))]
        # malloc block-level result var
        if isReduction:
            malloc_arrays += [
                    ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaMalloc'),
                                               [ast.CastExpr('void**', ast.UnaryExp(ast.IdentExp(dev_block_r), ast.UnaryExp.ADDRESSOF)),
                                                ast.BinOpExp(ast.IdentExp(gridx),
                                                             ast.FunCallExp(ast.IdentExp('sizeof'),[ast.IdentExp('double')]),
                                                             ast.BinOpExp.MUL)
                                                ]))]
            malloc_arrays += [
                    ast.AssignStmt(block_r,
                                   ast.CastExpr('double*',
                                                ast.FunCallExp(ast.IdentExp('malloc'),
                                               [ast.BinOpExp(ast.IdentExp(gridx),
                                                             ast.FunCallExp(ast.IdentExp('sizeof'),[ast.IdentExp('double')]),
                                                             ast.BinOpExp.MUL)
                                                ])))]
        # invoke device kernel function:
        # -- kernelFun<<<numOfBlocks,numOfThreads>>>(dev_vars, ..., dev_result);
        args = map(lambda x: ast.IdentExp(x), dev_ubounds + dev_lbi)
        kernell_call = ast.ExpStmt(ast.FunCallExp(ast.IdentExp(dev_kernel_name+'<<<dimGrid,dimBlock>>>'), args))
        
        # copy data from devices to host
        # -- cudaMemcpy(host_Y,dev_Y,host_arraysize*sizeof(double),cudaMemcpyDeviceToHost);
        memcpy_res_scl = []
        memcpy_res_arr = []
        for result_id in loop_lhs_ids:
            res_scalar_ids = filter(lambda x: x[1] == (dev+result_id), dev_scalar_ids)
            for res_scalar_id,dres_scalar_id in res_scalar_ids:
                memcpy_res_scl += [ast.ExpStmt(  ast.FunCallExp(ast.IdentExp('cudaMemcpy'),
                                                   [ast.IdentExp(res_scalar_id),
                                                    ast.IdentExp(dres_scalar_id),
                                                    ast.FunCallExp(ast.IdentExp('sizeof'),[ast.IdentExp('double')]),
                                                    ast.IdentExp('cudaMemcpyDeviceToHost')
                                                    ]))]
            res_array_ids  = filter(lambda x: x[1] == (dev+result_id), dev_array_ids)
            for res_array_id,dres_array_id in res_array_ids:
                memcpy_res_arr += [ast.ExpStmt(  ast.FunCallExp(ast.IdentExp('cudaMemcpy'),
                                                   [ast.IdentExp(res_array_id),
                                                    ast.IdentExp(dres_array_id),
                                                    ast.BinOpExp(ast.IdentExp(host_arraysize),
                                                                 ast.FunCallExp(ast.IdentExp('sizeof'),[ast.IdentExp('double')]),
                                                                 ast.BinOpExp.MUL),
                                                    ast.IdentExp('cudaMemcpyDeviceToHost')
                                                    ]))]
        # memcpy block-level result var
        if isReduction:
            memcpy_res_arr += [ast.ExpStmt(  ast.FunCallExp(ast.IdentExp('cudaMemcpy'),
                                                   [ast.IdentExp(block_r),
                                                    ast.IdentExp(dev_block_r),
                                                    ast.BinOpExp(ast.IdentExp(gridx),
                                                                 ast.FunCallExp(ast.IdentExp('sizeof'),[ast.IdentExp('double')]),
                                                                 ast.BinOpExp.MUL),
                                                    ast.IdentExp('cudaMemcpyDeviceToHost')
                                                    ]))]
        memcpy_result = memcpy_res_scl + memcpy_res_arr
        
        # reduce block-level results
        pp = []
        if isReduction:
            pp += [ast.VarDecl('int', ['i'])]
            pp += [ast.ForStmt(ast.BinOpExp(ast.IdentExp('i'), int0, ast.BinOpExp.EQ_ASGN),
                              ast.BinOpExp(ast.IdentExp('i'), ast.IdentExp(gridx), ast.BinOpExp.LT),
                              ast.UnaryExp(ast.IdentExp('i'), ast.UnaryExp.POST_INC),
                              ast.AssignStmt(loop_lhs_ids[0],
                                             ast.BinOpExp(ast.IdentExp(loop_lhs_ids[0]),
                                                          ast.ArrayRefExp(ast.IdentExp(block_r), ast.IdentExp('i')),
                                                          ast.BinOpExp.ADD)))]

        # free allocated variables
        dev_vars = dev_ubounds + dev_lbi
        free_vars = []
        for dvar in dev_vars:
            free_vars += [ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaFree'), [ast.IdentExp(dvar)]))]
        for hvar in host_ids:
            free_vars += [ast.ExpStmt(ast.FunCallExp(ast.IdentExp('free'), [ast.IdentExp(hvar)]))]
        # end marshal resources
        
        # cuda timing calls
        timeEventsDecl  = ast.VarDecl('cudaEvent_t', ['start', 'stop'])
        timeElapsedDecl = ast.VarDecl('float', ['orcuda_elapsedTime'])
        timeFileDecl    = ast.VarDecl('FILE*', ['orcuda_fp'])
        createStart = ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaEventCreate'),
                                                 [ast.UnaryExp(ast.IdentExp('start'), ast.UnaryExp.ADDRESSOF)]))
        createStop  = ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaEventCreate'),
                                                 [ast.UnaryExp(ast.IdentExp('stop'),  ast.UnaryExp.ADDRESSOF)]))
        recordStart = ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaEventRecord'),
                                                 [ast.IdentExp('start'), int0]))
        recordStop  = ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaEventRecord'),
                                                 [ast.IdentExp('stop'), int0]))
        syncStop    = ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaEventSynchronize'),
                                                 [ast.IdentExp('stop')]))
        calcElapsed = ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaEventElapsedTime'),
                                                 [ast.UnaryExp(ast.IdentExp('orcuda_elapsedTime'), ast.UnaryExp.ADDRESSOF),
                                                  ast.IdentExp('start'), ast.IdentExp('stop')]))
        timeFileOpen= ast.AssignStmt('orcuda_fp',
                                     ast.FunCallExp(ast.IdentExp('fopen'),
                                                 [ast.StringLitExp('orcuda_time.out'),
                                                  ast.StringLitExp('a')]))
        printElapsed= ast.ExpStmt(ast.FunCallExp(ast.IdentExp('fprintf'),
                                                 [ast.IdentExp('orcuda_fp'),
                                                  ast.StringLitExp('Kernel time @rep[%d]: %f ms'),
                                                  ast.IdentExp('orio_i'),
                                                  ast.IdentExp('orcuda_elapsedTime')]))
        destroyStart= ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaEventDestroy'),
                                                 [ast.IdentExp('start')]))
        destroyStop = ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaEventDestroy'),
                                                 [ast.IdentExp('stop')]))
        destroyTimeFP=ast.ExpStmt(ast.FunCallExp(ast.IdentExp('fclose'),
                                                 [ast.IdentExp('orcuda_fp')]))
        
        transformed_stmt = \
               ast.CompStmt([ast.Comment('declare device variables'),
                             dev_double_decls,
                             dev_int_decls,
                             dev_dim_decls,
                             ast.Comment('calculate device dimensions'),
                             init_gsize,
                             init_bsize,
                             ast.Comment('allocate device memory'),
                             malloc_ubound
                             ] +
                            malloc_scalars +
                            malloc_arrays +
                            [ast.Comment('copy data from host to devices'),
                             memcpy_ubound] +
                            memcopy_scalars +
                            memcpy_arrays +
                            [timeEventsDecl, timeElapsedDecl, timeFileDecl,
                             createStart, createStop, recordStart
                             ] +
                            [ast.Comment('invoke device kernel function'),
                             kernell_call
                             ] +
                            [recordStop, syncStop, calcElapsed, timeFileOpen, printElapsed,
                             destroyStart, destroyStop, destroyTimeFP
                             ] +
                            [ast.Comment('copy data from devices to host')] +
                             memcpy_result +
                             [ast.Comment('post-processing on the host')] +
                             pp +
                             [ast.Comment('free device memory')] +
                            free_vars
                            )
        
        return transformed_stmt
    

