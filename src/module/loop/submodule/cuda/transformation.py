#
# Contain the CUDA transformation module
#

import orio.module.loop.ast_lib.forloop_lib, orio.module.loop.ast_lib.common_lib
import orio.main.util.globals as g
import orio.module.loop.ast as ast

#----------------------------------------------------------------------------------------------------------------------

class Transformation:
    '''Code transformation'''

    def __init__(self, stmt, threadCount, blockCount):
        '''Instantiate a code transformation object'''
        self.stmt        = stmt
        self.threadCount = threadCount
        self.blockCount  = blockCount

    def transform(self):
        '''Transform the enclosed for-loop'''
        # get rid of compound statement that contains only a single statement
        while isinstance(self.stmt.stmt, ast.CompStmt) and len(self.stmt.stmt.stmts) == 1:
            self.stmt.stmt = self.stmt.stmt.stmts[0]
        
        # extract for-loop structure
        index_id, _, ubound_exp, _, loop_body = orio.module.loop.ast_lib.forloop_lib.ForLoopLib().extractForLoopInfo(self.stmt)

        loop_lib = orio.module.loop.ast_lib.common_lib.CommonLib()
        
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
        
        # add dereferences to all non-array id's in the loop body
        collectArrayIdents = lambda n: [n.exp.name] if (isinstance(n, ast.ArrayRefExp) and isinstance(n.exp, ast.IdentExp)) else []
        array_ids = loop_lib.collectNode(collectArrayIdents, loop_body)
        non_array_ids = list(lbi.difference(set(array_ids)))
        addDerefs2 = lambda n: ast.ParenthExp(ast.UnaryExp(n, ast.UnaryExp.DEREF)) if (isinstance(n, ast.IdentExp) and n.name in non_array_ids) else n
        loop_body2 = loop_lib.rewriteNode(addDerefs2, loop_body)

        # replace all array indices with thread id
        tid = 'tid'
        rewriteToTid = lambda x: ast.IdentExp(tid) if isinstance(x, ast.IdentExp) else x
        rewriteArrayIndices = lambda n: ast.ArrayRefExp(n.exp, loop_lib.rewriteNode(rewriteToTid, n.sub_exp)) if isinstance(n, ast.ArrayRefExp) else n
        loop_body3 = loop_lib.rewriteNode(rewriteArrayIndices, loop_body2)

        # generate the transformed statement
        # temp_id = FieldDecl('double*', 'orcuda_arg_'+str(Globals().getcounter())),
        decl_tid = ast.VarDecl('int', [tid])
        assign_tid = ast.AssignStmt(tid,
                                    ast.BinOpExp(ast.BinOpExp(ast.IdentExp('blockIdx.x'),
                                                 ast.IdentExp('blockDim.x'),
                                                 ast.BinOpExp.MUL),
                                         ast.IdentExp('threadIdx.x'),
                                         ast.BinOpExp.ADD)
                                    )
        if_stmt = ast.IfStmt(ast.BinOpExp(ast.IdentExp(tid), ubound_exp, ast.BinOpExp.LE), loop_body3)
        
        dev_kernel_name = 'orcuda_kern_'+str(g.Globals().getcounter())
        dev_kernel = ast.FunDecl(dev_kernel_name,
                                 'void',
                                 ['__global__'],
                                 ubound_id_decls + lbi_decls,
                                 ast.CompStmt([decl_tid,assign_tid,if_stmt]))
        
        # TODO: refactor, make this more graceful
        g.Globals().cunit_declarations += orio.module.loop.codegen.CodeGen('cuda').generator.generate(dev_kernel, '', '  ')
        
        
        
        # begin marshal resources
        # declare device variables
        # -- double* dev_X, *dev_Y, *dev_alpha;
        dev = 'dev_'
        dev_lbi = map(lambda x: dev+x, list(lbi))
        dev_lbi2 = map(lambda x: '*'+x, dev_lbi)
        dev_double_decls = ast.VarDecl('double', dev_lbi2)
        # -- int *dev_arraysize;
        dev_ubounds = map(lambda x: dev+x, ubound_ids)
        dev_int_decls = ast.VarDecl('int*', dev_ubounds)
        
        # calculate device dimensions
        # -- dim3 dimGrid, dimBlock;
        dev_dim_decls = ast.VarDecl('dim3', ['dimGrid', 'dimBlock'])
        gridx = 'dimGrid.x'
        blocx = 'dimBlock.x'
        host_arraysize = ubound_ids[0]
        dev_arraysize = dev + host_arraysize
        # -- dimGrid.x=ceil((float)host_arraysize/(float)THREADCOUNT);
        tcount = str(self.threadCount)
        init_gsize = ast.AssignStmt(gridx,
                                    ast.FunCallExp(ast.IdentExp('ceil'),
                                                   [ast.BinOpExp(ast.CastExpr('float', ast.IdentExp(host_arraysize)),
                                                                 ast.CastExpr('float', ast.IdentExp(tcount)),
                                                                 ast.BinOpExp.DIV)
                                                    ]))
        # -- dimBlock.x=THREADCOUNT;
        init_bsize = ast.AssignStmt(blocx, ast.IdentExp(tcount))

        #calc_dev_dims = ast.WhileStmt(ast.BinOpExp(ast.IdentExp(gridx), ast.IdentExp(str(self.blockCount)), ast.BinOpExp.GT),
        #                              ast.CompStmt([ast.AssignStmt(gridx, ast.BinOpExp(ast.IdentExp(gridx), ast.NumLitExp(2,ast.NumLitExp.INT), ast.BinOpExp.DIV)),
        #                                            ast.AssignStmt(blocx, ast.BinOpExp(ast.IdentExp(blocx), ast.NumLitExp(2,ast.NumLitExp.INT), ast.BinOpExp.MUL))
        #                                            ]))

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
            # -- cudaMalloc((void**)&dev_alpha,sizeof(double));
            malloc_scalars.append(
                        ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaMalloc'),
                                                   [ast.CastExpr('void**', ast.UnaryExp(ast.IdentExp(dsid), ast.UnaryExp.ADDRESSOF)),
                                                    ast.FunCallExp(ast.IdentExp('sizeof'), [ast.IdentExp('double')])
                                                    ]))
                                  )
            # -- cudaMemcpy(dev_alpha,&host_alpha,sizeof(double),cudaMemcpyHostToDevice);
            memcopy_scalars.append(
                        ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaMemcpy'),
                                                   [ast.IdentExp(dsid),
                                                    ast.UnaryExp(ast.IdentExp(sid), ast.UnaryExp.ADDRESSOF),
                                                    ast.FunCallExp(ast.IdentExp('sizeof'), [ast.IdentExp('double')]),
                                                    ast.IdentExp('cudaMemcpyHostToDevice')
                                                    ]))
                                   )

        dev_array_ids = map(lambda x: (x,dev+x), set(array_ids))
        malloc_arrays = []
        memcpy_arrays = []
        for aid,daid in dev_array_ids:
            # -- cudaMalloc((void**)&dev_X,host_arraysize*sizeof(double));
            malloc_arrays.append(
                        ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaMalloc'),
                                                   [ast.CastExpr('void**', ast.UnaryExp(ast.IdentExp(daid), ast.UnaryExp.ADDRESSOF)),
                                                    ast.BinOpExp(ast.IdentExp(host_arraysize),
                                                                 ast.FunCallExp(ast.IdentExp('sizeof'),[ast.IdentExp('double')]),
                                                                 ast.BinOpExp.MUL)
                                                    ]))
                                  )
            # -- cudaMemcpy(dev_X,host_X,host_arraysize*sizeof(double),cudaMemcpyHostToDevice);
            memcpy_arrays.append(
                        ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaMemcpy'),
                                                   [ast.IdentExp(daid),
                                                    ast.IdentExp(aid),
                                                    ast.BinOpExp(ast.IdentExp(host_arraysize),
                                                                 ast.FunCallExp(ast.IdentExp('sizeof'),[ast.IdentExp('double')]),
                                                                 ast.BinOpExp.MUL),
                                                    ast.IdentExp('cudaMemcpyHostToDevice')
                                                    ]))
                                   )
        
        # invoke device kernel function
        # -- kernAXPY<<<dimGrid,dimBlock>>>(dev_Y,dev_X,dev_arraysize,dev_alpha);
        args = map(lambda x: ast.IdentExp(x), dev_ubounds + dev_lbi)
        kernell_call = ast.ExpStmt(ast.FunCallExp(ast.IdentExp(dev_kernel_name+'<<<dimGrid,dimBlock>>>'), args))
        
        # copy data from devices to host
        # -- cudaMemcpy(host_Y,dev_Y,host_arraysize*sizeof(double),cudaMemcpyDeviceToHost);
        collectLhs = lambda n: [n.lhs.exp.name] if (isinstance(n, ast.BinOpExp) and 
                                                    (n.op_type == ast.BinOpExp.EQ_ASGN) and
                                                    isinstance(n.lhs, ast.ArrayRefExp) and 
                                                    isinstance(n.lhs.exp, ast.IdentExp)) else []
        loop_lhs = loop_lib.collectNode(collectLhs, loop_body)
        result_id = loop_lhs[0]
        res_array_ids = filter(lambda x: x[1] == (dev+result_id), dev_array_ids) # pairs of host and device id
        res_array_id = res_array_ids[0]
        memcpy_result = ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaMemcpy'),
                                                   [ast.IdentExp(res_array_id[0]),
                                                    ast.IdentExp(res_array_id[1]),
                                                    ast.BinOpExp(ast.IdentExp(host_arraysize),
                                                                 ast.FunCallExp(ast.IdentExp('sizeof'),[ast.IdentExp('double')]),
                                                                 ast.BinOpExp.MUL),
                                                    ast.IdentExp('cudaMemcpyDeviceToHost')
                                                    ]))
        # free device memory
        dev_vars = dev_ubounds + dev_lbi
        free_dev_vars = []
        for dvar in dev_vars:
            # -- cudaFree(dev_alpha);
            free_dev_vars.append(
                        ast.ExpStmt(ast.FunCallExp(ast.IdentExp('cudaFree'),
                                                   [ast.IdentExp(dvar)]))
                                  )
        # end marshal resources
        
        transformed_stmt = \
               ast.CompStmt([ast.Comment('declare device variables'),
                             dev_double_decls,
                             dev_int_decls,
                             dev_dim_decls,
                             ast.Comment('calculate device dimensions'),
                             init_gsize,
                             init_bsize,
                             #calc_dev_dims,
                             ast.Comment('allocate device memory'),
                             malloc_ubound
                             ] +
                            malloc_scalars +
                            malloc_arrays +
                            [ast.Comment('copy data from host to devices'),
                             memcpy_ubound] +
                            memcopy_scalars +
                            memcpy_arrays +
                            [ast.Comment('invoke device kernel function'),
                             kernell_call,
                             ast.Comment('copy data from devices to host'),
                             memcpy_result,
                             ast.Comment('free device memory')
                             ] +
                            free_dev_vars
                            )
        
        return transformed_stmt
    

