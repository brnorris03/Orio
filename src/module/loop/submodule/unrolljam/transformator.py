#
# Contain the transformation procedure
#

import sys
import module.loop.ast, module.loop.ast_lib.constant_folder, module.loop.ast_lib.forloop_lib

#-----------------------------------------

class Transformator:
    '''Code transformator'''

    def __init__(self, ufactor, do_jamming, stmt, init_cleanup_loop):
        '''To instantiate a code transformator object'''

        self.ufactor = ufactor
        self.do_jamming = do_jamming
        self.stmt = stmt
        self.init_cleanup_loop = init_cleanup_loop
        self.flib = module.loop.ast_lib.forloop_lib.ForLoopLib()
        self.cfolder = module.loop.ast_lib.constant_folder.ConstFolder()
        
    #----------------------------------------------------------

    def __addIdentWithExp(self, tnode, index_name, exp):
        '''Traverse the tree node and add any matching identifier with the provided expression'''

        if isinstance(exp, module.loop.ast.NumLitExp) and exp.val == 0:
            return tnode
        
        if isinstance(tnode, module.loop.ast.ExpStmt):
            if tnode.exp:
                tnode.exp = self.__addIdentWithExp(tnode.exp, index_name, exp)
            return tnode
    
        elif isinstance(tnode, module.loop.ast.CompStmt):
            tnode.stmts = [self.__addIdentWithExp(s, index_name, exp) for s in tnode.stmts]
            return tnode

        elif isinstance(tnode, module.loop.ast.IfStmt):
            tnode.test = self.__addIdentWithExp(tnode.test, index_name, exp)
            tnode.true_stmt = self.__addIdentWithExp(tnode.true_stmt, index_name, exp)
            if tnode.false_stmt:
                tnode.false_stmt = self.__addIdentWithExp(tnode.false_stmt, index_name, exp)
            return tnode

        elif isinstance(tnode, module.loop.ast.ForStmt):
            if tnode.init:
                tnode.init = self.__addIdentWithExp(tnode.init, index_name, exp)
            if tnode.test:
                tnode.test = self.__addIdentWithExp(tnode.test, index_name, exp)
            if tnode.iter:
                tnode.iter = self.__addIdentWithExp(tnode.iter, index_name, exp)
            tnode.stmt = self.__addIdentWithExp(tnode.stmt, index_name, exp)
            return tnode

        elif isinstance(tnode, module.loop.ast.TransformStmt):
            print 'internal error: unprocessed transform statement'
            sys.exit(1)

        elif isinstance(tnode, module.loop.ast.NumLitExp):
            return tnode

        elif isinstance(tnode, module.loop.ast.StringLitExp):
            return tnode

        elif isinstance(tnode, module.loop.ast.IdentExp):
            if tnode.name != index_name:
                return tnode
            else:
                add_exp = module.loop.ast.BinOpExp(tnode,
                                                   exp.replicate(),
                                                   module.loop.ast.BinOpExp.ADD)
                return module.loop.ast.ParenthExp(add_exp)

        elif isinstance(tnode, module.loop.ast.ArrayRefExp):
            tnode.exp = self.__addIdentWithExp(tnode.exp, index_name, exp)
            tnode.sub_exp = self.__addIdentWithExp(tnode.sub_exp, index_name, exp)
            return tnode
        
        elif isinstance(tnode, module.loop.ast.FunCallExp):
            tnode.exp = self.__addIdentWithExp(tnode.exp, index_name, exp)
            tnode.args = [self.__addIdentWithExp(a, index_name, exp) for a in tnode.args]
            return tnode

        elif isinstance(tnode, module.loop.ast.UnaryExp):
            tnode.exp = self.__addIdentWithExp(tnode.exp, index_name, exp)
            return tnode
        
        elif isinstance(tnode, module.loop.ast.BinOpExp):
            tnode.lhs = self.__addIdentWithExp(tnode.lhs, index_name, exp)
            tnode.rhs = self.__addIdentWithExp(tnode.rhs, index_name, exp)
            return tnode

        elif isinstance(tnode, module.loop.ast.ParenthExp):
            tnode.exp = self.__addIdentWithExp(tnode.exp, index_name, exp)
            return tnode        
        
        elif isinstance(tnode, module.loop.ast.NewAST):
            return tnode
        
        else:
            print 'internal error: unexpected AST type: "%s"' % tnode.__class__.__name__
            sys.exit(1)
    
    #-----------------------------------------

    def __jamStmtSeqs(self, stmt_seqs):
        '''Permute and jam the given statement sequences'''

        # check the validity of input parameter
        if len(stmt_seqs) == 0:
            print 'internal error: input statement sequences must not be empty'
            sys.exit(1)
        lg = None
        for seq in stmt_seqs:
            if len(seq) == 0:
                print 'internal error: a statement sequence must not be empty'
                sys.exit(1)
            if lg == None:
                lg = len(seq)
            elif len(seq) != lg:
                print 'internal error: non-uniform length of statement sequences'
                sys.exit(1)

        # permute the statement sequences
        permuted_stmt_seqs = zip(*stmt_seqs)
        
        # perform jamming
        jammed_stmts = [self.__jamStmts(list(seq)) for seq in permuted_stmt_seqs]
        
        # merge statements
        merged_stmts = []
        for s in jammed_stmts:
            while isinstance(s, module.loop.ast.CompStmt) and len(s.stmts) == 1:
                s = s.stmts[0]
            if isinstance(s, module.loop.ast.CompStmt):
                merged_stmts.extend(s.stmts)
            else:
                merged_stmts.append(s)

        # generate the final statement
        if len(merged_stmts) == 1:
            return merged_stmts[0]
        else:
            return module.loop.ast.CompStmt(merged_stmts)

    #-----------------------------------------

    def __jamStmts(self, stmts):
        '''Attempt to jam all the statements in the given list'''

        # in case of both input is null
        if len(stmts) == 0:
            print 'internal error: input statements cannot be null'
            sys.exit(1)

        # in case of a single statement
        if len(stmts) == 1:
            return stmts[0]
    
        # jam all compound statements
        if isinstance(stmts[0], module.loop.ast.CompStmt):
            for s in stmts:
                if not isinstance(s, module.loop.ast.ComptStmt):
                    print 'internal error: not all statements are compound'
                    sys.exit(1)
            return self.__jamStmtSeqs([s.stmts for s in stmts])
            
        # jam all identical for-loops
        elif isinstance(stmts[0], module.loop.ast.ForStmt):
            for s in stmts:
                if not isinstance(s, module.loop.ast.ForStmt):
                    print 'internal error: not all statements are for-loops'
                    sys.exit(1)
            init_exp = stmts[0].init
            test_exp = stmts[0].test
            iter_exp = stmts[0].iter
            stmt_seqs = []
            all_identical = True
            for s in stmts:
                if (str(init_exp) == str(s.init) and
                    str(test_exp) == str(s.test) and
                    str(iter_exp) == str(s.iter)):
                    while isinstance(s.stmt, module.loop.ast.CompStmt) and len(s.stmt.stmts) == 1:
                        s.stmt = s.stmt.stmts[0]
                    if isinstance(s.stmt, module.loop.ast.CompStmt):
                        stmt_seqs.append(s.stmt.stmts)
                    else:
                        stmt_seqs.append([s.stmt])
                else:
                    all_identical = False
                    break
            if all_identical:
                jammed_loop_body = self.__jamStmtSeqs(stmt_seqs)
                return module.loop.ast.ForStmt(init_exp, test_exp, iter_exp, jammed_loop_body)
            
        # return non-jammable statements
        return module.loop.ast.CompStmt(stmts)
        
    #-----------------------------------------

    def transform(self):
        '''To unroll-and-jam the enclosed for-loop'''

        # get rid of compound statement that contains only a single statement
        while isinstance(self.stmt, module.loop.ast.CompStmt) and len(self.stmt.stmts) == 1:
            self.stmt = self.stmt.stmts[0]
        
        # extract for-loop structure
        for_loop_info = self.flib.extractForLoopInfo(self.stmt)
        index_id, lbound_exp, ubound_exp, stride_exp, loop_body = for_loop_info
        
        # when ufactor = 1, no transformation will be applied
        if self.ufactor == 1:
            return self.flib.createForLoop(index_id, lbound_exp, ubound_exp,
                                           stride_exp, loop_body)
        
        # start generating the main unrolled loop
        # compute lower bound --> new_LB = LB
        new_lbound_exp = lbound_exp.replicate()
    
        # compute upper bound --> new_UB = UB-ST*(UF-1)
        it = module.loop.ast.BinOpExp(stride_exp.replicate(),
                                      module.loop.ast.NumLitExp(self.ufactor - 1,
                                                                module.loop.ast.NumLitExp.INT),
                                      module.loop.ast.BinOpExp.MUL)
        new_ubound_exp = module.loop.ast.BinOpExp(ubound_exp.replicate(),
                                                  it,
                                                  module.loop.ast.BinOpExp.SUB)
        new_ubound_exp = self.cfolder.fold(new_ubound_exp)
    
        # compute stride --> new_ST = UF*ST
        it = module.loop.ast.NumLitExp(self.ufactor, module.loop.ast.NumLitExp.INT)
        new_stride_exp = module.loop.ast.BinOpExp(it,
                                                  stride_exp.replicate(),
                                                  module.loop.ast.BinOpExp.MUL)
        new_stride_exp = self.cfolder.fold(new_stride_exp)
    
        # compute unrolled statements
        unrolled_stmt_seqs = []
        for i in range(0, self.ufactor):
            s = loop_body.replicate()
            it = module.loop.ast.NumLitExp(i, module.loop.ast.NumLitExp.INT)
            increment_exp = module.loop.ast.BinOpExp(it,
                                                     stride_exp.replicate(),
                                                     module.loop.ast.BinOpExp.MUL)
            increment_exp = self.cfolder.fold(increment_exp)
            ns = self.__addIdentWithExp(s, index_id.name, increment_exp)
            ns = self.cfolder.fold(ns)
            if isinstance(ns, module.loop.ast.CompStmt):
                unrolled_stmt_seqs.append(ns.stmts)
            else:
                unrolled_stmt_seqs.append([ns])

        # compute the unrolled loop body by jamming/fusing the unrolled statements
        if self.do_jamming:
            unrolled_loop_body = self.__jamStmtSeqs(unrolled_stmt_seqs)
        else:
            unrolled_stmts = reduce(lambda x,y: x+y, unrolled_stmt_seqs, [])
            unrolled_loop_body = module.loop.ast.CompStmt(unrolled_stmts)
            
        # generate the main unrolled loop
        main_loop = self.flib.createForLoop(index_id, new_lbound_exp, new_ubound_exp,
                                            new_stride_exp, unrolled_loop_body)
        
        # generate the cleanup-loop lower-bound expression
        t = module.loop.ast.BinOpExp(module.loop.ast.ParenthExp(ubound_exp.replicate()),
                                     module.loop.ast.NumLitExp(self.ufactor,
                                                               module.loop.ast.NumLitExp.INT),
                                     module.loop.ast.BinOpExp.MOD)
        cleanup_lbound_exp = module.loop.ast.BinOpExp(
            module.loop.ast.ParenthExp(ubound_exp.replicate()),
            module.loop.ast.ParenthExp(t),
            module.loop.ast.BinOpExp.SUB)
        cleanup_lbound_exp = self.cfolder.fold(cleanup_lbound_exp)
        if not self.init_cleanup_loop:
            cleanup_lbound_exp = None
        
        # generate the clean-up loop
        cleanup_loop = self.flib.createForLoop(index_id, cleanup_lbound_exp, ubound_exp,
                                               stride_exp, loop_body)
        
        # generate the transformed statement
        transformed_stmt = module.loop.ast.CompStmt([main_loop, cleanup_loop])

        # return the transformed statement
        return transformed_stmt


