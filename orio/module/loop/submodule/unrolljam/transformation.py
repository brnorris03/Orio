#
# Contain the transformation procedure
#

import sys
from orio.main.util.globals import *
import orio.module.loop.ast
import orio.module.loop.ast_lib.constant_folder
import orio.module.loop.ast_lib.forloop_lib
import orio.main.util.globals as g
from functools import reduce

#-----------------------------------------

class Transformation:
    '''Code transformation'''

    def __init__(self, ufactor, do_jamming, stmt, parallelize):
        '''To instantiate a code transformation object'''

        self.ufactor = ufactor
        self.do_jamming = do_jamming
        self.stmt = stmt
        self.parallelize = parallelize
        self.language = Globals().language
        self.flib = orio.module.loop.ast_lib.forloop_lib.ForLoopLib()
        self.cfolder = orio.module.loop.ast_lib.constant_folder.ConstFolder()
        self.newVarsOp = set([])  #tells you which operation is operated on the new set of variables (either none, plus, or multiply)
        self.varsToAdd = set([])  #tells you which variable names considered for introduction. Need ufactor to deduce the full set of variables introduced.
        self.varsNoAdd = set([])
        self.localvars = set([])
    #----------------------------------------------------------

    def __addIdentWithExp(self, tnode, index_name, exp):
        '''Traverse the tree node and add any matching identifier with the provided expression'''
            
        if isinstance(exp, orio.module.loop.ast.NumLitExp) and exp.val == 0:
            return tnode
        
        if isinstance(tnode, orio.module.loop.ast.ExpStmt):
            if tnode.exp:
                tnode.exp = self.__addIdentWithExp(tnode.exp, index_name, exp)
            return tnode
 
        if isinstance(tnode, orio.module.loop.ast.GotoStmt):
            return tnode
            
        elif isinstance(tnode, orio.module.loop.ast.CompStmt):
            tnode.stmts = [self.__addIdentWithExp(s, index_name, exp) for s in tnode.stmts]
            return tnode

        elif isinstance(tnode, orio.module.loop.ast.IfStmt):
            tnode.test = self.__addIdentWithExp(tnode.test, index_name, exp)
            tnode.true_stmt = self.__addIdentWithExp(tnode.true_stmt, index_name, exp)
            if tnode.false_stmt:
                tnode.false_stmt = self.__addIdentWithExp(tnode.false_stmt, index_name, exp)
            return tnode

        elif isinstance(tnode, orio.module.loop.ast.ForStmt):
            if tnode.init:
                tnode.init = self.__addIdentWithExp(tnode.init, index_name, exp)
            if tnode.test:
                tnode.test = self.__addIdentWithExp(tnode.test, index_name, exp)
            if tnode.iter:
                tnode.iter = self.__addIdentWithExp(tnode.iter, index_name, exp)
            tnode.stmt = self.__addIdentWithExp(tnode.stmt, index_name, exp)
            return tnode

        elif isinstance(tnode, orio.module.loop.ast.TransformStmt):
            err('orio.module.loop.submodule.unrolljam.transformation internal error: unprocessed transform statement')

        elif isinstance(tnode, orio.module.loop.ast.NumLitExp):
            return tnode

        elif isinstance(tnode, orio.module.loop.ast.StringLitExp):
            return tnode

        elif isinstance(tnode, orio.module.loop.ast.IdentExp):
            if tnode.name != index_name:
                if self.ufactor == 1:
                    return tnode
                else:
                    if tnode.name in self.varsToAdd:
                        k = orio.module.loop.ast.IdentExp(tnode.name+str(exp))
                        return k
                    else:
                        return tnode
            else:
                add_exp = orio.module.loop.ast.BinOpExp(tnode,
                                                   exp.replicate(),
                                                   orio.module.loop.ast.BinOpExp.ADD)
                return orio.module.loop.ast.ParenthExp(add_exp)

        elif isinstance(tnode, orio.module.loop.ast.ArrayRefExp):
            if isinstance(tnode.exp, orio.module.loop.ast.ArrayRefExp):
                tnode.exp = self.__addIdentWithExp(tnode.exp, index_name, exp)
            tnode.sub_exp = self.__addIdentWithExp(tnode.sub_exp, index_name, exp)
            return tnode
        
        elif isinstance(tnode, orio.module.loop.ast.FunCallExp):
            tnode.exp = self.__addIdentWithExp(tnode.exp, index_name, exp)
            tnode.args = [self.__addIdentWithExp(a, index_name, exp) for a in tnode.args]
            return tnode

        elif isinstance(tnode, orio.module.loop.ast.UnaryExp):
            tnode.exp = self.__addIdentWithExp(tnode.exp, index_name, exp)
            return tnode
        
        elif isinstance(tnode, orio.module.loop.ast.BinOpExp):              
            tnode.lhs = self.__addIdentWithExp(tnode.lhs, index_name, exp)
            tnode.rhs = self.__addIdentWithExp(tnode.rhs, index_name, exp)
            return tnode

        elif isinstance(tnode, orio.module.loop.ast.ParenthExp):
            tnode.exp = self.__addIdentWithExp(tnode.exp, index_name, exp)
            return tnode        
        
        elif isinstance(tnode, orio.module.loop.ast.NewAST):
            return tnode
        
        elif isinstance(tnode, orio.module.loop.ast.Comment):
            return tnode  
              
        else:
            err('orio.module.loop.submodule.unrolljam.transformation.__addIdentWithExp internal error: unexpected AST type: "%s"' % tnode.__class__.__name__)
    
    #-----------------------------------------
    
    
    def __computeNewVarsIntro(self, tnode):
        
        binOpExprs = self.__analyzeForNewVars(tnode) #has the form set([ lfsName, (rhsName1, rhsName2, ...)]
        
        # there can only be one operation, and this operation has to be either addition or multiplication (they are both commutative and associative)
        if len(self.newVarsOp) != 1 or (next(iter(self.newVarsOp)) != orio.module.loop.ast.BinOpExp.ADD and next(iter(self.newVarsOp)) != orio.module.loop.ast.BinOpExp.MUL):
            #self.introNewVars = False
            return
        
        for exp1 in binOpExprs:
            if exp1[0] in self.varsNoAdd:
                continue
            if exp1[0] in exp1[1]:   #there's modification of variable
                for exp2 in binOpExprs:
                    if exp2 is exp1:
                        continue
                    if exp1[0] in exp2[1] and exp1[0] != exp2[0]:  #modification with alias
                        self.varsNoAdd |= set([exp1[0]])
                        self.varsToAdd -= set([exp1[0]])
                        break
                else:  #modification to itself but no alias
                    self.varsToAdd |= set([exp1[0]])   
            else:   #there's at least one non-modification of variable (just read)
                self.varsNoAdd |= set([exp1[0]])
                self.varsToAdd -= set([exp1[0]])
            
    
    
    
    
    #-----------------------------------------
    
    
    def __analyzeForNewVars(self, tnode):
        '''Traverse the tree node to reach every top level binary operation expression to obtain parsed info for later computation of new variables to introduce'''
        binOpExprs = set([])
        
        if isinstance(tnode, orio.module.loop.ast.ExpStmt):
            binOpExprs |= self.__analyzeForNewVars(tnode.exp)
            
        elif isinstance(tnode, orio.module.loop.ast.CompStmt):
            for s in tnode.stmts:
                binOpExprs |= self.__analyzeForNewVars(s)

        elif isinstance(tnode, orio.module.loop.ast.IfStmt):
            binOpExprs |= self.__analyzeForNewVars(tnode.test)
            binOpExprs |= self.__analyzeForNewVars(tnode.true_stmt)
            if tnode.false_stmt:
                binOpExprs |= self.__analyzeForNewVars(tnode.false_stmt)

        elif isinstance(tnode, orio.module.loop.ast.ForStmt):
            if tnode.init:
                binOpExprs |= self.__analyzeForNewVars(tnode.init)
            if tnode.test:
                binOpExprs |= self.__analyzeForNewVars(tnode.test)
            if tnode.iter:
                binOpExprs |= self.__analyzeForNewVars(tnode.iter)
            binOpExprs |= self.__analyzeForNewVars(tnode.stmt)

        elif isinstance(tnode, orio.module.loop.ast.TransformStmt):
            err('orio.module.loop.submodule.unrolljam.transformation internal error: unprocessed transform statement')
        
        elif isinstance(tnode, orio.module.loop.ast.BinOpExp):
            lfsVarName = '@ArrayVarNotConsidered@'
            if tnode.op_type != orio.module.loop.ast.BinOpExp.EQ_ASGN:
                return binOpExprs
                #err('first operation of a top level binary op expression is not equal assignment???')
            if not isinstance(tnode.lhs, orio.module.loop.ast.IdentExp):
                if not isinstance(tnode.lhs, orio.module.loop.ast.ArrayRefExp):
                    err('left hand side of a top level binary op expression is not an identifer expression or array ref exp???')
            else:
                lfsVarName = tnode.lhs.name
            
            tup = tuple(self.__extractFromBinOp(tnode.rhs))
            binOpExprs |= set([(lfsVarName, tup)])
        elif isinstance(tnode, orio.module.loop.ast.UnaryExp):
            lfsVarName = '@ArrayVarNotConsidered@'
            
            if not isinstance(tnode.exp, orio.module.loop.ast.IdentExp):
                if not isinstance(tnode.exp, orio.module.loop.ast.ArrayRefExp):
                    err('the sub level exp of top level unary expression is not an identifer expression or array ref exp???')
            else:
                lfsVarName = tnode.exp.name
                
            tup = tuple(self.__extractFromBinOp(tnode.exp))
            binOpExprs |= set([(lfsVarName, tup)])
            
        elif isinstance(tnode, orio.module.loop.ast.ParenthExp):
            binOpExprs |= self.__analyzeForNewVars(tnode.exp)
              
        elif isinstance(tnode, orio.module.loop.ast.Container):
            binOpExprs |= self.__analyzeForNewVars(tnode.ast)
              
        elif isinstance(tnode, orio.module.loop.ast.VarDecl):
            return binOpExprs
        
        elif isinstance(tnode, orio.module.loop.ast.Comment):
            return binOpExprs

        elif isinstance(tnode, orio.module.loop.ast.Pragma):
            return binOpExprs

        else:
            err('orio.module.loop.submodule.unrolljam.transformation.__analyzeForNewVars internal error: unexpected AST type: "%s"' % tnode.__class__.__name__)
            
        return binOpExprs
    
    
    
    
    
    
    
    #-----------------------------------------
    
    
    
    
    
    def __extractFromBinOp(self, tnode):
        '''extract variable names used'''
        var_names = set([])
        
        if isinstance(tnode, orio.module.loop.ast.NumLitExp):
            return var_names
        if isinstance(tnode, orio.module.loop.ast.ParenthExp):
            var_names |= self.__extractFromBinOp(tnode.exp)
        elif isinstance(tnode, orio.module.loop.ast.ArrayRefExp):
            var_names |= self.__extractFromBinOp(tnode.exp)
            var_names |= self.__extractFromBinOp(tnode.sub_exp)
        elif isinstance(tnode, orio.module.loop.ast.IdentExp):
            var_names |= set([tnode.name])
        elif isinstance(tnode, orio.module.loop.ast.FunCallExp):
            return var_names
        elif isinstance(tnode, orio.module.loop.ast.BinOpExp):
            if isinstance(tnode.lhs, orio.module.loop.ast.ParenthExp):
                var_names |= self.__extractFromBinOp(tnode.lhs.exp)
            elif isinstance(tnode.lhs, orio.module.loop.ast.ArrayRefExp):
                var_names |= self.__extractFromBinOp(tnode.lhs.exp)
                var_names |= self.__extractFromBinOp(tnode.lhs.sub_exp)
            elif isinstance(tnode.lhs, orio.module.loop.ast.BinOpExp):
                var_names |= self.__extractFromBinOp(tnode.lhs)
            elif isinstance(tnode.lhs, orio.module.loop.ast.IdentExp):
                var_names |= set([tnode.lhs.name])
            elif isinstance(tnode.lhs, orio.module.loop.ast.FunCallExp):
                var_names = reduce(set.union, list(map(self.__extractFromBinOp, tnode.lhs.args)), var_names)
            elif isinstance(tnode.lhs, orio.module.loop.ast.NumLitExp):
                pass
            else:
                err('orio.module.loop.submodule.unrolljam.transformation.__extractFromBinOp.BinOpExp internal error: unexpected AST type: "%s", lhs: %s' % (tnode.lhs.__class__.__name__,tnode.lhs))
                
            if isinstance(tnode.rhs, orio.module.loop.ast.ParenthExp):
                var_names |= self.__extractFromBinOp(tnode.rhs.exp)
            elif isinstance(tnode.rhs, orio.module.loop.ast.ArrayRefExp):
                var_names |= self.__extractFromBinOp(tnode.rhs.exp)
                var_names |= self.__extractFromBinOp(tnode.rhs.sub_exp)
            elif isinstance(tnode.rhs, orio.module.loop.ast.BinOpExp):
                var_names |= self.__extractFromBinOp(tnode.rhs)
            elif isinstance(tnode.rhs, orio.module.loop.ast.IdentExp):
                var_names |= set([tnode.rhs.name])
            elif isinstance(tnode.rhs, orio.module.loop.ast.FunCallExp):
                var_names = reduce(set.union, list(map(self.__extractFromBinOp, tnode.rhs.args)), var_names)
            elif isinstance(tnode.rhs, orio.module.loop.ast.NumLitExp):
                pass
            else:
                err('orio.module.loop.submodule.unrolljam.transformation.__extractFromBinOp internal error: unexpected AST type: "%s", rhs: %s' % (tnode.rhs.__class__.__name__,tnode.rhs))
                
            self.newVarsOp |= set([tnode.op_type])
        else:
            err('orio.module.loop.submodule.unrolljam.transformation.__extractFromBinOp. internal error: unexpected AST type: "%s", node: %s' % (tnode.__class__.__name__,tnode))
    
        return var_names
    
    #-----------------------------------------

    def __jamStmts(self, stmtss):
        '''Jam/fuse statements whenever possible'''

        if len(stmtss) == 0:
            return orio.module.loop.ast.CompStmt([])
        if len(stmtss) == 1:
            return orio.module.loop.ast.CompStmt(stmtss[0])

        num = len(stmtss[0])
        for stmts in stmtss:
            assert(num == len(stmts)), 'internal error: unequal length of statement list'

        is_jam_valid = True
        contain_loop = False
        for i in range(num):
            s1 = None
            for stmts in stmtss:
                if s1 == None:
                    s1 = stmts[i]
                    if isinstance(s1, orio.module.loop.ast.ForStmt):
                        contain_loop = True
                elif isinstance(s1, orio.module.loop.ast.ForStmt):
                    s2 = stmts[i]
                    assert(isinstance(s2, orio.module.loop.ast.ForStmt)), 'internal error: not a loop statement'
                    if not (str(s1.init) == str(s2.init) and str(s1.test) == str(s2.test) and str(s1.iter) == str(s2.iter)):
                        is_jam_valid = False
        if is_jam_valid:
            if not contain_loop:
                is_jam_valid = False

        if not is_jam_valid:
            n_stmts = []
            for stmts in stmtss:
                n_stmts.extend(stmts)
            return orio.module.loop.ast.CompStmt(n_stmts)

        n_stmts = []
        for stmts in zip(*stmtss):
            if isinstance(stmts[0], orio.module.loop.ast.ForStmt):
                l_stmtss = []
                for s in stmts:
                    if isinstance(s.stmt, orio.module.loop.ast.CompStmt):
                        l_stmtss.append(s.stmt.stmts)
                    else:
                        l_stmtss.append([s.stmt])
                loop = stmts[0].replicate()
                loop.stmt = self.__jamStmts(l_stmtss)
                n_stmts.append(loop)
            else:
                n_stmts.extend(stmts)
        return orio.module.loop.ast.CompStmt(n_stmts)
        
    #-----------------------------------------

    def transform(self):
        '''To unroll-and-jam the enclosed for-loop'''

        # get rid of compound statement that contains only a single statement
        while isinstance(self.stmt.stmt, orio.module.loop.ast.CompStmt) and len(self.stmt.stmt.stmts) == 1:
            self.stmt.stmt = self.stmt.stmt.stmts[0]
        
        # extract for-loop structure
        for_loop_info = self.flib.extractForLoopInfo(self.stmt)
        index_id, lbound_exp, ubound_exp, stride_exp, loop_body = for_loop_info

#        if not index_id.name in self.localvars:
#            index_decl = ast.VarDecl('int', [index_id.name])
#            self.localvars.add(index_id.name)
#            debug('loop.unrolljam.transformation: generating var decl: %s %s'% (int, index_id.name),self)
        
        # when ufactor = 1, no transformation will be applied
        if self.ufactor == 1:
            orig_loop = self.flib.createForLoop(index_id, lbound_exp, ubound_exp,
                                                stride_exp, loop_body, meta="original")
            if self.parallelize:
                inames = self.flib.getLoopIndexNames(orig_loop)
                inames_str = ','.join(inames)
                if inames:
                    omp_pragma = orio.module.loop.ast.Pragma('omp parallel for private(%s)' % inames_str)
                else:
                    omp_pragma = orio.module.loop.ast.Pragma('omp parallel for')
                return orio.module.loop.ast.CompStmt([omp_pragma, orig_loop])
            else:
                return orio.module.loop.ast.CompStmt([orig_loop])
        
        # start generating the orio.main.unrolled loop
        # compute lower bound --> new_LB = LB
        new_lbound_exp = lbound_exp.replicate()
    
        # compute upper bound --> new_UB = UB-ST*(UF-1)
        it = orio.module.loop.ast.BinOpExp(stride_exp.replicate(),
                                      orio.module.loop.ast.NumLitExp(self.ufactor - 1,
                                                                orio.module.loop.ast.NumLitExp.INT),
                                      orio.module.loop.ast.BinOpExp.MUL)
        new_ubound_exp = orio.module.loop.ast.BinOpExp(ubound_exp.replicate(),
                                                  it,
                                                  orio.module.loop.ast.BinOpExp.SUB)
        new_ubound_exp = self.cfolder.fold(new_ubound_exp)
    
        # compute stride --> new_ST = UF*ST
        it = orio.module.loop.ast.NumLitExp(self.ufactor, orio.module.loop.ast.NumLitExp.INT)
        new_stride_exp = orio.module.loop.ast.BinOpExp(it,
                                                  stride_exp.replicate(),
                                                  orio.module.loop.ast.BinOpExp.MUL)
        new_stride_exp = self.cfolder.fold(new_stride_exp)
    
        
        s = loop_body.replicate()
        #obtain info about whether or not to introduce new variables
        self.__computeNewVarsIntro(s)
    
        # compute unrolled statements
        unrolled_stmt_seqs = []
        
        for i in range(0, self.ufactor):
            s = loop_body.replicate()
            it = orio.module.loop.ast.NumLitExp(i, orio.module.loop.ast.NumLitExp.INT)
            increment_exp = orio.module.loop.ast.BinOpExp(it,
                                                     stride_exp.replicate(),
                                                     orio.module.loop.ast.BinOpExp.MUL)
            increment_exp = self.cfolder.fold(increment_exp)
            ns = self.__addIdentWithExp(s, index_id.name, increment_exp)
            ns = self.cfolder.fold(ns)
            if isinstance(ns, orio.module.loop.ast.CompStmt):
                unrolled_stmt_seqs.append(ns.stmts)
            else:
                unrolled_stmt_seqs.append([ns])

        # compute the unrolled loop body by jamming/fusing the unrolled statements
        if self.do_jamming:
            unrolled_loop_body = self.__jamStmts(unrolled_stmt_seqs)
        else:
            unrolled_stmts = reduce(lambda x,y: x+y, unrolled_stmt_seqs, [])
            unrolled_loop_body = orio.module.loop.ast.CompStmt(unrolled_stmts)
            
        # generate the orio.main.unrolled loop
        
        lbound_name = 'orio_lbound'+str(g.Globals().getcounter())
        lbound_name_exp = orio.module.loop.ast.IdentExp(lbound_name)
        lbound_init = orio.module.loop.ast.VarDeclInit('int', lbound_name_exp, new_lbound_exp)

        loop = self.flib.createForLoop(index_id, new_lbound_exp, new_ubound_exp,
                                            new_stride_exp, unrolled_loop_body, meta="main")
        
        # generate the cleanup-loop lower-bound expression
        # if self.parallelize or self.language == 'fortran':
        t = orio.module.loop.ast.BinOpExp(orio.module.loop.ast.ParenthExp(ubound_exp.replicate()), 
                                          orio.module.loop.ast.ParenthExp(lbound_exp.replicate()),
                                          orio.module.loop.ast.BinOpExp.SUB)
        t = orio.module.loop.ast.BinOpExp(t, orio.module.loop.ast.NumLitExp(1, orio.module.loop.ast.NumLitExp.INT), orio.module.loop.ast.BinOpExp.ADD)
        t = orio.module.loop.ast.BinOpExp(orio.module.loop.ast.ParenthExp(t),
                                          orio.module.loop.ast.NumLitExp(self.ufactor, orio.module.loop.ast.NumLitExp.INT),
                                          orio.module.loop.ast.BinOpExp.MOD)
        t = orio.module.loop.ast.BinOpExp(orio.module.loop.ast.ParenthExp(ubound_exp.replicate()),
                                          orio.module.loop.ast.ParenthExp(t),
                                          orio.module.loop.ast.BinOpExp.SUB)
        cleanup_lbound_exp = orio.module.loop.ast.BinOpExp(orio.module.loop.ast.ParenthExp(t),
                                                           orio.module.loop.ast.NumLitExp(1, orio.module.loop.ast.NumLitExp.INT),
                                                           orio.module.loop.ast.BinOpExp.ADD)
        
                                          
        #t = orio.module.loop.ast.BinOpExp(orio.module.loop.ast.ParenthExp(ubound_exp.replicate()),
        #                            orio.module.loop.ast.NumLitExp(self.ufactor,
        #                                                         orio.module.loop.ast.NumLitExp.INT),
        #                         orio.module.loop.ast.BinOpExp.MOD)
        #cleanup_lbound_exp = orio.module.loop.ast.BinOpExp(
         #                            orio.module.loop.ast.ParenthExp(ubound_exp.replicate()),
          #                           orio.module.loop.ast.ParenthExp(t),
           #                          orio.module.loop.ast.BinOpExp.SUB)
        cleanup_lbound_exp = self.cfolder.fold(cleanup_lbound_exp)
        
    
        #else:
            #cleanup_lbound_exp = None
        # the above if else conditions are removed to make CUDA submodule works, which needs a lower bound.
        # Not sure why there is an if else condition in the first place.
        
        # generate the clean-up loop
        cleanup_lbound_name = 'orio_lbound'+str(g.Globals().getcounter())
        cleanup_lbound_name_exp = orio.module.loop.ast.IdentExp(cleanup_lbound_name)
        cleanup_lbound_init = orio.module.loop.ast.VarDeclInit('int', cleanup_lbound_name_exp, cleanup_lbound_exp)
        
        cleanup_loop = self.flib.createForLoop(index_id, cleanup_lbound_exp, ubound_exp,
                                               stride_exp, loop_body, meta="cleanup")
        
        
        
        # generate the transformed statement
        if self.parallelize:
            inames = self.flib.getLoopIndexNames(loop)
            inames_str = ','.join(inames)
            if inames:
                omp_pragma = orio.module.loop.ast.Pragma('omp parallel for private(%s)' % inames_str)
            else:
                omp_pragma = orio.module.loop.ast.Pragma('omp parallel for')     
            stmts = [omp_pragma, loop, cleanup_loop]
        else:
            stmts = [loop, cleanup_loop]
        transformed_stmt = orio.module.loop.ast.CompStmt(stmts)

        # return the transformed statement
        return transformed_stmt


