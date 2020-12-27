#
# The implementation of the code transformation
#

import sys
from orio.main.util.globals import *
from . import ast, ast_util

#-------------------------------------------------

class Transformation:
    '''The code transformation that performs several code optimizations'''

    def __init__(self, unroll, vectorize, scalar_replacement, constant_folding):
        '''To instantiate a code transformation'''
        
        self.ast_util = ast_util.ASTUtil()

        # booleans to switch on/off optimizations
        self.unroll = unroll
        self.vectorize = vectorize
        self.scalar_replacement = scalar_replacement
        self.constant_folding = constant_folding   # constant-folding optimization on array index 
                                                   # expressions (strict assumption: expressions 
                                                   # must be affine function)
        
    #----------------------------------------------

    def __normalizeExp(self, exp):
        '''
        To convert <list> and <tuple> back to addition and multiplication (respectively), after
        performing constant folding
        '''

        if isinstance(exp, ast.NumLitExp):
            if exp.val < 0:
                exp.val *= -1
                return ast.UnaryExp(exp, ast.UnaryExp.MINUS)
            else:
                return exp

        elif isinstance(exp, ast.StringLitExp):
            return exp

        elif isinstance(exp, ast.IdentExp):
            return exp

        elif isinstance(exp, ast.ArrayRefExp):
            exp.exp = self.__normalizeExp(exp.exp)
            exp.sub_exp = self.__normalizeExp(exp.sub_exp)
            return exp

        elif isinstance(exp, ast.FunCallExp):
            exp.exp = self.__normalizeExp(exp.exp)
            exp.args = [self.__normalizeExp(a) for a in exp.args]
            return exp

        elif isinstance(exp, ast.UnaryExp):
            exp.exp = self.__normalizeExp(exp.exp)
            return exp

        elif isinstance(exp, ast.BinOpExp):
            exp.lhs = self.__normalizeExp(exp.lhs)
            exp.rhs = self.__normalizeExp(exp.rhs)
            return exp

        elif isinstance(exp, ast.ParenthExp):
            exp.exp = self.__normalizeExp(exp.exp)
            return exp

        # addition
        elif isinstance(exp, list): 
            n_exp = []
            for e in exp:
                n_exp.append(self.__normalizeExp(e))
            exp = n_exp
            lhs = exp[0]
            for e in exp[1:]:
                if isinstance(e, ast.UnaryExp) and e.op_type == ast.UnaryExp.MINUS:                   
                    lhs = ast.BinOpExp(lhs, e.exp, ast.BinOpExp.SUB)
                else:
                    lhs = ast.BinOpExp(lhs, e, ast.BinOpExp.ADD)
            return lhs

        # multiplication
        elif isinstance(exp, tuple):
            exp = list(exp)
            n_exp = []
            for e in exp:
                n_exp.append(self.__normalizeExp(e))
            exp = n_exp
            sign = 1
            n_exp = []
            for e in exp:
                if isinstance(e, ast.UnaryExp) and e.op_type == ast.UnaryExp.MINUS:
                    n_exp.append(e.exp)
                    sign *= -1
                else:
                    n_exp.append(e)
            exp = n_exp
            lhs = ast.BinOpExp(exp[0], exp[1], ast.BinOpExp.MUL)
            for e in exp[2:]:
                lhs = ast.BinOpExp(lhs, e, ast.BinOpExp.MUL)
            if sign == -1:
                return ast.UnaryExp(lhs, ast.UnaryExp.MINUS)
            return lhs

        else:
            err('orio.module.ortildriver.transformation internal error: unknown type of expression: %s' % 
                   exp.__class__.__name__)

    #----------------------------------------------
        
    def __foldConstantExp(self, exp, up_sign):
        '''
        To perform a simple (not full-fledged) constant folding optimization on the given expression
        A list is used to represent a k-ary addition.
        A tuple is used to represent a k-ary multiplication.
        '''

        if isinstance(exp, ast.NumLitExp):
            if up_sign == -1:
                exp.val *= -1
            return exp
            
        elif isinstance(exp, ast.StringLitExp):
            if up_sign == -1:
                return ast.UnaryExp(exp, ast.UnaryExp.MINUS)
            return exp

        elif isinstance(exp, ast.IdentExp):
            if up_sign == -1:
                return ast.UnaryExp(exp, ast.UnaryExp.MINUS)
            return exp

        elif isinstance(exp, ast.ArrayRefExp):
            exp.exp = self.__foldConstantExp(exp.exp, 1)
            exp.sub_exp = self.__foldConstantExp(exp.sub_exp, 1)
            if up_sign == -1:
                return ast.UnaryExp(exp, ast.UnaryExp.MINUS)
            return exp
            
        elif isinstance(exp, ast.FunCallExp):
            exp.exp = self.__foldConstantExp(exp.exp, 1)
            exp.args = [self.__foldConstantExp(a, 1) for a in exp.args]
            if up_sign == -1:
                return ast.UnaryExp(exp, ast.UnaryExp.MINUS)
            return exp

        elif isinstance(exp, ast.UnaryExp):
            if exp.op_type == ast.UnaryExp.MINUS:
                up_sign *= -1
            return  self.__foldConstantExp(exp.exp, up_sign)

        elif isinstance(exp, ast.BinOpExp):

            if exp.op_type in (ast.BinOpExp.ADD, ast.BinOpExp.SUB):
                if exp.op_type == ast.BinOpExp.ADD:
                    if up_sign == -1:
                        lhs_sign = -1
                        rhs_sign = -1
                    else:
                        lhs_sign = 1
                        rhs_sign = 1
                else:
                    if up_sign == -1:
                        lhs_sign = -1
                        rhs_sign = 1
                    else:
                        lhs_sign = 1
                        rhs_sign = -1
                lhs = self.__foldConstantExp(exp.lhs, lhs_sign)
                rhs = self.__foldConstantExp(exp.rhs, rhs_sign)
                ops = []
                for e in [lhs, rhs]:
                    if isinstance(e, list):
                        ops.extend(e)
                    else:
                        ops.append(e)
                add_num = 0
                n_ops = []
                for o in ops:
                    if isinstance(o, ast.NumLitExp):
                        add_num += o.val
                    else:
                        n_ops.append(o)
                ops = n_ops
                if add_num != 0:
                    ops.append(ast.NumLitExp(add_num, ast.NumLitExp.INT))
                if len(ops) == 0:
                    return ast.NumLitExp(0, ast.NumLitExp.INT)
                elif len(ops) == 1:
                    return ops[0]
                else:
                    return ops
                
            elif exp.op_type == ast.BinOpExp.MUL:
                lhs = self.__foldConstantExp(exp.lhs, up_sign)
                rhs = self.__foldConstantExp(exp.rhs, 1)
                ops1 = []
                ops2 = []
                if isinstance(lhs, list):
                    ops1.extend(lhs)
                else:
                    ops1.append(lhs)
                if isinstance(rhs, list):
                    ops2.extend(rhs)
                else:
                    ops2.append(rhs)
                ops = []
                for o1 in ops1:
                    if isinstance(o1, tuple):
                        o1 = list(o1)
                    else:
                        o1 = [o1]
                    for o2 in ops2:
                        if isinstance(o2, tuple):
                            o2 = list(o2)
                        else:
                            o2 = [o2]
                        o1 = [o.replicate() for o in o1]
                        o2 = [o.replicate() for o in o2]
                        o = []
                        mul_num = 1
                        for i in (o1+o2):
                            if isinstance(i, ast.NumLitExp):
                                mul_num *= i.val
                            else:
                                o.append(i)
                        if mul_num != 0:
                            if mul_num != 1:
                                o.insert(0, ast.NumLitExp(mul_num, ast.NumLitExp.INT))
                            if len(o) == 0:
                                ops.append(ast.NumLitExp(1, ast.NumLitExp.INT))
                            elif len(o) == 1:
                                ops.append(o[0])
                            else:
                                ops.append(tuple(o))
                add_num = 0
                n_ops = []
                for o in ops:
                    if isinstance(o, ast.NumLitExp):
                        add_num += o.val
                    else:
                        n_ops.append(o)
                ops = n_ops
                if add_num != 0:
                    ops.append(ast.NumLitExp(add_num, ast.NumLitExp.INT))
                if len(ops) == 0:
                    return ast.NumLitExp(0, ast.NumLitExp.INT)
                elif len(ops) == 1:
                    return ops[0]
                else:
                    return ops

            else:
                err('orio.module.ortildriver.transformation: constant folding cannot handle binary operations other ' +
                       'than +,-,*')

        elif isinstance(exp, ast.ParenthExp):
            return self.__foldConstantExp(exp.exp, up_sign)
        
        else:
            err('orio.module.ortildriver.transformation internal error: unknown type of expression: %s' % 
                   exp.__class__.__name__)

    #----------------------------------------------

    def __foldConstant(self, exp):
        '''To perform a simple (not full-fledged) constant folding optimization'''

        exp = self.__foldConstantExp(exp, 1)
        exp = self.__normalizeExp(exp)
        return exp

    #----------------------------------------------

    def __addIdentWithConstant(self, tnode, iname, constant):
        '''Add with the given constant all identifiers that match to the specified name'''
        
        if isinstance(tnode, ast.NumLitExp):
            return tnode

        elif isinstance(tnode, ast.StringLitExp):
            return tnode

        elif isinstance(tnode, ast.IdentExp):
            if tnode.name == iname:
                a = ast.BinOpExp(tnode.replicate(), ast.NumLitExp(constant, ast.NumLitExp.INT), 
                                 ast.BinOpExp.ADD)
                return ast.ParenthExp(a)
            else:
                return tnode

        elif isinstance(tnode, ast.ArrayRefExp):
            tnode.exp = self.__addIdentWithConstant(tnode.exp, iname, constant)
            tnode.sub_exp = self.__addIdentWithConstant(tnode.sub_exp, iname, constant)
            if self.constant_folding:
                tnode.exp = self.__foldConstant(tnode.exp)
                tnode.sub_exp = self.__foldConstant(tnode.sub_exp)
            return tnode

        elif isinstance(tnode, ast.FunCallExp):
            tnode.exp = self.__addIdentWithConstant(tnode.exp, iname, constant)
            tnode.args = [self.__addIdentWithConstant(a, iname, constant) for a in tnode.args]
            return tnode

        elif isinstance(tnode, ast.UnaryExp):
            tnode.exp = self.__addIdentWithConstant(tnode.exp, iname, constant)
            return tnode

        elif isinstance(tnode, ast.BinOpExp):
            tnode.lhs = self.__addIdentWithConstant(tnode.lhs, iname, constant)
            tnode.rhs = self.__addIdentWithConstant(tnode.rhs, iname, constant)
            return tnode

        elif isinstance(tnode, ast.ParenthExp):
            tnode.exp = self.__addIdentWithConstant(tnode.exp, iname, constant)
            return tnode

        elif isinstance(tnode, ast.ExpStmt):
            if tnode.exp:
                tnode.exp = self.__addIdentWithConstant(tnode.exp, iname, constant)
            return tnode

        elif isinstance(tnode, ast.CompStmt):
            tnode.stmts = [self.__addIdentWithConstant(s, iname, constant) for s in tnode.stmts]
            return tnode

        elif isinstance(tnode, ast.IfStmt):
            tnode.test = self.__addIdentWithConstant(tnode.test, iname, constant)
            tnode.true_stmt = self.__addIdentWithConstant(tnode.true_stmt, iname, constant)
            if tnode.false_stmt:
                tnode.false_stmt = self.__addIdentWithConstant(tnode.false_stmt, iname, constant)
            return tnode

        elif isinstance(tnode, ast.ForStmt):
            if tnode.init:
                tnode.init = self.__addIdentWithConstant(tnode.init, iname, constant)
            if tnode.test:
                tnode.test = self.__addIdentWithConstant(tnode.test, iname, constant)
            if tnode.iter:
                tnode.iter = self.__addIdentWithConstant(tnode.iter, iname, constant)
            tnode.stmt = self.__addIdentWithConstant(tnode.stmt, iname, constant)
            return tnode

        else:
            err('orio.module.ortildriver.transformation internal error: unknown type of AST: %s' % tnode.__class__.__name__)

    #----------------------------------------------

    def __unroll(self, stmt, unroll_factor_table):
        '''
        To apply loop unrolling on the fully rectangularly tiled loop.
        Assumption: the given loop is perfectly nested, and all loops are fully rectangularly tiled.
        '''
        
        # expression statement
        if isinstance(stmt, ast.ExpStmt):
            return stmt

        # compound statement     
        elif isinstance(stmt, ast.CompStmt):
            ustmts = []
            for s in stmt.stmts:
                us = self.__unroll(s, unroll_factor_table)
                if isinstance(us, ast.CompStmt):
                    ustmts.extend(us.stmts)
                else:
                    ustmts.append(us)
            stmt.stmts = ustmts
            return stmt

        # if statement    
        elif isinstance(stmt, ast.IfStmt):
            return stmt

        # for loop statement     
        elif isinstance(stmt, ast.ForStmt):
            
            # extract this loop structure
            id,_,_,_,lbody = self.ast_util.getForLoopInfo(stmt)

            # recursively unroll its loop body
            ubody = self.__unroll(lbody, unroll_factor_table)
            
            # unroll the loop body
            ufactor = unroll_factor_table[id.name]
            ustmts = []
            for i in range(0, ufactor):
                s = ubody.replicate()
                s = self.__addIdentWithConstant(s, id.name, i)
                if isinstance(s, ast.CompStmt):
                    ustmts.extend(s.stmts)
                else:
                    ustmts.append(s)

            # return the unrolled body
            return ast.CompStmt(ustmts)

        # unknown statement      
        else:
            err('orio.module.ortildriver.transformation internal error: unknown type of statement: %s' % 
                   stmt.__class__.__name__)
 
    #----------------------------------------------

    def __replaceScalar(self, tnode):
        '''
        To replace identical array references for the purpose of exposing and maximizing 
        register reuse.
        '''
        
        # count the occurences of each array references
        count_table = {}
        aref_seq = []
        self.__countArrRefs(tnode, count_table, aref_seq)

        # replace only array references that occur more than once
        repl_arefs = [x for x in aref_seq if count_table[str(x)] > 1]

        # generate new variable names for the more-than-once array references
        new_var_names = [('aref%s' % i) for i in range(1, len(repl_arefs)+1)]

        # create the replacement table
        replace_table = {}
        for a,v in zip(repl_arefs, new_var_names): 
            replace_table[str(a)] = v

        # replace the array references that occur more than once with the given replacement identifier
        tnode = self.__replaceArrRefs(tnode, replace_table)

        # create identifier assignments and array element stores
        type_name = 'double'
        assgn_code = '\n'
        store_code = '\n'
        for a,v in zip(repl_arefs, new_var_names):
            assgn_code += '  %s %s = %s; \n' % (type_name, v, a)
            store_code += '  %s = %s; \n' % (a, v)
            

        # return all needed information about scalar replacement
        return (tnode, assgn_code, store_code)
            
    #----------------------------------------------

    def __replaceArrRefs(self, tnode, replace_table):
        '''To replace some array references with specified identifiers'''

        if isinstance(tnode, ast.NumLitExp):
            return tnode
            
        elif isinstance(tnode, ast.StringLitExp):
            return tnode

        elif isinstance(tnode, ast.IdentExp):
            return tnode

        elif isinstance(tnode, ast.ArrayRefExp):
            aref_str = str(tnode)
            if aref_str in replace_table:
                iname = replace_table[aref_str]
                return ast.IdentExp(iname)
            else:
                return tnode

        elif isinstance(tnode, ast.FunCallExp):
            tnode.exp = self.__replaceArrRefs(tnode.exp, replace_table)
            tnode.args = [self.__replaceArrRefs(a, replace_table) for a in tnode.args]
            return tnode

        elif isinstance(tnode, ast.UnaryExp):
            tnode.exp = self.__replaceArrRefs(tnode.exp, replace_table)
            return tnode

        elif isinstance(tnode, ast.BinOpExp):
            tnode.lhs = self.__replaceArrRefs(tnode.lhs, replace_table)
            tnode.rhs = self.__replaceArrRefs(tnode.rhs, replace_table)
            return tnode

        elif isinstance(tnode, ast.ParenthExp):
            tnode.exp = self.__replaceArrRefs(tnode.exp, replace_table)
            return tnode

        elif isinstance(tnode, ast.ExpStmt):
            if tnode.exp:
                tnode.exp = self.__replaceArrRefs(tnode.exp, replace_table)
            return tnode

        elif isinstance(tnode, ast.CompStmt):
            tnode.stmts = [self.__replaceArrRefs(s, replace_table) for s in tnode.stmts]
            return tnode

        elif isinstance(tnode, ast.IfStmt):
            tnode.test = self.__replaceArrRefs(tnode.test, replace_table)
            tnode.true_stmt = self.__replaceArrRefs(tnode.true_stmt, replace_table)
            if tnode.false_stmt:
                tnode.false_stmt = self.__replaceArrRefs(tnode.false_stmt, replace_table)
            return tnode

        elif isinstance(tnode, ast.ForStmt):
            if tnode.init:
                tnode.init = self.__replaceArrRefs(tnode.init, replace_table)
            if tnode.test:
                tnode.test = self.__replaceArrRefs(tnode.test, replace_table)
            if tnode.iter:
                tnode.iter = self.__replaceArrRefs(tnode.iter, replace_table)
            tnode.stmt = self.__replaceArrRefs(tnode.stmt, replace_table)
            return tnode

        else:
            err('orio.module.ortildriver.transformation internal error:OrTilDriver: unknown type of AST: %s' % tnode.__class__.__name__)
 
    #----------------------------------------------

    def __countArrRefs(self, tnode, count_table, aref_seq):
        '''To count the number of occurences of each array reference'''
        
        if isinstance(tnode, ast.NumLitExp):
            return

        elif isinstance(tnode, ast.StringLitExp):
            return

        elif isinstance(tnode, ast.IdentExp):
            return

        elif isinstance(tnode, ast.ArrayRefExp):
            aref_str = str(tnode)
            if aref_str in count_table:
                count_table[aref_str] += 1
            else:
                count_table[aref_str] = 1
                aref_seq.append(tnode.replicate())

        elif isinstance(tnode, ast.FunCallExp):
            self.__countArrRefs(tnode.exp, count_table, aref_seq)
            for a in tnode.args:
                self.__countArrRefs(a, count_table, aref_seq)

        elif isinstance(tnode, ast.UnaryExp):
            self.__countArrRefs(tnode.exp, count_table, aref_seq)

        elif isinstance(tnode, ast.BinOpExp):
            self.__countArrRefs(tnode.lhs, count_table, aref_seq)
            self.__countArrRefs(tnode.rhs, count_table, aref_seq)

        elif isinstance(tnode, ast.ParenthExp):
            self.__countArrRefs(tnode.exp, count_table, aref_seq)

        elif isinstance(tnode, ast.ExpStmt):
            if tnode.exp:
                self.__countArrRefs(tnode.exp, count_table, aref_seq)

        elif isinstance(tnode, ast.CompStmt):
            for s in tnode.stmts:
                self.__countArrRefs(s, count_table, aref_seq)

        elif isinstance(tnode, ast.IfStmt):
            self.__countArrRefs(tnode.test, count_table, aref_seq)
            self.__countArrRefs(tnode.true_stmt, count_table, aref_seq)
            if tnode.false_stmt:
                self.__countArrRefs(tnode.false_stmt, count_table, aref_seq)

        elif isinstance(tnode, ast.ForStmt):
            if tnode.init:
                self.__countArrRefs(tnode.init, count_table, aref_seq)
            if tnode.test:
                self.__countArrRefs(tnode.test, count_table, aref_seq)
            if tnode.iter:
                self.__countArrRefs(tnode.iter, count_table, aref_seq)
            self.__countArrRefs(tnode.stmt, count_table, aref_seq)

        else:
            err('orio.module.ortildriver.transformation internal error:OrTilDriver: unknown type of AST: %s' % tnode.__class__.__name__)
 
    #----------------------------------------------

    def transform(self, iter_names, iter_vals, stmt):
        '''To apply code optimizations on the given fully rectangularly tiled loop'''
        
        # create a table that maps each tile size variable to its corresponding unroll factors
        unroll_factor_table = dict(list(zip(iter_names, iter_vals)))

        # apply loop unrolling
        if self.unroll:
            stmt = self.__unroll(stmt, unroll_factor_table)

        # apply scalar replacement
        if self.scalar_replacement:
            (stmt, assgn_code, store_code) = self.__replaceScalar(stmt)
            
        # generate the transformed code
        transformed_code = str(stmt)

        # insert identifier assignments and array stores for scalar replacements
        if self.scalar_replacement:
            start_pos = transformed_code.index('{')
            end_pos = transformed_code.index('}')
            transformed_code = (transformed_code[:start_pos+1] + assgn_code + 
                                transformed_code[start_pos+1:end_pos] + store_code +
                                transformed_code[end_pos:])

        # apply vectorization
        if self.vectorize:
            transformed_code = '\n#pragma ivdep \n#pragma vector always \n' + transformed_code

        # insert the initializations for the used loop iterators
        if self.unroll:
            iter_init_code = '\n'
            for i in iter_names:
                iter_init_code += '  %s = %st1; \n' % (i, i)
            transformed_code = iter_init_code + transformed_code

        # return the transformed loop
        return transformed_code


