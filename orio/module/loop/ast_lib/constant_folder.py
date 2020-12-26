#
# A class to perform constant folding optimization
#

import sys
import orio.module.loop.ast
from orio.main.util.globals import *
from operator import itemgetter

#-----------------------------------------

class ConstFolder:
    '''A constant-folding optimizer'''

    def __init__(self):
        '''To instantiate a constant-folding optimizer'''
        pass

    #------------------------------------------

    def fold(self, tnode):
        '''To perform a constant folding on the given AST'''
        
        if isinstance(tnode, orio.module.loop.ast.Stmt):
            return self.__constantFoldStmt(tnode)

        elif isinstance(tnode, orio.module.loop.ast.Exp):
            return self.__constantFoldExp(tnode)

        elif isinstance(tnode, orio.module.loop.ast.NewAST):
            return tnode
        
        elif isinstance(tnode, orio.module.loop.ast.Comment):
            return tnode
        
        else:
            err('orio.module.loop.ast_lib.constant_folder internal error: unexpected AST type: "%s"' % tnode.__class__.__name__)

    #-----------------------------------------

    def __constantFoldStmt(self, stmt):
        '''Perform constant folding on the given statement'''

        if isinstance(stmt, orio.module.loop.ast.ExpStmt):
            stmt.exp = self.__constantFoldExp(stmt.exp)
            return stmt

        if isinstance(stmt, orio.module.loop.ast.GotoStmt):
            return stmt
        
        elif isinstance(stmt, orio.module.loop.ast.CompStmt):
            stmt.stmts = [self.__constantFoldStmt(s) for s in stmt.stmts]
            return stmt

        elif isinstance(stmt, orio.module.loop.ast.IfStmt):
            stmt.test = self.__constantFoldExp(stmt.test)
            stmt.true_stmt = self.__constantFoldStmt(stmt.true_stmt)
            if stmt.false_stmt:
                stmt.false_stmt = self.__constantFoldStmt(stmt.false_stmt)
            return stmt

        elif isinstance(stmt, orio.module.loop.ast.ForStmt):
            if stmt.init:
                stmt.init = self.__constantFoldExp(stmt.init)
            if stmt.test:
                stmt.test = self.__constantFoldExp(stmt.test)
            if stmt.iter:
                stmt.iter = self.__constantFoldExp(stmt.iter)
            stmt.stmt = self.__constantFoldStmt(stmt.stmt)
            return stmt

        elif isinstance(stmt, orio.module.loop.ast.TransformStmt):
            err('orio.module.loop.ast_lib.constant_folder:%s: a constant folding is never applied to a transformation statement' %
                   stmt.line_no)

        elif isinstance(stmt, orio.module.loop.ast.Comment):
            return stmt
        
        elif isinstance(stmt, orio.module.loop.ast.NewAST):
            return stmt
         
        elif isinstance(stmt, orio.module.loop.ast.Comment):
            return stmt
           
        else:
            err('orio.module.loop.ast_lib.constant_folder internal error: unexpected AST type: "%s"' % stmt.__class__.__name__)
        
    #-----------------------------------------

    def __flattenAdditiveTree(self, exp, sign):
        '''Flatten an additive expression tree'''
        
        if (isinstance(exp, orio.module.loop.ast.BinOpExp) and
            exp.op_type in (orio.module.loop.ast.BinOpExp.ADD, orio.module.loop.ast.BinOpExp.SUB)):
            if exp.op_type == orio.module.loop.ast.BinOpExp.ADD:
                if sign > 0:
                    lsign = 1
                    rsign = 1
                else:
                    lsign = -1
                    rsign = -1
            else:
                if sign > 0:
                    lsign = 1
                    rsign = -1
                else:
                    lsign = -1
                    rsign = 1
            lnodes = self.__flattenAdditiveTree(exp.lhs, lsign)
            rnodes = self.__flattenAdditiveTree(exp.rhs, rsign)
            return lnodes + rnodes

        elif isinstance(exp, orio.module.loop.ast.ParenthExp):
            nodes = self.__flattenAdditiveTree(exp.exp, sign)
            if len(nodes) == 1:
                cur_sign, cur_node = nodes[0]
                return [(cur_sign, orio.module.loop.ast.ParenthExp(cur_node))]
            else:
                return nodes
        
        else:
            return [(sign, exp)]
    
    #-----------------------------------------

    def __flattenMultiplicativeTree(self, exp, sign):
        '''Flatten a multiplicative expression tree'''

        if (isinstance(exp, orio.module.loop.ast.BinOpExp) and
            exp.op_type in (orio.module.loop.ast.BinOpExp.MUL, orio.module.loop.ast.BinOpExp.DIV)):
            if exp.op_type == orio.module.loop.ast.BinOpExp.MUL:
                if sign > 0:
                    lsign = 1
                    rsign = 1
                else:
                    lsign = -1
                    rsign = -1
            else:
                if sign > 0:
                    lsign = 1
                    rsign = -1
                else:
                    lsign = -1
                    rsign = 1
            lnodes = self.__flattenMultiplicativeTree(exp.lhs, lsign)
            rnodes = self.__flattenMultiplicativeTree(exp.rhs, rsign)
            return lnodes + rnodes
        
        elif isinstance(exp, orio.module.loop.ast.ParenthExp):
            nodes = self.__flattenMultiplicativeTree(exp.exp, sign)
            if len(nodes) == 1:
                cur_sign, cur_node = nodes[0]
                return [(cur_sign, orio.module.loop.ast.ParenthExp(cur_node))]
            else:
                return nodes
            
        else:
            return [(sign, exp)]
    
    #-----------------------------------------

    def __constantFoldExp(self, exp):
        '''Perform constant folding on the given expression'''

        if isinstance(exp, orio.module.loop.ast.NumLitExp):
            return exp
        
        elif isinstance(exp, orio.module.loop.ast.StringLitExp):
            return exp
        
        elif isinstance(exp, orio.module.loop.ast.IdentExp):
            return exp
        
        elif isinstance(exp, orio.module.loop.ast.ArrayRefExp):
            exp.exp = self.__constantFoldExp(exp.exp)
            exp.sub_exp = self.__constantFoldExp(exp.sub_exp)
            return exp

        elif isinstance(exp, orio.module.loop.ast.FunCallExp):
            exp.exp = self.__constantFoldExp(exp.exp)
            exp.args = [self.__constantFoldExp(a) for a in exp.args]
            return exp
        
        elif isinstance(exp, orio.module.loop.ast.UnaryExp):
            exp.exp = self.__constantFoldExp(exp.exp)
            if isinstance(exp.exp, orio.module.loop.ast.NumLitExp):
                if exp.op_type == orio.module.loop.ast.UnaryExp.PLUS:
                    return exp.exp
                elif exp.op_type == orio.module.loop.ast.UnaryExp.MINUS:
                    exp.exp.val = -exp.exp.val
                    return exp.exp
                else:
                    return exp
            else:
                return exp

        elif isinstance(exp, orio.module.loop.ast.BinOpExp):
            exp.lhs = self.__constantFoldExp(exp.lhs)
            exp.rhs = self.__constantFoldExp(exp.rhs)
            
            if exp.op_type in (orio.module.loop.ast.BinOpExp.ADD, orio.module.loop.ast.BinOpExp.SUB):

                # flatten the expression tree
                nodes = self.__flattenAdditiveTree(exp.lhs, 1)
                if exp.op_type == orio.module.loop.ast.BinOpExp.ADD:
                    nodes += self.__flattenAdditiveTree(exp.rhs, 1)
                else:
                    nodes += self.__flattenAdditiveTree(exp.rhs, -1)

                # add all numeric nodes
                num_val = 0.0
                non_num_nodes = []
                for sign, node in nodes:
                    if isinstance(node, orio.module.loop.ast.NumLitExp):
                        if sign > 0:
                            num_val += node.val
                        else:
                            num_val -= node.val
                    else:
                        non_num_nodes.append((sign, node))

                # put a positive non-numeric node at the first element
                #non_num_nodes.sort(lambda x,y: -cmp(x[0],y[0]))
                non_num_nodes = sorted(non_num_nodes, key=itemgetter(0), reverse=True)
                # build the numeric expression
                num_exp = None
                if num_val != 0:
                    if num_val % 1 == 0:
                        num_exp = orio.module.loop.ast.NumLitExp(int(num_val),
                                                            orio.module.loop.ast.NumLitExp.INT)
                    else:
                        num_exp = orio.module.loop.ast.NumLitExp(num_val, orio.module.loop.ast.NumLitExp.FLOAT)

                # build the non-numeric expression
                non_num_exp = None
                for sign, node in non_num_nodes:
                    if non_num_exp:
                        if sign > 0:
                            non_num_exp = orio.module.loop.ast.BinOpExp(non_num_exp, node,
                                                                   orio.module.loop.ast.BinOpExp.ADD)
                        else:
                            non_num_exp = orio.module.loop.ast.BinOpExp(non_num_exp, node,
                                                                   orio.module.loop.ast.BinOpExp.SUB)
                    else:
                        non_num_exp = node
                        if sign < 0:
                            if num_exp:
                                non_num_exp = orio.module.loop.ast.BinOpExp(num_exp, non_num_exp,
                                                                       orio.module.loop.ast.BinOpExp.SUB)
                                num_exp = None
                            else:
                                if isinstance(non_num_exp, orio.module.loop.ast.BinOpExp):
                                    non_num_exp = orio.module.loop.ast.ParenthExp(non_num_exp)
                                non_num_exp =orio.module.loop.ast.UnaryExp(non_num_exp,
                                                                      orio.module.loop.ast.UnaryExp.MINUS)

                # build the final expression
                if num_exp and non_num_exp:
                    if num_exp.val < 0:
                        num_exp.val = -num_exp.val
                        return orio.module.loop.ast.BinOpExp(non_num_exp, num_exp,
                                                        orio.module.loop.ast.BinOpExp.SUB)
                    else:
                        return orio.module.loop.ast.BinOpExp(non_num_exp, num_exp,
                                                        orio.module.loop.ast.BinOpExp.ADD)
                elif num_exp:
                    return num_exp
                elif non_num_exp:
                    return non_num_exp
                else:
                    return orio.module.loop.ast.NumLitExp(0, orio.module.loop.ast.NumLitExp.INT)
                
            elif exp.op_type in (orio.module.loop.ast.BinOpExp.MUL, orio.module.loop.ast.BinOpExp.DIV):

                # flatten the expression tree
                nodes = self.__flattenMultiplicativeTree(exp.lhs, 1)
                if exp.op_type == orio.module.loop.ast.BinOpExp.MUL:
                    nodes += self.__flattenMultiplicativeTree(exp.rhs, 1)
                else:
                    nodes += self.__flattenMultiplicativeTree(exp.rhs, -1)

                # add all numeric nodes
                num_val = 1.0
                dividend_nodes = []
                divisor_nodes = []
                for sign, node in nodes:
                    if isinstance(node, orio.module.loop.ast.NumLitExp):
                        if sign > 0:
                            num_val *= node.val
                        else:
                            num_val /= node.val
                    else:
                        if sign > 0:
                            dividend_nodes.append((sign, node))
                        else:
                            divisor_nodes.append((sign, node))

                # build the numeric expression
                num_exp = None
                if num_val != 1:
                    if num_val % 1 == 0:
                        num_exp = orio.module.loop.ast.NumLitExp(int(num_val),
                                                            orio.module.loop.ast.NumLitExp.INT)
                    else:
                        num_exp = orio.module.loop.ast.NumLitExp(num_val, orio.module.loop.ast.NumLitExp.FLOAT)
                        
                # build the dividend expression
                dividend_exp = num_exp
                for sign, node in dividend_nodes:
                    assert (sign > 0), 'dividends must have positive signs'
                    if dividend_exp:
                        dividend_exp = orio.module.loop.ast.BinOpExp(dividend_exp, node,
                                                                orio.module.loop.ast.BinOpExp.MUL)
                    else:
                        dividend_exp = node

                # build the divisor expression
                divisor_exp = None
                need_parenth = False 
                for sign, node in divisor_nodes:
                    assert (sign < 0), 'divisors must have negative signs'
                    if divisor_exp:
                        divisor_exp = orio.module.loop.ast.BinOpExp(divisor_exp, node,
                                                               orio.module.loop.ast.BinOpExp.MUL)
                        need_parenth = True
                    else:
                        divisor_exp = node
                if need_parenth:
                    divisor_exp = orio.module.loop.ast.ParenthExp(divisor_exp)

                # build the final expression
                if num_exp and num_exp.val == 0:
                    return orio.module.loop.ast.NumLitExp(0, orio.module.loop.ast.NumLitExp.INT) 
                elif dividend_exp and divisor_exp:
                    return orio.module.loop.ast.BinOpExp(dividend_exp, divisor_exp,
                                                    orio.module.loop.ast.BinOpExp.DIV)
                elif dividend_exp:
                    return dividend_exp
                elif divisor_exp:
                    return orio.module.loop.ast.BinOpExp(
                        orio.module.loop.ast.NumLitExp(1, orio.module.loop.ast.NumLitExp.INT),
                        divisor_exp,
                        orio.module.loop.ast.BinOpExp.DIV)
                else:
                    return orio.module.loop.ast.NumLitExp(1, orio.module.loop.ast.NumLitExp.INT)

            elif exp.op_type == orio.module.loop.ast.BinOpExp.MOD:
                if (isinstance(exp.lhs, orio.module.loop.ast.NumLitExp) and
                    isinstance(exp.rhs, orio.module.loop.ast.NumLitExp)):
                    new_val = exp.lhs.val % exp.rhs.val
                    if new_val % 1 == 0:
                        return orio.module.loop.ast.NumLitExp(int(new_val), orio.module.loop.ast.NumLitExp.INT)
                    else:
                        return orio.module.loop.ast.NumLitExp(new_val, orio.module.loop.ast.NumLitExp.FLOAT)
                else:
                    return exp
            else:
                return exp

        elif isinstance(exp, orio.module.loop.ast.ParenthExp):
            exp.exp = self.__constantFoldExp(exp.exp)
            return exp
        
        elif isinstance(exp, orio.module.loop.ast.NewAST):
            return exp
        
        elif isinstance(exp, orio.module.loop.ast.Comment):
            return exp

        else:
            err('orio.module.loop.ast_lib.constant_folder internal error: unexpected AST type: "%s"' % exp.__class__.__name__)
    
