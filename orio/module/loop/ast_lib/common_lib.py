#
# Contain a class that provides a set of common library functions for AST processing
#

import sys
import orio.module.loop.ast
import orio.main.util.globals as g

#-----------------------------------------------------------
 
class CommonLib:
    '''A common library set for AST processing'''
    
    def __init__(self):
        '''To instantiate a common library object'''
        pass

    #-------------------------------------------------------

    def replaceIdent(self, tnode, iname_from, iname_to):
        '''Replace the names of all matching identifiers with the given name'''

        if isinstance(tnode, orio.module.loop.ast.NumLitExp):
            return tnode
            
        elif isinstance(tnode, orio.module.loop.ast.StringLitExp):
            return tnode
            
        elif isinstance(tnode, orio.module.loop.ast.IdentExp):
            if tnode.name == iname_from:
                tnode.name = iname_to
            return tnode
            
        elif isinstance(tnode, orio.module.loop.ast.ArrayRefExp):
            tnode.exp = self.replaceIdent(tnode.exp, iname_from, iname_to)
            tnode.sub_exp = self.replaceIdent(tnode.sub_exp, iname_from, iname_to)
            return tnode
            
        elif isinstance(tnode, orio.module.loop.ast.FunCallExp):
            tnode.exp = self.replaceIdent(tnode.exp, iname_from, iname_to)
            tnode.args = [self.replaceIdent(a, iname_from, iname_to) for a in tnode.args]
            return tnode
            
        elif isinstance(tnode, orio.module.loop.ast.UnaryExp):
            tnode.exp = self.replaceIdent(tnode.exp, iname_from, iname_to)
            return tnode
            
        elif isinstance(tnode, orio.module.loop.ast.BinOpExp):
            tnode.lhs = self.replaceIdent(tnode.lhs, iname_from, iname_to)
            tnode.rhs = self.replaceIdent(tnode.rhs, iname_from, iname_to)
            return tnode
            
        elif isinstance(tnode, orio.module.loop.ast.ParenthExp):
            tnode.exp = self.replaceIdent(tnode.exp, iname_from, iname_to)
            return tnode
            
        elif isinstance(tnode, orio.module.loop.ast.ExpStmt):
            if tnode.exp:
                tnode.exp = self.replaceIdent(tnode.exp, iname_from, iname_to)
            return tnode
            
        elif isinstance(tnode, orio.module.loop.ast.CompStmt):
            tnode.stmts = [self.replaceIdent(s, iname_from, iname_to) for s in tnode.stmts]
            return tnode
            
        elif isinstance(tnode, orio.module.loop.ast.IfStmt):
            tnode.test = self.replaceIdent(tnode.test, iname_from, iname_to)
            tnode.true_stmt = self.replaceIdent(tnode.true_stmt, iname_from, iname_to)
            if tnode.false_stmt:
                tnode.false_stmt = self.replaceIdent(tnode.false_stmt, iname_from, iname_to)
            return tnode
            
        elif isinstance(tnode, orio.module.loop.ast.ForStmt):
            if tnode.init:
                tnode.init = self.replaceIdent(tnode.init, iname_from, iname_to)
            if tnode.test:
                tnode.test = self.replaceIdent(tnode.test, iname_from, iname_to)
            if tnode.iter:
                tnode.iter = self.replaceIdent(tnode.iter, iname_from, iname_to)
            tnode.stmt = self.replaceIdent(tnode.stmt, iname_from, iname_to)
            return tnode

        elif isinstance(tnode, orio.module.loop.ast.TransformStmt):
            g.err('orio.module.loop.ast_lib.common_lib internal error:  unexpected AST type: "%s"' % tnode.__class__.__name__)
        
        elif isinstance(tnode, orio.module.loop.ast.NewAST):
            return tnode

        elif isinstance(tnode, orio.module.loop.ast.Comment):
            return tnode

        else:
            g.err('orio.module.loop.ast_lib.common_lib internal error:  unexpected AST type: "%s"' % tnode.__class__.__name__)
        
    #-------------------------------------------------------

    def containIdentName(self, exp, iname):
        '''
        Check if the given expression contains an identifier whose name matches to the given name
        '''

        if exp == None:
            return False
        
        if isinstance(exp, orio.module.loop.ast.NumLitExp):
            return False
        
        elif isinstance(exp, orio.module.loop.ast.StringLitExp):
            return False
        
        elif isinstance(exp, orio.module.loop.ast.IdentExp):
            return exp.name == iname
        
        elif isinstance(exp, orio.module.loop.ast.ArrayRefExp):
            return self.containIdentName(exp.exp, iname) or self.containIdentName(exp.sub_exp, iname)
        
        elif isinstance(exp, orio.module.loop.ast.FunCallExp):
            has_match = reduce(lambda x,y: x or y,
                               [self.containIdentName(a, iname) for a in exp.args],
                               False)
            return self.containIdentName(exp.exp, iname) or has_match
        
        elif isinstance(exp, orio.module.loop.ast.UnaryExp):
            return self.containIdentName(exp.exp, iname)
        
        elif isinstance(exp, orio.module.loop.ast.BinOpExp):
            return self.containIdentName(exp.lhs, iname) or self.containIdentName(exp.rhs, iname)
        
        elif isinstance(exp, orio.module.loop.ast.ParenthExp):
            return self.containIdentName(exp.exp, iname)
        
        elif isinstance(exp, orio.module.loop.ast.NewAST):
            return False
        
        elif isinstance(exp, orio.module.loop.ast.Comment):
            return False

        else:
            g.err('orio.module.loop.ast_lib.common_lib internal error:  unexpected AST type: "%s"' % exp.__class__.__name__)
            
    #-------------------------------------------------------

    def isComplexExp(self, exp):
        '''
        To determine if the given expression is complex. Simple expressions contain only a variable
        or a number or a string.
        '''
        
        if isinstance(exp, orio.module.loop.ast.NumLitExp):
            return False
        
        # a rare case
        elif isinstance(exp, orio.module.loop.ast.StringLitExp):
            return False
        
        elif isinstance(exp, orio.module.loop.ast.IdentExp):
            return False
        
        # a rare case
        elif isinstance(exp, orio.module.loop.ast.ArrayRefExp):
            return True
        
        elif isinstance(exp, orio.module.loop.ast.FunCallExp):
            return True
        
        elif isinstance(exp, orio.module.loop.ast.UnaryExp):
            return self.isComplexExp(exp.exp)
        
        elif isinstance(exp, orio.module.loop.ast.BinOpExp):
            return True
        
        elif isinstance(exp, orio.module.loop.ast.ParenthExp):
            return self.isComplexExp(exp.exp)
        
        # a rare case
        elif isinstance(exp, orio.module.loop.ast.NewAST):
            return True
        
        elif isinstance(exp, orio.module.loop.ast.Comment):
            return False

        else:
            g.err('orio.module.loop.ast_lib.common_lib internal error:  unexpected AST type: "%s"' % exp.__class__.__name__)
            
    #------------------------------------------------------------------------------------------------------------------

    def collectNode(self, f, n):
        ''' Collect within the given node a list using the given function: pre-order traversal. '''
        
        if isinstance(n, orio.module.loop.ast.NumLitExp):
            return f(n)
        
        elif isinstance(n, orio.module.loop.ast.StringLitExp):
            return f(n)
        
        elif isinstance(n, orio.module.loop.ast.IdentExp):
            return f(n)
        
        elif isinstance(n, orio.module.loop.ast.ArrayRefExp):
            return f(n) + self.collectNode(f, n.exp) + self.collectNode(f, n.sub_exp)

        elif isinstance(n, orio.module.loop.ast.FunCallExp):
            return reduce(lambda x,y: x + y,
                          [self.collectNode(f, a) for a in n.args],
                          f(n))
        
        elif isinstance(n, orio.module.loop.ast.CastExpr):
            return f(n) + self.collectNode(f, n.expr)
        
        elif isinstance(n, orio.module.loop.ast.UnaryExp):
            return f(n) + self.collectNode(f, n.exp)
        
        elif isinstance(n, orio.module.loop.ast.BinOpExp):
            return f(n) + self.collectNode(f, n.lhs) + self.collectNode(f, n.rhs)
        
        elif isinstance(n, orio.module.loop.ast.ParenthExp):
            return f(n) + self.collectNode(f, n.exp)
        
        elif isinstance(n, orio.module.loop.ast.Comment):
            return f(n) + self.collectNode(f, n.text)
        
        elif isinstance(n, orio.module.loop.ast.ExpStmt):
            return f(n) + self.collectNode(f, n.exp)
        
        elif isinstance(n, orio.module.loop.ast.GotoStmt):
            return f(n) + self.collectNode(f, n.target)
        elif isinstance(n, orio.module.loop.ast.VarDecl):
            return f(n)
        elif isinstance(n, orio.module.loop.ast.VarDeclInit):
            return f(n)
        elif isinstance(n, orio.module.loop.ast.CompStmt):
            return reduce(lambda x,y: x + y,
                          [self.collectNode(f, a) for a in n.stmts],
                          f(n))
        
        elif isinstance(n, orio.module.loop.ast.IfStmt):
            result = self.collectNode(f, n.test) + self.collectNode(f, n.true_stmt)
            if n.false_stmt:
                result += self.collectNode(f, n.false_stmt)
            return result
        
        elif isinstance(n, orio.module.loop.ast.ForStmt):
            result = []
            if n.init:
                result += self.collectNode(f, n.init)
            if n.test:
                result += self.collectNode(f, n.test)
            if n.iter:
                result += self.collectNode(f, n.iter)
            result += self.collectNode(f, n.stmt)
            return result
        
        elif isinstance(n, orio.module.loop.ast.AssignStmt):
            return f(n) + self.collectNode(f, n.var) + self.collectNode(f, n.exp)
        
        elif isinstance(n, orio.module.loop.ast.TransformStmt):
            return f(n) + self.collectNode(f, n.name) + self.collectNode(f, n.args) + self.collectNode(f, n.stmt)

        else:
            g.err('orio.module.loop.ast_lib.common_lib.collectNode: unexpected AST type: "%s"' % n.__class__.__name__)
            
    #-------------------------------------------------------

    def rewriteNode(self, r, n):
        ''' Rewrite the given node with the given rewrite function: post-order traversal, in-place update. '''
        
        if isinstance(n, orio.module.loop.ast.NumLitExp):
            return r(n)
        
        elif isinstance(n, orio.module.loop.ast.StringLitExp):
            return r(n)
        
        elif isinstance(n, orio.module.loop.ast.IdentExp):
            return r(n)
        
        elif isinstance(n, orio.module.loop.ast.VarDecl):
            return r(n)
        
        elif isinstance(n, orio.module.loop.ast.ArrayRefExp):
            n.exp = self.rewriteNode(r, n.exp)
            n.sub_exp = self.rewriteNode(r, n.sub_exp)
            return r(n)

        elif isinstance(n, orio.module.loop.ast.FunCallExp):
            n.exp = self.rewriteNode(r, n.exp)
            n.args = map(lambda x: self.rewriteNode(r, x), n.args)
            return r(n)
        
        elif isinstance(n, orio.module.loop.ast.UnaryExp):
            n.exp = self.rewriteNode(r, n.exp)
            return r(n)
        
        elif isinstance(n, orio.module.loop.ast.BinOpExp):
            n.lhs = self.rewriteNode(r, n.lhs)
            n.rhs = self.rewriteNode(r, n.rhs)
            return r(n)
        
        elif isinstance(n, orio.module.loop.ast.ParenthExp):
            n.exp = self.rewriteNode(r, n.exp)
            return r(n)
        
        elif isinstance(n, orio.module.loop.ast.Comment):
            n.text = self.rewriteNode(r, n.text)
            return r(n)
        
        elif isinstance(n, orio.module.loop.ast.ExpStmt):
            n.exp = self.rewriteNode(r, n.exp)
            return r(n)
        
        elif isinstance(n, orio.module.loop.ast.GotoStmt):
            n.target = self.rewriteNode(r, n.target)
            return r(n)
        
        elif isinstance(n, orio.module.loop.ast.CompStmt):
            n.stmts = map(lambda x: self.rewriteNode(r, x), n.stmts)
            return r(n)
        
        elif isinstance(n, orio.module.loop.ast.IfStmt):
            n.test = self.rewriteNode(r, n.test)
            n.true_stmt = self.rewriteNode(r, n.true_stmt)
            if n.false_stmt:
                n.false_stmt = self.rewriteNode(r, n.false_stmt)
            return r(n)
        
        elif isinstance(n, orio.module.loop.ast.ForStmt):
            if n.init:
                n.init = self.rewriteNode(r, n.init)
            if n.test:
                n.test = self.rewriteNode(r, n.test)
            if n.iter:
                n.iter = self.rewriteNode(r, n.iter)
            n.stmt = self.rewriteNode(r, n.stmt)
            return r(n)
        
        elif isinstance(n, orio.module.loop.ast.AssignStmt):
            n.var = self.rewriteNode(r, n.var)
            n.exp = self.rewriteNode(r, n.exp)
            return r(n)
        
        elif isinstance(n, orio.module.loop.ast.TransformStmt):
            n.name = self.rewriteNode(r, n.name)
            n.args = self.rewriteNode(r, n.args)
            n.stmt = self.rewriteNode(r, n.stmt)
            return r(n)
        
        else:
            g.err('orio.module.loop.ast_lib.common_lib.rewriteNode: unexpected AST type: "%s"' % n.__class__.__name__)

#-----------------------------------------------------------------------------------------------------------------------


