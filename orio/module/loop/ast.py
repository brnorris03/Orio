#
# The classes for the abstract syntax tree (AST)
#
#  AST 
#   |
#   +-- Exp 
#   |    |
#   |    +-- NumLitExp
#   |    +-- StringLitExp
#   |    +-- IdentExp
#   |    +-- ArrayRefExp 
#   |    +-- FunCallExp 
#   |    +-- UnaryExp 
#   |    +-- BinOpExp 
#   |    +-- ParenthExp
#   |
#   +-- Stmt 
#   |    |
#   |    +-- ExpStmt 
#   |    +-- CompStmt 
#   |    +-- IfStmt 
#   |    +-- ForStmt 
#   |    +-- AssignStmt
#   |    +-- TransformStmt 
#   |
#   +-- NewAST 
#        |
#        +-- VarDecl
#        +-- VarDeclInit
#        +-- FieldDecl 
#        +-- FunDecl
#        +-- Pragma 
#        +-- Container
#
# - The NewAST is an AST used only in the output code generation. Such separation is needed to
#   simplify the input language.
#
from orio.module.loop import codegen
from orio.main.util.globals import Globals
import copy

#-----------------------------------------------
# AST - Abstract Syntax Tree
#-----------------------------------------------

class AST:

    def __init__(self, line_no = '', parent = None, meta={}):
        '''Create an abstract syntax tree node'''
        self.line_no = line_no           # may be null (i.e. empty string)
        self.parent = parent
        self.meta = copy.deepcopy(meta)
        self.initMeta('uses')
        self.initMeta('defs')
        self.id = str(Globals().incrementCounter())
        self.temp = None
        
    def replicate(self):
        '''Replicate this abstract syntax tree node'''
        raise NotImplementedError('%s: abstract function "replicate" not implemented' %
                                  self.__class__.__name__)
        
    def accept(self, visitor, params={}):
        '''Visitor pattern accept function'''
        visitor.visit(self,params)

    def initMeta(self, key, val=0):
        self.meta[key] = val

    def updateMeta(self, key, val=1):
        if self.meta.get(key):
            self.meta[key] += val
        else:
            self.meta[key] = val 
            
    def getMeta(self, key):
        if self.meta.get(key): return self.meta[key]
        else: return 0
                     
    def __repr__(self):
        '''Return a string representation for this AST object'''
        return codegen.CodeGen().generate(self)

    def __str__(self):
        '''Return a string representation for this AST object'''
        return repr(self)
    
#-----------------------------------------------
# Expression
#-----------------------------------------------

class Exp(AST):

    def __init__(self, line_no = '', parent = None, meta={}):
        '''Create an expression'''
        AST.__init__(self, line_no, parent, meta)

#-----------------------------------------------
# Number Literal
#-----------------------------------------------

class NumLitExp(Exp):

    INT = 1
    FLOAT = 2
    
    def __init__(self, val, lit_type, line_no = '', parent = None, meta={}):
        '''Create a numeric literal'''
        Exp.__init__(self, line_no, parent)
        self.val = val
        self.lit_type = lit_type
        self.meta = meta

    def replicate(self):
        '''Replicate this abstract syntax tree node'''
        return NumLitExp(self.val, self.lit_type, self.line_no, meta=copy.deepcopy(self.meta))
        
#-----------------------------------------------
# String Literal
#-----------------------------------------------

class StringLitExp(Exp):

    def __init__(self, val, line_no = '', meta={}):
        '''Create a string literal'''
        Exp.__init__(self, line_no, meta)
        self.val = val

    def replicate(self):
        '''Replicate this abstract syntax tree node'''
        return StringLitExp(self.val, self.line_no, meta=copy.deepcopy(self.meta))
        
#-----------------------------------------------
# Identifier
#-----------------------------------------------

class IdentExp(Exp):

    def __init__(self, name, line_no = '', meta={}):
        '''Create an identifier'''
        Exp.__init__(self, line_no, meta)
        self.name = name
        
    def replicate(self):
        '''Replicate this abstract syntax tree node'''
        return IdentExp(self.name, self.line_no, meta=copy.deepcopy(self.meta))

#-----------------------------------------------
# Array Reference
#-----------------------------------------------

class ArrayRefExp(Exp):

    def __init__(self, exp, sub_exp, line_no = '', meta={}):
        '''Create an array reference'''
        Exp.__init__(self, line_no, meta)
        self.exp = exp
        self.sub_exp = sub_exp

    def replicate(self):
        '''Replicate this abstract syntax tree node'''
        return ArrayRefExp(self.exp.replicate(), self.sub_exp.replicate(), 
                           self.line_no, meta=copy.deepcopy(self.meta))
        
#-----------------------------------------------
# Function Call
#-----------------------------------------------

class FunCallExp(Exp):

    def __init__(self, exp, args, line_no = '', meta={}):
        '''Create a function call'''
        Exp.__init__(self, line_no, meta)
        self.exp = exp
        self.args = args
        
    def replicate(self):
        '''Replicate this abstract syntax tree node'''
        return FunCallExp(self.exp.replicate(), 
                          [a.replicate() for a in self.args], 
                          self.line_no, meta=copy.deepcopy(self.meta))

#-----------------------------------------------
# Unary Expression
#-----------------------------------------------

class UnaryExp(Exp):
    PLUS = 1
    MINUS = 2
    LNOT = 3
    PRE_INC = 4
    PRE_DEC = 5
    POST_INC = 6
    POST_DEC = 7
    DEREF = 8
    ADDRESSOF = 9

    def __init__(self, exp, op_type, line_no = '', meta={}):
        '''Create a unary operation expression'''
        Exp.__init__(self, line_no, meta)
        self.exp = exp
        self.op_type = op_type

    def replicate(self):
        '''Replicate this abstract syntax tree node'''
        return UnaryExp(self.exp.replicate(), self.op_type, self.line_no, meta=copy.deepcopy(self.meta))

#-----------------------------------------------
# Binary Operation
#-----------------------------------------------

class BinOpExp(Exp):
    MUL = 1
    DIV = 2
    MOD = 3
    ADD = 4
    SUB = 5
    LT = 6
    GT = 7
    LE = 8
    GE = 9
    EQ = 10
    NE = 11
    LOR = 12
    LAND = 13
    COMMA = 14
    EQ_ASGN = 15
    ASGN_ADD = 16
    ASGN_SHR = 17
    ASGN_SHL = 18
    BAND = 19
    SHR = 20
    BOR = 21

    def __init__(self, lhs, rhs, op_type, line_no = '', meta={}):
        '''Create a binary operation expression'''
        Exp.__init__(self, line_no, meta)
        self.lhs = lhs
        self.rhs = rhs
        self.op_type = op_type
        if op_type == self.EQ_ASGN:
            self.lhs.updateMeta('defs')
            self.rhs.updateMeta('uses')
        pass


    def replicate(self):
        '''Replicate this abstract syntax tree node'''
        self.lhs
        return BinOpExp(self.lhs.replicate(), self.rhs.replicate(), 
                        self.op_type, self.line_no, meta=copy.deepcopy(self.meta))

#-----------------------------------------------
# Ternary Operation
#-----------------------------------------------
class TernaryExp(Exp):
    def __init__(self, test, true_expr, false_expr, line_no = '', meta={}):
        '''Create a ternary operation expression'''
        Exp.__init__(self, line_no, meta)
        self.test = test
        self.true_expr = true_expr
        self.false_expr = false_expr

    def replicate(self):
        '''Replicate this abstract syntax tree node'''
        return TernaryExp(self.test.replicate(), self.true_expr.replicate(), 
                          self.false_expr.replicate(), self.line_no, meta=copy.deepcopy(self.meta))

#-----------------------------------------------
# Parenthesized Expression
#-----------------------------------------------

class ParenthExp(Exp):

    def __init__(self, exp, line_no = '', meta={}):
        '''Create a parenthesized expression'''
        Exp.__init__(self, line_no, meta)
        self.exp = exp

    def replicate(self):
        '''Replicate this abstract syntax tree node'''
        return ParenthExp(self.exp.replicate(), self.line_no, meta=copy.deepcopy(self.meta))
        
#-----------------------------------------------
# Comments
#-----------------------------------------------
class Comment(AST):

    def __init__(self, comment, line_no = '', meta={}):
        AST.__init__(self, line_no, meta)
        self.text = comment

    def replicate(self):
        '''Replicates the comment node'''
        return Comment(self.text, self.line_no, meta=copy.deepcopy(self.meta))

#-----------------------------------------------
# Statement
#-----------------------------------------------

class Stmt(AST):

    def __init__(self, line_no = '', label=None, meta={}):
        '''Create a statement'''
        AST.__init__(self, line_no, meta)
        self.label = label
        self.meta = copy.deepcopy(meta)
    
    def setLabel(self, label):
        self.label = label
    
    def getLabel(self):
        return self.label
    

#-----------------------------------------------
# Expression Statement
#-----------------------------------------------

class ExpStmt(Stmt):

    def __init__(self, exp, line_no = '', label=None, meta={}):
        '''Create an expression statement'''
        Stmt.__init__(self, line_no, label, meta)
        self.exp = exp         # may be null

    def replicate(self):
        '''Replicate this abstract syntax tree node'''
        r_e = self.exp
        if r_e:
            r_e = r_e.replicate()
        return ExpStmt(r_e, self.line_no, self.label, meta=copy.deepcopy(self.meta))

class GotoStmt(Stmt):
    def __init__(self, target, line_no = '', label=None, meta={}):
        '''Create an expression statement'''
        Stmt.__init__(self, line_no, label, meta)
        self.target = target

    def replicate(self):
        '''Replicate this abstract syntax tree node'''
        return GotoStmt(self.target, self.line_no, self.label, meta=copy.deepcopy(self.meta))
     
#-----------------------------------------------
# Compound Statement
#-----------------------------------------------

class CompStmt(Stmt):

    def __init__(self, stmts, line_no = '', label=None, meta={}):
        '''Create a compound statement'''
        Stmt.__init__(self, line_no, label, meta)
        self.stmts = stmts
        for s in self.stmts: s.parent = self
        if not self.meta.get('id') and line_no: self.meta['id'] = 'loop_' + line_no

    def replicate(self):
        '''Replicate this abstract syntax tree node'''
        return CompStmt([s.replicate() for s in self.stmts], 
                        self.line_no, self.label, meta=copy.deepcopy(self.meta))
    
#-----------------------------------------------
# If-Then-Else
#-----------------------------------------------

class IfStmt(Stmt):

    def __init__(self, test, true_stmt, false_stmt = None, line_no = '', label=None, meta={}):
        '''Create an if statement'''
        Stmt.__init__(self, line_no, label, meta)
        self.test = test
        self.true_stmt = true_stmt
        self.false_stmt = false_stmt           # may be null

    def replicate(self):
        '''Replicate this abstract syntax tree node'''
        f_s = self.false_stmt
        if f_s:
            f_s = f_s.replicate()
        return IfStmt(self.test.replicate(), self.true_stmt.replicate(),
                       f_s, self.line_no, self.label, meta=copy.deepcopy(self.meta))

#-----------------------------------------------
# For Loop
#-----------------------------------------------

class ForStmt(Stmt):

    def __init__(self, init, test, itr, stmt, line_no = '', label=None, meta={}, parent=None):
        '''Create a for-loop statement'''
        Stmt.__init__(self, line_no, label, meta)
        self.init = init      # may be null
        self.test = test      # may be null
        self.iter = itr      # may be null
        self.stmt = stmt
        self.parent = parent
        if not self.meta.get('id') and line_no: self.meta['id'] = 'loop_' + line_no

    def replicate(self):
        '''Replicate this abstract syntax tree node'''
        r_in = self.init
        r_t = self.test
        r_it = self.iter
        if r_in:
            r_in = r_in.replicate()
        if r_t:
            r_t = r_t.replicate()
        if r_it:
            r_it = r_it.replicate()
        return ForStmt(r_in, r_t, r_it, self.stmt.replicate(), #label='loop_' + self.line_no
                       line_no=self.line_no, meta=copy.deepcopy(self.meta), parent=self.parent)

#-----------------------------------------------
# Assignment
#-----------------------------------------------

class AssignStmt(Stmt):

    def __init__(self, var, exp, line_no = '', label=None, meta={}):
        '''Create an assign ment statement.'''
        #TODO: this does not appear to be used anywhere, assignemnts are treated
        # as binary operators
        Stmt.__init__(self, line_no, label, meta)
        self.var = var
        self.exp = exp

    def replicate(self):
        '''Replicate this node'''
        self.var.updateMeta('defs')
        newexp = self.exp.replicate()
        newexp.updateMeta('uses')
        return AssignStmt(self.var, newexp, self.line_no, self.label, meta=copy.deepcopy(self.meta))

#-----------------------------------------------
# Transformation
#-----------------------------------------------

class TransformStmt(Stmt):

    def __init__(self, name, args, stmt, line_no = '', label=None, meta={}):
        '''Create a transformation statement'''
        Stmt.__init__(self, line_no, label, meta)
        self.name = name
        self.args = args
        self.stmt = stmt

    def replicate(self):
        '''Replicate this abstract syntax tree node'''
        return TransformStmt(self.name, self.args[:], self.stmt.replicate(), 
                             self.line_no, meta=copy.deepcopy(self.meta))

#-----------------------------------------------
# New AST
#-----------------------------------------------

class NewAST(AST):

    def __init__(self, line_no = '', meta={}):
        '''Create a newly-added statement'''
        AST.__init__(self, line_no, meta)

#-----------------------------------------------
# Variable Declaration
#-----------------------------------------------

class VarDecl(NewAST):

    def __init__(self, type_name, var_names, line_no = '', meta={}, qual=''):
        '''Create a variable declaration'''
        NewAST.__init__(self, line_no, meta)
        self.type_name = type_name
        self.var_names = var_names
        self.qualifier = qual


    def replicate(self):
        '''Replicate this abstract syntax tree node'''
        return VarDecl(self.type_name, self.var_names[:],
                       self.line_no, meta=copy.deepcopy(self.meta), qual=self.qualifier)

class VarDeclInit(NewAST):

    def __init__(self, type_name, var_name, init_exp, line_no = '', meta={}, qual=''):
        '''Create an initializing variable declaration'''
        NewAST.__init__(self, line_no, meta)
        self.type_name = type_name
        self.var_name  = var_name
        self.init_exp  = init_exp
        self.qualifier = qual

    def replicate(self):
        return VarDeclInit(self.type_name, self.var_name, self.init_exp,
                           self.line_no, meta=copy.deepcopy(self.meta), qual=self.qualifier)


class DeclStmt(NewAST):
    def __init__(self):
        self.decls = []

    def append(self, decl):
        self.decls.append(decl)

    def vars(self):
        return self.decls

#-----------------------------------------------
# Field Declaration
#-----------------------------------------------

class FieldDecl(NewAST):

    def __init__(self, ty, name, line_no = '', meta={}):
        '''Create a field declaration'''
        NewAST.__init__(self, line_no, meta)
        self.ty = ty
        self.name = name

    def replicate(self):
        '''Replicate this abstract syntax tree node'''
        return FieldDecl(self.ty, self.name, self.line_no, meta=copy.deepcopy(self.meta))

#-----------------------------------------------
# Function Declaration
#-----------------------------------------------

class FunDecl(NewAST):

    def __init__(self, name, return_type, modifiers, params, body, line_no = '', meta={}):
        '''Create a function declaration'''
        NewAST.__init__(self, line_no, meta)
        self.name = name
        self.return_type = return_type
        self.modifiers = modifiers
        self.params = params
        self.body = body # a body should be a compound stmt

    def replicate(self):
        '''Replicate this abstract syntax tree node'''
        return FunDecl(self.fun_name, self.return_type, self.modifiers[:], 
                       self.params[:], self.body.replicate(), self.line_no,
                       meta=copy.deepcopy(self.meta))

#-----------------------------------------------
# Pragma Directive
#-----------------------------------------------

class Pragma(NewAST):

    def __init__(self, pstring, line_no = '', meta={}):
        '''Create a pragma directive'''
        NewAST.__init__(self, line_no, meta)
        self.pstring = pstring

    def replicate(self):
        '''Replicate this abstract syntax tree node'''
        return Pragma(self.pstring, self.line_no, meta=copy.deepcopy(self.meta))

#-----------------------------------------------
# Container
#-----------------------------------------------

class Container(NewAST):

    def __init__(self, ast, line_no = '', meta={}):
        '''Create a container AST (to protect the contained AST from any code transformations)'''
        NewAST.__init__(self, line_no, meta)
        self.ast = ast

    def replicate(self):
        '''Replicate this abstract syntax tree node'''
        return Container(self.ast.replicate(), self.line_no, meta=copy.deepcopy(self.meta))

#-----------------------------------------------
# While Loop
#-----------------------------------------------

class WhileStmt(NewAST):

    def __init__(self, test, stmt, line_no = '', meta={}):
        NewAST.__init__(self, line_no, meta)
        self.test = test
        self.stmt = stmt
    
    def replicate(self):
        return WhileStmt(self.test.replicate(), self.stmt.replicate(), 
                         self.line_no, meta=copy.deepcopy(self.meta))

#-----------------------------------------------
# Cast expression
#-----------------------------------------------

class CastExpr(NewAST):

    def __init__(self, ty, expr, line_no = '', meta={}):
        NewAST.__init__(self, line_no, meta)
        self.ctype = ty
        self.expr = expr
    
    def replicate(self):
        return CastExpr(self.ctype, self.expr.replicate(), self.line_no, 
                        meta=copy.deepcopy(self.meta))


if __name__ == '__main__':
    i = IdentExp('Hi')
    print(i.name)