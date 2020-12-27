#
# The class definitions of the Abstract Syntax Tree (AST)
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
#        |
#        +-- ExpStmt 
#        +-- CompStmt 
#        +-- IfStmt 
#        +-- ForStmt 
#

import sys
from . import pprinter

#-----------------------------------------------
# AST - Abstract Syntax Tree
#-----------------------------------------------

class AST:

    def __init__(self):
        '''Create an abstract syntax tree node'''
        pass
        
    def replicate(self):
        '''Replicate this abstract syntax tree node'''
        raise NotImplementedError('%s: abstract function "replicate" not implemented' %
                                  self.__class__.__name__)

    def __repr__(self):
        '''Return a string representation for this AST object'''
        return pprinter.PrettyPrinter().pprint(self)

    def __str__(self):
        '''Return a string representation for this AST object'''
        return repr(self)

#-----------------------------------------------
# Expression
#-----------------------------------------------

class Exp(AST):

    def __init__(self):
        '''Create an expression'''
        AST.__init__(self)

#-----------------------------------------------
# Number Literal
#-----------------------------------------------

class NumLitExp(Exp):

    INT = 1
    FLOAT = 2
    
    def __init__(self, val, lit_type):
        '''Create a numeric literal'''
        Exp.__init__(self)
        self.val = val
        self.lit_type = lit_type

    def replicate(self):
        '''Replicate this abstract syntax tree node'''
        return NumLitExp(self.val, self.lit_type)
        
#-----------------------------------------------
# String Literal
#-----------------------------------------------

class StringLitExp(Exp):

    def __init__(self, val):
        '''Create a string literal'''
        Exp.__init__(self)
        self.val = val

    def replicate(self):
        '''Replicate this abstract syntax tree node'''
        return StringLitExp(self.val)
        
#-----------------------------------------------
# Identifier
#-----------------------------------------------

class IdentExp(Exp):

    def __init__(self, name):
        '''Create an identifier'''
        Exp.__init__(self)
        self.name = name
        
    def replicate(self):
        '''Replicate this abstract syntax tree node'''
        return IdentExp(self.name)

#-----------------------------------------------
# Array Reference
#-----------------------------------------------

class ArrayRefExp(Exp):

    def __init__(self, exp, sub_exp):
        '''Create an array reference'''
        Exp.__init__(self)
        self.exp = exp
        self.sub_exp = sub_exp

    def replicate(self):
        '''Replicate this abstract syntax tree node'''
        return ArrayRefExp(self.exp.replicate(), self.sub_exp.replicate())
        
#-----------------------------------------------
# Function Call
#-----------------------------------------------

class FunCallExp(Exp):

    def __init__(self, exp, args):
        '''Create a function call'''
        Exp.__init__(self)
        self.exp = exp
        self.args = args
        
    def replicate(self):
        '''Replicate this abstract syntax tree node'''
        return FunCallExp(self.exp.replicate(), [a.replicate() for a in self.args])

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

    def __init__(self, exp, op_type):
        '''Create a unary operation expression'''
        Exp.__init__(self)
        self.exp = exp
        self.op_type = op_type

    def replicate(self):
        '''Replicate this abstract syntax tree node'''
        return UnaryExp(self.exp.replicate(), self.op_type)

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

    def __init__(self, lhs, rhs, op_type):
        '''Create a binary operation expression'''
        Exp.__init__(self)
        self.lhs = lhs
        self.rhs = rhs
        self.op_type = op_type

    def replicate(self):
        '''Replicate this abstract syntax tree node'''
        return BinOpExp(self.lhs.replicate(), self.rhs.replicate(), self.op_type)

#-----------------------------------------------
# Parenthesized Expression
#-----------------------------------------------

class ParenthExp(Exp):

    def __init__(self, exp):
        '''Create a parenthesized expression'''
        Exp.__init__(self)
        self.exp = exp

    def replicate(self):
        '''Replicate this abstract syntax tree node'''
        return ParenthExp(self.exp.replicate())
        
#-----------------------------------------------
# Statement
#-----------------------------------------------

class Stmt(AST):

    def __init__(self):
        '''Create a statement'''
        AST.__init__(self)

#-----------------------------------------------
# Expression Statement
#-----------------------------------------------

class ExpStmt(Stmt):

    def __init__(self, exp):
        '''Create an expression statement'''
        Stmt.__init__(self)
        self.exp = exp         # may be null

    def replicate(self):
        '''Replicate this abstract syntax tree node'''
        r_e = self.exp
        if r_e:
            r_e = r_e.replicate()
        return ExpStmt(r_e)

#-----------------------------------------------
# Compound Statement
#-----------------------------------------------

class CompStmt(Stmt):

    def __init__(self, stmts):
        '''Create a compound statement'''
        Stmt.__init__(self)
        self.stmts = stmts

    def replicate(self):
        '''Replicate this abstract syntax tree node'''
        return CompStmt([s.replicate() for s in self.stmts])
    
#-----------------------------------------------
# If-Then-Else
#-----------------------------------------------

class IfStmt(Stmt):

    def __init__(self, test, true_stmt, false_stmt = None):
        '''Create an if statement'''
        Stmt.__init__(self)
        self.test = test
        self.true_stmt = true_stmt
        self.false_stmt = false_stmt           # may be null

    def replicate(self):
        '''Replicate this abstract syntax tree node'''
        f_s = self.false_stmt
        if f_s:
            f_s = f_s.replicate()
        return IfStmt(self.test.replicate(), self.true_stmt.replicate(), f_s)

#-----------------------------------------------
# For Loop
#-----------------------------------------------

class ForStmt(Stmt):

    def __init__(self, init, test, iter, stmt):
        '''Create a for-loop statement'''
        Stmt.__init__(self)
        self.init = init      # may be null
        self.test = test      # may be null
        self.iter = iter      # may be null
        self.stmt = stmt

        # This is a hack! These are fields with string values, used to mark loops. 
        # An instance of using these fields is to mark whether this loop iterates 
        # over full rectangular tiles.
        self.start_label = ''
        self.end_label = ''

    def replicate(self):
        '''Replicate this abstract syntax tree node'''
        r_in = self.init
        r_ts = self.test
        r_it = self.iter
        if r_in:
            r_in = r_in.replicate()
        if r_ts:
            r_ts = r_ts.replicate()
        if r_it:
            r_it = r_it.replicate()
        f = ForStmt(r_in, r_ts, r_it, self.stmt.replicate())
        f.start_label = self.start_label
        f.end_label = self.end_label
        return f


