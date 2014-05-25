'''
Created on Oct 19, 2010

@author: norris
'''

#
# The classes for the abstract syntax tree (ASTNode)
#
#  ASTNode 
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
#   |    +-- TransformStmt 
#   |
#   +-- NewAST 
#        |
#        +-- VarDecl 
#        +-- Pragma 
#        +-- Container
#
# - The NewAST is an ASTNode used only in the output code generation. Such separation is needed to
#   simplify the input language.
#

import os
import orio.main.parsers.fcodegen
from orio.main.util.globals import *
import orio.tool.graphlib.graph as graph

# ----------------------------------------------
# AST - Abstract Syntax Tree class
# ----------------------------------------------
class AST(graph.Graph):
    '''Abstract Syntax Tree class.'''
    def __init__(self, name):
        '''Create a graph'''
        graph.Graph.__init__(self, name)

    #@staticmethod
    def bfs(start, reverse=False, actions=[]):
        '''
        Returns a breadth-first search list of vertices connected by edges of edgetype. Note that 
        this is less general than the graphlib graph breadth_first_search method used in 
        the BVertex dependents() and dependencies() methods.

        @param start_v the initial vertex
        @param reverse the order of the search
        @param edgetype a list of edge types to consider (the possible types are "extends", "implements", "requires", "contains")
        '''
        #unprocessed = [start_v]
        if isinstance(start,list):
            unprocessed = start
        else:
            unprocessed = [start]
        visited = []
        while unprocessed:
            v = unprocessed.pop(0)
            if v not in visited:
                visited.append(v)
                if reverse: 
                    elist = []
                    for e in v.in_e:
                        if e.action in actions: elist.extend(e.src_v)
                    if elist: unprocessed.extend(elist)
                else: 
                    elist = []
                    for e in v.out_e:
                        if e.action in actions: elist.extend(e.dest_v)
                    if elist: unprocessed.extend(elist)
        return visited
    bfs = staticmethod(bfs)


    #@staticmethod
    def dfs(start, visitor = None, reverse=False, actions=[]):
        '''
        Returns a depth-first search list of vertices connected by edges of edgetype. Note that 
        this is less general than the graphlib graph depth_first_search method.

        @param start_v the initial vertex
        @param visitor a flag indicating whether the visit() method should be called on the vertex.
        @param reverse the order of the search
        @param edgetype a list of edge types to consider (the possible types are "extends", "implements", "requires", "contains")
        '''
        if isinstance(start,list):
            unprocessed = start
        else: 
            unprocessed = [start]
        visited = []
        while unprocessed:
            v = unprocessed.pop(0)
            if v not in visited:
                if visitor:
                    v.visitor()
                visited.append(v)
                if reverse: 
                    elist = []
                    #print str(v)
                    for e in v.in_e:
                        if e.action in actions: elist.extend(e.src_v)
                    if elist: unprocessed.extend(elist)
                else: 
                    elist = []
                    for e in v.out_e:
                        if e.action in actions: elist.extend(e.dest_v)
                    if elist: unprocessed.extend(elist)
        return visited
    dfs = staticmethod(dfs)  
    
    # ---------- end AST class ------------------------------
     
#-----------------------------------------------
# ASTNode - Abstract Syntax Tree node
#-----------------------------------------------

class ASTNode(graph.Vertex):

    def __init__(self, line_no = ''):
        '''Create an abstract syntax tree node'''
        self.line_no = line_no           # may be null (i.e. empty string)
        self.filename = ''
        
    def clone(self):
        '''Replicate this abstract syntax tree node'''
        raise NotImplementedError('%s: abstract function "replicate" not implemented' %
                                  self.__class__.__name__)

    def __repr__(self):
        '''Return a string representation for this ASTNode object'''
        return orio.main.parsers.fcodegen.CodeGen().generate(self)

    def __str__(self):
        '''Return a string representation for this ASTNode object'''
        return repr(self)

    
#-----------------------------------------------
# Subroutines and functions
#-----------------------------------------------
class SubroutineDeclaration(ASTNode):
    def __init__(self, header, varrefs, body, function=False):
        '''
        @param header: a tuple of (name, arglist)
        @param body: a tuple of (subroutine statements string, span in source file); 
                The span is a (startpos, endpos) tuple.
                
        '''
        self.name = header[0]
        self.arglist = header[1]
        self.varrefs = varrefs
        self.body = body[0]
        self.bodyspan = body[1]         # the location span
        self.function = function        # designates whether subroutine is function or procedure
        return
    
    def __repr__(self):
        buf = 'subroutine:'
        buf += str(self.name) + '\n' + str(self.arglist) + '\n' + str(self.varrefs) + \
            '\n' + str(self.body) + '\n' + str(self.bodyspan)
        return buf
    
    def inline(self, params):
        '''
        Rewrite the body to contain actual arguments in place of the formal parameters.
        @param params: the list of actual parameters in the same order as the formal 
                    parameters in the subroutine definition
        '''
        starpos = self.bodyspan[0]
        for v in self.varrefs:
            arg = v[0]      # the variable name
            argspan = v[1]  # begin and end position 
            
        return
        
    
    # end of class SubroutineDefinition
    
class SubroutineDefinition(SubroutineDeclaration):
    pass
#-----------------------------------------------
# Expression
#-----------------------------------------------

class Exp(ASTNode):

    def __init__(self, line_no = ''):
        '''Create an expression'''
        ASTNode.__init__(self, line_no)

#-----------------------------------------------
# Number Literal
#-----------------------------------------------

class NumLitExp(Exp):

    INT = 1
    FLOAT = 2
    
    def __init__(self, val, lit_type, line_no = ''):
        '''Create a numeric literal'''
        Exp.__init__(self, line_no)
        self.val = val
        self.lit_type = lit_type

    def clone(self):
        '''Replicate this abstract syntax tree node'''
        return NumLitExp(self.val, self.lit_type, self.line_no)
        
#-----------------------------------------------
# String Literal
#-----------------------------------------------

class StringLitExp(Exp):

    def __init__(self, val, line_no = ''):
        '''Create a string literal'''
        Exp.__init__(self, line_no)
        self.val = val

    def clone(self):
        '''Replicate this abstract syntax tree node'''
        return StringLitExp(self.val, self.line_no)
        
#-----------------------------------------------
# Identifier
#-----------------------------------------------

class IdentExp(Exp):

    def __init__(self, name, line_no = ''):
        '''Create an identifier'''
        Exp.__init__(self, line_no)
        self.name = name
        
    def clone(self):
        '''Replicate this abstract syntax tree node'''
        return IdentExp(self.name, self.line_no)

#-----------------------------------------------
# Array Reference
#-----------------------------------------------

class ArrayRefExp(Exp):

    def __init__(self, exp, sub_exp, line_no = ''):
        '''Create an array reference'''
        Exp.__init__(self, line_no)
        self.exp = exp
        self.sub_exp = sub_exp

    def clone(self):
        '''Replicate this abstract syntax tree node'''
        return ArrayRefExp(self.exp.replicate(), self.sub_exp.replicate(), self.line_no)
        
#-----------------------------------------------
# Function Call
#-----------------------------------------------

class FunCallExp(Exp):

    def __init__(self, exp, args, line_no = ''):
        '''Create a function call'''
        Exp.__init__(self, line_no)
        self.exp = exp
        self.args = args
        
    def clone(self):
        '''Replicate this abstract syntax tree node'''
        return FunCallExp(self.exp.replicate(), [a.replicate() for a in self.args], self.line_no)

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

    def __init__(self, exp, op_type, line_no = ''):
        '''Create a unary operation expression'''
        Exp.__init__(self, line_no)
        self.exp = exp
        self.op_type = op_type

    def clone(self):
        '''Replicate this abstract syntax tree node'''
        return UnaryExp(self.exp.replicate(), self.op_type, self.line_no)

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

    def __init__(self, lhs, rhs, op_type, line_no = ''):
        '''Create a binary operation expression'''
        Exp.__init__(self, line_no)
        self.lhs = lhs
        self.rhs = rhs
        self.op_type = op_type

    def clone(self):
        '''Replicate this abstract syntax tree node'''
        return BinOpExp(self.lhs.replicate(), self.rhs.replicate(), self.op_type, self.line_no)

#-----------------------------------------------
# Parenthesized Expression
#-----------------------------------------------

class ParenthExp(Exp):

    def __init__(self, exp, line_no = ''):
        '''Create a parenthesized expression'''
        Exp.__init__(self, line_no)
        self.exp = exp

    def clone(self):
        '''Replicate this abstract syntax tree node'''
        return ParenthExp(self.exp.replicate(), self.line_no)
        
#-----------------------------------------------
# Statement
#-----------------------------------------------

class Stmt(ASTNode):

    def __init__(self, line_no = ''):
        '''Create a statement'''
        ASTNode.__init__(self, line_no)

#-----------------------------------------------
# Expression Statement
#-----------------------------------------------

class ExpStmt(Stmt):

    def __init__(self, exp, line_no = ''):
        '''Create an expression statement'''
        Stmt.__init__(self, line_no)
        self.exp = exp         # may be null

    def clone(self):
        '''Replicate this abstract syntax tree node'''
        r_e = self.exp
        if r_e:
            r_e = r_e.replicate()
        return ExpStmt(r_e, self.line_no)

#-----------------------------------------------
# Compound Statement
#-----------------------------------------------

class CompStmt(Stmt):

    def __init__(self, stmts, line_no = ''):
        '''Create a compound statement'''
        Stmt.__init__(self, line_no)
        self.stmts = stmts

    def clone(self):
        '''Replicate this abstract syntax tree node'''
        return CompStmt([s.replicate() for s in self.stmts], self.line_no)
    
#-----------------------------------------------
# If-Then-Else
#-----------------------------------------------

class IfStmt(Stmt):

    def __init__(self, test, true_stmt, false_stmt = None, line_no = ''):
        '''Create an if statement'''
        Stmt.__init__(self, line_no)
        self.test = test
        self.true_stmt = true_stmt
        self.false_stmt = false_stmt           # may be null

    def clone(self):
        '''Replicate this abstract syntax tree node'''
        f_s = self.false_stmt
        if f_s:
            f_s = f_s.replicate()
        return IfStmt(self.test.replicate(), self.true_stmt.replicate(), f_s, self.line_no)

#-----------------------------------------------
# Transformation
#-----------------------------------------------

class TransformStmt(Stmt):

    def __init__(self, name, args, stmt, line_no = ''):
        '''Create a transformation statement'''
        Stmt.__init__(self, line_no)
        self.name = name
        self.args = args
        self.stmt = stmt

    def clone(self):
        '''Replicate this abstract syntax tree node'''
        return TransformStmt(self.name, self.args[:], self.stmt.replicate(), self.line_no)



# ==========================================================
# AST edge
# ==========================================================
class ASTEdge(graph.DirEdge):
    def __init__(self, v1, v2, graph=None, name=None):
        if name is None:
            # generate as unique a name as possible
            name = v1.name + ':' + v2.name
        self.name = name
        debug("Creating edge from %s to %s" % (v1.name,v2.name),level=5)
        graph.DirEdge.__init__(self, name, v1, v2)

        if graph != None:
            if v1.name not in graph.v.keys(): graph.add_v(v1)
            if v2.name not in graph.v.keys(): graph.add_v(v2)
            if self.name not in graph.e.keys(): graph.add_e(self)

        pass        
