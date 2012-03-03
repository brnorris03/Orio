#
# The classes for the abstract syntax tree (AST)
#
#  Node 
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
#   +-- NewNode
#        |
#        +-- VarDecl
#        +-- FieldDecl 
#        +-- FunDecl
#        +-- Pragma 
#        +-- Container
#
# - The NewNode is an AST node used only in the output code generation. Such separation is needed to
#   simplify the input language.
#

import sys

class Node(object):
    ''' Abstract base class for AST nodes. '''
    def __init__(self, line_no = ''):
        '''Create an abstract syntax tree node'''
        self.line_no = line_no           # may be null (i.e. empty string)
        self.kids = []
        
    def replicate(self):
        '''Replicate this node'''
        raise NotImplementedError('%s: abstract function "replicate" not implemented' % self.__class__.__name__)

    def __repr__(self):
        '''Return a string representation for this AST object'''
        return orio.module.loop.codegen.CodeGen().generator.generate(self)

    def __str__(self):
        '''Return a string representation for this AST object'''
        return repr(self)
    
    def show(self, buf=sys.stdout, offset=0, attrnames=False, showcoord=False):
        """ Pretty print the Node and all its attributes and
            children (recursively) to a buffer.
            
            file:   
                Open IO buffer into which the Node is printed.
            
            offset: 
                Initial offset (amount of leading spaces) 
            
            attrnames:
                True if you want to see the attribute names in
                name=value pairs. False to only see the values.
            
            showcoord:
                Do you want the coordinates of each Node to be
                displayed.
        """
        lead = ' ' * offset
        buf.write(lead + self.__class__.__name__+': ')

        if self.attr_names:
            if attrnames:
                nvlist = [(n, getattr(self,n)) for n in self.attr_names]
                attrstr = ', '.join('%s=%s' % nv for nv in nvlist)
            else:
                vlist = [getattr(self, n) for n in self.attr_names]
                attrstr = ', '.join('%s' % v for v in vlist)
            buf.write(attrstr)

        if showcoord:
            buf.write(' (at %s)' % self.line_no)
        buf.write('\n')

        for c in self.kids:
            c.show(buf, offset + 2, attrnames, showcoord)


class NodeVisitor(object):
    """ A base NodeVisitor class for visiting AST nodes. 
        Subclass it and define your own visit_XXX methods, where
        XXX is the class name you want to visit with these 
        methods.
        
        For example:
        
        class ConstantVisitor(NodeVisitor):
            def __init__(self):
                self.values = []
            
            def visit_Constant(self, node):
                self.values.append(node.value)

        Creates a list of values of all the constant nodes 
        encountered below the given node. To use it:
        
        cv = ConstantVisitor()
        cv.visit(node)
        
        Notes:
        
        *   generic_visit() will be called for AST nodes for which 
            no visit_XXX method was defined. 
        *   The children of nodes for which a visit_XXX was 
            defined will not be visited - if you need this, call
            generic_visit() on the node. 
            You can use:
                NodeVisitor.generic_visit(self, node)
        *   Modeled after Python's own AST visiting facilities
            (the ast module of Python 3.0)
    """
    def visit(self, node):
        """ Visit a node. 
        """
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)
        
    def generic_visit(self, node):
        """ Called if no explicit visitor function exists for a 
            node. Implements preorder visiting of the node.
        """
        for c in node.kids:
            self.visit(c)

#-----------------------------------------------
# Expression
#-----------------------------------------------

class Exp(Node):

    def __init__(self, line_no = ''):
        super(Exp, self).__init__(line_no)

#-----------------------------------------------
# Number Literal
#-----------------------------------------------

class NumLitExp(Exp):

    INT = 1
    FLOAT = 2
    
    def __init__(self, val, lit_type, line_no = ''):
        super(NumLitExp, self).__init__(line_no)
        self.val = val
        self.lit_type = lit_type
        self.kids = [val, lit_type]

    def replicate(self):
        return NumLitExp(self.val, self.lit_type, self.line_no)
        
#-----------------------------------------------
# String Literal
#-----------------------------------------------

class StringLitExp(Exp):

    def __init__(self, val, line_no = ''):
        super(StringLitExp, self).__init__(line_no)
        self.val = val
        self.kids = [val]

    def replicate(self):
        return StringLitExp(self.val, self.line_no)
        
#-----------------------------------------------
# Identifier
#-----------------------------------------------

class IdentExp(Exp):

    def __init__(self, name, line_no = ''):
        super(IdentExp, self).__init__(line_no)
        self.name = name
        self.kids = [name]
        
    def replicate(self):
        return IdentExp(self.name, self.line_no)

#-----------------------------------------------
# Array Reference
#-----------------------------------------------

class ArrayRefExp(Exp):

    def __init__(self, exp, sub_exp, line_no = ''):
        super(ArrayRefExp, self).__init__(line_no)
        self.exp = exp
        self.sub_exp = sub_exp
        self.kids = [exp, sub_exp]

    def replicate(self):
        return ArrayRefExp(self.exp.replicate(), self.sub_exp.replicate(), self.line_no)
        
#-----------------------------------------------
# Function Call
#-----------------------------------------------

class FunCallExp(Exp):

    def __init__(self, exp, args, line_no = ''):
        super(FunCallExp, self).__init__(line_no)
        self.exp = exp
        self.args = args
        self.kids = [exp, args]
        
    def replicate(self):
        return FunCallExp(self.exp.replicate(), [a.replicate() for a in self.args], self.line_no)

#-----------------------------------------------
# Unary Operation Expression
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
        super(UnaryExp, self).__init__(line_no)
        self.exp = exp
        self.op_type = op_type
        self.kids = [exp, op_type]

    def replicate(self):
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
        super(BinOpExp, self).__init__(line_no)
        self.lhs = lhs
        self.rhs = rhs
        self.op_type = op_type
        self.kids = [lhs, rhs, op_type]

    def replicate(self):
        return BinOpExp(self.lhs.replicate(), self.rhs.replicate(), self.op_type, self.line_no)

#-----------------------------------------------
# Parenthesized Expression
#-----------------------------------------------

class ParenthExp(Exp):

    def __init__(self, exp, line_no = ''):
        super(ParenthExp, self).__init__(line_no)
        self.exp = exp
        self.kids = [exp]

    def replicate(self):
        return ParenthExp(self.exp.replicate(), self.line_no)
        
#-----------------------------------------------
# Comments
#-----------------------------------------------

class Comment(Node):

    def __init__(self, comment, line_no = ''):
        super(Comment, self).__init__(line_no)
        self.text = comment
        self.kids = [comment]

    def replicate(self):
        return Comment(self.text, self.line_no)

#-----------------------------------------------
# Statement
#-----------------------------------------------

class Stmt(Node):

    def __init__(self, line_no = '', label=None):
        super(Stmt, self).__init__(line_no)
        self.label = None
    
    def setLabel(self, label):
        self.label = label
    
    def getLabel(self):
        return self.label
    
#-----------------------------------------------
# Expression Statement
#-----------------------------------------------

class ExpStmt(Stmt):

    def __init__(self, exp, line_no = '', label=None):
        super(ExpStmt, self).__init__(line_no, label)
        self.exp = exp         # may be null
        self.kids = [exp]

    def replicate(self):
        r_e = self.exp
        if r_e:
            r_e = r_e.replicate()
        return ExpStmt(r_e, self.line_no, self.label)

#-----------------------------------------------
# Goto
#-----------------------------------------------

class GotoStmt(Stmt):

    def __init__(self, target, line_no = '', label=None):
        super(GotoStmt, self).__init__(line_no, label)
        self.target = target
        self.kids = [target]

    def replicate(self):
        return GotoStmt(self.target, self.line_no, self.label)
     
#-----------------------------------------------
# Compound Statement
#-----------------------------------------------

class CompStmt(Stmt):

    def __init__(self, stmts, line_no = '', label=None):
        super(CompStmt, self).__init__(line_no, label)
        self.stmts = stmts
        self.kids = [stmts]

    def replicate(self):
        return CompStmt([s.replicate() for s in self.stmts], self.line_no, self.label)
    
#-----------------------------------------------
# If-Then-Else
#-----------------------------------------------

class IfStmt(Stmt):

    def __init__(self, test, true_stmt, false_stmt = None, line_no = '', label=None):
        super(IfStmt, self).__init__(line_no, label)
        self.test = test
        self.true_stmt = true_stmt
        self.false_stmt = false_stmt           # may be null
        self.kids = [test, true_stmt, false_stmt]

    def replicate(self):
        f_s = self.false_stmt
        if f_s:
            f_s = f_s.replicate()
        return IfStmt(self.test.replicate(), self.true_stmt.replicate(), f_s, self.line_no, self.label)

#-----------------------------------------------
# For Loop
#-----------------------------------------------

class ForStmt(Stmt):

    def __init__(self, init, test, itr, stmt, line_no = '', label=None):
        super(ForStmt, self).__init__(line_no, label)
        self.init = init      # may be null
        self.test = test      # may be null
        self.iter = itr       # may be null
        self.stmt = stmt
        self.kids = [init, test, itr, stmt]

    def replicate(self):
        r_in = self.init
        r_t  = self.test
        r_it = self.iter
        if r_in:
            r_in = r_in.replicate()
        if r_t:
            r_t  = r_t.replicate()
        if r_it:
            r_it = r_it.replicate()
        return ForStmt(r_in, r_t, r_it, self.stmt.replicate(), self.line_no, self.label)

#-----------------------------------------------
# Assignment
#-----------------------------------------------

class AssignStmt(Stmt):

    def __init__(self, var, exp, line_no = '', label=None):
        super(AssignStmt, self).__init__(line_no, label)
        self.var = var
        self.exp = exp
        self.kids = [var, exp]

    def replicate(self):
        return AssignStmt(self.var, self.exp.replicate(), self.line_no, self.label)

#-----------------------------------------------
# Transformation Statement
#-----------------------------------------------

class TransformStmt(Stmt):

    def __init__(self, name, args, stmt, line_no = '', label=None):
        super(TransformStmt, self).__init__(line_no, label)
        self.name = name
        self.args = args
        self.stmt = stmt
        self.kids = [name, args, stmt]

    def replicate(self):
        return TransformStmt(self.name, self.args[:], self.stmt.replicate(), self.line_no)

#-----------------------------------------------
# New or output node
#-----------------------------------------------

class NewNode(Node):

    def __init__(self, line_no = ''):
        super(NewNode, self).__init__(line_no)

#-----------------------------------------------
# Variable Declaration
#-----------------------------------------------

class VarDecl(NewNode):

    def __init__(self, type_name, var_names, line_no = ''):
        super(VarDecl, self).__init__(line_no)
        self.type_name = type_name
        self.var_names = var_names
        self.kids = [type_name, var_names]

    def replicate(self):
        return VarDecl(self.type_name, self.var_names[:], self.line_no)

#-----------------------------------------------
# Field Declaration
#-----------------------------------------------

class FieldDecl(NewNode):

    def __init__(self, ty, name, line_no = ''):
        super(FieldDecl, self).__init__(line_no)
        self.ty = ty
        self.name = name
        self.kids = [ty, name]

    def replicate(self):
        return FieldDecl(self.ty, self.name, self.line_no)

#-----------------------------------------------
# Function Declaration
#-----------------------------------------------

class FunDecl(NewNode):

    def __init__(self, name, return_type, modifiers, params, body, line_no = ''):
        super(FunDecl, self).__init__(line_no)
        self.name = name
        self.return_type = return_type
        self.modifiers = modifiers
        self.params = params
        self.body = body # a body should be a compound stmt
        self.kids[name, return_type, modifiers, params, body]

    def replicate(self):
        return FunDecl(self.fun_name, self.return_type, self.modifiers[:], self.params[:], self.body.replicate(), self.line_no)

#-----------------------------------------------
# Pragma Directive
#-----------------------------------------------

class Pragma(NewNode):

    def __init__(self, pstring, line_no = ''):
        super(Pragma, self).__init__(line_no)
        self.pstring = pstring
        self.kids = [pstring]

    def replicate(self):
        return Pragma(self.pstring, self.line_no)

#-----------------------------------------------
# Container node (to protect the contained node from any code transformations)
#-----------------------------------------------

class Container(NewNode):

    def __init__(self, ast, line_no = ''):
        super(Container, self).__init__(line_no)
        self.ast = ast
        self.kids = [ast]

    def replicate(self):
        return Container(self.ast.replicate(), self.line_no)


