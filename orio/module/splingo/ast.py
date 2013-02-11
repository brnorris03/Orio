#==============================================================================
#  AST
#==============================================================================

import sys

#----------------------------------------------------------------------------------------------------------------------
class Node(object):
    ''' Abstract base class for AST nodes. '''

    def __init__(self, coord=None):
        '''Create an abstract syntax tree node'''
        self.coord = coord
        self.kids = []
        
    def __repr__(self):
        '''Return a string representation for this AST object'''
        import orio.module.splingo.printer as printer
        return printer.Printer().generate(self)

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
            buf.write(' (at %s)' % self.coord)
        buf.write('\n')

        for c in self.kids:
            c.show(buf, offset + 2, attrnames, showcoord)
#----------------------------------------------------------------------------------------------------------------------



#----------------------------------------------------------------------------------------------------------------------
class NodeVisitor(object):
    def visit(self, node):
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        for k in node.kids:
            self.visit(k)

    def rewriteTD(self, r, n):
        nn = r(n)
        kids = getattr(nn, 'kids', False)
        if kids:
            nkids = []
            for k in nn.kids:
                nkids += [self.rewriteTD(r, k)]
            nn.kids = nkids
        elif isinstance(nn, list):
            nn = map(lambda e: self.rewriteTD(r, e), nn)
        return nn

    def collectTD(self, f, n):
        acc = f(n)
        kids = getattr(n, 'kids', [])
        for k in kids:
            acc += self.collectTD(f, k)
        return acc

    def rewriteBU(self, r, n):
        kids = getattr(n, 'kids', False)
        if kids:
            nkids = []
            for k in n.kids:
                nkids += [self.rewriteBU(r, k)]
            n.kids = nkids
        elif isinstance(n, list):
            return map(lambda e: self.rewriteBU(r, e), n)
        return r(n)
#----------------------------------------------------------------------------------------------------------------------



#----------------------------------------------------------------------------------------------------------------------
class Comment(Node):

    def __init__(self, comment, coord=None):
        super(Comment, self).__init__(coord)
        self.kids = [comment]

#----------------------------------------------------------------------------------------------------------------------
class Exp(Node):

    def __init__(self, coord=None):
        super(Exp, self).__init__(coord)

#----------------------------------------------------------
class LitExp(Exp):
    #BOOL = 1
    INT = 2
    FLOAT = 3
    STRING = 4
    
    def __init__(self, val, lit_type, coord=None):
        super(LitExp, self).__init__(coord)
        self.kids = [lit_type, val]

#----------------------------------------------------------
class IdentExp(Exp):

    def __init__(self, name, coord=None):
        super(IdentExp, self).__init__(coord)
        self.kids = [name]

#----------------------------------------------------------
class ArrayRefExp(Exp):

    def __init__(self, exp, sub, coord=None):
        super(ArrayRefExp, self).__init__(coord)
        self.kids = [exp, sub]

#----------------------------------------------------------
class CallExp(Exp):

    def __init__(self, exp, args, coord=None):
        super(CallExp, self).__init__(coord)
        self.kids = [exp, args]

#----------------------------------------------------------
class UnaryExp(Exp):
    PLUS = 1
    MINUS = 2
    LNOT = 3
    TRANSPOSE = 4
    PRE_INC = 10
    PRE_DEC = 11
    POST_INC = 12
    POST_DEC = 13

    def __init__(self, op_type, exp, coord=None):
        super(UnaryExp, self).__init__(coord)
        self.kids = [op_type, exp]

#----------------------------------------------------------
class BinOpExp(Exp):
    PLUS = 1
    MINUS = 2
    MULT = 3
    DIV = 4
    MOD = 5
    LT = 10
    GT = 11
    LE = 12
    GE = 13
    EE = 14
    NE = 15
    LOR = 20
    LAND = 21
    EQ = 30
    EQPLUS = 31
    EQMINUS = 32
    EQMULT = 33
    EQDIV = 34

    def __init__(self, op_type, lhs, rhs, coord=None):
        super(BinOpExp, self).__init__(coord)
        self.kids = [op_type, lhs, rhs]

#----------------------------------------------------------
class ParenExp(Exp):

    def __init__(self, exp, coord=None):
        super(ParenExp, self).__init__(coord)
        self.kids = [exp]



#----------------------------------------------------------------------------------------------------------------------
class Stmt(Node):

    def __init__(self, coord=None):
        super(Stmt, self).__init__(coord)
        self.label = None

#----------------------------------------------------------
class ExpStmt(Stmt):

    def __init__(self, exp, coord=None):
        super(ExpStmt, self).__init__(coord)
        self.kids = [exp]

#----------------------------------------------------------
class CompStmt(Stmt):

    def __init__(self, stmts, coord=None):
        super(CompStmt, self).__init__(coord)
        self.kids = stmts

#----------------------------------------------------------
class IfStmt(Stmt):

    def __init__(self, test, true_stmt, false_stmt=None, coord=None):
        super(IfStmt, self).__init__(coord)
        self.kids = [test, true_stmt, false_stmt]

#----------------------------------------------------------
class ForStmt(Stmt):

    def __init__(self, init, test, itr, stmt, coord=None):
        super(ForStmt, self).__init__(coord)
        self.kids = [init, test, itr, stmt]

#----------------------------------------------------------
class WhileStmt(Stmt):

    def __init__(self, test, stmt, coord=None):
        super(WhileStmt, self).__init__(coord)
        self.kids = [test, stmt]

#----------------------------------------------------------
class VarInit(Stmt):

    def __init__(self, var_name, init_exp=None, coord=None):
        super(VarInit, self).__init__(coord)
        self.kids = [var_name, init_exp]

#----------------------------------------------------------
class VarDec(Stmt):

    def __init__(self, type_name, var_inits, isAtomic, coord=None):
        super(VarDec, self).__init__(coord)
        self.isAtomic = isAtomic
        self.kids = [type_name, var_inits]

#----------------------------------------------------------
class ParamDec(Stmt):

    def __init__(self, ty, name, coord=None):
        super(ParamDec, self).__init__(coord)
        self.kids = [ty, name]

#----------------------------------------------------------
class FunDec(Stmt):

    def __init__(self, name, return_type, modifiers, param_decs, body, coord=None):
        super(FunDec, self).__init__(coord)
        self.kids = [name, return_type, modifiers, param_decs, body]

#----------------------------------------------------------
class TransformStmt(Stmt):

    def __init__(self, name, args, stmt, coord=None):
        super(TransformStmt, self).__init__(coord)
        self.kids = [name, args, stmt]


