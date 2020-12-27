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
        
    def __repr__(self):
        '''Return a string representation for this AST object'''
        import orio.module.splingo.printer as printer
        return printer.Printer().pp(self)

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

#----------------------------------------------------------------------------------------------------------------------



#----------------------------------------------------------------------------------------------------------------------
class NodeVisitor(object):

    def rewriteTD(self, r, n):
        nn = r(n)
        d = getattr(nn, '__dict__', False)
        if d:
          for k in list(d.keys()):
            d[k] = self.rewriteTD(r, d[k])
        elif isinstance(nn,list):
          nn = [self.rewriteTD(r, e) for e in nn]
        return nn

    def collectTD(self, f, n):
        acc = f(n)
        d = getattr(n, '__dict__', [])
        for k in list(d.keys()):
          acc += self.collectTD(f, d[k])
        return acc

    def rewriteBU(self, r, n):
        d = getattr(n, '__dict__', False)
        if d:
          dd = {}
          for k in list(d.keys()):
            dd[k] = self.rewriteBU(r, d[k])
          n.__dict__ = dd
        elif isinstance(n, list):
            return [self.rewriteBU(r, e) for e in n]
        return r(n)
#----------------------------------------------------------------------------------------------------------------------



#----------------------------------------------------------------------------------------------------------------------
class Comment(Node):

    def __init__(self, comment, coord=None):
        super(Comment, self).__init__(coord)
        self.comment = comment

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
        self.lit_type = lit_type
        self.val = val

#----------------------------------------------------------
class IdentExp(Exp):

    def __init__(self, name, coord=None):
        super(IdentExp, self).__init__(coord)
        self.name = name

#----------------------------------------------------------
class QualIdentExp(Exp):

    def __init__(self, name, ident, coord=None):
        super(QualIdentExp, self).__init__(coord)
        self.name = name
        self.qual = ident

#----------------------------------------------------------
class ArrayRefExp(Exp):

    def __init__(self, exp, sub, coord=None):
        super(ArrayRefExp, self).__init__(coord)
        self.exp = exp
        self.sub = sub

#----------------------------------------------------------
class CallExp(Exp):

    def __init__(self, exp, args, coord=None):
        super(CallExp, self).__init__(coord)
        self.exp  = exp
        self.args = args

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
        self.oper = op_type
        self.exp  = exp

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
    COMMA = 40

    def __init__(self, op_type, lhs, rhs, coord=None):
        super(BinOpExp, self).__init__(coord)
        self.oper = op_type
        self.lhs  = lhs
        self.rhs  = rhs

#----------------------------------------------------------
class ParenExp(Exp):

    def __init__(self, exp, coord=None):
        super(ParenExp, self).__init__(coord)
        self.exp = exp



#----------------------------------------------------------------------------------------------------------------------
class Stmt(Node):

    def __init__(self, coord=None):
        super(Stmt, self).__init__(coord)
        self.label = None

#----------------------------------------------------------
class ExpStmt(Stmt):

    def __init__(self, exp, coord=None):
        super(ExpStmt, self).__init__(coord)
        self.exp  = exp

#----------------------------------------------------------
class CompStmt(Stmt):

    def __init__(self, stmts, coord=None):
        super(CompStmt, self).__init__(coord)
        self.stmts = stmts

#----------------------------------------------------------
class IfStmt(Stmt):

    def __init__(self, test, true_stmt, false_stmt=None, coord=None):
        super(IfStmt, self).__init__(coord)
        self.test = test
        self.then_s = true_stmt
        self.else_s = false_stmt

#----------------------------------------------------------
class ForStmt(Stmt):

    def __init__(self, init, test, itr, stmt, coord=None):
        super(ForStmt, self).__init__(coord)
        self.init = init
        self.test = test
        self.itr  = itr
        self.stmt = stmt

#----------------------------------------------------------
class WhileStmt(Stmt):

    def __init__(self, test, stmt, coord=None):
        super(WhileStmt, self).__init__(coord)
        self.test = test
        self.stmt = stmt

#----------------------------------------------------------
class VarInit(Stmt):

    def __init__(self, var_name, init_exp=None, coord=None):
        super(VarInit, self).__init__(coord)
        self.var_name = var_name
        self.init_exp = init_exp

#----------------------------------------------------------
class VarDec(Stmt):

    def __init__(self, type_name, var_inits, isAtomic, quals=[], coord=None):
        super(VarDec, self).__init__(coord)
        self.isAtomic = isAtomic
        self.type_name = type_name
        self.var_inits = var_inits
        self.quals = quals

#----------------------------------------------------------
class ParamDec(Stmt):

    def __init__(self, ty, name, coord=None):
        super(ParamDec, self).__init__(coord)
        self.ty   = ty
        self.name = name

#----------------------------------------------------------
class FunDec(Stmt):

    def __init__(self, name, return_type, modifiers, param_decs, body, coord=None):
        super(FunDec, self).__init__(coord)
        self.name   = name
        self.rtype  = return_type
        self.quals  = modifiers
        self.params = param_decs
        self.body   = body

#----------------------------------------------------------
class TransformStmt(Stmt):

    def __init__(self, name, args, stmt, coord=None):
        super(TransformStmt, self).__init__(coord)
        self.name = name
        self.args = args
        self.stmt = stmt


