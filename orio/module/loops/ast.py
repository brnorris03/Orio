import sys

#----------------------------------------------------------------------------------------------------------------------
class Node(object):
    ''' Abstract base class of AST nodes. '''

    def __init__(self, coord=None):
        '''Create an abstract syntax tree node'''
        self.coord = coord
        self.kids = []
        
    def __repr__(self):
        '''Return a string representation for this AST object'''
        import orio.module.loops.printer as printer
        return printer.CodeGen_C().generate(self)

    def __str__(self):
        '''Return a string representation of this object'''
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
        kids = getattr(n, 'kids', False)
        if kids:
            nkids = []
            for k in n.kids:
                nkids += [self.rewriteTD(r, k)]
            nn.kids = nkids
        elif isinstance(nn, list):
            nn = [self.rewriteTD(r, e) for e in nn]
        return nn

    def rewriteBU(self, r, n):
        kids = getattr(n, 'kids', False)
        if kids:
            nkids = []
            for k in n.kids:
                nkids += [self.rewriteBU(r, k)]
            n.kids = nkids
        nn = r(n)
        return nn

    def collectTD(self, f, n):
        acc = f(n)
        kids = getattr(n, 'kids', [])
        for k in kids:
            acc += self.collectTD(f, k)
        return acc
#----------------------------------------------------------------------------------------------------------------------



#----------------------------------------------------------------------------------------------------------------------
class Comment(Node):

    def __init__(self, comment, coord=None):
        super(Comment, self).__init__(coord)
        self.text = comment
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
    ARRAY = 5
    
    def __init__(self, lit_type, val, coord=None):
        super(LitExp, self).__init__(coord)
        self.lit_type = lit_type
        self.val = val
        self.kids = [lit_type, val]

#----------------------------------------------------------
class IdentExp(Exp):

    def __init__(self, name, coord=None):
        super(IdentExp, self).__init__(coord)
        self.name = name

#----------------------------------------------------------
class ArrayRefExp(Exp):

    def __init__(self, exp, sub, coord=None):
        super(ArrayRefExp, self).__init__(coord)
        self.exp = exp # head
        self.sub = sub # offset
        self.kids = [exp, sub]

#----------------------------------------------------------
class CallExp(Exp):

    def __init__(self, exp, args, coord=None):
        super(CallExp, self).__init__(coord)
        self.exp = exp   # fname
        self.args = args # fargs
        self.kids = [exp, args]

#----------------------------------------------------------
class CastExp(Exp):

    def __init__(self, castto, exp, coord=None):
        super(CastExp, self).__init__(coord)
        self.castto = castto
        self.exp = exp
        self.kids = [castto, exp]

#----------------------------------------------------------
class UnaryExp(Exp):
    PLUS      = 1
    MINUS     = 2
    LNOT      = 3
    BNOT      = 4
    SIZEOF    = 5
    PRE_INC   = 10
    PRE_DEC   = 11
    POST_INC  = 12
    POST_DEC  = 13
    DEREF     = 20
    ADDRESSOF = 21

    def __init__(self, op_type, exp, coord=None):
        super(UnaryExp, self).__init__(coord)
        self.op_type = op_type
        self.exp = exp
        self.kids = [op_type, exp]

#----------------------------------------------------------
class BinOpExp(Exp):
    PLUS    = 1
    MINUS   = 2
    MULT    = 3
    DIV     = 4
    MOD     = 5
    LT      = 10
    GT      = 11
    LE      = 12
    GE      = 13
    EE      = 14
    NE      = 15
    LOR     = 20
    LAND    = 21
    EQ      = 30
    PLUSEQ  = 31
    MINUSEQ = 32
    MULTEQ  = 33
    DIVEQ   = 34
    MODEQ   = 35
    COMMA   = 40
    BOR     = 50
    BAND    = 51
    BXOR    = 52
    BSHL    = 53
    BSHR    = 54
    BSHLEQ  = 55
    BSHREQ  = 56
    BANDEQ  = 57
    BXOREQ  = 58
    BOREQ   = 59
    DOT     = 60
    SELECT  = 61

    def __init__(self, op_type, lhs, rhs, coord=None):
        super(BinOpExp, self).__init__(coord)
        self.op_type = op_type
        self.lhs = lhs
        self.rhs = rhs
        self.kids = [op_type, lhs, rhs]

#----------------------------------------------------------
class TernaryExp(Exp):

    def __init__(self, test, true_exp, false_exp, coord=None):
        super(IfStmt, self).__init__(coord)
        self.test = test
        self.true_exp = true_exp
        self.false_exp = false_exp
        self.kids = [test, true_exp, false_exp]

#----------------------------------------------------------
class ParenExp(Exp):

    def __init__(self, exp, coord=None):
        super(ParenExp, self).__init__(coord)
        self.exp = exp
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
        self.exp = exp
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
        self.test = test
        self.true_stmt = true_stmt
        self.false_stmt = false_stmt
        self.kids = [test, true_stmt, false_stmt]

#----------------------------------------------------------
class ForStmt(Stmt):

    def __init__(self, init, test, itr, stmt, coord=None):
        super(ForStmt, self).__init__(coord)
        self.init = init
        self.test = test
        self.iter = itr
        self.stmt = stmt
        self.kids = [init, test, itr, stmt]

#----------------------------------------------------------
class WhileStmt(Stmt):

    def __init__(self, test, stmt, coord=None):
        super(WhileStmt, self).__init__(coord)
        self.test = test
        self.stmt = stmt
        self.kids = [test, stmt]

#----------------------------------------------------------
class VarDec(Stmt):

    def __init__(self, type_name, var_inits, isnested=False, coord=None):
        super(VarDec, self).__init__(coord)
        self.type_name = type_name
        self.var_inits = var_inits
        self.isnested = isnested
        self.kids = [type_name, var_inits]

#----------------------------------------------------------
class ParamDec(Stmt):

    def __init__(self, ty, name, coord=None):
        super(ParamDec, self).__init__(coord)
        self.ty = ty
        self.name = name
        self.kids = [ty, name]

#----------------------------------------------------------
class FunDec(Stmt):

    def __init__(self, name, return_type, modifiers, param_decs, body, coord=None):
        super(FunDec, self).__init__(coord)
        self.kids = [name, return_type, modifiers, param_decs, body]
        self.name = self.kids[0]
        self.return_type = self.kids[1]
        self.modifiers = self.kids[2]
        self.params = self.kids[3]
        self.body = self.kids[4]

#----------------------------------------------------------
class TransformStmt(Stmt):

    def __init__(self, name, args, stmt, coord=None):
        super(TransformStmt, self).__init__(coord)
        self.name = name
        self.args = args
        self.stmt = stmt
        self.kids = [name, args, stmt]

#----------------------------------------------------------
class Pragma(Stmt):

    def __init__(self, pstring, coord=None):
        super(Pragma, self).__init__(coord)
        self.pstring = pstring
        self.kids = [pstring]
#----------------------------------------------------------------------------------------------------------------------


