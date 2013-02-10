#==============================================================================
# Transformation module
#==============================================================================
from orio.module.splingo.ast import *

#------------------------------------------------------------------------------
class Rewriter(object):
  '''Code transformation implementation'''

  def __init__(self):
    '''Instantiate a code transformation object'''
    
    self.st = {}

  #----------------------------------------------------------------------------
  def transform(self, n):
    '''Apply code transformations'''
    
    trav = NodeVisitor()
    
    # todo: containers should be type-parametric and type-inferred with double as default
    def param2ty(conty, ty):
      def f(n):
        if isinstance(n, ParamDec):
          if n.ty.name == conty:
            self.st[n.name] = ty
            return ParamDec(IdentExp(ty), n.name)
          else:
            return n
        else:
          return n
      return f

    def isAssign(n):
      if isinstance(n, ExpStmt) and isinstance(n.exp, BinOpExp) and n.exp.op_type == BinOpExp.EQ:
        return True
      else:
        return False

    def expandVecs():
      def f(n):
        if isAssign(n):
          ivar = IdentExp('i')
          rv = ForStmt(VarDec('int', [BinOpExp(BinOpExp.EQ, ivar, LitExp(LitExp.INT, 0))], False),
                       BinOpExp(BinOpExp.LT, ivar, IdentExp('n')),
                       UnaryExp(UnaryExp.POST_INC, ivar),
                       n
                       )
          return rv
        else:
          return n
      return f
    
    n1 = trav.rewriteTD(param2ty('scalar', 'double'), n)
    n2 = trav.rewriteTD(param2ty('vector', 'double[]'), n1)
    nf = trav.rewriteTD(expandVecs(), n2)

    return nf
