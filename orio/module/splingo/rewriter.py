#==============================================================================
# Transformation unit
#==============================================================================
from orio.module.splingo.ast import *
import orio.module.splingo.parser as parser

#------------------------------------------------------------------------------
def s2t(sym, s):
  parser.start = sym
  return parser.parse(s)

#------------------------------------------------------------------------------
class Rewriter(object):
  '''Code transformation implementation'''

  def __init__(self):
    '''Instantiate a code transformation object'''
    
    self.st = {} # symbol table
    self.existVecs = False

  #----------------------------------------------------------------------------
  def transform(self, n):
    '''Apply code transformations'''
    
    # abbreviations
    trav = NodeVisitor()
    
    # todo: containers should be type-parametric and type-inferred with double as default
    def param2ty(conty, ty, toMarkVecs):
      def f(n):
        if isinstance(n, ParamDec):
          if n.kids[0].kids[0] == conty:
            self.st[str(n.kids[1])] = ty
            if toMarkVecs and n.kids[0].kids[0] == 'vector':
              self.existVecs = True
            return ParamDec(IdentExp(ty), n.kids[1])
          else:
            return n
        else:
          return n
      return f

    def isAssign(n):
      if isinstance(n, ExpStmt) and isinstance(n.kids[0], BinOpExp) and n.kids[0].kids[0] == BinOpExp.EQ:
        return True
      else:
        return False

    def expandVecs(n):
      if isAssign(n):
        def g(t1): # add indexing exprs to vector refs
          if isinstance(t1, IdentExp) and self.st.has_key(t1.kids[0]) and self.st[t1.kids[0]] == 'double*':
            return ArrayRefExp(t1, IdentExp('i'))
          else:
            return t1
        n1 = trav.rewriteBU(g, n)
        return CompStmt([s2t('stmt', 'int i;'),
                        ForStmt(s2t('exp', 'i = 0'), s2t('exp', 'i<n'), s2t('exp', 'i++'), n1)
                        ])
      elif isinstance(n, FunDec) and self.existVecs: # add container length parameter
        n.kids[3] = [ParamDec(IdentExp('int'), IdentExp('n'))] + n.kids[3]

        if isinstance(n.kids[4].kids[0].kids, list): # peel off nested compound stmt in the body
          n.kids[4].kids = n.kids[4].kids[0].kids
        return n
      else:
        return n
    
    n1 = trav.rewriteTD(param2ty('scalar', 'double',  False), n)
    nf = trav.rewriteTD(param2ty('vector', 'double*', True), n1)
    if self.existVecs:
      nf = trav.rewriteBU(expandVecs, nf)

    return nf
