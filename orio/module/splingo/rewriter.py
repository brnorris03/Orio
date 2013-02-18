#==============================================================================
# Transformation unit
#==============================================================================
from orio.module.splingo.ast import *
import orio.module.splingo.parser as parser
import operator

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
    self.st['itrs'] = [{},{},{}]
    self.existVecs = False
    self.existMats = False
    self.applyOnce = {}

  #----------------------------------------------------------------------------
  def transform(self, n):
    '''Apply code transformations'''
    
    # abbreviations
    trav = NodeVisitor()
    
    # todo: containers should be type-parametric and type-inferred with double as default
    def param2ty(conty, ty, toMarkVecs, len=None):
      def f(n):
        if isinstance(n, ParamDec) and n.ty.name == conty:
          self.st[str(n.name)] = {'srcty':conty, 'genty':ty, 'len':len}
          if toMarkVecs:
            self.conLengths = len
            if n.ty.name == 'vector':
              self.existVecs = True
            elif n.ty.name == 'matrix':
              self.existMats = True
              #if isinstance(n.ty, QualIdentExp):
              #  print n.ty.qual
          return ParamDec(IdentExp(ty), n.name)
        else:
          return n
      return f

    def isAssign(n):
      return isinstance(n, ExpStmt) and isinstance(n.exp, BinOpExp) and n.exp.oper == BinOpExp.EQ

    def getForStmt(idx, beg, end, upd, stmt):
      return ForStmt(s2t('exp', idx + ' = ' + beg), s2t('exp', idx + '<' + end), s2t('exp', upd), stmt)

    def expandCons(n):
      if isAssign(n):
        def g(t1):
          '''Add indexing exprs to container refs'''
          if isinstance(t1, IdentExp) and self.st.has_key(t1.name):
            ninfo = self.st[t1.name]
            if ninfo['srcty'] == 'vector':
              if self.existMats and t1.name == n.exp.lhs.name:
                return ArrayRefExp(t1, IdentExp('i2'))
              else:
                self.st['itrs'][0].update({ninfo['len'][0]: 'i1'})
                return ArrayRefExp(t1, IdentExp('i1'))
            elif ninfo['srcty'] == 'matrix':
              self.st['itrs'][0].update({ninfo['len'][0]: 'i1'})
              self.st['itrs'][1].update({ninfo['len'][1]: 'i2'})
              sub = s2t('exp', ninfo['len'][0] + ' * i2 + i1')
              return ArrayRefExp(t1, sub)
            else:
              return t1
          else:
            return t1
        n1 = trav.rewriteBU(g, n)

        dims = reduce(operator.concat, map(dict.keys,   self.st['itrs']), [])
        itrs = reduce(operator.concat, map(dict.values, self.st['itrs']), [])
        decl_itrs = s2t('stmt', 'int ' + ', '.join(itrs) + ';')
        
        if self.existMats:
          if not self.applyOnce.get('cacheLhs', None):
            cachedLhs = None
            def cacheLhs(t2):
              if isAssign(t2):
                cachedLhs = t2.exp.lhs
                t2.exp.lhs = IdentExp('tmp1')
                t2.exp.oper = BinOpExp.EQPLUS
                self.applyOnce['cacheLhs'] = cachedLhs
                return t2
              else:
                return t2
            n1 = trav.rewriteTD(cacheLhs, n1)
          return CompStmt([decl_itrs,
                           s2t('stmt', 'register double tmp1;'),
                           getForStmt(itrs[1], '0', dims[1], itrs[1] + '++',
                                      CompStmt([s2t('stmt', 'tmp1 = 0.0'),
                                                getForStmt(itrs[0], '0', dims[0], itrs[0] + '++', n1),
                                                s2t('stmt', repr(self.applyOnce['cacheLhs']) + ' = tmp1')
                                                ]))
                          ])
        else:
          return CompStmt([decl_itrs,
                           getForStmt(itrs[0], '0', dims[0], itrs[0] + '++', n1)
                          ])
      elif isinstance(n, FunDec):
        # add container length parameters
        lengths = map(lambda x: ParamDec(IdentExp('int'), IdentExp(x)),
                      reduce(operator.concat, map(dict.keys, self.st['itrs']), []))
        n.params = lengths + n.params

        # peel off nested compound stmt in the body
        if len(n.body.stmts) == 1 and isinstance(n.body.stmts[0], CompStmt):
          n.body = n.body.stmts[0]
        return n
      else:
        return n
    
    n1 = trav.rewriteTD(param2ty('scalar', 'double',  False), n)
    n2 = trav.rewriteTD(param2ty('vector', 'double*', True, ['dim1']), n1)
    nf = trav.rewriteTD(param2ty('matrix', 'double*', True, ['dim1','dim2']), n2)
    if self.existVecs or self.existMats:
      nf = trav.rewriteBU(expandCons, nf)

    return nf


