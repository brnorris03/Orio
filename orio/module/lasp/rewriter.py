#==============================================================================
# Transformation module
#==============================================================================
import orio.module.lasp.ast as ast

#------------------------------------------------------------------------------
class Rewriter(object):
  '''Code transformation implementation'''

  def __init__(self):
    '''Instantiate a code transformation object'''

  #----------------------------------------------------------------------------
  def transform(self, n):
    '''Apply code transformations'''
    
    scalar2double = lambda n: ast.ParamDec(ast.IdentExp('double'), n.name)\
                              if (isinstance(n, ast.ParamDec) and n.ty.name == 'scalar')\
                              else n
    nn = ast.NodeVisitor().rewriteTD(scalar2double, n)
    return nn
