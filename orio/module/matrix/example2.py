#!/usr/bin/env python

# Parse a BTO file

from mparser import MParser
import sys

mparser = MParser(yacc_debug=False,printToStderr=False)

program = '''
GEMVER
in
  A : matrix(column), u1 : vector(column), u2 : vector(column), 
  v1 : vector(column), v2 : vector(column),
  a : scalar, b : scalar,
  y : vector(column), z : vector(column)
out
  B : matrix(column), x : vector(column), w : vector(column)
{
  B = A + u1 * v1' + u2 * v2'
  x = b * (B' * y) + z
  w = a * (B * x)
}
'''
theresult = None


print program
try:
  theresult = mparser._processString(program)
except:
  pass

 
# Errors are stored in the mparser.lex.errors list 
if theresult and len(mparser.mlex.errors) == 0:
  print('Successfully parsed program.')
else:
  print ('*** Errors\n', ' '.join(mparser.mlex.errors))

    

