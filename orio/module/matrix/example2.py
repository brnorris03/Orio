#!/usr/bin/env python

# Parse a BTO file

from matrixparser import MParser
import sys

mparser = MParser(debug=0,printToStderr=False)

program = '''
GEMVER
in
  A : column matrix, u1 : vector, u2 : vector, v1 : vector, v2 : vector,
  a : scalar, b : scalar,
  y : vector, z : vector
out
  B : column matrix, x : vector, w : vector
{
  B = A + u1 * v1' + u2 * v2'
  x = b * (B' * y) + z 
  w = a * (B * x)
}
'''
theresult = None


print program
try:
  theresult = mparser.processString(program) 
except:
  pass

 
# Errors are stored in the mparser.lex.errors list 
if theresult and len(mparser.lex.errors) == 0:
  print >>sys.stdout, 'Successfully parsed program.'
else:
  print '*** Errors\n', ' '.join(mparser.lex.errors)

    

