#!/usr/bin/env python

# Parse a BTO file

from matrixparser import MParser
import sys

mparser = MParser(debug=0,printToStderr=False)
filename = 'test/vadd.m'
theresult = None

try:
  theresult = mparser.processFile(filename) 
except:
  pass

 
# Errors are stored in the mparser.lex.errors list 
if len(mparser.lex.errors) == 0:
  print >>sys.stdout, 'Successfully parsed %s' % filename
else:
  print '*** Errors\n', ' '.join(mparser.lex.errors)

    

