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
  sys.stdout.write('Successfully parsed %s' % filename)
else:
  sys.stderr.write('*** Errors\n %s' % ' '.join(mparser.lex.errors)

    

