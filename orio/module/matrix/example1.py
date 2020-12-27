#!/usr/bin/env python

# Parse a BTO file

from .mparser import MParser
import sys

mparser = MParser(yacc_debug=False, printToStderr=False)
filename = 'example/vadd.m'
theresult = None

try:
  theresult = mparser._processFile(filename)
except:
  pass

 
# Errors are stored in the mparser.lex.errors list 
if len(mparser.mlex.errors) == 0:
  print(('Successfully parsed %s' % filename))
else:
  print(('*** Errors\n', ' '.join(mparser.mlex.errors)))

    

