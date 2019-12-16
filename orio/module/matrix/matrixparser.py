#!/usr/bin/env python
'''
Created on Mar 8, 2012

@author: norris
'''

import os,sys
from lexer import *
from parser import *
from orio.main.util.globals import *

class MParser:
  ''' 
  BTO Parser
  '''
  
  def __init__(self, debug=0, outputdir='.', printToStderr=True):
    import parser as matrixparser
    self.parser = matrixparser.setup(debug=debug, outputdir=outputdir)
    self.lex = MatrixLexer()
    self.lex.build(printToStderr=printToStderr, optimize=1, 
                   lextab=os.path.join("MatrixParser.lextab"))
    self.errorlog = []
    self.debug = debug
    self.lex.errors = matrixparser.errors
    
  def processString(self, input=''):
    if input == '' or input.isspace(): 
      return None
    else:
      return self.parser.parse(input, lexer=self.lex.lexer, debug=self.debug)

  def processFile(self, inputfile=''):
    if not os.path.exists(inputfile):
      self.error(0,"Input file not found: %s" % inputfile)
      return None
    else:
      f = open(inputfile,"r")
      s = f.read()
      f.close()

      return self.parser.parse(s, lexer=self.lex.lexer, debug=self.debug)
  

  def error(self, msg):
    self.errorlog.append(msg)
    if printToStderr:
      err(msg)


        
if __name__ == '__main__':
  
  mparser = MParser(debug=0,printToStderr=False)
  
  for i in range(1, len(sys.argv)):
    debug("[parse] About to parse %s" % sys.argv[i], obj=mparser)
    os.system('cat %s' % sys.argv[i])
    theresult = mparser.processFile(sys.argv[i])
    if theresult and len(mparser.lex.errors)==0:
      debug('[parser] Successfully parsed %s' % sys.argv[i], obj=mparser)
    
    debug('All variables and their types:', obj=mparser)
    for key,val in getVars().items():
      debug("%s : %s" % (key,val))

    debug('***Errors\n'+ str(mparser.lex.errors))




