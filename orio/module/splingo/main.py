#!/usr/bin/env python 

import sys
from . import parser, printer, rewriter

#----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

  # process cmd-line switches
  f = open(sys.argv[-1], "r")
  s = f.read()
  f.close()

  # parse
  ast = parser.parse(s)

  # lower
  ast2 = rewriter.Rewriter().transform(ast)
  
  # tune
  
  # pretty-print
  text = printer.Printer().pp(ast2, '')
  print(text)
  f = open(sys.argv[-1] + '.c', 'w')
  f.write(text)
  f.close()
