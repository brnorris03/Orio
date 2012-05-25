'''
Created on Mar 8, 2012

@author: norris
'''

import os
from orio.module.matrix.lexer import *

class MatrixParser:
    ''' 
    BTO Parser
    '''
    
    def __init__(self, debug=0, outputdir='.'):
        import orio.module.matrix.parser as matrixparser
        self.parser = matrixparser.setup(debug=debug, outputdir=outputdir)
        self.lex = MatrixLexer()
        self.lex.build(optimize=1, lextab=os.path.join("orio.module.matrix.lextab"))
        
    def printAST(self):
        '''
        Print the parsed AST (mainly used for debugging)
        '''
        
