'''
Created on Mar 8, 2012

@author: norris
'''

import sys
from orio.module.matrix.visitors.depthfirstvisitor import DepthFirstVisitor

class Printer(DepthFirstVisitor):
    ''' 
    Prints AST to stdout 
    '''
    
    def __init__(self, tab='    ', outstream = sys.stdout, initialIndent=0, commentExcludeList=[]):
        self.tab = tab
        self.indentSize = initialIndent
        self.output = outstream
        self.packages = []
        self.declOnly = [0]
        self.commentExcludes = commentExcludeList
        DepthFirstVisitor.__init__(self)
