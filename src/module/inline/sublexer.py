#!/usr/bin/env python
'''
Created on Oct 10, 2010

@author: norris
'''

import sys, re
import orio.tool.ply.lex as lex
from orio.main.util.globals import *

class SubroutineLexer:
    def __init__(self):
        self.subroutines = []
        self.substart = 0
        self.insubroutine = False
        self.subre = re.compile(r'\s*subroutine.*[\n\r]?')
        pass
    
    def reset(self, fname=''):
        self.subroutines = []
        self.substart = 0
        self.insubroutine = False        
        pass
    
    states = (
              ('header','exclusive'),
              )
    
    tokens = ('SUBROUTINE','ENDSUBROUTINE','EXECSTMT', 'ID', 'LPAREN', 'RPAREN', 'COMMA')


    def t_SUBROUTINE(self,t):
        r'SUBROUTINE\s*[A-Za-z_][\w_]*'
        self.insubroutine = True
        self.beforestmts = True
        self.substart = t.lexpos
        t.lexer.begin('header')
        return t

    # Any whitespace character: equiv. to set [ \t\n\r\f\v]
    def t_INITIAL_WS(self,t):
        r'[ \t\r\f]'
        pass

    # Primitive way to catch first executable statement
    def t_INITIAL_header_EXECSTMT(self,t):
        r'[\w_]+\s*='
        if self.insubroutine:
            if self.beforestmts: 
                self.substart = t.lexer.lexdata.rfind('\n',0,t.lexpos)
            self.beforestmts = False
        t.lexer.begin('INITIAL')
        
    def t_ENDSUBROUTINE(self,t):
        r'END\s*SUBROUTINE'
        if self.insubroutine:
            self.insubroutine = False
            self.subroutines.append(t.lexer.lexdata[self.substart:t.lexpos])
        return t
    
    # identifiers
    def t_header_ID(self, t):
        r'[A-Za-z_][\w_]*'
        #t.type = reserved_map.get(t.value,'ID')
        return t
    
    
    # delimeters
    t_header_LPAREN           = r'\('
    t_header_RPAREN           = r'\)'
    t_header_COMMA            = r','
    
    def t_INITIAL_header_error(self, t):
        t.lexer.skip(1)
        pass

    
    
    # instantiate lexer
    def build(self,**kwargs):
        self.lexer = lex.lex(object=self, reflags=re.IGNORECASE, **kwargs)  

    # Test it output
    def test(self,data):
        self.lexer.input(data)
        while 1:
            tok = self.lexer.token()
            if not tok: break
            debug(tok,level=5)
        print 'subroutine bodies:'
        for i in self.subroutines:
            print i
        
    def setFileName(self, filename):
        self.filename = filename
        pass
    
#lexer = parse.ply.lex.lex(optimize = 0)
# Main for debugging the lexer:
if __name__ == "__main__":
    import sys
    # Build the lexer and try it out
    Globals().verbose = True
    l = SubroutineLexer()
    l.build(debug=1)           # Build the lexer
    for i in range(1, len(sys.argv)):
        info("About to lex %s" % sys.argv[i])
        f = open(sys.argv[i],"r")
        # Use Intel compiler rules for Fortran file suffix to determine fixed vs free form
        l.reset()
        l.setFileName(sys.argv[i])
        #l.determineFileFormat(sys.argv[i])
        s = f.read()
        f.close()
        # info("Contents of %s: %s" % (sys.argv[i], s))
        if s == '' or s.isspace(): sys.exit(0)
        l.test(s)
        info('Done processing %s.' % sys.argv[i])

