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
        self.incontinuation = False
        self.lineno = 0
        self.subre = re.compile(r'\s*subroutine.*[\n\r]?')
        pass
    
    def reset(self, fname=''):
        self.subroutines = []
        self.substart = 0
        self.insubroutine = False       
        self.lineno = 0 
        pass
    
    states = (
              ('header','exclusive'),
              ('body','inclusive')
              )
    
    tokens = ('SUBROUTINE','ENDSUBROUTINE','EXECSTMT', 'ID', 'LPAREN', 'RPAREN', 'COMMA', 'LINECOMMENT')

    # reserved words
    keywords = ('INTEGER', 'REAL', 'COMPLEX', 'CHARACTER', 'LOGICAL',
                'ABSTRACT', 'ALLOCATABLE', 'ALLOCATE', 'ASSIGNMENT', 
                'ASSOCIATE', 'ASYNCHRONOUS', 'BACKSPACE', 'BLOCK',
                'BLOCKDATA', 'CALL', 'CASE', 'CLASS', 'CLOSE', 'COMMON',
                'CONTAINS', 'CONTINUE', 'CYCLE', 'DATA', 'DEFAULT',
                'DEALLOCATE', 'DEFERRED', 'DO', 'DOUBLE', 'DOUBLEPRECISION',
                'DOUBLECOMPLEX', 'ELEMENTAL', 'ELSE', 'ELSEIF', 'ELSEWHERE', 
                'ENTRY', 'ENUM', 'ENUMERATOR', 'EQUIVALENCE', 'EXIT', 
                'EXTENDS', 'EXTERNAL', 'FILE', 'FINAL', 'FLUSH', 'FORALL', 
                'FORMAT', 'FORMATTED', 'FUNCTION', 'GENERIC', 'GO', 
                'GOTO', 'IF', 'IMPLICIT', 'IMPORT', 'IN', 'INOUT', 
                'INTENT', 'INTERFACE', 'INTRINSIC', 'INQUIRE', 'MODULE',
                'NAMELIST', 'NONE', 'NON_INTRINSIC', 'NON_OVERRIDABLE', 
                'NOPASS', 'NULLIFY', 'ONLY', 'OPEN', 'OPERATOR', 
                'OPTIONAL', 'OUT', 'PARAMETER', 'PASS', 'PAUSE',
                'POINTER', 'PRECISION', 'PRINT', 'PRIVATE', 'PROCEDURE', 'PROGRAM',
                'PROTECTED', 'PUBLIC', 'PURE', 'READ', 'RECURSIVE',
                'RESULT', 'RETURN', 'REWIND', 'SAVE', 'SELECT',
                'SELECTCASE', 'SELECTTYPE', 'SEQUENCE', 'STOP',
                'SUBROUTINE', 'TARGET', 'THEN', 'TO', 'TYPE', 
                'UNFORMATTED', 'USE', 'VALUE', 'VOLATILE', 'WAIT', 
                'WHERE', 'WHILE', 'WRITE', 
                'ENDASSOCIATE', 'ENDBLOCK', 'ENDBLOCKDATA', 'ENDDO',
                'ENDENUM', 'ENDFORALL', 'ENDFILE', 'ENDFUNCTION',
                'ENDIF', 'ENDINTERFACE', 'ENDMODULE', 'ENDPROGRAM', 
                'ENDSELECT', 'ENDSUBROUTINE', 'ENDTYPE', 'ENDWHERE',
                'END',                
                'DIMENSION', 'KIND', 'LEN', 'BIND')
    
    # Any whitespace character: equiv. to set [ \t\n\r\f\v]
    def t_INITIAL_body_WS(self,t):
        r'[ \t\r\f]'
        pass
    
    def t_INITIAL_header_LINECOMMENT(self,t):
        r'![^\n\r]*'
        pass

    # count newlines
    def t_INITIAL_header_NEWLINE(self, t):
        r'\n+'
        self.lineno += t.value.count('\n')
        
    def t_SUBROUTINE(self,t):
        r'SUBROUTINE'
        self.insubroutine = True
        self.beforestmts = True
        self.substart = t.lexpos
        t.lexer.begin('header')
        return t

    # Primitive way to catch first executable statement
    def t_INITIAL_header_EXECSTMT(self,t):
        r'[\w_]+\s*='
        first = False
        if self.insubroutine:
            if self.beforestmts: 
                self.substart = t.lexer.lexdata.rfind('\n',0,t.lexpos)
                first = True
            self.beforestmts = False
        t.lexer.begin('body')
        
    def t_INITIAL_body_ENDSUBROUTINE(self,t):
        r'END\s*SUBROUTINE'
        if self.insubroutine:
            self.insubroutine = False
            t.value = (t.lexer.lexdata[self.substart:t.lexpos],(self.substart,t.lexpos))
            self.subroutines.append(t.value)
        t.lexer.begin('INITIAL')
        return t
    
    # identifiers
    def t_header_body_ID(self, t):
        r'[A-Za-z_][\w_]*'
        #t.type = reserved_map.get(t.value,'ID')
        if not t.value.upper() in self.keywords:
            return t
    
    # Ignore continuation characters
    def t_header_CONTINUE_CHAR(self, t):
        r'&'
        pass   # eat it

    
    # delimeters
    t_header_LPAREN           = r'\('
    t_header_COMMA            = r','
    
    def t_header_RPAREN(self,t):
        r'\)'
        # after capturing arguments, revert to the default sloppy lexer
        t.lexer.begin('INITIAL')        
        return t
    
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

