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
        self.lexer = None
        self.subre = re.compile(r'\s*subroutine.*[\n\r]?')
        self.alphanum = re.compile(r'\w*')
        self.subroutines = []
        self.scopes = []
        self.substart = -1
        self.ininterface = False
        self.insubroutine = False
        self.infunction = False
        self.incoutinuation = False     
        self.beforestmts = True
        self.nestedsubroutine = 0
        self.nestedinterface = 0
        self.lineno = 0 
        self.reset()
        pass
    
    def reset(self, fname=''):
        self.setFileName(fname)
        if self.lexer: self.lexer.lineno = 0
        self.subroutines = []
        self.scopes = []
        self.substart = -1
        self.ininterface = False
        self.insubroutine = False
        self.infunction = False
        self.incoutinuation = False     
        self.beforestmts = True
        self.nestedsubroutine = 0
        self.nestedinterface = 0
        self.lineno = 0 
        if self.lexer: self.lexer.begin('INITIAL') # Revert to the initial state
        pass
    
    states = (
              ('header','exclusive'),
              ('declarations','exclusive'),
              ('body','inclusive'),
              ('interface','exclusive'),
              ('sstring', 'exclusive'),
              ('dstring', 'exclusive'),
              )

    keywords = ['INTEGER', 'REAL', 'COMPLEX', 'CHARACTER', 'LOGICAL',
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
                'END', 'IMPLICITNONE',        
                'DIMENSION', 'KIND', 'LEN', 'BIND']
    
    # reserved words
    #global reserved
    reserved = {}
    for r in keywords:
        reserved[r.lower()] = r
        
    tokens = keywords + [\
        # literals (identifier, integer constant, float constant, string constant)
        'ID', 'DIGIT_STRING', 'FCONST', 'SCONST_D', 'SCONST_S', 
        'ICONST', 'PSCONST_D', 'PSCONST_S',       # partial strings
        'BCONST', 'HCONST_S', 'HCONST_D', 'OCONST_S', 'OCONST_D',

    
        # operators (+,-,*,/,%,||,&&,!,<,<=,>,>=,==,!=)
        'PLUS', 'MINUS', 'TIMES', 'DIVIDE', 'MOD',
        'OR', 'AND', 'NOT', 'TRUE', 'FALSE', 'EQV', 'NEQV',
        'LT', 'LE', 'GT', 'GE', 'EQ', 'NE', 'EQ_EQ', 'GREATERTHAN_EQ',
        'LESSTHAN_EQ', 'SLASH_EQ', 'EQ_GT', 'LESSTHAN', 'GREATERTHAN',
        'DEFINED_UNARY_OP',
    
        # assignment (=, *=, /=, %=, +=, -=)
        'EQUALS', 'ASSIGN',
    
        # delimeters ( ) [ ] { } , ; :
        'LPAREN', 'RPAREN',
        'LBRACKET', 'RBRACKET',
        'LBRACE', 'RBRACE',
        'COMMA', 'SEMICOLON', 'COLON', 'PERIOD', 
        
        # Other
        'WS','COLON_COLON', 'SLASH_SLASH', 'UNDERSCORE',
        'CONTINUE_CHAR', 'LINECOMMENT', 'PREPROCESSOR',
        'MISC_CHAR', 
        'EXECSTMT'
        ]

#                 'COLON', 'COLON_COLON', 'COMMA', 'CONTINUATION', 
#                 'SCONST_S', 'SCONST_D', 'PSCONST_S', 'PSCONST_D',
#                 'DOUBLEQUOTE',
#                 'EQ', 'EQ_GT', 'EXECSTMT', 
#                 'ID', 'LPAREN', 'LINECOMMENT', 
#                 'QUOTE', 'RPAREN', 'PREPROCESSOR', 
#                 'SEMICOLON', 'SLASH']

    # delimeters
    t_header_declarations_body_LPAREN       = r'\('
    t_header_declarations_COMMA             = r','
    t_INITIAL_declarations_header_SEMICOLON = r';'
    t_interface_COLON_COLON                 = r'::'
    
    # operators
    t_declarations_body_PLUS                = r'\+'
    t_declarations_body_MINUS               = r'-'
    t_declarations_body_TIMES               = r'\*'
    t_declarations_body_DIVIDE              = r'/'
    t_declarations_body_MOD                 = r'%'
    t_declarations_body_SLASH_SLASH         = r'//'     # concatenation of char values
    
    
    # relational operators
    t_declarations_body_LESSTHAN            = r'<'
    t_declarations_body_LESSTHAN_EQ         = r'<='
    t_declarations_body_GREATERTHAN         = r'>'
    t_declarations_body_GREATERTHAN_EQ      = r'>='
    t_declarations_body_EQ_GT               = r'=>'
    t_declarations_body_EQ_EQ               = r'=='
    t_declarations_body_SLASH_EQ            = r'/='
    
    t_declarations_body_EQ                  = r'\.EQ\.'
    t_declarations_body_NE                  = r'\.NE\.'
    t_declarations_body_LT                  = r'\.LT\.'
    t_declarations_body_LE                  = r'\.LE\.'
    t_declarations_body_GT                  = r'\.GT\.'
    t_declarations_body_GE                  = r'\.GE\.'
    
    # R428 .TRUE.[_kind-param] or .FALSE.[_kind-param]
    t_declarations_body_TRUE             = r'\.TRUE\.(_([A-Za-z]\w*))?'
    t_declarations_body_FALSE            = r'\.FALSE\.(_([A-Za-z]\w*))?'
    
    t_declarations_body_OR              = r'\.OR\.'
    t_declarations_body_AND             = r'\.AND\.'
    t_declarations_body_NOT             = r'\.NOT\.'
    t_declarations_body_EQV             = r'\.EQV\.'
    t_declarations_body_NEQV            = r'\.NEQV\.'
    
    
    # assignment operators
    t_declarations_body_EQUALS           = r'='
    
    # defined unary operator; R703
    t_declarations_body_DEFINED_UNARY_OP = r'\.[A-Za-z]+\.'
    
    # delimeters
    t_declarations_body_LPAREN           = r'\('
    t_declarations_body_RPAREN           = r'\)'
    t_declarations_body_LBRACKET         = r'\['
    t_declarations_body_RBRACKET         = r'\]'
    t_declarations_body_LBRACE           = r'\{'
    t_declarations_body_RBRACE           = r'\}'
    t_declarations_body_COMMA            = r','
    t_declarations_body_COLON            = r':'
    t_declarations_body_COLON_COLON      = r'::'
    t_declarations_body_PERIOD           = r'\.'
    
    def t_INITIAL_declarations_bpody_SLASH(self, t):
        r'/'
        pass
        
    def t_PROGRAM(self, t):
        r'PROGRAM'
        t.lexer.begin('declarations')
        return t

    def t_MODULE(self, t):
        r'MODULE'
        t.lexer.begin('declarations')
        return t
   
    # Any whitespace character: equiv. to set [ \t\n\r\f\v]
    def t_INITIAL_declarations_body_WS(self,t):
        r'[ \t\r\f]'
        pass
    
    # Skip preprocessor lines (remember that we are parsing an already preprocessed file)
    def t_INITIAL_declarations_interface_body_PREPROCESSOR(self, t):
        r'\#\w+[^\n\r]*'
        pass
    
    #------------ String literals --------------------------
    
    # string literal (with single quotes); R427
    #t_SCONST_S   = r'\'([^\\\n]|(\\.))*?\''
    def t_INITIAL_header_declarations_body_SCONST_S(self,t):
        r'(([A-Za-z]\w*)_)?\'([^\\\n]|(\\.))*\''
        pass
    
    # string literal (with double quotes); R427
    # char-literal-constant is [kind-param]'[rep-char]...' or [kind-param]"[rep-char]..."
    def t_INITIAL_header_declarations_body_SCONST_D(self,t):
        r'(([A-Za-z]\w*)_)?\"([^\\\n]|(\\.))*\"'
        pass
    
    # Partial double-quoted string literal (continued on a different line)
    def t_INITIAL_header_declarations_body_PSCONST_D(self, t):                   
        r'(([A-Za-z]\w*)_)?\"([^\\\n\"]|(\\.))*&\s*\n'
        t.lexer.lineno += 1
        t.value = t.value.strip().strip('&').strip('"')
        t.lexer.push_state('dstring')
        pass
    
    # The end of a partial double-quoted string literal (continued from a previous line)
    def t_dstring_PSCONST_D(self, t):
        r'\s*&?([^\\\n]|(\\.))*\"'
        t.value = t.value.strip().strip('&').strip('"')
        t.lexer.pop_state()
        pass
    
    # partial single-quoted string literal (continued on another line)
    def t_INITIAL_header_declarations_body_PSCONST_S(self, t):
        r'(([A-Za-z]\w*)_)?\'([^\\\n\']|(\\.))*&\s*\n'
        t.lexer.lineno += 1
        t.value = t.value.strip().strip('&').strip("'")
        t.lexer.push_state('sstring')
        pass
    
    # the end of a partial single-quoted string literal (continued from another line)
    def t_sstring_PSCONST_S(self, t):
        r'\s*&?([^\\\n]|(\\.))*\''
        t.value = t.value.strip().strip('&').strip("'")
        t.lexer.pop_state()
        pass

    #------- end string literals ---------------------
    
    def t_INITIAL_interface_SUBROUTINE(self,t):
        r'SUBROUTINE'
        self.insubroutine = True
        self.beforestmts = True
        t.lexer.begin('header')
        return t
    
    def t_declarations_body_SUBROUTINE(self,t):
        r'SUBROUTINE'
        self.nestedsubroutine += 1
        pass

    def t_INITIAL_interface_FUNCTION(self,t):
        r'FUNCTION'
        self.infunction = True
        self.beforestmts = True
        t.lexer.begin('header')
        return t
    
    def t_declarations_body_FUNCTION(self,t):
        r'FUNCTION'
        self.nestedsubroutine += 1
        pass

    def t_INITIAL_header_declarations_body_interface_ENDSUBROUTINE(self,t):
        r'END\s*SUBROUTINE'
        if self.nestedsubroutine == 0:
            if self.insubroutine:
                if t.lexpos >= 0:
                    t.value = (t.lexer.lexdata[self.substart:t.lexpos],(self.substart,t.lexpos))
                else:
                    # This means that the subroutine contained no executable statements
                    t.value = ('',(t.lexpos,t.lexpos))
                self.subroutines.append(t.value)
            self.insubroutine = False
            self.substart = -1
            self.beforestmts = True
            if not self.ininterface:
                t.lexer.begin('INITIAL')
            else: 
                t.lexer.begin('interface')
            return t
        else:
            self.nestedsubroutine -= 1
        pass
    
    def t_INITIAL_declarations_body_interface_ENDFUNCTION(self,t):
        r'END\s*FUNCTION'
        if self.nestedsubroutine == 0:
            if self.infunction:
                if t.lexpos >= 0:
                    t.value = (t.lexer.lexdata[self.substart:t.lexpos],(self.substart,t.lexpos))
                else:
                    # This means that the subroutine contained no executable statements
                    t.value = ('',(t.lexpos,t.lexpos))
                self.subroutines.append(t.value)
            self.infunction = False
            self.substart = -1
            self.beforestmts = True
            if not self.ininterface:
                t.lexer.begin('INITIAL')
            else:
                t.lexer.begin('interface')
            return t
        else:
            self.nestedsubroutine -= 1
        pass    
    
    def t_interface_ENDINTERFACE(self,t):
        r'END\s*INTERFACE'
        if self.nestedinterface == 0:
            self.ininterface = False
            t.lexer.begin('declarations')
            print 'ENTERING DECLARATIONS MODE'
        else:
            self.nestedinterface -= 1
        print 'END INTERFACE',t.lexer.lineno, t.value, self.nestedinterface
        return t
    
    def t_INITIAL_declarations_interface_IMPLICITNONE(self,t):
        r'IMPLICIT\s*NONE'
        return t
    
    def t_INITIAL_declarations_interface_IMPLICIT(self, t):
        r'IMPLICIT'
        return t

    def t_INITIAL_declarations_body_ENDPROGRAM(self, t):
        r'END\s*PROGRAM'
        t.lexer.begin('INITIAL')
        return t

    def t_INITIAL_declarations_body_ENDMODULE(self, t):
        r'END\s*MODULE'
        t.lexer.begin('INITIAL')
        return t

    # Primitive way to catch first executable statement
    def t_INITIAL_declarations_interface_EXECSTMT(self,t):
        r'if\s*\(|do\s*while\s*\(|call\s+|[A-Za-z_]+\w*\s*='
        first = False
        if self.insubroutine or self.infunction:
            bol,line = self.getLine(t)      # beginning of line position, and the line
            #print 'checking whether executable:', bol, line.strip(), self.beforestmts,  line.strip().find('::')
            if self.beforestmts and not self.incontinuation and len(line) and line.strip().find('::') == -1:
                self.substart = bol
                first = True
                self.beforestmts = False
        if first: 
            #print 'ENTERING BODY'
            t.lexer.begin('body')
        pass
              
    def t_declarations_interface_ID(self, t):
        r'[A-Za-z_][\w_]*'
        #t.type = self.reserved.get(t.value,'ID')
        if t.value.lower() == 'interface':
            self.handleInterface(t)
            t.lexer.begin('interface')
        #if t.value.lower() in ['RECURSIVE','PURE','ELEMENTAL']:
        t.type = self.reserved.get(t.value.lower(),'ID')
        return t
    
    def handleInterface(self, t):
        t.type = 'INTERFACE'
        self.ininterface = True
        self.insubroutine = False
        self.infunction = False
     

    def t_interface_FUNCTION(self,t):
        r'FUNCTION'
        pass
    
    def t_INITIAL_declarations_INCLUDE(self, t):
        r'INCLUDE\s+[^\n\r]*'
        pass
    
    
    def t_INITIAL_header_declarations_LINECOMMENT(self,t):
        r'![^\n\r]*'
        pass

    # count newlines
    def t_INITIAL_header_declarations_interface_body_NEWLINE(self, t):
        r'\n+'
        self.incontinuation = False
        t.lexer.lineno += len(t.value)
    
    def t_header_RPAREN(self,t):
        r'\)'
        # after capturing arguments, revert to the default sloppy lexer
        t.lexer.begin('declarations')   
        return t
    
    # Ignore continuation characters
    def t_header_interface_declarations_CONTINUATION(self,t):
        r'&[\n]'
        self.incontinuation = True
        t.lexer.lineno += 1
        pass
    
    def t_declarations_BLOCKDATA(self, t):
        r'block\s*data'
        pass
    
    def t_declarations_ENDBLOCKDATA(self, t):
        r'end\s*block\s*data'
        pass


    # Various end tokens
    def t_body_ENDIF(self, t):
        r'end\s*if'
        pass
 
    # Various end tokens
    def t_body_ENDDO(self, t):
        r'end\s*do'
        pass
    
    # Various end tokens
    def t_body_ENDWHERE(self, t):
        r'end\s*where'
        pass
    
    # Various end tokens
    def t_body_ENDSELECT(self, t):
        r'end\s*select'
        pass
    
    # Various end tokens
    def t_body_ENDASSOCIATE(self, t):
        r'end\s*associate'
        pass
        
    def t_body_SEMICOLON(self,t):
        r';'
        pass
    
    # identifiers
    def t_INITIAL_header_body_ID(self, t):
        r'[A-Za-z_][\w_]*[^\w]'
        #t.type = reserved_map.get(t.value,'ID')
        if self.insubroutine or self.infunction:
            if t.value.lower() in ['use', 'implicit']:
                t.lexer.begin('declarations')
                # rewind
                t.lexpos -= len(t.value)
            elif t.value.lower() == 'end' and not t.lexer.lexdata[t.lexpos +len(t.value):].strip():
                # sloppy subroutine end at the end of the file -- this is not sufficient to cover all cases!
                # TODO: match all ends in the future 
                if self.insubroutine:
                    t.value = 'end subroutine'
                    t.type = 'ENDSUBROUTINE'
                elif self.insfunction:
                    t.value = 'end function'
                    t.type = 'ENDFUNCTION'
        t.lexer.lexpos -= 1
        return t
    
    # ------------------------------
    # Error handlers
    # -----------------------------   
    def t_INITIAL_error(self, t):
        t.lexer.skip(1)
        pass
   
    def t_header_interface_body_error(self, t):
        t.lexer.skip(1)
        pass
    
    def t_declarations_error(self,t):
        t.lexer.skip(1)
        pass
    
    def t_dstring_sstring_error(self,t):
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
    
    def getFileName(self):
        return self.filename
    
    def getLine(self,t):
        bol = t.lexer.lexdata.rfind('\n',0,t.lexpos)
        line = t.lexer.lexdata[bol:t.lexer.lexdata.find('\n',bol+1)]
        return bol,line
    
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
        # Use Intel compiler rules for Fortran file suffix to determine fixed vs free form
        l.reset()
        l.setFileName(sys.argv[i])
        #l.determineFileFormat(sys.argv[i])
        f = os.popen('gfortran -E %s' % sys.argv[i])
        #f = open(sys.argv[i],"r")
        s = f.read()
        f.close()
        # info("Contents of %s: %s" % (sys.argv[i], s))
        if s == '' or s.isspace(): sys.exit(0)
        l.test(s)
        info('Done processing %s.' % sys.argv[i])

