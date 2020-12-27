#!/usr/bin/env python
#
# File: $id$
# @Package: orio
# @version: $Revision$
# @lastrevision: $Date$
# @modifiedby: $LastChangedBy$
# @lastmodified: $LastChangedDate$
#
# Description: Fortran 2003 lexer (supports Fortran 77 and later)
# 
# Produced at Argonne National Laboratory
# Author: Boyana Norris (norris@mcs.anl.gov)
# Copyright (c) 2009 UChicago Argonne, LLC, Operator of Argonne National Laboratory
# ("Argonne").  Argonne, a U.S. Department of Energy Office
# of Science laboratory, is operated under Contract No. DE-AC02-06CH11357. 
# The U.S. Government has rights to use, reproduce, and distribute this software. 
# NEITHER THE GOVERNMENT NOR UCHICAGO ARGONNE, LLC MAKES ANY WARRANTY, EXPRESS OR 
# IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. It is provided 
# "as is" without express or implied warranty. Permission is hereby granted to 
# use, reproduce, prepare derivative works, and to redistribute to others, 
# as long as this original copyright notice is retained.
#
# This software is loosely based, but distinct from, OpenFortran: 
#  OpenFortran Copyright (c) 2005, 2006 Los Alamos National Security, LLC.  
#
# For detailed license information refer to the LICENSE file in the top-level
# Orio source directory.

import sys, re
import orio.tool.ply.lex as lex
from orio.main.util.globals import *

class FLexer:
    
    def __init__(self):
        self.currentline = ''
        self.filename = ''
        self.lineno = 1
        self.informat = False
        self.incontinuation = False
        self.fixedform = False
        self.needs_preprocessing = False
        self.incomment = False
        self.code_start = -1
        self.subroutines = []
        self.insubroutine = False
        self.eolre = re.compile(r'[\n\r]')
        self.testval = ''
        
    def reset(self, fname=''):
        self.currentline = ''
        self.filename = ''
        self.lineno = 1
        self.informat = False
        self.incontinuation = False
        self.fixedform = False
        self.needs_preprocessing = False
        self.incomment = False
        self.code_start = -1
        self.subroutines = []
        self.insubroutine = False
        self.testval = ''
        if fname:
            self.setFileName(fname)
            self.determineFileFormat(fname)
        
    #------------------------------------------------
    
    states = (
          ('misc', 'exclusive'),
          ('sstring', 'exclusive'),
          ('dstring', 'exclusive'),
          ('fcode', 'exclusive')
         )

   
    #------------------------------------------------
    
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
    
    tokens = keywords + ( \
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
        'COMMA', 'SEMI', 'COLON', 'PERIOD', 
        
        # Other
        'WS','COLON_COLON', 'SLASH_SLASH', 'UNDERSCORE',
        'CONTINUE_CHAR', 'LINECOMMENT', 'PREPROCESS_LINE',
        'MISC_CHAR', 
        )
    
    # operators
    t_PLUS             = r'\+'
    t_MINUS            = r'-'
    t_TIMES            = r'\*'
    t_DIVIDE           = r'/'
    t_MOD              = r'%'
    t_SLASH_SLASH      = r'//'     # concatenation of char values
    
    
    # relational operators
    t_LESSTHAN         = r'<'
    t_LESSTHAN_EQ      = r'<='
    t_GREATERTHAN      = r'>'
    t_GREATERTHAN_EQ   = r'>='
    t_EQ_GT            = r'=>'
    t_EQ_EQ            = r'=='
    t_SLASH_EQ         = r'/='
    
    t_EQ               = r'\.EQ\.'
    t_NE               = r'\.NE\.'
    t_LT               = r'\.LT\.'
    t_LE               = r'\.LE\.'
    t_GT               = r'\.GT\.'
    t_GE               = r'\.GE\.'
    
    # R428 .TRUE.[_kind-param] or .FALSE.[_kind-param]
    t_TRUE             = r'\.TRUE\.(_([A-Za-z]\w*))?'
    t_FALSE            = r'\.FALSE\.(_([A-Za-z]\w*))?'
    
    t_OR              = r'\.OR\.'
    t_AND             = r'\.AND\.'
    t_NOT             = r'\.NOT\.'
    t_EQV             = r'\.EQV\.'
    t_NEQV            = r'\.NEQV\.'
    
    
    # assignment operators
    t_EQUALS           = r'='
    
    # defined unary operator; R703
    t_DEFINED_UNARY_OP = r'\.[A-Za-z]+\.'
    
    # delimeters
    t_LPAREN           = r'\('
    t_RPAREN           = r'\)'
    t_LBRACKET         = r'\['
    t_RBRACKET         = r'\]'
    t_LBRACE           = r'\{'
    t_RBRACE           = r'\}'
    t_COMMA            = r','
    t_COLON            = r':'
    t_COLON_COLON      = r'::'
    t_SEMI             = r';'
    t_PERIOD           = r'\.'
    t_UNDERSCORE       = r'_'
    
    # types
    t_INTEGER          = 'INTEGER'
    t_REAL             = 'REAL'
    t_COMPLEX          = 'COMPLEX'
    t_CHARACTER        = 'CHARACTER'
    t_LOGICAL          = 'LOGICAL'
    
    t_ABSTRACT         = 'ABSTRACT'
    t_ALLOCATABLE      = 'ALLOCATABLE'
    t_ALLOCATE         = 'ALLOCATE'      ;
    t_ASSIGNMENT       = 'ASSIGNMENT'    
    # ASSIGN statements are a deleted feature.
    t_ASSIGN           = 'ASSIGN'        
    t_ASSOCIATE        = 'ASSOCIATE'     
    t_ASYNCHRONOUS     = 'ASYNCHRONOUS'  
    t_BACKSPACE        = 'BACKSPACE'     
    t_BLOCK            = 'BLOCK'         
    t_BLOCKDATA        = 'BLOCKDATA'     
    t_CALL             = 'CALL'          
    t_CASE             = 'CASE'          
    t_CLASS            = 'CLASS'         
    t_CLOSE            = 'CLOSE'         
    t_COMMON           = 'COMMON'        
    t_CONTAINS         = 'CONTAINS'      
    t_CONTINUE         = 'CONTINUE'      
    t_CYCLE            = 'CYCLE'         
    t_DATA             = 'DATA'          
    t_DEFAULT          = 'DEFAULT'       
    t_DEALLOCATE       = 'DEALLOCATE'    
    t_DEFERRED         = 'DEFERRED'      
    t_DO               = 'DO'            
    t_DOUBLE           = 'DOUBLE'        
    t_DOUBLEPRECISION  = 'DOUBLEPRECISION' 
    t_DOUBLECOMPLEX    = 'DOUBLECOMPLEX' 
    t_ELEMENTAL        = 'ELEMENTAL'     
    t_ELSE             = 'ELSE'          
    t_ELSEIF           = 'ELSEIF'        
    t_ELSEWHERE        = 'ELSEWHERE'     
    t_ENTRY            = 'ENTRY'         
    t_ENUM             = 'ENUM'          
    t_ENUMERATOR       = 'ENUMERATOR'    
    t_EQUIVALENCE      = 'EQUIVALENCE'   
    t_EXIT             = 'EXIT'          
    t_EXTENDS          = 'EXTENDS'       
    t_EXTERNAL         = 'EXTERNAL'      
    t_FILE             = 'FILE'          
    t_FINAL            = 'FINAL'         
    t_FLUSH            = 'FLUSH'         
    t_FORALL           = 'FORALL'        
    t_FORMATTED        = 'FORMATTED'     
    t_FUNCTION         = 'FUNCTION'      
    t_GENERIC          = 'GENERIC'       
    t_GO               = 'GO'            
    t_GOTO             = 'GOTO'          
    t_IF               = 'IF'            
    t_IMPORT           = 'IMPORT'        
    t_IN               = 'IN'            
    t_INOUT            = 'INOUT'         
    t_INTENT           = 'INTENT'        
    t_INTERFACE        = 'INTERFACE'     
    t_INTRINSIC        = 'INTRINSIC'     
    t_INQUIRE          = 'INQUIRE'       
    t_MODULE           = 'MODULE'        
    t_NAMELIST         = 'NAMELIST'      
    t_NONE             = 'NONE'          
    t_NON_INTRINSIC    = 'NON_INTRINSIC' 
    t_NON_OVERRIDABLE  = 'NON_OVERRIDABLE'
    t_NOPASS           = 'NOPASS'        
    t_NULLIFY          = 'NULLIFY'       
    t_ONLY             = 'ONLY'          
    t_OPEN             = 'OPEN'          
    t_OPERATOR         = 'OPERATOR'      
    t_OPTIONAL         = 'OPTIONAL'      
    t_OUT              = 'OUT'           
    t_PARAMETER        = 'PARAMETER'     
    t_PASS             = 'PASS'          
    t_PAUSE            = 'PAUSE'         
    t_POINTER          = 'POINTER'       
    t_PRINT            = 'PRINT'         
    t_PRECISION        = 'PRECISION'     
    t_PRIVATE          = 'PRIVATE'       
    t_PROCEDURE        = 'PROCEDURE'     
    t_PROGRAM          = 'PROGRAM'       
    t_PROTECTED        = 'PROTECTED'     
    t_PUBLIC           = 'PUBLIC'        
    t_PURE             = 'PURE'          
    t_READ             = 'READ'          
    t_RECURSIVE        = 'RECURSIVE'     
    t_RESULT           = 'RESULT'             
    t_RETURN           = 'RETURN' 
    t_REWIND           = 'REWIND'        
    t_SAVE             = 'SAVE'          
    t_SELECT           = 'SELECT'        
    t_SELECTCASE       = 'SELECTCASE'    
    t_SELECTTYPE       = 'SELECTTYPE'    
    t_SEQUENCE         = 'SEQUENCE'      
    t_STOP             = 'STOP'          
    t_SUBROUTINE       = 'SUBROUTINE'    
    t_TARGET           = 'TARGET'        
    t_THEN             = 'THEN'          
    t_TO               = 'TO'            
    t_TYPE             = 'TYPE'          
    t_UNFORMATTED      = 'UNFORMATTED'   
    t_USE              = 'USE'           
    t_VALUE            = 'VALUE'         
    t_VOLATILE         = 'VOLATILE'      
    t_WAIT             = 'WAIT'          
    t_WHERE            = 'WHERE'         
    t_WHILE            = 'WHILE'         
    t_WRITE            = 'WRITE'         
    
    t_ENDASSOCIATE     = 'ENDASSOCIATE'  
    t_ENDBLOCK         = 'ENDBLOCK'      
    t_ENDBLOCKDATA     = 'ENDBLOCKDATA'  
    t_ENDDO            = 'ENDDO'         
    t_ENDENUM          = 'ENDENUM'       
    t_ENDFORALL        = 'ENDFORALL'     
    t_ENDFILE          = 'ENDFILE'       
    t_ENDFUNCTION      = 'ENDFUNCTION'   
    t_ENDIF            = 'ENDIF'         
    t_ENDINTERFACE     = 'ENDINTERFACE'  
    t_ENDMODULE        = 'ENDMODULE'     
    t_ENDPROGRAM       = 'ENDPROGRAM'    
    t_ENDSELECT        = 'ENDSELECT' 
    t_ENDSUBROUTINE    = 'ENDSUBROUTINE'    
    t_ENDTYPE          = 'ENDTYPE'       
    t_ENDWHERE         = 'ENDWHERE'      
    
    # END is in a longer rule below
    #t_END              = 'END'
            
    
    t_DIMENSION        = 'DIMENSION'     
    
    t_KIND             = 'KIND' 
    t_LEN              = 'LEN' 
    
    t_BIND             = 'BIND' 
    
    
    # ignored characters
    t_ignore           = '\r\t\x0c'
    
    # reserved words
    global reserved_map
    reserved_map = {}
    for r in keywords:
        reserved_map[r.lower()] = r
    
        
    t_LINECOMMENT = r'![^\n\r]*'
    
    t_PREPROCESS_LINE = r'\#[^\n\r]*'
        
    # identifiers
    def t_ID(self, t):
        r'[A-Za-z_][\w_]*'
        t.type = reserved_map.get(t.value,'ID')
        return t
    
    def t_FORMAT(self, t):
        r'FORMAT'
        self.inFormat = True
        t.lexer.push_state('format')
        return t
        
    # Limited initial subroutine body detection (between 
    def t_INITIAL_fcode_IMPLICIT(self, t):
        r'IMPLICIT'
        t.lexer.code_start = t.lexpos
        t.lexer.insubroutine = True
        t.lexer.push_state('fcode')
        return t
        
    def t_INITIAL_fcode_END(self,t):
        r'END'
        if self.subre.match(t.lexer.lexdata[t.lexpos:]) and self.insubroutine:
            t.lexer.insubroutine = False
            t.lexer.subroutines.append(t.lexer.lexdata[t.lexer.code_start:t.lexpos].strip())
            t.lexer.pop_state()  # leave the 'fcode' state
        return t
        
    # integer constant, label, and part of kind; R409
    t_DIGIT_STRING = r'\d+'
    
    # binary literal; R408
    t_BCONST     = r'([bB]\'\d+\') | ([bB]"\d+")'
    
    # integer literal; R405
    t_ICONST     = r'\d+(_[A-Za-z]\w*)?'
    
    # floating literal; R417
    t_FCONST     = r'(\d*\.\d+([EedD][+-]?(\d*_[A-Za-z]\w*|\d+))?)'
                        
    # string literal (with double quotes); R427
    # char-literal-constant is [kind-param]'[rep-char]...' or [kind-param]"[rep-char]..."
    t_INITIAL_SCONST_D   = r'(([A-Za-z]\w*)_)?\"([^\\\n]|(\\.))*\"'
    
    # Partial double-quoted string literal (continued on a different line)
    def t_INITIAL_PSCONST_D(self, t):                   
        r'(([A-Za-z]\w*)_)?\"([^\\\n\"]|(\\.))*&\s*\n'
        self.lineno += 1
        t.value = t.value.strip().strip('&').strip('"')
        t.lexer.push_state('dstring')
        return t
    
    # The end of a partial double-quoted string literal (continued from a previous line)
    def t_dstring_PSCONST_D(self, t):
        r'\s*&?([^\\\n]|(\\.))*\"'
        t.value = t.value.strip().strip('&').strip('"')
        t.lexer.pop_state()
        return t
 
    # string literal (with single quotes); R427
    #t_SCONST_S   = r'\'([^\\\n]|(\\.))*?\''
    t_INITIAL_SCONST_S = r'(([A-Za-z]\w*)_)?\'([^\\\n]|(\\.))*\''
    
    # partial single-quoted string literal (continued on another line)
    def t_INITIAL_PSCONST_S(self, t):
        r'(([A-Za-z]\w*)_)?\'([^\\\n\']|(\\.))*&\s*\n'
        self.lineno += 1
        t.value = t.value.strip().strip('&').strip("'")
        t.lexer.push_state('sstring')
        return t
    
    # the end of a partial single-quoted string literal (continued from another line)
    def t_sstring_PSCONST_S(self, t):
        r'\s*&?([^\\\n]|(\\.))*\''
        t.value = t.value.strip().strip('&').strip("'")
        t.lexer.pop_state()
        return t
    
    # octal constant (with double quotes); R413
    t_OCONST_D   = r'[oO]"[0-7]+"'
    
    # octal constant (with single quotes); R413
    t_OCONST_S   = r'[oO]\'[0-7]+\''
    
    # hex constant (with double quotes); R414
    t_HCONST_D   = r'[zZ]"[\dA-Fa-f]+"'
    
    # hex constant (with double quotes); R414
    t_HCONST_S   = r'[zZ]\'[\dA-Fa-f]+\''
    
    # Any whitespace character: equiv. to set [ \t\n\r\f\v]
    def t_INITIAL_WS(self,t):
        r'[ \t\r\f]'
        pass
        
    def t_CONTINUE_CHAR(self, t):
        r'&'
        # Fortran continuation is evil, especially when a comment can be snuck in between 
        # the line being continued -- see handling in string token ruless
        self.incontinuation = True
                
        pass   # eat it

    # count newlines
    def t_NEWLINE(self, t):
        r'\n+'
        self.lineno += t.value.count('\n')
        if self.fixedform and not self.incontinuation:
            t.lexer.push_state('misc')
        self.incontinuation = False
        
    # A catch-all rule for the case when any character is used in column 1 or 6
    # to designate a comment or continuation. The rules in the "misc" state are
    # the only fixed mode-specific part of the lexer.
    def t_misc_MISC_CHAR(self, t):
        r'\s*\S'
        # Check the first non whitespace character -- significant in fixed form only
        col = self.find_column(t.lexer.lexdata, t)
        if self.fixedform:
            if col == 1 and len(t.value) == 1 and t.value != ' ':
                self.incomment = True
                # Pretend this is a freeform comment and hand it over to the rest of the lexer
                
                eolmatch = self.eolre.search(t.lexer.lexdata[t.lexer.lexpos:])
                #debug('MISC_CHAR: %s\n' % str(t.lexer.lexdata[t.lexer.lexpos-1:]))
                if eolmatch:
                    eol = t.lexer.lexpos + eolmatch.start()
                    t.value = '!' + t.lexer.lexdata[t.lexer.lexpos:eol]
                    #debug('tvalue=%s' % t.value)
                    t.type = 'LINECOMMENT'
                    t.lexer.lexpos = eol
                    t.lexer.pop_state()
                    return t
            elif len(t.value) == 6 and t.value[-1] != ' ':
                self.incontinuation = True
                t.lexer.pop_state()
                return  # discard the token
            else:
                # No idea what we are parsing, revert
                t.lexer.lexpos -= len(t.value)

        if not self.fixedform and t.value == '&':
            self.incontinuation = True
            self.lexer.lexpos -= 1   # let the continuation rule handle it
        
        t.lexer.pop_state()     # revert to freeform
        
        return   # for debugging (to see this token), change this to 'return t'
            
    # syntactical error (shared by both states)
    def t_misc_INITIAL_error(self, t):
        self.raise_error(t)

    def t_sstring_error(self,t):
        self.raise_error(t)
    
    def raise_error(self,t):
        col = self.find_column(t.lexer.lexdata, t)
        err('orio.main.parsers.flexer: *** Fortran parse error in %s: illegal character (%s) at line %s, column %s' \
                % (self.filename, t.value[0], self.lineno, col))
        # lexing errors are fatal
        #t.lexer.skip(1) # this makes lexing errors non-fatal
        
        
        
    # --------------------------------- End of token rules ------------------------------------
    # &-&-&-&-&-&-&-&-&-&-&-&-&-&-&-&-&-&-&-&-&-&-&-&-&-&-&-&-&-&-&-&-&-&-&-&-&-&-&-&-&-&-&-&-&
      

    # Compute column. 
    #     input is the input text string
    #     token is a token instance
    def find_column(self,input,token):
        i = token.lexpos
        while i > 0:
            if input[i] == '\n': 
                break
            i -= 1
        column = (token.lexpos - i) 
        return column

       
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
        print('subroutines:',self.subroutines)
    
    def determineFileFormat(self, filename):
        # Use the filename suffix to determine whether this is fixed form or free form (same as the Intel compiler):
        #  o Filenames with the suffix .f90 are interpreted as free-form Fortran 95/90 source files.
        #  o Filenames with the suffix .f, .for, or .ftn are interpreted as fixed-form Fortran source  files.
        #  o Filenames  with  the  suffix .fpp, .F, .FOR, .FTN, or .FPP are interpreted as fixed-form Fortran
        #    source files, which must be preprocessed by the fpp preprocessor before being compiled.
        suffix = filename[filename.rfind('.'):]
        if suffix.lower() in ['.f', '.for', '.ftn']:
            self.fixedform = True
            self.lexer.push_state('misc')
        if suffix.isupper():
            self.needs_preprocessing = True
        
        pass
    
    def setFileName(self, filename):
        self.filename = filename
        pass
    
    # ---------------- end of class FLexer  -------------------------
        
#lexer = parse.ply.lex.lex(optimize = 0)
# Main for debugging the lexer:
if __name__ == "__main__":
    import sys
    # Build the lexer and try it out
    Globals().verbose = True
    l = FLexer()
    l.build(debug=1)           # Build the lexer
    for i in range(1, len(sys.argv)):
        info("About to lex %s" % sys.argv[i])
        f = open(sys.argv[i],"r")
        # Use Intel compiler rules for Fortran file suffix to determine fixed vs free form
        l.reset()
        l.setFileName(sys.argv[i])
        l.determineFileFormat(sys.argv[i])
        s = f.read()
        f.close()
        # info("Contents of %s: %s" % (sys.argv[i], s))
        if s == '' or s.isspace(): sys.exit(0)
        l.test(s)
        info('Done processing %s.' % sys.argv[i])


    
# --------------- end of lexer -------------------
