#!/usr/bin/env python

import sys, os
import orio.tool.ply.lex, orio.tool.ply.yacc
import orio.main.util.globals as g
import orio.module.loops.ast as ast

#----------------------------------------------------------------------------------------------------------------------
class LoopsLexer:

    def __init__(self):
        pass

    keywords = [
        'if', 'else',
        'for', #'do', 'while',
        #'return', 'break', 'continue', 'goto', 
        #'switch', 'case', 'default',

        'char', 'short', 'int', 'long', 'float', 'double',
        'sizeof', #'signed', 'unsigned',

        #'auto', 'register', 'static', 'extern',
        #'const', 'restrict', 'volatile',

        #'void', 'inline',
        #'enum', 'struct', 'typedef', 'union',

        'transform'
    ]

    reserved = {}
    for k in keywords:
        reserved[k] = k.upper()
    
    tokens = list(reserved.values()) + [
        # literals (identifier, integer, float, string)
        'ID', 'ICONST', 'FCONST', 'SCONST',
        
        # operators
        'LOR', 'LAND', 'LNOT',
        'LT', 'LE', 'GT', 'GE', 'EE', 'NE',
        'BSHL', 'BSHR',
        'SELECT',
    
        # assignment operators
        'EQ', 'MULTEQ', 'DIVEQ', 'MODEQ', 'PLUSEQ', 'MINUSEQ',
        'BSHLEQ', 'BSHREQ', 'BANDEQ', 'BXOREQ', 'BOREQ',
    
        'PP', 'MM', # increment/decrement (++,--)
        'LINECOMMENT'
    ]


    # A string containing ignored characters (spaces and tabs)
    t_ignore = ' \t'
    t_LINECOMMENT = r'[\#!][^\n\r]*'
    
    # operators
    t_LOR     = r'\|\|'
    t_LAND    = r'&&'
    t_LNOT    = r'!'
    t_LT      = r'<'
    t_GT      = r'>'
    t_LE      = r'<='
    t_GE      = r'>='
    t_EE      = r'=='
    t_NE      = r'!='
    t_BSHL    = r'<<'
    t_BSHR    = r'>>'
    t_SELECT  = r'->'
    
    # assignment operators
    t_EQ      = r'='
    t_MULTEQ  = r'\*='
    t_DIVEQ   = r'/='
    t_MODEQ   = r'%='
    t_PLUSEQ  = r'\+='
    t_MINUSEQ = r'-='
    t_BSHLEQ  = r'<<='
    t_BSHREQ  = r'>>='
    t_BANDEQ  = r'&='
    t_BXOREQ  = r'^='
    t_BOREQ   = r'\|='
    
    # increment/decrement
    t_PP      = r'\+\+'
    t_MM      = r'--'
    
    literals = "+-*/%()[]{},;:.~&|^?"

    # integer literal
    t_ICONST  = r'\d+'
    
    # floating literal
    t_FCONST  = r'((\d+)(\.\d*)([eE](\+|-)?(\d+))? | (\d+)[eE](\+|-)?(\d+))'
    
    # string literal
    t_SCONST  = r'\"([^\\\n]|(\\.))*?\"|\'([^\\\n]|(\\.))*?\''
    
    def t_ID(self, t):
        r'[A-Za-z_]([A-Za-z0-9_\.]*[A-Za-z0-9_]+)*'
        t.type = self.reserved.get(t.value, 'ID')
        return t
    
    def t_NEWLINE(self, t):
        r'\n+'
        t.lexer.lineno += len(t.value)
    
    def t_error(self, t):
        g.err('orio.module.loops.lexer: illegal character (%s) at line %s' % (t.value[0], t.lexer.lineno))
    
    def build(self, **kwargs):
        self.lexer = orio.tool.ply.lex.lex(module=self, **kwargs)
    
    def test(self, data):
        self.lexer.input(data)
        while 1:
            tok = self.lexer.token()
            if not tok: break
            print(tok)
    
    def input(self, data):
        return self.lexer.input(data)
    
    def token(self):
        return self.lexer.token()
#----------------------------------------------------------------------------------------------------------------------



#----------------------------------------------------------------------------------------------------------------------
tokens = LoopsLexer.tokens
start = 'annotation'
__start_line_no = 1
#----------------------------------------------------------------------------------------------------------------------
def p_annotation(p):
    '''annotation : statements'''
    p[0] = p[1]

def p_statements_1(p):
    'statements : empty'
    p[0] = p[1]
    
def p_statements_2(p):
    '''statements : statements statement'''
    p[1].append(p[2])
    p[0] = p[1]
    
def p_statement(p):
    '''statement : line_comment
                 | expression_statement
                 | compound_statement
                 | conditional_statement
                 | iteration_statement
                 | transform_statement '''
    p[0] = p[1]

def p_line_comment(p):
    'line_comment : LINECOMMENT'
    p[0] = ast.Comment(p[1], p.lineno(1))

def p_expression_statement(p):
    '''expression_statement : expression_opt ';' '''
    p[0] = ast.ExpStmt(p[1], p.lineno(1))

def p_compound_statement(p):
    '''compound_statement : '{' statements '}' '''
    p[0] = ast.CompStmt(p[2], p.lineno(1))

def p_conditional_statement_1(p):
    '''conditional_statement : IF '(' expr ')' statement'''
    p[0] = ast.IfStmt(p[3], p[5], None, p.lineno(1))
    
def p_conditional_statement_2(p):
    '''conditional_statement : IF '(' expr ')' statement ELSE statement'''
    p[0] = ast.IfStmt(p[3], p[5], p[7], p.lineno(1))

def p_iteration_statement(p):
    '''iteration_statement : FOR '(' expression_opt ';' expression_opt ';' expression_opt ')' statement'''
    p[0] = ast.ForStmt(p[3], p[5], p[7], p[9], p.lineno(1))

def p_transform_statement(p):
    '''transform_statement : TRANSFORM ID '(' transform_args ')' statement'''
    p[0] = ast.TransformStmt(p[2], p[4], p[6], p.lineno(1))

def p_transform_statement2(p):
    '''transform_statement : TRANSFORM ID '(' transform_args ')' '''
    p[0] = ast.TransformStmt(p[2], p[4], None, p.lineno(1))

def p_transform_args1(p):
    '''transform_args : empty'''
    p[0] = []

def p_transform_args2(p):
    '''transform_args : transform_arg'''
    p[0] = [p[1]]

def p_transform_args3(p):
    '''transform_args : transform_args ',' transform_arg'''
    p[1].append(p[3])
    p[0] = p[1]

def p_transform_arg(p):
    '''transform_arg : ID EQ expr'''
    p[0] = [p[1], p[3], p.lineno(1)]


#------------------------------------------------------------------------------
precedence = (
    ('left', ','),
    ('right', 'EQ', 'PLUSEQ', 'MINUSEQ', 'MULTEQ', 'DIVEQ', 'MODEQ', 'BSHLEQ', 'BSHREQ', 'BANDEQ', 'BXOREQ', 'BOREQ'),
    ('left', 'LOR'),
    ('left', 'LAND'),
    ('left', '|'),
    ('left', '^'),
    ('left', '&'),
    ('left', 'EE', 'NE'),
    ('left', 'LT', 'GT', 'LE', 'GE'),
    ('left', 'BSHL', 'BSHR'),
    ('left', '+', '-'),
    ('left', '*', '/', '%'),
    ('right', 'LNOT', '~', 'PP', 'MM', 'UPLUS', 'UMINUS', 'DEREF', 'ADDRESSOF', 'SIZEOF'),
    ('left', '.', 'SELECT')
)

#------------------------------------------------------------------------------
def p_expression_opt_1(p):
    'expression_opt :'
    p[0] = None

def p_expression_opt_2(p):
    'expression_opt : expr'
    p[0] = p[1]

def p_expr_dec(p):
    '''expr : tyexpr expr'''
    p[0] = ast.VarDec(p[1], [p[2]], True, p.lineno(1))

def p_tyexpr(p):
    '''tyexpr : ty stars'''
    p[0] = [p[1]] + p[2]

def p_stars1(p):
    '''stars : empty'''
    p[0] = p[1]

def p_stars2(p):
    '''stars : stars '*' '''
    p[1].append(p[2])
    p[0] = p[1]

def p_ty(p):
    '''ty : CHAR
          | SHORT
          | INT
          | LONG
          | FLOAT
          | DOUBLE '''
    p[0] = p[1]

def p_expr_seq(p):
    '''expr : expr ',' expr'''
    p[0] = ast.BinOpExp(ast.BinOpExp.COMMA, p[1], p[3], p.lineno(1))

def p_expr_ternary(p):
    '''expr : expr '?' expr ':' expr'''
    p[0] = ast.TernaryExp(p[1], p[3], p[5], p.lineno(1))

#------------------------------------------------------------------------------
def p_expr_assign1(p):
    '''expr : expr EQ expr'''
    p[0] = ast.BinOpExp(ast.BinOpExp.EQ, p[1], p[3], p.lineno(1))

def p_expr_assign2(p):
    '''expr : expr MULTEQ expr'''
    p[0] = ast.BinOpExp(ast.BinOpExp.MULTEQ, p[1], p[3], p.lineno(1))

def p_expr_assign3(p):
    '''expr : expr DIVEQ expr'''
    p[0] = ast.BinOpExp(ast.BinOpExp.DIVEQ, p[1], p[3], p.lineno(1))

def p_expr_assign4(p):
    '''expr : expr MODEQ expr'''
    p[0] = ast.BinOpExp(ast.BinOpExp.MODEQ, p[1], p[3], p.lineno(1))

def p_expr_assign5(p):
    '''expr : expr PLUSEQ expr'''
    p[0] = ast.BinOpExp(ast.BinOpExp.PLUSEQ, p[1], p[3], p.lineno(1))

def p_expr_assign6(p):
    '''expr : expr MINUSEQ expr'''
    p[0] = ast.BinOpExp(ast.BinOpExp.MINUSEQ, p[1], p[3], p.lineno(1))

def p_expr_assign7(p):
    '''expr : expr BSHLEQ expr'''
    p[0] = ast.BinOpExp(ast.BinOpExp.BSHLEQ, p[1], p[3], p.lineno(1))

def p_expr_assign8(p):
    '''expr : expr BSHREQ expr'''
    p[0] = ast.BinOpExp(ast.BinOpExp.BSHREQ, p[1], p[3], p.lineno(1))

def p_expr_assign9(p):
    '''expr : expr BANDEQ expr'''
    p[0] = ast.BinOpExp(ast.BinOpExp.BANDEQ, p[1], p[3], p.lineno(1))

def p_expr_assign10(p):
    '''expr : expr BXOREQ expr'''
    p[0] = ast.BinOpExp(ast.BinOpExp.BXOREQ, p[1], p[3], p.lineno(1))

def p_expr_assign11(p):
    '''expr : expr BOREQ expr'''
    p[0] = ast.BinOpExp(ast.BinOpExp.BOREQ, p[1], p[3], p.lineno(1))

#------------------------------------------------------------------------------
def p_expr_log1(p):
    'expr : expr LOR expr'
    p[0] = ast.BinOpExp(ast.BinOpExp.LOR, p[1], p[3], p.lineno(1))

def p_expr_log2(p):
    'expr : expr LAND expr'
    p[0] = ast.BinOpExp(ast.BinOpExp.LAND, p[1], p[3], p.lineno(1))

#------------------------------------------------------------------------------
def p_expr_bit1(p):
    '''expr : expr '|' expr'''
    p[0] = ast.BinOpExp(ast.BinOpExp.BOR, p[1], p[3], p.lineno(1))

def p_expr_bit2(p):
    '''expr : expr '^' expr'''
    p[0] = ast.BinOpExp(ast.BinOpExp.BXOR, p[1], p[3], p.lineno(1))

def p_expr_bit3(p):
    '''expr : expr '&' expr'''
    p[0] = ast.BinOpExp(ast.BinOpExp.BAND, p[1], p[3], p.lineno(1))

#------------------------------------------------------------------------------
def p_expr_rel1(p):
    'expr : expr EE expr'
    p[0] = ast.BinOpExp(ast.BinOpExp.EE, p[1], p[3], p.lineno(1))

def p_expr_rel2(p):
    'expr : expr NE expr'
    p[0] = ast.BinOpExp(ast.BinOpExp.NE, p[1], p[3], p.lineno(1))

def p_expr_rel3(p):
    'expr : expr LT expr'
    p[0] = ast.BinOpExp(ast.BinOpExp.LT, p[1], p[3], p.lineno(1))

def p_expr_rel4(p):
    'expr : expr GT expr'
    p[0] = ast.BinOpExp(ast.BinOpExp.GT, p[1], p[3], p.lineno(1))

def p_expr_rel5(p):
    'expr : expr LE expr'
    p[0] = ast.BinOpExp(ast.BinOpExp.LE, p[1], p[3], p.lineno(1))

def p_expr_rel6(p):
    'expr : expr GE expr'
    p[0] = ast.BinOpExp(ast.BinOpExp.GE, p[1], p[3], p.lineno(1))

#------------------------------------------------------------------------------
def p_expr_shift1(p):
    '''expr : expr BSHL expr'''
    p[0] = ast.BinOpExp(ast.BinOpExp.BSHL, p[1], p[3], p.lineno(1))

def p_expr_shift2(p):
    '''expr : expr BSHR expr'''
    p[0] = ast.BinOpExp(ast.BinOpExp.BSHR, p[1], p[3], p.lineno(1))

#------------------------------------------------------------------------------
def p_expr_add1(p):
    '''expr : expr '+' expr'''
    p[0] = ast.BinOpExp(ast.BinOpExp.PLUS, p[1], p[3], p.lineno(1))

def p_expr_add2(p):
    '''expr : expr '-' expr'''
    p[0] = ast.BinOpExp(ast.BinOpExp.MINUS, p[1], p[3], p.lineno(1))

#------------------------------------------------------------------------------
def p_expr_mult1(p):
    '''expr : expr '*' expr'''
    p[0] = ast.BinOpExp(ast.BinOpExp.MULT, p[1], p[3], p.lineno(1))

def p_expr_mult2(p):
    '''expr : expr '/' expr'''
    p[0] = ast.BinOpExp(ast.BinOpExp.DIV, p[1], p[3], p.lineno(1))

def p_expr_mult3(p):
    '''expr : expr '%' expr'''
    p[0] = ast.BinOpExp(ast.BinOpExp.MOD, p[1], p[3], p.lineno(1))

#------------------------------------------------------------------------------
def p_expr_select1(p):
    '''expr : expr '.' expr'''
    p[0] = ast.BinOpExp(ast.BinOpExp.DOT, p[1], p[3], p.lineno(1))

def p_expr_select2(p):
    '''expr : expr SELECT expr'''
    p[0] = ast.BinOpExp(ast.BinOpExp.SELECT, p[1], p[3], p.lineno(1))

#------------------------------------------------------------------------------
def p_expr_pre1(p):
    'expr : PP expr'
    p[0] = ast.UnaryExp(ast.UnaryExp.PRE_INC, p[2], p.lineno(1))

def p_expr_pre2(p):
    'expr : MM expr'
    p[0] = ast.UnaryExp(ast.UnaryExp.PRE_DEC, p[2], p.lineno(1))

def p_expr_pre3(p):
    '''expr : '+' expr %prec UPLUS'''
    p[0] = ast.UnaryExp(ast.UnaryExp.PLUS, p[2], p.lineno(1))

def p_expr_pre4(p):
    '''expr : '-' expr %prec UMINUS'''
    p[0] = ast.UnaryExp(ast.UnaryExp.MINUS, p[2], p.lineno(1))

def p_expr_pre5(p):
    '''expr : LNOT expr'''
    p[0] = ast.UnaryExp(ast.UnaryExp.LNOT, p[2], p.lineno(1))

def p_expr_pre6(p):
    '''expr : '~' expr'''
    p[0] = ast.UnaryExp(ast.UnaryExp.BNOT, p[2], p.lineno(1))

def p_expr_pre7(p):
    '''expr : '*' expr %prec DEREF'''
    p[0] = ast.UnaryExp(ast.UnaryExp.DEREF, p[2], p.lineno(1))

def p_expr_pre8(p):
    '''expr : '&' expr %prec ADDRESSOF'''
    p[0] = ast.UnaryExp(ast.UnaryExp.ADDRESSOF, p[2], p.lineno(1))

def p_expr_pre9(p):
    '''expr : SIZEOF expr'''
    p[0] = ast.UnaryExp(ast.UnaryExp.SIZEOF, p[2], p.lineno(1))

#------------------------------------------------------------------------------
def p_expr_arrayref(p):
    '''expr : primary '[' expr ']' '''
    p[0] = ast.ArrayRefExp(p[1], p[3], p.lineno(1))

def p_expr_funcall(p):
    '''expr : expr '(' arg_exprs ')' '''
    p[0] = ast.CallExp(p[1], p[3], p.lineno(1))

def p_arg_exprs_1(p):
    'arg_exprs : empty'
    p[0] = p[1]

def p_arg_exprs_2(p):
    'arg_exprs : expr'
    p[0] = [p[1]]

def p_arg_exprs_3(p):
    '''arg_exprs : arg_exprs ',' expr''' 
    p[1].append(p[3])
    p[0] = p[1]

def p_expr_cast(p):
    '''expr : '(' tyexpr ')' expr '''
    p[0] = ast.CastExp(p[2], p[4], p.lineno(1))

#------------------------------------------------------------------------------
def p_expr_post1(p):
    'expr : expr PP'
    p[0] = ast.UnaryExp(ast.UnaryExp.POST_INC, p[1], p.lineno(1))

def p_expr_post2(p):
    'expr : expr MM'
    p[0] = ast.UnaryExp(ast.UnaryExp.POST_DEC, p[1], p.lineno(1))

#------------------------------------------------------------------------------
def p_expr_primary0(p):
    'expr : primary'
    p[0] = p[1]

def p_expr_primary1(p):
    'primary : ID'
    p[0] = ast.IdentExp(p[1], p.lineno(1))

def p_expr_primary2(p):
    'primary : ICONST'
    p[0] = ast.LitExp(ast.LitExp.INT, int(p[1]), p.lineno(1))

def p_expr_primary3(p):
    'primary : FCONST'
    p[0] = ast.LitExp(ast.LitExp.FLOAT, float(p[1]), p.lineno(1))

def p_expr_primary4(p):
    'primary : SCONST'
    p[0] = ast.LitExp(ast.LitExp.STRING, p[1], p.lineno(1))

def p_expr_primary5(p):
    '''primary : '(' expr ')' '''
    p[0] = ast.ParenExp(p[2], p.lineno(1))

def p_expr_primary6(p):
    '''primary : '[' arg_exprs ']' '''
    p[0] = ast.LitExp(ast.LitExp.ARRAY, p[2], p.lineno(1))


#----------------------------------------------------------------------------------------------------------------------
# utility funs
# Compute column. 
#     input is the input text string
#     token is a token instance
def find_column(inputtxt,token):
    last_cr = inputtxt[:token.lexpos].rfind('\n') # count backwards until you reach a newline
    if last_cr < 0:
        last_cr = 0
    column = (token.lexpos - last_cr) + 1
    return column

def p_empty(p):
    'empty :'
    p[0] = []

def p_error(p):
    col = find_column(p.lexer.lexdata,p)
    g.err(__name__+": unexpected token-type '%s', token-value '%s' at line %s, column %s" % (p.type, p.value, p.lineno, col))
#----------------------------------------------------------------------------------------------------------------------



#----------------------------------------------------------------------------------------------------------------------
def parse(start_line_no, text):

    # set the starting line number
    global __start_line_no
    __start_line_no = start_line_no

    l = LoopsLexer()
    l.build(debug=0)
    l.lexdata = text
    
    parser = orio.tool.ply.yacc.yacc(method='LALR', debug=0, optimize=1, write_tables=0)
    theresult = parser.parse(text, lexer=l, debug=0)
    return theresult



#----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    for i in range(1, len(sys.argv)):
        #print "About to lex %s" % sys.argv[i]
        f = open(sys.argv[i], "r")
        s = f.read()
        f.close()
        #print "Contents of %s:\n%s" % (sys.argv[i], s)
        # Test the lexer; just print out all tokens founds
        #l.test(s)
        
        tree = parse(1, s)
        print(tree)
        print('[parser] Successfully parsed %s' % sys.argv[i], file=sys.stderr)


