#
# Syntax and grammar specifications of a subset of C language
#

import sys
import ast, tool.ply.lex, tool.ply.yacc
from orio.main.util.globals import *

#------------------------------------------------

# reserved words
reserved = ['IF', 'ELSE', 'FOR']

tokens = reserved + [

    # literals (identifier, integer constant, float constant, string constant)
    'ID', 'ICONST', 'FCONST', 'SCONST',

    # operators (+,-,*,/,%,||,&&,!,<,<=,>,>=,==,!=)
    'PLUS', 'MINUS', 'TIMES', 'DIVIDE', 'MOD',
    'LOR', 'LAND', 'LNOT',
    'LT', 'LE', 'GT', 'GE', 'EQ', 'NE',

    # assignment (=, *=, /=, %=, +=, -=)
    'EQUALS', 'TIMESEQUAL', 'DIVEQUAL', 'MODEQUAL', 'PLUSEQUAL', 'MINUSEQUAL',

    # increment/decrement (++,--)
    'PLUSPLUS', 'MINUSMINUS',

    # delimeters ( ) [ ] { } , ;
    'LPAREN', 'RPAREN',
    'LBRACKET', 'RBRACKET',
    'LBRACE', 'RBRACE',
    'COMMA', 'SEMI',
    ]

# operators
t_PLUS             = r'\+'
t_MINUS            = r'-'
t_TIMES            = r'\*'
t_DIVIDE           = r'/'
t_MOD              = r'%'
t_LOR              = r'\|\|'
t_LAND             = r'&&'
t_LNOT             = r'!'
t_LT               = r'<'
t_GT               = r'>'
t_LE               = r'<='
t_GE               = r'>='
t_EQ               = r'=='
t_NE               = r'!='

# assignment operators
t_EQUALS           = r'='
t_TIMESEQUAL       = r'\*='
t_DIVEQUAL         = r'/='
t_MODEQUAL         = r'%='
t_PLUSEQUAL        = r'\+='
t_MINUSEQUAL       = r'-='

# increment/decrement
t_PLUSPLUS         = r'\+\+'
t_MINUSMINUS       = r'--'

# delimeters
t_LPAREN           = r'\('
t_RPAREN           = r'\)'
t_LBRACKET         = r'\['
t_RBRACKET         = r'\]'
t_LBRACE           = r'\{'
t_RBRACE           = r'\}'
t_COMMA            = r','
t_SEMI             = r';'

# ignored characters
t_ignore = ' \t'

# reserved words
reserved_map = {}
for r in reserved:
    reserved_map[r.lower()] = r

# identifiers
def t_ID(t):
    r'[A-Za-z_]\w*'
    t.type = reserved_map.get(t.value,'ID')
    return t

# integer literal
t_ICONST     = r'\d+'

# floating literal
t_FCONST     = r'((\d+)(\.\d*)([eE](\+|-)?(\d+))? | (\d+)[eE](\+|-)?(\d+))'

# string literal (with double quotes)
t_SCONST     = r'\"([^\\\n]|(\\.))*?\"'

# newlines
def t_NEWLINE(t):
    r'\n+'
    t.lineno += t.value.count('\n')
    
# syntactical error
def t_error(t):
    err('orio.module.tilic.code_parser: Tilic: syntactical error: "%s"' % (t.value[0]))
    
#------------------------------------------------

# program
def p_program(p):
    'program : statement_list_opt'
    p[0] = p[1]
    
# statement-list
def p_statement_list_opt_1(p):
    'statement_list_opt :'
    p[0] = []
    
def p_statement_list_opt_2(p):
    'statement_list_opt : statement_list'
    p[0] = p[1]
    
def p_statement_list_1(p):
    'statement_list : statement'
    p[0] = [p[1]]
    
def p_statement_list_2(p):
    'statement_list : statement_list statement'
    p[1].append(p[2])
    p[0] = p[1]
    
# statement:
def p_statement(p):
    '''statement : expression_statement
                 | compound_statement
                 | selection_statement
                 | iteration_statement
                 '''
    p[0] = p[1]
    
# expression-statement:
def p_expression_statement(p):
    'expression_statement : expression_opt SEMI'
    p[0] = ast.ExpStmt(p[1])

# compound-statement:
def p_compound_statement(p):
    'compound_statement : LBRACE statement_list_opt RBRACE'
    p[0] = ast.CompStmt(p[2])

# selection-statement
# Note:
#   This grammar clause results in a shift/reduce conflict. However, such conflict is
#   not harmful, because PLY resolves such conflict in favor of shifting.
def p_selection_statement_1(p):
    'selection_statement : IF LPAREN expression RPAREN statement'
    p[0] = ast.IfStmt(p[3], p[5], None)
    
def p_selection_statement_2(p):
    'selection_statement : IF LPAREN expression RPAREN statement ELSE statement'
    p[0] = ast.IfStmt(p[3], p[5], p[7])

# iteration-statement
def p_iteration_statement(p):
    'iteration_statement : FOR LPAREN expression_opt SEMI expression_opt SEMI expression_opt RPAREN statement'
    p[0] = ast.ForStmt(p[3], p[5], p[7], p[9])

# expression:
def p_expression_opt_1(p):
    'expression_opt :'
    p[0] = None

def p_expression_opt_2(p):
    'expression_opt : expression'
    p[0] = p[1]

def p_expression_1(p):
    'expression : assignment_expression'
    p[0] = p[1]

def p_expression_2(p):
    'expression : expression COMMA assignment_expression'
    p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.COMMA)

# assignment_expression:
def p_assignment_expression_1(p):
    'assignment_expression : logical_or_expression'
    p[0] = p[1]

def p_assignment_expression_2(p):
    'assignment_expression : unary_expression assignment_operator assignment_expression'
    if (p[2] == '='):
        p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.EQ_ASGN)
    elif p[2] in ('*=', '/=', '%=', '+=', '-='):
        lhs = p[1].replicate()
        rhs = None
        if (p[2] == '*='):
            rhs = ast.BinOpExp(p[1], p[3], ast.BinOpExp.MUL)
        elif (p[2] == '/='):
            rhs = ast.BinOpExp(p[1], p[3], ast.BinOpExp.DIV)
        elif (p[2] == '%='):
            rhs = ast.BinOpExp(p[1], p[3], ast.BinOpExp.MOD)
        elif (p[2] == '+='):
            rhs = ast.BinOpExp(p[1], p[3], ast.BinOpExp.ADD)
        elif (p[2] == '-='):
            rhs = ast.BinOpExp(p[1], p[3], ast.BinOpExp.SUB)
        else:
            err('orio.module.tilic.code_parser internal error: unknown assignment operator')
        p[0] = ast.BinOpExp(lhs, rhs, ast.BinOpExp.EQ_ASGN)
    else:
        err('orio.module.tilic.code_parser internal error: unknown assignment operator')

# assignment-operator:
def p_assignment_operator(p):
    '''assignment_operator : EQUALS
                           | TIMESEQUAL
                           | DIVEQUAL
                           | MODEQUAL
                           | PLUSEQUAL
                           | MINUSEQUAL
                           '''
    p[0] = p[1]

# logical-or-expression
def p_logical_or_expression_1(p):
    'logical_or_expression : logical_and_expression'
    p[0] = p[1]

def p_logical_or_expression_2(p):
    'logical_or_expression : logical_or_expression LOR logical_and_expression'
    p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.LOR)

# logical-and-expression
def p_logical_and_expression_1(p):
    'logical_and_expression : equality_expression'
    p[0] = p[1]

def p_logical_and_expression_2(p):
    'logical_and_expression : logical_and_expression LAND equality_expression'
    p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.LAND)

# equality-expression:
def p_equality_expression_1(p):
    'equality_expression : relational_expression'
    p[0] = p[1]

def p_equality_expression_2(p):
    'equality_expression : equality_expression equality_operator relational_expression'
    if p[2] == '==':
        p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.EQ)
    elif p[2] == '!=':
        p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.NE)
    else:
        err('orio.module.tilic.code_parser internal error: unknown equality operator')

# equality-operator:
def p_equality_operator(p):
    '''equality_operator : EQ
                         | NE'''
    p[0] = p[1]

# relational-expression:
def p_relational_expression_1(p):
    'relational_expression : additive_expression'
    p[0] = p[1]

def p_relational_expression_2(p):
    'relational_expression : relational_expression relational_operator additive_expression'
    if (p[2] == '<'):
        p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.LT)
    elif (p[2] == '>'):
        p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.GT)
    elif (p[2] == '<='):
        p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.LE)
    elif (p[2] == '>='):
        p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.GE)
    else:
        err('orio.module.tilic.code_parser internal error: unknown relational operator')
        
# relational-operator
def p_relational_operator(p):
    '''relational_operator : LT
                           | GT
                           | LE
                           | GE'''
    p[0] = p[1]

# additive-expression
def p_additive_expression_1(p):
    'additive_expression : multiplicative_expression'
    p[0] = p[1]

def p_additive_expression_2(p):
    'additive_expression : additive_expression additive_operator multiplicative_expression'
    if (p[2] == '+'):
        p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.ADD)
    elif (p[2] == '-'):
        p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.SUB)
    else:
        err('orio.module.tilic.code_parser internal error: unknown additive operator' )

# additive-operator:
def p_additive_operator(p):
    '''additive_operator : PLUS
                         | MINUS'''
    p[0] = p[1]

# multiplicative-expression
def p_multiplicative_expression_1(p):
    'multiplicative_expression : unary_expression'
    p[0] = p[1]

def p_multiplicative_expression_2(p):
    'multiplicative_expression : multiplicative_expression multiplicative_operator unary_expression'
    if (p[2] == '*'):
        p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.MUL)
    elif (p[2] == '/'):
        p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.DIV)
    elif (p[2] == '%'):
        p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.MOD)
    else:
        err('orio.module.tilic.code_parser internal error: unknown multiplicative operator')

# multiplicative-operator
def p_multiplicative_operator(p):
    '''multiplicative_operator : TIMES
                               | DIVIDE
                               | MOD'''
    p[0] = p[1]

# unary-expression:
def p_unary_expression_1(p):
    'unary_expression : postfix_expression'
    p[0] = p[1]

def p_unary_expression_2(p):
    'unary_expression : PLUSPLUS unary_expression'
    p[0] = ast.UnaryExp(p[2], ast.UnaryExp.PRE_INC)

def p_unary_expression_3(p):
    'unary_expression : MINUSMINUS unary_expression'
    p[0] = ast.UnaryExp(p[2], ast.UnaryExp.PRE_DEC)

def p_unary_expression_4(p):
    'unary_expression : unary_operator unary_expression'
    if p[1] == '+':
        p[0] = ast.UnaryExp(p[2], ast.UnaryExp.PLUS)
    elif p[1] == '-':
        p[0] = ast.UnaryExp(p[2], ast.UnaryExp.MINUS)
    elif p[1] == '!':
        p[0] = ast.UnaryExp(p[2], ast.UnaryExp.LNOT)
    else:
        err('orio.module.tilic.code_parser internal error: unknown unary operator')

# unary-operator
def p_unary_operator(p):
    '''unary_operator : PLUS
                      | MINUS
                      | LNOT '''
    p[0] = p[1]

# postfix-expression
def p_postfix_expression_1(p):
    'postfix_expression : primary_expression'
    p[0] = p[1]

def p_postfix_expression_2(p):
    'postfix_expression : postfix_expression LBRACKET expression RBRACKET'
    p[0] = ast.ArrayRefExp(p[1], p[3])

def p_postfix_expression_3(p):
    'postfix_expression : postfix_expression LPAREN argument_expression_list_opt RPAREN'
    p[0] = ast.FunCallExp(p[1], p[3])

def p_postfix_expression_4(p):
    'postfix_expression : postfix_expression PLUSPLUS'
    p[0] = ast.UnaryExp(p[1], ast.UnaryExp.POST_INC)

def p_postfix_expression_5(p):
    'postfix_expression : postfix_expression MINUSMINUS'
    p[0] = ast.UnaryExp(p[1], ast.UnaryExp.POST_DEC)

# primary-expression
def p_primary_expression_1(p):
    'primary_expression : ID'
    p[0] = ast.IdentExp(p[1])

def p_primary_expression_2(p):
    'primary_expression : ICONST'
    val = int(p[1])
    p[0] = ast.NumLitExp(val, ast.NumLitExp.INT)

def p_primary_expression_3(p):
    'primary_expression : FCONST'
    val = float(p[1])
    p[0] = ast.NumLitExp(val, ast.NumLitExp.FLOAT)

def p_primary_expression_4(p):
    'primary_expression : SCONST'
    p[0] = ast.StringLitExp(p[1])

def p_primary_expression_5(p):
    '''primary_expression : LPAREN expression RPAREN'''
    p[0] = ast.ParenthExp(p[2])

# argument-expression-list:
def p_argument_expression_list_opt_1(p):
    'argument_expression_list_opt :'
    p[0] = []
     
def p_argument_expression_list_opt_2(p):
    'argument_expression_list_opt : argument_expression_list'
    p[0] = p[1]

def p_argument_expression_list_1(p):
    'argument_expression_list : assignment_expression' 
    p[0] = [p[1]]

def p_argument_expression_list_2(p):
    'argument_expression_list : argument_expression_list COMMA assignment_expression' 
    p[1].append(p[3])
    p[0] = p[1]

# grammatical error
def p_error(p):
    err('orio.module.tilic.code_parser: Tilic: grammatical error: "%s"' % p.value)

#------------------------------------------------

def getParser():
    '''Create the parser basing on the above C-subset language'''

    # create the lexer and parser
    lexer = tool.ply.lex.lex()
    parser = tool.ply.yacc.yacc(method='LALR', debug=0)

    # return the parser
    return parser
    
