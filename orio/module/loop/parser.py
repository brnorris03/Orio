#
# Contain syntax and grammar specifications of the annotations language
#

import sys
from orio.module.loop import ast
import orio.tool.ply.lex
import orio.tool.ply.yacc
from orio.main.util.globals import *

#------------------------------------------------

__start_line_no = 1
__line_no = 1

#------------------------------------------------

# reserved words
reserved = ['IF', 'ELSE', 'FOR', 'TRANSFORM', 'NOT', 'AND', 'OR', 'GOTO', 'CONST',
            'REGISTER', 'VOLATILE']

tokens = reserved + [

    # literals (identifier, integer constant, float constant, string constant)
    'ID', 'ICONST', 'FCONST', 'SCONST_D', 'SCONST_S',

    # operators (+,-,*,/,%,||,&&,!,<,<=,>,>=,==,!=)
    'PLUS', 'MINUS', 'TIMES', 'DIVIDE', 'MOD',
    'LOR', 'LAND', 'LNOT',
    'LT', 'LE', 'GT', 'GE', 'EQ', 'NE',

    # assignment (=, *=, /=, %=, +=, -=)
    'EQUALS', 'TIMESEQUAL', 'DIVEQUAL', 'MODEQUAL', 'PLUSEQUAL', 'MINUSEQUAL',

    # increment/decrement (++,--)
    'PLUSPLUS', 'MINUSMINUS',
    

    # delimeters ( ) [ ] { } , ; :
    'LPAREN', 'RPAREN',
    'LBRACKET', 'RBRACKET',
    'LBRACE', 'RBRACE',
    'COMMA', 'SEMI', 'COLON', 'PERIOD',
    'LINECOMMENT'
    ]

# Comments
t_LINECOMMENT    = r'\#.*'
      
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
t_COLON            = r':'
t_PERIOD           = r'\.'

# ignored characters
t_ignore = ' \t'

# reserved words
reserved_map = {}
for r in reserved:
    reserved_map[r.lower()] = r

# identifiers
def t_ID(t):
    r'[A-Za-z_]([_\.\w]*[_\w]+)*'
    t.type = reserved_map.get(t.value,'ID')
    return t

# integer literal
t_ICONST     = r'\d+'

# floating literal
t_FCONST     = r'((\d+)(\.\d*)([eE](\+|-)?(\d+))? | (\d+)[eE](\+|-)?(\d+))'

# string literal (with double quotes)
t_SCONST_D   = r'\"([^\\\n]|(\\.))*?\"'

# string literal (with single quotes)
t_SCONST_S   = r'\'([^\\\n]|(\\.))*?\''

def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)
    
# syntactical error
def t_error(t):
    err('orio.module.loop.parser: %s: syntactical error: "%s"' % ((t.lineno + __start_line_no - 1), t.value[0]))
    
#------------------------------------------------


# annotation
def p_annotation(p):
    'annotation : statement_list'
    p[0] = p[1]
    
# statement-list
def p_statement_list_1(p):
    'statement_list :'
    p[0] = []

    
def p_statement_list_2(p):
    'statement_list : statement'
    p[0] = [p[1]]
    
def p_statement_list_3(p):
    'statement_list : statement_list statement'
    p[1].append(p[2])
    p[0] = p[1]
    
# statement:
def p_statement(p):
    '''statement : labeled_statement
                 | goto_statement
                 | expression_statement
                 | compound_statement
                 | selection_statement
                 | iteration_statement
                 | transformation_statement
                 | line_comment
                 | declaration
                 '''
    p[0] = p[1]

    
# labeled statement (limited)
def p_labeled_statement(p):
    'labeled_statement : label COLON statement'
    p[3].setLabel(p[1])
    p[0] = p[3]

def p_label(p):
    '''label : ID 
            | ICONST
    '''
    p[0] = p[1]
    
# line comment
def p_line_comment(p):
    'line_comment : LINECOMMENT'
    p[0] = ast.Comment(p[1], line_no=str(p.lineno(1) + __start_line_no - 1))
    
#def p_optional_line_comment(p):
#    '''optional_line_comment : line_comment
#                            | empty
#                            '''
#    if p[1]: 
#        p[0] = ast.Comment(p[1], line_no=str(p.lineno(1) + __start_line_no - 1))
#    p[0] = None

def p_declaration_1(p):
    'declaration : typename var_decl_list SEMI'
    p[0] = ast.DeclStmt()  # list of declarations
    for var in p[2]:
        if len(var) == 1:
            p[0].append(ast.VarDecl(type_name=p[1],var_names=var))
        else:
            p[0].append(ast.VarDeclInit(type_name=p[1],var_name=var[0],init_exp=var[1]))

def p_declaration_2(p):
    'declaration : qual typename var_decl_list SEMI'
    p[0] = ast.DeclStmt()  # list of declarations
    for var in p[3]:
        if len(var) == 1:
            p[0].append(ast.VarDecl(type_name=p[2],var_names=var,qual=p[1]))
        else:
            p[0].append(ast.VarDeclInit(type_name=p[2],var_name=var[0],init_exp=var[1],qual=p[1]))

def p_typename(p):
    'typename : ID'
    p[0] = p[1]

def p_qual(p):
    """qual : CONST
            | REGISTER
            | VOLATILE
    """
    p[0] = p[1]

def p_var_decl_list_1(p):
    'var_decl_list : var_decl'
    p[0] = [p[1]]

def p_var_decl_list_2(p):
    'var_decl_list : var_decl_list var_decl'
    p[1].append(p[2])
    p[0] = p[1]

def p_var_decl_1(p):
    'var_decl    : ID'
    p[0] = [ast.IdentExp(p[1])]

def p_var_decl_2(p):
    'var_decl    : ID EQUALS expression'
    p[0] = (ast.IdentExp(p[1]), p[3])


# expression-statement:
def p_expression_statement(p):
    'expression_statement : expression_opt SEMI'
    p[0] = ast.ExpStmt(p[1], line_no=str(p.lineno(1) + __start_line_no - 1))
    
def p_goto_statement(p):
    'goto_statement : GOTO label SEMI'
    p[0] = ast.GotoStmt(p[2], line_no=str(p.lineno(1) + __start_line_no - 1))

# compound-statement:
def p_compound_statement(p):
    'compound_statement : LBRACE statement_list RBRACE'
    p[0] = ast.CompStmt(p[2], line_no=str(p.lineno(1) + __start_line_no - 1))
    
# selection-statement
# Note:
#   This results in a shift/reduce conflict. However, such conflict is not harmful
#   because PLY resolves such conflict in favor of shifting.
def p_selection_statement_1(p):
    'selection_statement : IF LPAREN expression RPAREN statement'
    p[0] = ast.IfStmt(p[3], p[5], None, line_no=str(p.lineno(1) + __start_line_no - 1))
    
def p_selection_statement_2(p):
    'selection_statement : IF LPAREN expression RPAREN statement ELSE statement'
    p[0] = ast.IfStmt(p[3], p[5], p[7], line_no=str(p.lineno(1) + __start_line_no - 1))

# iteration-statement
def p_iteration_statement(p):
    'iteration_statement : FOR LPAREN expression_opt SEMI expression_opt SEMI expression_opt RPAREN statement'
    p[0] = ast.ForStmt(p[3], p[5], p[7], p[9], line_no=str(p.lineno(1) + __start_line_no - 1))

# transformation-statement
def p_transformation_statement(p):
    'transformation_statement : TRANSFORM ID LPAREN transformation_argument_list_opt RPAREN statement'
    p[0] = ast.TransformStmt(p[2], p[4], p[6], line_no=str(p.lineno(1) + __start_line_no - 1))

def p_transformation_statement2(p):
    'transformation_statement : TRANSFORM ID LPAREN transformation_argument_list_opt RPAREN'
    p[0] = ast.TransformStmt(p[2], p[4], None, line_no=str(p.lineno(1) + __start_line_no - 1))

# transformation-argument-list
def p_transformation_argument_list_opt_1(p):
    'transformation_argument_list_opt :'
    p[0] = []

def p_transformation_argument_list_opt_2(p):
    'transformation_argument_list_opt : transformation_argument_list'
    p[0] = p[1]

def p_transformation_argument_list_1(p):
    'transformation_argument_list : transformation_argument'
    p[0] = [p[1]]

def p_transformation_argument_list_2(p):
    'transformation_argument_list : transformation_argument_list COMMA transformation_argument'
    p[1].append(p[3])
    p[0] = p[1]

# transformation-argument
def p_transformation_argument(p):
    'transformation_argument : ID EQUALS py_expression'
    p[0] = [p[1], p[3], str(p.lineno(1) + __start_line_no - 1)]

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
    p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.COMMA, line_no=str(p.lineno(1) + __start_line_no - 1))

def p_expression_3(p):
    'expression : typename ID'
    p[0] = ast.VarDecl(p[1], [p[2]], line_no=str(p.lineno(1) + __start_line_no - 1))

#def p_expression_4(p):
#    'expression : ID ID EQUALS expression'
#    p[0] = ast.VarDeclInit(p[1], ast.IdentExp(p[2]), p[4], line_no=str(p.lineno(1) + __start_line_no - 1))

# assignment_expression:
def p_assignment_expression_1(p):
    'assignment_expression : logical_or_expression'
    p[0] = p[1]

def p_assignment_expression_2(p):
    'assignment_expression : unary_expression assignment_operator assignment_expression'
    if (p[2] == '='):
        p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.EQ_ASGN, line_no=str(p.lineno(1) + __start_line_no - 1))
    elif p[2] in ('*=', '/=', '%=', '+=', '-='):
        lhs = p[1].replicate()
        rhs = None
        if (p[2] == '*='):
            rhs = ast.BinOpExp(p[1], p[3], ast.BinOpExp.MUL, line_no=str(p.lineno(1) + __start_line_no - 1))
        elif (p[2] == '/='):
            rhs = ast.BinOpExp(p[1], p[3], ast.BinOpExp.DIV, line_no=str(p.lineno(1) + __start_line_no - 1))
        elif (p[2] == '%='):
            rhs = ast.BinOpExp(p[1], p[3], ast.BinOpExp.MOD, line_no=str(p.lineno(1) + __start_line_no - 1))
        elif (p[2] == '+='):
            rhs = ast.BinOpExp(p[1], p[3], ast.BinOpExp.ADD, line_no=str(p.lineno(1) + __start_line_no - 1))
        elif (p[2] == '-='):
            rhs = ast.BinOpExp(p[1], p[3], ast.BinOpExp.SUB, line_no=str(p.lineno(1) + __start_line_no - 1))
        else:
            err('orio.module.loop.parser internal error: missing case for assignment operator')
        p[0] = ast.BinOpExp(lhs, rhs, ast.BinOpExp.EQ_ASGN, line_no=str(p.lineno(1) + __start_line_no - 1))
    else:
        err('orio.module.loop.parser internal error: unknown assignment operator')

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
    p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.LOR, line_no=str(p.lineno(1) + __start_line_no - 1))

# logical-and-expression
def p_logical_and_expression_1(p):
    'logical_and_expression : equality_expression'
    p[0] = p[1]

def p_logical_and_expression_2(p):
    'logical_and_expression : logical_and_expression LAND equality_expression'
    p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.LAND, line_no=str(p.lineno(1) + __start_line_no - 1))

# equality-expression:
def p_equality_expression_1(p):
    'equality_expression : relational_expression'
    p[0] = p[1]

def p_equality_expression_2(p):
    'equality_expression : equality_expression equality_operator relational_expression'
    if p[2] == '==':
        p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.EQ, line_no=str(p.lineno(1) + __start_line_no - 1))
    elif p[2] == '!=':
        p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.NE, line_no=str(p.lineno(1) + __start_line_no - 1))
    else:
        err('orio.module.loop.parser internal error: unknown equality operator')

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
        p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.LT, line_no=str(p.lineno(1) + __start_line_no - 1))
    elif (p[2] == '>'):
        p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.GT, line_no=str(p.lineno(1) + __start_line_no - 1))
    elif (p[2] == '<='):
        p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.LE, line_no=str(p.lineno(1) + __start_line_no - 1))
    elif (p[2] == '>='):
        p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.GE, line_no=str(p.lineno(1) + __start_line_no - 1))
    else:
        err('orio.module.loop.parser internal error: unknown relational operator')
        
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
        p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.ADD, line_no=str(p.lineno(1) + __start_line_no - 1))
    elif (p[2] == '-'):
        p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.SUB, line_no=str(p.lineno(1) + __start_line_no - 1))
    else:
        err('orio.module.loop.parser internal error: unknown additive operator' )

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
        p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.MUL, line_no=str(p.lineno(1) + __start_line_no - 1))
    elif (p[2] == '/'):
        p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.DIV, line_no=str(p.lineno(1) + __start_line_no - 1))
    elif (p[2] == '%'):
        p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.MOD, line_no=str(p.lineno(1) + __start_line_no - 1))
    else:
        err('orio.module.loop.parser internal error: unknown multiplicative operator')

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
    p[0] = ast.UnaryExp(p[2], ast.UnaryExp.PRE_INC, line_no=str(p.lineno(1) + __start_line_no - 1))

def p_unary_expression_3(p):
    'unary_expression : MINUSMINUS unary_expression'
    p[0] = ast.UnaryExp(p[2], ast.UnaryExp.PRE_DEC, line_no=str(p.lineno(1) + __start_line_no - 1))

def p_unary_expression_4(p):
    'unary_expression : unary_operator unary_expression'
    if p[1] == '+':
        p[0] = ast.UnaryExp(p[2], ast.UnaryExp.PLUS, line_no=str(p.lineno(1) + __start_line_no - 1))
    elif p[1] == '-':
        p[0] = ast.UnaryExp(p[2], ast.UnaryExp.MINUS, line_no=str(p.lineno(1) + __start_line_no - 1))
    elif p[1] == '!':
        p[0] = ast.UnaryExp(p[2], ast.UnaryExp.LNOT, line_no=str(p.lineno(1) + __start_line_no - 1))
    else:
        err('orio.module.loop.parser internal error: unknown unary operator')

def p_unary_expression_5(p):
    'unary_expression : LPAREN ID RPAREN unary_expression'
    p[0] = ast.CastExpr(p[2], p[4], line_no=str(p.lineno(1) + __start_line_no - 1))

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
    p[0] = ast.ArrayRefExp(p[1], p[3], line_no=str(p.lineno(1) + __start_line_no - 1))

def p_postfix_expression_3(p):
    'postfix_expression : postfix_expression LPAREN argument_expression_list_opt RPAREN'
    p[0] = ast.FunCallExp(p[1], p[3], line_no=str(p.lineno(1) + __start_line_no - 1))

def p_postfix_expression_4(p):
    'postfix_expression : postfix_expression PLUSPLUS'
    p[0] = ast.UnaryExp(p[1], ast.UnaryExp.POST_INC, line_no=str(p.lineno(1) + __start_line_no - 1))

def p_postfix_expression_5(p):
    'postfix_expression : postfix_expression MINUSMINUS'
    p[0] = ast.UnaryExp(p[1], ast.UnaryExp.POST_DEC, line_no=str(p.lineno(1) + __start_line_no - 1))

# primary-expression
def p_primary_expression_1(p):
    'primary_expression : ID'
    p[0] = ast.IdentExp(p[1], line_no=str(p.lineno(1) + __start_line_no - 1))

def p_primary_expression_2(p):
    'primary_expression : ICONST'
    val = int(p[1])
    p[0] = ast.NumLitExp(val, ast.NumLitExp.INT, line_no=str(p.lineno(1) + __start_line_no - 1))

def p_primary_expression_3(p):
    'primary_expression : FCONST'
    val = float(p[1])
    p[0] = ast.NumLitExp(val, ast.NumLitExp.FLOAT, line_no=str(p.lineno(1) + __start_line_no - 1))

def p_primary_expression_4(p):
    'primary_expression : SCONST_D'
    p[0] = ast.StringLitExp(p[1], line_no=str(p.lineno(1) + __start_line_no - 1))

def p_primary_expression_5(p):
    '''primary_expression : LPAREN expression RPAREN'''
    p[0] = ast.ParenthExp(p[2], line_no=str(p.lineno(1) + __start_line_no - 1))

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

    line,col = find_column(p.lexer.lexdata,p)
    pos = (col-1)*' '
    err("[orio.module.loop.parser] unexpected symbol '%s' at line %s, column %s:\n\t%s\n\t%s^" \
        % (p.value, p.lexer.lineno, col, line, pos))
   
    #err('orio.module.loop.parser: %s: grammatical error: "%s"' % ((p.lineno + __start_line_no - 1), p.value))

# Compute column. 
#     input is the input text string
#     token is a token instance
def find_column(input,token):
    i = token.lexpos
    startline = input[:i].rfind('\n')
    endline = startline + input[startline+1:].find('\n') 
    line = input[startline+1:endline+1]
    while i > 0:
        if input[i] == '\n': break
        i -= 1
    column = (token.lexpos - i)
    return line, column

#------------------------------------------------

# Below is a grammar subset of Python expression
# py-expression
def p_py_expression_1(p):
    'py_expression : py_m_expression'
    p[0] = p[1]

def p_py_expression_2(p):
    'py_expression : py_conditional_expression'
    p[0] = p[1]

# py-expression-list
def p_py_expression_list_opt_1(p):
    'py_expression_list_opt : '
    p[0] = ''

def p_py_expression_list_opt_2(p):
    'py_expression_list_opt : py_expression_list'
    p[0] = p[1]

def p_py_expression_list_1(p):
    'py_expression_list : py_expression'
    p[0] = p[1]

def p_py_expression_list_2(p):
    'py_expression_list : py_expression_list COMMA py_expression'
    p[0] = p[1] + p[2] + p[3]

# py-conditional-expression
def p_py_conditional_expression(p):
    'py_conditional_expression : py_m_expression IF py_m_expression ELSE py_m_expression'
    p[0] = p[1] + ' ' + p[2] + ' ' + p[3] + ' ' + p[4] + ' ' + p[5]

# py-m-expression
def p_py_m_expression_1(p):
    'py_m_expression : py_u_expression'
    p[0] = p[1]

def p_py_m_expression_2(p):
    'py_m_expression : py_m_expression py_binary_operator py_u_expression'
    p[0] = p[1] + ' ' +  p[2] + ' ' + p[3]

# py-binary-operator
def p_py_binary_operator(p):
    '''py_binary_operator : PLUS
                          | MINUS
                          | TIMES
                          | DIVIDE
                          | MOD
                          | LT
                          | GT
                          | LE
                          | GE
                          | EQ
                          | NE
                          | AND
                          | OR'''
    p[0] = p[1]

# py-u-expression
def p_py_u_expression_1(p):
    'py_u_expression : py_primary'
    p[0] = p[1]

def p_py_u_expression_2(p):
    '''py_u_expression : PLUS py_u_expression
                       | MINUS py_u_expression
                       | NOT py_u_expression'''
    p[0] = p[1] + ' ' + p[2]

# py-primary
def p_py_primary(p):
    '''py_primary : py_atom
                  | py_subscription
                  | py_attribute_ref
                  | py_call
                  | py_list_display
                  | py_dict_display'''
    p[0] = p[1]

# py-subscription
def p_py_subscription(p):
    'py_subscription : py_primary LBRACKET py_expression_list RBRACKET'
    p[0] = p[1] + p[2] + p[3] + p[4]

# py-attribute-ref
def p_py_attribute_ref(p):
    'py_attribute_ref : py_primary PERIOD ID'
    p[0] = p[1] + p[2] + p[3]

# py-call
def p_py_call(p):
    'py_call : py_primary LPAREN py_expression_list_opt RPAREN'
    p[0] = p[1] + p[2] + p[3] + p[4]

# py-list-display
def p_py_list_display(p):
    'py_list_display : LBRACKET py_expression_list_opt RBRACKET'
    p[0] = p[1] + p[2] + p[3]

# py-dict-display
def p_py_dict_display(p):
    'py_dict_display : LBRACE py_key_datum_list_opt RBRACE'
    p[0] = p[1] + p[2] + p[3]

# py-key-datum-list
def p_py_key_datum_list_opt_1(p):
    'py_key_datum_list_opt : '
    p[0] = ''

def p_py_key_datum_list_opt_2(p):
    'py_key_datum_list_opt : py_key_datum_list'
    p[0] = p[1]

def p_py_key_datum_list_1(p):
    'py_key_datum_list : py_key_datum'
    p[0] = p[1]

def p_py_key_datum_list_2(p):
    'py_key_datum_list : py_key_datum_list COMMA py_key_datum'
    p[0] = p[1] + p[2] + p[3]

# py-key-datum
def p_py_key_datum(p):
    'py_key_datum : py_expression COLON py_expression'
    p[0] = p[1] + p[2] + p[3]

# py-atom
def p_py_atom_1(p):
    '''py_atom : ID
               | ICONST
               | FCONST
               | SCONST_D
               | SCONST_S'''
    p[0] = p[1]

def p_py_atom_2(p):
    'py_atom : LPAREN py_expression_list_opt RPAREN'
    p[0] = p[1] + p[2] + p[3]

#def p_empty(p):
#    'empty : '
#    p[0] = None
    
#------------------------------------------------

def getParser(start_line_no):
    '''Create the parser for the annotations language'''

    # set the starting line number
    global __start_line_no
    __start_line_no = start_line_no
    global __line_no
    __line_no = start_line_no

    # create the lexer and parser
    lexer = orio.tool.ply.lex.lex()
    parser = orio.tool.ply.yacc.yacc(method='LALR', debug=0, optimize=1, write_tables=0)

    # return the parser
    return parser
    


