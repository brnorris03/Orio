#!/usr/bin/env python
#
# Parser
#
import sys, os
import orio.tool.ply.lex, orio.tool.ply.yacc
import orio.main.util.globals as g
import orio.module.splingo.ast as ast

#----------------------------------------------------------------------------------------------------------------------
# LEXER
class SpLingoLexer:
  def __init__(self):
    pass

  keywords = [
    'scalar', 'vector', 'matrix',
    'in', 'out', #'inout',
    'dia',
    'register', 'auto', 'extern', 'static'
  ]

  reserved = {}
  for k in keywords:
    reserved[k] = k.upper()

  tokens = list(reserved.values()) + [
    # identifiers and literals
    'ID', 'ILIT', 'FLIT', 'SLIT',

    # operators (||,&&,<=,>=,==,!=)
    #'LOR', 'LAND', 'LNOT',
    #LE', GE', 'EE', 'NE',

    # shorthand assignment (*=, /=, %=, +=, -=)
    #'MULTEQ', 'DIVEQ', 'MODEQ', 'PLUSEQ', 'MINUSEQ',

    # increment/decrement (++,--)
    'PP', 'MM',
    'SLCOMMENT', 'MLCOMMENT'
  ]

  # A string containing ignored characters (spaces and tabs)
  t_ignore = ' \t'

  # operators
  #t_LOR     = r'\|\|'
  #t_LAND    = r'&&'
  #t_LE      = r'<='
  #t_GE      = r'>='
  #t_EE      = r'=='
  #t_NE      = r'!='
  
  # shorthand assignment operators
  #t_MULTEQ  = r'\*='
  #t_DIVEQ   = r'/='
  #t_MODEQ   = r'%='
  #t_PLUSEQ  = r'\+='
  #t_MINUSEQ = r'-='
  
  # increment/decrement
  t_PP      = r'\+\+'
  t_MM      = r'--'
  
  literals = "+-*/%()[]{},;:'.=<>!"
  
  # integer literal
  t_ILIT    = r'\d+'
  
  # floating literal
  t_FLIT    = r'((\d+)(\.\d*)([eE](\+|-)?(\d+))? | (\d+)[eE](\+|-)?(\d+))'
  
  # string literal
  t_SLIT    = r'\"([^\\\n]|(\\.))*?\"'

  def t_ID(self, t):
    r'[A-Za-z_]([A-Za-z0-9_]*[A-Za-z0-9_]+)*'
    t.type = self.reserved.get(t.value, 'ID')
    return t

  def t_SLCOMMENT(self, t):
    r'//.*'
    t.value = t.value[2:]
    return t

  def t_MLCOMMENT(self, t):
    r'/\*[^/]*\*/'
    t.value = t.value[2:-2]
    return t

  def t_NEWLINE(self, t):
    r'\n+'
    t.lexer.lineno += len(t.value)
  
  def t_error(self, t):
    g.err('%s: illegal character (%s) at line %s' % (self.__class__, t.value[0], t.lexer.lineno))

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
# GRAMMAR
tokens = SpLingoLexer.tokens
start = 'prog'
parser = None
elixir = None
def p_prog_a(p):
    '''prog : sid IN params OUT params '{' stmts '}' '''
    p[0] = ast.FunDec(p[1], ast.IdentExp('void'), [], p[3]+p[5], p[7])
    #print codegen.CodeGen().generate(p[0], '', '  ')

def p_prog_b(p):
    '''prog : sid IN params            '{' stmts '}'
            | sid           OUT params '{' stmts '}' '''
    p[0] = ast.FunDec(p[1], ast.IdentExp('void'), [], p[3], p[5])

def p_params(p):
    '''params : param
              | params ',' param'''
    if len(p) == 2:
      p[0] = [p[1]]
    else:
      p[0] = p[1] + [p[3]]

def p_param(p):
    '''param : sid ':' type'''
    #tname = reduce(lambda acc,item: acc+'.'+item, p[3][1:], p[3][0])
    p[0] = ast.ParamDec(p[3], p[1])

def p_type(p):
    '''type : MATRIX subty
            | VECTOR
            | SCALAR'''
    if len(p) == 3:
      if p[2] is None:
        p[0] = ast.IdentExp(p[1])
      else:
        p[0] = ast.QualIdentExp(p[1], p[2])
    else:
      p[0] = ast.IdentExp(p[1])

def p_subty(p):
    '''subty : "." DIA
            | empty'''
    if p[1] is None:
      p[0] = None
    else:
      p[0] = ast.IdentExp(p[2])

#------------------------------------------------------------------------------
def p_stmts(p):
    '''stmts : stmt
             | stmts stmt'''
    if len(p) == 2:
      p[0] = ast.CompStmt([p[1]])
    else:
      p[0] = ast.CompStmt(p[1].stmts + [p[2]])

def p_stmt_eq(p):
    '''stmt : exp '''
    #TODO: ensure correpondence of stored coordinates to file positions
    coord = p.lineno(1)
    p[0] = ast.ExpStmt(p[1], coord)

def p_stmt_dec(p):
    '''stmt :       sid exp ';'
            | quals sid exp ';' '''
    if len(p) == 4:
      p[0] = ast.VarDec(p[1], [p[2]], True, [], p.lineno(3))
    else:
      p[0] = ast.VarDec(p[2], [p[3]], True, p[1], p.lineno(4))

def p_stmt_comment(p):
    '''stmt : comment'''
    p[0] = p[1]

def p_comment(p):
    '''comment : SLCOMMENT
               | MLCOMMENT'''
    p[0] = ast.Comment(p[1], p.lineno(1))

def p_quals(p):
    '''quals : qual
             | quals qual'''
    if len(p) == 2:
      p[0] = [p[1]]
    else:
      p[0] = p[1].append(p[2])

def p_qual(p):
    '''qual : REGISTER
            | AUTO
            | EXTERN
            | STATIC '''
    p[0] = p[1]

#------------------------------------------------------------------------------

precedence = (
    ('left', ','),
    ('right', '='),
    ('left', '<', '>'),
    ('left', '+', '-'),
    ('left', '*', '/', '%'),
    ('left', 'PP', 'MM')
)
#------------------------------------------------------------------------------
def p_exp_primary(p):
    '''exp : primary'''
    p[0] = p[1]

def p_exp_paren(p):
    '''exp : '(' exp ')' '''
    p[0] = ast.ParenExp(p[2], p.lineno(1))

def p_exp_seq(p):
    '''exp : exp ',' exp'''
    p[0] = ast.BinOpExp(ast.BinOpExp.COMMA, p[1], p[3], p.lineno(2))

def p_exp_eq(p):
    '''exp : exp '=' exp'''
    p[0] = ast.BinOpExp(ast.BinOpExp.EQ, p[1], p[3], p.lineno(2))

def p_exp_plus(p):
    '''exp : exp '+' exp'''
    p[0] = ast.BinOpExp(ast.BinOpExp.PLUS, p[1], p[3], p.lineno(2))

def p_exp_minus(p):
    '''exp : exp '-' exp'''
    p[0] = ast.BinOpExp(ast.BinOpExp.MINUS, p[1], p[3], p.lineno(2))

def p_exp_mult(p):
    '''exp : exp '*' exp'''
    p[0] = ast.BinOpExp(ast.BinOpExp.MULT, p[1], p[3], p.lineno(2))

def p_exp_div(p):
    '''exp : exp '/' exp'''
    p[0] = ast.BinOpExp(ast.BinOpExp.DIV, p[1], p[3], p.lineno(2))

def p_exp_mod(p):
    '''exp : exp '%' exp'''
    p[0] = ast.BinOpExp(ast.BinOpExp.MOD, p[1], p[3], p.lineno(2))

def p_exp_lt(p):
    '''exp : exp '<' exp'''
    p[0] = ast.BinOpExp(ast.BinOpExp.LT, p[1], p[3], p.lineno(2))

def p_exp_gt(p):
    '''exp : exp '>' exp'''
    p[0] = ast.BinOpExp(ast.BinOpExp.GT, p[1], p[3], p.lineno(2))

def p_exp_uminus(p):
    '''exp : '-' exp'''
    p[0] = ast.UnaryExp(ast.UnaryExp.MINUS, p[2], p.lineno(1))

def p_exp_transpose(p):
    '''exp : exp "'" '''
    p[0] = ast.UnaryExp(ast.UnaryExp.TRANSPOSE, p[1], p.lineno(2))

def p_exp_postpp(p):
    '''exp : exp PP '''
    p[0] = ast.UnaryExp(ast.UnaryExp.POST_INC, p[1], p.lineno(2))

def p_exp_postmm(p):
    '''exp : exp MM '''
    p[0] = ast.UnaryExp(ast.UnaryExp.POST_DEC, p[1], p.lineno(2))

def p_exp_array(p):
    '''exp : exp '[' exp ']' '''
    p[0] = ast.ArrayRefExp(p[1], p[3], p.lineno(2))
#------------------------------------------------------------------------------


def p_primary_id(p):
    'primary : sid'
    p[0] = p[1]

def p_primary_ilit(p):
    'primary : ILIT'
    p[0] = ast.LitExp(int(p[1]), ast.LitExp.INT, p.lineno(1))

def p_primary_flit(p):
    'primary : FLIT'
    p[0] = ast.LitExp(float(p[1]), ast.LitExp.FLOAT, p.lineno(1))

def p_primary_slit(p):
    'primary : SLIT'
    p[0] = ast.LitExp(p[1], ast.LitExp.STRING, p.lineno(1))

def p_sid(p):
    'sid : ID'
    p[0] = ast.IdentExp(p[1], p.lineno(1))

def p_empty(p):
    'empty : '
    p[0] = None

def p_error(p):
    g.err("orio.module.splingo.parser: error in input line #%s, at token-type '%s', token-value '%s'" % (p.lineno, p.type, p.value))
#----------------------------------------------------------------------------------------------------------------------



#----------------------------------------------------------------------------------------------------------------------
def parse(text):
  '''Lex, parse and create the HL AST for the DSL text'''

  global elixir
  if elixir is None:
    elixir = SpLingoLexer()
    elixir.build(debug=0, optimize=1)

  parser = orio.tool.ply.yacc.yacc(debug=0, optimize=1, write_tables=0,
                                   errorlog=orio.tool.ply.yacc.NullLogger()
                                   )
  return parser.parse(text, lexer=elixir, debug=0)


  
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

    parse(s)
    print('[parser] Successfully parsed %s' % sys.argv[i], file=sys.stderr)


