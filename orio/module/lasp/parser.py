#
# Parser
#
import sys
import orio.tool.ply.lex
import orio.main.util.globals as g

#----------------------------------------------------------------------------------------------------------------------
# LEXER
class LASPLexer:
  def __init__(self):
    pass

  keywords = [
    'scalar', 'vector', 'matrix',
    'in', 'inout', 'out'
  ]

  reserved = {}
  for r in keywords:
    reserved[r] = r.upper()

  tokens = list(reserved.values()) + [
    # identifiers and literals
    'ID', 'ILIT', 'FLIT', 'SLIT',

    # operators (+,-,*,/,%,||,&&,!,<,<=,>,>=,==,!=)
    'LOR', 'LAND', 'LNOT',
    'LT', 'LE', 'GT', 'GE', 'EE', 'NE',

    # assignment (=, *=, /=, %=, +=, -=)
    'EQ', 'MULTEQ', 'DIVEQ', 'MODEQ', 'PLUSEQ', 'MINUSEQ',

    # increment/decrement (++,--)
    'PP', 'MM'
  ]

  # A string containing ignored characters (spaces and tabs)
  t_ignore = ' \t'
  t_ignore_LCOMMENT = r'\#.*'
  
  literals = "+-*/%()[]{},;:'."
  
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
  
  # assignment operators
  t_EQ      = r'='
  t_MULTEQ  = r'\*='
  t_DIVEQ   = r'/='
  t_MODEQ   = r'%='
  t_PLUSEQ  = r'\+='
  t_MINUSEQ = r'-='
  
  # increment/decrement
  t_PP      = r'\+\+'
  t_MM      = r'--'
  
  # integer literal
  t_ILIT    = r'\d+'
  
  # floating literal
  t_FLIT    = r'((\d+)(\.\d*)([eE](\+|-)?(\d+))? | (\d+)[eE](\+|-)?(\d+))'
  
  # string literal
  t_SLIT    = r'\"([^\\\n]|(\\.))*?\"'

  def t_ID(self, t):
    r'[A-Za-z_]([A-Za-z0-9_\.]*[A-Za-z0-9_]+)*'
    t.type = self.reserved.get(t.value, 'ID')
    return t

  def t_newline(self, t):
    r'\n+'
    t.lexer.lineno += len(t.value)
  
  def t_error(self, t):
    g.err('orio.module.lasp.parser.lexer: illegal character (%s) at line %s' % (t.value[0], t.lexer.lineno))

  def build(self, **kwargs):
    self.lexer = orio.tool.ply.lex.lex(module=self, **kwargs)
  
  def test(self, data):
    self.lexer.input(data)
    while 1:
      tok = self.lexer.token()
      if not tok: break
      print tok

#----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
  l = LASPLexer()
  l.build(debug=0, optimize=0)
  for i in range(1, len(sys.argv)):
    #print "About to lex %s" % sys.argv[i]
    f = open(sys.argv[i], "r")
    s = f.read()
    f.close()
    #print "Contents of %s: %s" % (sys.argv[i], s)
    # Test the lexer; just print out all tokens founds
    l.test(s)


