# 
# A PLY-based parser for the TSpec (Tuning Specifier)
#
import re
import orio.tool.ply.lex, orio.tool.ply.yacc
import orio.main.util.globals as g

#----------------------------------------------------------------------------------------------------------------------
# LEXER
# reserved keywords
keywords = [
    'def', 'arg', 'param', 'decl', 'let', 'spec', 'constraint', 'option',
    'build', 'build_command', 'prebuild_command', 'postbuild_command', 'postrun_command', 'batch_command', 'status_command', 'num_procs', 'libs',
    'input_params', 'input_vars', 'static', 'dynamic', 'void', 'char', 'short', 'int', 'long', 'float', 'double', '__device__',
    'performance_params', 'performance_counter', 'power', 'cmdline_params', 'method', 'repetitions',
    'search', 'time_limit', 'total_runs', 'use_z3', 'resume', 'algorithm',
    'init_file', 'decl_file',
    'exhaustive_start_coord',
    'msimplex_reflection_coef', 'msimplex_expansion_coef',
    'msimplex_contraction_coef', 'msimplex_shrinkage_coef', 'msimplex_size', 'msimplex_x0',
    'simplex_reflection_coef', 'simplex_expansion_coef',
    'simplex_contraction_coef', 'simplex_shrinkage_coef', 'simplex_local_distance', 'simplex_x0',
    'cudacfg_instmix',
    'validation', 'validation_file', 'expected_output',
    'macro', 'performance_test_code', 'skeleton_test_code', 'skeleton_code_file',
    'other', 'device_spec_file',
]

# map of reserved keywords
reserved = {}
for r in keywords:
    reserved[r] = r.upper()

# tokens
tokens = list(reserved.values()) + ['ID', 'EQ','EXPR', 'STRING', 'EXPR_IDX']

states = (
    ('pyexpr','inclusive'), # lexer state to match arbitrary expressions as raw strings
)

# A string containing ignored characters (spaces and tabs)
t_ignore = ' \t'

literals = ";[]{}"

# PLY-Lex note: when building the master regular expression, rules are added in the following order:
# 1. All tokens defined by functions are added in the same order as they appear in the lexer file.
# 2. Tokens defined by strings are added next by sorting them in order of decreasing regular expression length (longer expressions are added first).

# count newlines
def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)

def t_ID(t):
    r'[A-Za-z_]([A-Za-z0-9_\.]*[A-Za-z0-9_]+)*'
    t.type = reserved.get(t.value, 'ID')    # Check for reserved words
    #print('lexed %s:%s' %(t.type,t.value))
    return t

def t_EQ(t):
    r'='
    t.lexer.begin('pyexpr')
    return t

def t_pyexpr_EXPR(t):
    r'[^;]+'
    t.lexer.lineno += t.value.count('\n')
    t.lexer.begin('INITIAL')           
    #print('lexed expr:%s' %t.value)
    return t

def t_EXPR_IDX(t):
    r'\[([^\]])+\]'
    # remove leading and trailing brackets
    t.value = t.lexer.lexdata[(t.lexer.lexpos-len(t.value)+1):t.lexer.lexpos-1]
    return t

def t_STRING(t):
    r'"[^"]*"'
    # String literal using double quotes
    return t

# Error handling rule
def t_error(t):
    g.err('orio.main.tspec.pparser.lexer: illegal character (%s) at line %s' % (t.value[0], t.lexer.lineno))
#----------------------------------------------------------------------------------------------------------------------



#----------------------------------------------------------------------------------------------------------------------
# GRAMMAR
# file spec statements: start symbol when tspecs are in an external file
start = 'fspecs'
def p_fspecs(p):
    ''' fspecs : fspec fspecs
               | fspec
    '''
    if len(p) == 3:
        p[0] = [p[1]] + p[2]
    else:
        p[0] = p[1]

#----------------------------------------------------------------------------------------------------------------------
# file spec statement
def p_fspec1(p):
    ''' fspec : let '''
    p[0] = p[1]

def p_fspec2(p):
    ''' fspec : SPEC ID '{' specs '}' '''
    p[0] = p[4]

#----------------------------------------------------------------------------------------------------------------------
# spec body statements: start symbol when tspecs are embedded into annotations
def p_specs(p):
    ''' specs : spec specs
              | spec
    '''
    if len(p) == 3:
        p[0] = [p[1]] + p[2]
    else:
        p[0] = [p[1]]

#----------------------------------------------------------------------------------------------------------------------
# specification statement
def p_spec(p):
    ''' spec : def
             | let
    '''
    p[0] = p[1]

#----------------------------------------------------------------------------------------------------------------------
# definition statement
def p_def(p):
    ''' def : DEF deftype '{' stmts '}' '''
    p[0] = (p[1], p.lineno(1), p[2], p[4])

#----------------------------------------------------------------------------------------------------------------------
# types of a definition statement
def p_def_type(p):
    ''' deftype : BUILD
                | PERFORMANCE_PARAMS
                | PERFORMANCE_COUNTER
                | POWER
                | CMDLINE_PARAMS
                | INPUT_PARAMS
                | INPUT_VARS
                | SEARCH
                | VALIDATION
                | PERFORMANCE_TEST_CODE
                | OTHER
    '''
    p[0] = (p[1], p.lineno(1))

#----------------------------------------------------------------------------------------------------------------------
# definition body statements
def p_stmts(p):
    ''' stmts : stmt ';' stmts
              | stmt ';'
    '''
    if len(p) == 4:
        p[0] = [p[1]] + p[3]
    else:
        p[0] = [p[1]]

#----------------------------------------------------------------------------------------------------------------------
# definition body statement
def p_stmt(p):
    ''' stmt : let
             | arg
             | param
             | option
             | constraint
             | decl
    '''
    p[0] = p[1]

#----------------------------------------------------------------------------------------------------------------------
# let statement
def p_let(p):
    ''' let : LET ID EQ EXPR '''
    p[0] = (p[1], p.lineno(1), (p[2], p.lineno(2)), (p[4], p.lineno(4)))

#----------------------------------------------------------------------------------------------------------------------
# argument statement
def p_arg(p):
    ''' arg : ARG argtype EQ EXPR '''
    p[0] = (p[1], p.lineno(1), p[2], (p[4], p.lineno(4)))

#----------------------------------------------------------------------------------------------------------------------
# types of argument statements
def p_arg_type(p):
    ''' argtype : BUILD_COMMAND
                | PREBUILD_COMMAND
                | POSTBUILD_COMMAND
                | POSTRUN_COMMAND
                | BATCH_COMMAND
                | STATUS_COMMAND
                | NUM_PROCS
                | METHOD
                | REPETITIONS
                | ALGORITHM
                | TIME_LIMIT
                | TOTAL_RUNS
                | USE_Z3
                | RESUME
                | LIBS
                | INIT_FILE
                | DECL_FILE
                | EXHAUSTIVE_START_COORD
                | MSIMPLEX_EXPANSION_COEF
                | MSIMPLEX_REFLECTION_COEF
                | MSIMPLEX_CONTRACTION_COEF
                | MSIMPLEX_SHRINKAGE_COEF
                | MSIMPLEX_SIZE
                | MSIMPLEX_X0   
                | SIMPLEX_EXPANSION_COEF
                | SIMPLEX_REFLECTION_COEF
                | SIMPLEX_CONTRACTION_COEF
                | SIMPLEX_SHRINKAGE_COEF
                | SIMPLEX_LOCAL_DISTANCE    
                | SIMPLEX_X0
                | CUDACFG_INSTMIX
                | VALIDATION_FILE
                | EXPECTED_OUTPUT
                | SKELETON_TEST_CODE
                | SKELETON_CODE_FILE
                | OTHER
                | DEVICE_SPEC_FILE
    '''
    p[0] = (p[1], p.lineno(1))

#----------------------------------------------------------------------------------------------------------------------
# parameter statement
def p_param(p):
    ''' param : PARAM ID brackets EQ EXPR '''
    is_range = p[3]
    p[0] = (p[1], p.lineno(1), (p[2], p.lineno(2)), is_range, (p[5], p.lineno(5)) )

#----------------------------------------------------------------------------------------------------------------------
# constraint statement
def p_constraint(p):
    ''' constraint : CONSTRAINT ID EQ EXPR '''
    p[0] = (p[1], p.lineno(1), (p[2], p.lineno(2)), (p[4], p.lineno(4)))

#----------------------------------------------------------------------------------------------------------------------
# command-line option
def p_option(p):
    ''' option : OPTION STRING EQ EXPR '''
    is_range = p[3]
    p[0] = (p[1], p.lineno(1), (p[2], p.lineno(2)), True, (p[4], p.lineno(4)) )

#----------------------------------------------------------------------------------------------------------------------
# declaration statement
def p_decl(p):
    ''' decl : DECL dyst mods type ID arrsizes EQ EXPR
             | DECL dyst mods type ID arrsizes
    '''
    id_name = (p[5], p.lineno(5))
    types = p[2] + p[3] + p[4]
    if len(p) == 7:
        p[0] = (p[1], p.lineno(1), id_name, types, p[6], (None, None))
    else:
        p[0] = (p[1], p.lineno(1), id_name, types, p[6], (p[8], p.lineno(8)))

#----------------------------------------------------------------------------------------------------------------------
# optional static or dynamic modifier
def p_dyst(p):
    ''' dyst : DYNAMIC
             | STATIC
             | empty
    '''
    if p[1] != None:
        p[0] = [(p[1], p.lineno(1))]
    else:
        p[0] = []

#----------------------------------------------------------------------------------------------------------------------
# optional static or dynamic modifier
def p_mods(p):
    ''' mods : __DEVICE__
             | empty
    '''
    if p[1] != None:
        p[0] = [(p[1], p.lineno(1))]
    else:
        p[0] = []

#----------------------------------------------------------------------------------------------------------------------
# data type of a declaration
def p_type(p):
    ''' type : VOID
             | CHAR
             | SHORT
             | INT
             | LONG
             | FLOAT
             | DOUBLE
             | MACRO
    '''
    p[0] = [(p[1], p.lineno(1))]

#----------------------------------------------------------------------------------------------------------------------
# optional array brackets
def p_brackets(p):
    ''' brackets : '[' ']' brackets
                 | empty
    '''
    if p[1] == None:
        p[0] = False
    else:
        p[0] = True
    
#----------------------------------------------------------------------------------------------------------------------
# optional array size expressions
def p_arrsizes(p):
    ''' arrsizes : EXPR_IDX arrsizes
                 | EXPR_IDX
                 | empty
    '''
    if len(p) == 3:
        p[0] = [(p[1], p.lineno(1))] + p[2]
    elif p[1] == None:
        p[0] = []
    else:
        p[0] = [(p[1], p.lineno(1))]

#----------------------------------------------------------------------------------------------------------------------
# empty/epsilon production
def p_empty(p):
    'empty :'
    pass

#----------------------------------------------------------------------------------------------------------------------
# Error rule for parse errors
def p_error(p):
    g.err("orio.main.tspec.pparser.parser: error in input line #%s, at token-type '%s', token-value '%s'"
          % (p.lineno, p.type, p.value))
#----------------------------------------------------------------------------------------------------------------------



#----------------------------------------------------------------------------------------------------------------------
def getParser(start_symbol):
    '''Create the parser'''
    _ = orio.tool.ply.lex.lex()
    parser = orio.tool.ply.yacc.yacc(method='LALR', debug=0, start=start_symbol, check_recursion=0, tabmodule="pparsetab", optimize=1, write_tables=0)
    return parser


#--------------------------------------------------------------------------------

class TSpecParser:
    '''The parser of the TSpec language'''

    def __init__(self):
        '''To instantiate a TSpec parser'''
        pass
        
    #----------------------------------------------------------------------------

    def __parse(self, code, line_no, start_symbol):
        '''To parse the given code and return a sequence of statements'''

        # append multiple newlines to imitate the actual line number
        code = ('\n' * (line_no-1)) + code

        # append a newline on the given code
        code += '\n'

        # remove all comments
        code = re.sub(r'#.*?\n', '\n', code)

        # create the parser
        p = getParser(start_symbol)
        
        # parse the tuning specifications
        stmt_seq = p.parse(code)

        # return the statement sequence
        return stmt_seq


    #----------------------------------------------------------------------------

    def parseProgram(self, code, line_no = 1):
        '''To parse the given program body and return a sequence of statements'''
        return self.__parse(code, line_no, 'fspecs')

    #----------------------------------------------------------------------------

    def parseSpec(self, code, line_no = 1):
        '''To parse the given specification body and return a sequence of statements'''
        return self.__parse(code, line_no, 'specs')


