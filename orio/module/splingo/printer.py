#==============================================================================
# The unparser of the AST nodes
#==============================================================================

import orio.main.util.globals as g
import orio.module.splingo.ast as ast

#------------------------------------------------------------------------------
class Printer(object):
    '''The printer of AST nodes'''

    def __init__(self):
        pass

    #--------------------------------------------------------------------------
    def generate(self, tnode, indent='  ', extra_indent='  '):
        '''To generate code that corresponds to the given AST'''

        s = ''
        if isinstance(tnode, ast.Comment):
            s += indent
            if tnode.kids[0]:
                s += '/*' + tnode.kids[0] + '*/'
            s += '\n'
        
        elif isinstance(tnode, ast.LitExp):
            if tnode.kids[0] == ast.LitExp.STRING:
                s += '"' + str(tnode.kids[1]) + '"'
            else:
                s += str(tnode.kids[1])

        elif isinstance(tnode, ast.IdentExp):
            s += str(tnode.kids[0])

        elif isinstance(tnode, ast.ArrayRefExp):
            s += self.generate(tnode.kids[0], indent, extra_indent)
            s += '[' + self.generate(tnode.kids[1], indent, extra_indent) + ']'

        elif isinstance(tnode, ast.CallExp):
            s += self.generate(tnode.kids[0], indent, extra_indent) + '('
            s += ','.join(map(lambda x: self.generate(x, indent, extra_indent), tnode.kids[1]))
            s += ')'

        elif isinstance(tnode, ast.UnaryExp):
            s = self.generate(tnode.kids[1], indent, extra_indent)
            if   tnode.kids[0] == tnode.PLUS: s = '+' + s
            elif tnode.kids[0] == tnode.MINUS: s = '-' + s
            elif tnode.kids[0] == tnode.LNOT: s = '!' + s
            elif tnode.kids[0] == tnode.TRANSPOSE: s += "'"
            elif tnode.kids[0] == tnode.PRE_INC: s = ' ++' + s
            elif tnode.kids[0] == tnode.PRE_DEC: s = ' --' + s
            elif tnode.kids[0] == tnode.POST_INC: s += '++ '
            elif tnode.kids[0] == tnode.POST_DEC: s += '-- '
            else: g.err('%s: unknown unary operator type: %s' % (self.__class__, tnode.kids[0]))

        elif isinstance(tnode, ast.BinOpExp):
            s += self.generate(tnode.kids[1], indent, extra_indent)
            if   tnode.kids[0] == tnode.PLUS: s += '+'
            elif tnode.kids[0] == tnode.MINUS: s += '-'
            elif tnode.kids[0] == tnode.MULT: s += '*'
            elif tnode.kids[0] == tnode.DIV: s += '/'
            elif tnode.kids[0] == tnode.MOD: s += '%'
            elif tnode.kids[0] == tnode.LT: s += '<'
            elif tnode.kids[0] == tnode.GT: s += '>'
            elif tnode.kids[0] == tnode.LE: s += '<='
            elif tnode.kids[0] == tnode.GE: s += '>='
            elif tnode.kids[0] == tnode.EE: s += '=='
            elif tnode.kids[0] == tnode.NE: s += '!='
            elif tnode.kids[0] == tnode.LOR: s += '||'
            elif tnode.kids[0] == tnode.LAND: s += '&&'
            elif tnode.kids[0] == tnode.EQ: s += '='
            elif tnode.kids[0] == tnode.EQPLUS: s += '+='
            elif tnode.kids[0] == tnode.EQMINUS: s += '-='
            elif tnode.kids[0] == tnode.EQMULT: s += '*='
            elif tnode.kids[0] == tnode.EQDIV: s += '/='
            else: g.err('%s: unknown binary operator type: %s' % (self.__class__, tnode.kids[0]))
            s += self.generate(tnode.kids[2], indent, extra_indent)

        elif isinstance(tnode, ast.ParenExp):
            s += '(' + self.generate(tnode.kids[0], indent, extra_indent) + ')'

        elif isinstance(tnode, ast.ExpStmt):
            s += indent + self.generate(tnode.kids[0], indent, extra_indent) + ';\n'

        elif isinstance(tnode, ast.CompStmt):
            s += indent + '{\n'
            for stmt in tnode.kids:
                s += self.generate(stmt, indent + extra_indent, extra_indent)
            s += indent + '}\n'

        elif isinstance(tnode, ast.IfStmt):
            s += indent + 'if (' + self.generate(tnode.kids[0], indent, extra_indent) + ') '
            if isinstance(tnode.kids[1], ast.CompStmt):
                tstmt_s = self.generate(tnode.kids[1], indent, extra_indent)
                s += tstmt_s[tstmt_s.index('{'):]
                if tnode.kids[2]:
                    s = s[:-1] + ' else '
            else:
                s += '\n'
                s += self.generate(tnode.kids[1], indent + extra_indent, extra_indent)
                if tnode.kids[2]:
                    s += indent + 'else '
            if tnode.kids[2]:
                if isinstance(tnode.kids[2], ast.CompStmt):
                    tstmt_s = self.generate(tnode.kids[2], indent, extra_indent)
                    s += tstmt_s[tstmt_s.index('{'):]
                else:
                    s += '\n'
                    s += self.generate(tnode.kids[2], indent + extra_indent, extra_indent)

        elif isinstance(tnode, ast.ForStmt):
            #if tnode.getLabel(): s += tnode.getLabel() + ':'
            s += indent + 'for ('
            if tnode.kids[0]:
                s += self.generate(tnode.kids[0], indent, extra_indent)
            s += '; '
            if tnode.kids[1]:
                s += self.generate(tnode.kids[1], indent, extra_indent)
            s += '; '
            if tnode.kids[2]:
                s += self.generate(tnode.kids[2], indent, extra_indent)
            s += ') '
            if isinstance(tnode.kids[3], ast.CompStmt): 
                stmt_s = self.generate(tnode.kids[3], indent, extra_indent)
                s += stmt_s[stmt_s.index('{'):]
            else:
                s += '\n'
                s += self.generate(tnode.kids[3], indent + extra_indent, extra_indent)

        elif isinstance(tnode, ast.WhileStmt):
            s += indent + 'while (' + self.generate(tnode.kids[0], indent, extra_indent)
            s += ') '
            if isinstance(tnode.kids[1], ast.CompStmt): 
                stmt_s = self.generate(tnode.kids[1], indent, extra_indent)
                s += stmt_s[stmt_s.index('{'):]
            else:
                s += '\n'
                s += self.generate(tnode.kids[1], indent + extra_indent, extra_indent)

        elif isinstance(tnode, ast.VarInit):
            s += self.generate(tnode.kids[0], indent, extra_indent)
            if tnode.kids[1]:
                s += '=' + self.generate(tnode.kids[1], indent, extra_indent)

        elif isinstance(tnode, ast.VarDec):
            s += str(tnode.kids[0]) + ' '
            s += ', '.join(map(self.generate, tnode.kids[1]))
            if tnode.isAtomic:
                s = indent + s + ';\n'

        elif isinstance(tnode, ast.ParamDec):
            s += self.generate(tnode.kids[0], indent, extra_indent) + ' ' + self.generate(tnode.kids[1], indent, extra_indent)

        elif isinstance(tnode, ast.FunDec):
            s += ' '.join(tnode.kids[2]) + ''
            s += self.generate(tnode.kids[1], indent, extra_indent) + ' '
            s += self.generate(tnode.kids[0], indent, extra_indent) + '('
            s += ', '.join(map(self.generate, tnode.kids[3])) + ') '
            s += self.generate(tnode.kids[4], indent, extra_indent)

        elif isinstance(tnode, ast.TransformStmt):
            g.err('%s: a transformation statement is never generated as an output' % self.__class__)

        else:
            g.err('%s: unrecognized type of AST: (%s, %s)' % (self.__class__, tnode.__class__.__name__,tnode))

        return s
    #--------------------------------------------------------------------------
