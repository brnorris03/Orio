#
# The code generator (i.e. unparser) for the AST classes for OpenCL
#

from orio.module.loop import ast
import orio.main.util.globals as g
from orio.module.loop.codegen import CodeGen_C

class CodeGen_OpenCL (CodeGen_C):
    '''The code generator for the AST classes'''

    def __init__(self, language='opencl'):
        '''To instantiate a code generator'''
        self.arrayref_level = 0
        pass

    #----------------------------------------------

    def generate(self, tnode, indent = '  ', extra_indent = '  '):
        '''To generate code that corresponds to the given AST'''

        s = ''

        if isinstance(tnode, ast.NumLitExp):
            s += str(tnode.val)

        elif isinstance(tnode, ast.StringLitExp):
            s += '"' + str(tnode.val) + '"'

        elif isinstance(tnode, ast.IdentExp):
            s += str(tnode.name)

        elif isinstance(tnode, ast.ArrayRefExp):
            s += self.generate(tnode.exp, indent, extra_indent)
            s += '[' + self.generate(tnode.sub_exp, indent, extra_indent) + ']'

        elif isinstance(tnode, ast.FunCallExp):
            s += self.generate(tnode.exp, indent, extra_indent) + '('
            s += ','.join([self.generate(x, indent, extra_indent) for x in tnode.args])
            s += ')'

        elif isinstance(tnode, ast.UnaryExp):
            s = self.generate(tnode.exp, indent, extra_indent)
            if tnode.op_type == tnode.PLUS:
                s = '+' + s
            elif tnode.op_type == tnode.MINUS:
                s = '-' + s
            elif tnode.op_type == tnode.LNOT:
                s = '!' + s
            elif tnode.op_type == tnode.PRE_INC:
                s = ' ++' + s
            elif tnode.op_type == tnode.PRE_DEC:
                s = ' --' + s
            elif tnode.op_type == tnode.POST_INC:
                s = s + '++ '
            elif tnode.op_type == tnode.POST_DEC:
                s = s + '-- '
            elif tnode.op_type == tnode.DEREF:
                s = '*' + s
            elif tnode.op_type == tnode.ADDRESSOF:
                s = '&' + s
            else:
                g.err('orio.module.loop.codegen_opencl internal error: unknown unary operator type: %s' % tnode.op_type)

        elif isinstance(tnode, ast.BinOpExp):
            s += self.generate(tnode.lhs, indent, extra_indent)
            if tnode.op_type == tnode.MUL:
                s += '*'
            elif tnode.op_type == tnode.DIV:
                s += '/'
            elif tnode.op_type == tnode.MOD:
                s += '%'
            elif tnode.op_type == tnode.ADD:
                s += '+'
            elif tnode.op_type == tnode.SUB:
                s += '-'
            elif tnode.op_type == tnode.LT:
                s += '<'
            elif tnode.op_type == tnode.GT:
                s += '>'
            elif tnode.op_type == tnode.LE:
                s += '<='
            elif tnode.op_type == tnode.GE:
                s += '>='
            elif tnode.op_type == tnode.EQ:
                s += '=='
            elif tnode.op_type == tnode.NE:
                s += '!='
            elif tnode.op_type == tnode.LOR:
                s += '||'
            elif tnode.op_type == tnode.LAND:
                s += '&&'
            elif tnode.op_type == tnode.COMMA:
                s += ','
            elif tnode.op_type == tnode.EQ_ASGN:
                s += '='
            elif tnode.op_type == tnode.ASGN_ADD:
                s += '+='
            elif tnode.op_type == tnode.ASGN_SHR:
                s += '>>='
            elif tnode.op_type == tnode.ASGN_SHL:
                s += '<<='
            elif tnode.op_type == tnode.BAND:
                s += '&'
            elif tnode.op_type == tnode.SHR:
                s += '>>'
            elif tnode.op_type == tnode.BOR:
                s += '|'
            else:
                g.err('orio.module.loop.codegen_opencl internal error: unknown binary operator type: %s' % tnode.op_type)
            s += self.generate(tnode.rhs, indent, extra_indent)

        elif isinstance(tnode, ast.TernaryExp):
            s += self.generate(tnode.test, indent, extra_indent) + '?'
            s += self.generate(tnode.true_expr,  indent, extra_indent) + ':'
            s += self.generate(tnode.false_expr, indent, extra_indent)

        elif isinstance(tnode, ast.ParenthExp):
            s += '(' + self.generate(tnode.exp, indent, extra_indent) + ')'

        elif isinstance(tnode, ast.Comment):
            s += indent
            if tnode.text:
                s += '/*' + tnode.text + '*/'
            s += '\n'
            
        elif isinstance(tnode, ast.ExpStmt):
            if tnode.getLabel(): s += tnode.getLabel() + ':'
            s += indent
            if tnode.exp:
                s += self.generate(tnode.exp, indent, extra_indent)
            s += ';\n'

        elif isinstance(tnode, ast.GotoStmt):
            if tnode.getLabel(): s += tnode.getLabel() + ':'
            s += indent
            if tnode.target:
                s += 'goto ' + tnode.target + ';\n'
                
        elif isinstance(tnode, ast.CompStmt):
            s += indent + '{\n'
            for stmt in tnode.stmts:
                s += self.generate(stmt, indent + extra_indent, extra_indent)
            s += indent + '}\n'

        elif isinstance(tnode, ast.IfStmt):
            if tnode.getLabel(): s += tnode.getLabel() + ':'
            s += indent + 'if (' + self.generate(tnode.test, indent, extra_indent) + ') '
            if isinstance(tnode.true_stmt, ast.CompStmt):
                tstmt_s = self.generate(tnode.true_stmt, indent, extra_indent)
                s += tstmt_s[tstmt_s.index('{'):]
                if tnode.false_stmt:
                    s = s[:-1] + ' else '
            else:
                s += '\n'
                s += self.generate(tnode.true_stmt, indent + extra_indent, extra_indent)
                if tnode.false_stmt:
                    s += indent + 'else '
            if tnode.false_stmt:
                if isinstance(tnode.false_stmt, ast.CompStmt):
                    tstmt_s = self.generate(tnode.false_stmt, indent, extra_indent)
                    s += tstmt_s[tstmt_s.index('{'):]
                else:
                    s += '\n'
                    s += self.generate(tnode.false_stmt, indent + extra_indent, extra_indent)

        elif isinstance(tnode, ast.ForStmt):
            if tnode.getLabel(): s += tnode.getLabel() + ':'
            s += indent + 'for ('
            if tnode.init:
                if isinstance(tnode.init, ast.VarDeclInit):
                  s += str(tnode.init.type_name) + ' '
                  s += self.generate(tnode.init.var_name, indent, extra_indent)
                  s += '=' + self.generate(tnode.init.init_exp, indent, extra_indent)
                else:
                  s += self.generate(tnode.init, indent, extra_indent)
            s += '; '
            if tnode.test:
                s += self.generate(tnode.test, indent, extra_indent)
            s += '; '
            if tnode.iter:
                s += self.generate(tnode.iter, indent, extra_indent)
            s += ') '
            if isinstance(tnode.stmt, ast.CompStmt): 
                stmt_s = self.generate(tnode.stmt, indent, extra_indent)
                s += stmt_s[stmt_s.index('{'):]
            else:
                s += '\n'
                s += self.generate(tnode.stmt, indent + extra_indent, extra_indent)

        elif isinstance(tnode, ast.AssignStmt):
            if tnode.getLabel(): s += tnode.getLabel() + ':'
            s += indent + tnode.var + '='
            s += self.generate(tnode.exp, indent, extra_indent)
            s += ';\n'
            
        elif isinstance(tnode, ast.TransformStmt):
            g.err('orio.module.loop.codegen_opencl internal error: a transformation statement is never generated as an output')

        elif isinstance(tnode, ast.VarDecl):
            s += indent + str(tnode.type_name) + ' '
            if isinstance(tnode.var_names[0], ast.IdentExp): 
                s += ', '.join(map(self.generate, tnode.var_names))
            else:
                s += ', '.join(tnode.var_names)
            s += ';\n'
 
        elif isinstance(tnode, ast.VarDeclInit):
            s += indent + str(tnode.type_name) + ' '
            s += self.generate(tnode.var_name, indent, extra_indent)
            s += '=' + self.generate(tnode.init_exp, indent, extra_indent)
            s += ';\n'

        elif isinstance(tnode, ast.FieldDecl):
            s += tnode.ty + ' '
            if isinstance(tnode.name, ast.IdentExp):
                s+= tnode.name.name
            else:
                s += tnode.name

        elif isinstance(tnode, ast.FunDecl):
            s += indent + ' '.join(tnode.modifiers) + ' '
            s += tnode.return_type + ' '
            s += tnode.name + '('
            s += ', '.join(map(self.generate, tnode.params)) + ') '
            s += self.generate(tnode.body, indent, extra_indent)

        elif isinstance(tnode, ast.Pragma):
            s += indent + '#pragma ' + str(tnode.pstring) + '\n'

        elif isinstance(tnode, ast.Container):
            s += self.generate(tnode.ast, indent, extra_indent)

        elif isinstance(tnode, ast.WhileStmt):
            s += indent + 'while (' + self.generate(tnode.test, indent, extra_indent)
            s += ') '
            if isinstance(tnode.stmt, ast.CompStmt): 
                stmt_s = self.generate(tnode.stmt, indent, extra_indent)
                s += stmt_s[stmt_s.index('{'):]
            else:
                s += '\n'
                s += self.generate(tnode.stmt, indent + extra_indent, extra_indent)

        elif isinstance(tnode, ast.CastExpr):
            s += '(' + tnode.ctype + ')'
            s += self.generate(tnode.expr, indent, extra_indent)
        
        else:
            g.err('orio.module.loop.codegen_opencl internal error: unrecognized type of AST: %s' % tnode.__class__.__name__)

        return s

