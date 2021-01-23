#
# The code generator (i.e. unparser) for the AST classes
#

from orio.module.loop import ast
import orio.main.util.globals as g

#-------------------------------------------------

class CodeGen:
    '''The code generator for the AST classes'''

    def __init__(self, language='C'):
        '''Instantiates a code generator'''
        self.generator = None
        
        if language.lower() in ['c','c++','cxx']:
            self.generator = CodeGen_C()
        elif language.lower() in ['f', 'f90', 'fortran']:
            self.generator = CodeGen_F()
        elif language.lower() in ['cuda']:
            from orio.module.loop.codegen_cuda import CodeGen_CUDA
            self.generator = CodeGen_CUDA()
        elif language.lower() in ['opencl']:
            from orio.module.loop.codegen_opencl import CodeGen_OpenCL
            self.generator = CodeGen_OpenCL()
        else:
            g.err('orio.module.loop.codegen: Unknown language specified for code generation: %s' % language)
        pass

    #----------------------------------------------

    def generate(self, tnode, indent = '  ', extra_indent = '  '):
        '''Generates code that corresponds to the given AST'''

        return self.generator.generate(tnode,indent,extra_indent)


class CodeGen_C (CodeGen):
    '''The code generator for the AST classes'''

    def __init__(self):
        '''To instantiate a code generator'''
        self.arrayref_level = 0
        self.alldecls = set([])
        self.ids = []
        pass

    #----------------------------------------------

    def generate(self, tnode, indent = '  ', extra_indent = '  '):
        '''To generate code that corresponds to the given AST'''

        s = ''

        if isinstance(tnode, ast.NumLitExp):
            s += str(tnode.val)

        elif isinstance(tnode, ast.StringLitExp):
            s += str(tnode.val)

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
                g.err('orio.module.loop.codegen internal error: unknown unary operator type: %s' % tnode.op_type)

        elif isinstance(tnode, ast.BinOpExp):
            s += self.generate(tnode.lhs, indent, extra_indent)
            if tnode.op_type == tnode.MUL:
                s += ' * '
            elif tnode.op_type == tnode.DIV:
                s += ' / '
            elif tnode.op_type == tnode.MOD:
                s += ' % '
            elif tnode.op_type == tnode.ADD:
                s += ' + '
            elif tnode.op_type == tnode.SUB:
                s += ' - '
            elif tnode.op_type == tnode.LT:
                s += ' < '
            elif tnode.op_type == tnode.GT:
                s += ' > '
            elif tnode.op_type == tnode.LE:
                s += ' <= '
            elif tnode.op_type == tnode.GE:
                s += ' >= '
            elif tnode.op_type == tnode.EQ:
                s += ' == '
            elif tnode.op_type == tnode.NE:
                s += ' != '
            elif tnode.op_type == tnode.LOR:
                s += ' || '
            elif tnode.op_type == tnode.LAND:
                s += ' && '
            elif tnode.op_type == tnode.COMMA:
                s += ', '
            elif tnode.op_type == tnode.EQ_ASGN:
                #print "(((((( Binop: tnode.lhs.meta=%s, tnode.rhs.meta=%s ))))) " \
                #    % (str(tnode.lhs.meta),str(tnode.rhs.meta))

                s += ' = '
            else:
                g.err('orio.module.loop.codegen internal error: unknown binary operator type: %s' % tnode.op_type)
            s += self.generate(tnode.rhs, indent, extra_indent)

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
            try:
                tmp = tnode.meta.get('id')
                fake_loop = False
                #if tmp and (not tmp in self.ids):
                if tmp and g.Globals().marker_loops:
                    #self.ids.append(tmp)
                    fake_loop = True
                    #s += tmp + ': \n'
                    fake_scope_loop = 'for (int %s=0; %s < 1; %s++)' % (tmp, tmp, tmp)
                    s += indent + fake_scope_loop
                    old_indent = indent
                    indent += extra_indent
                s += indent + '{\n'

                self.alldecls = set([])
                for stmt in tnode.stmts:
                    g.debug('generating code for stmt type: %s' % stmt.__class__.__name__, obj=self,level=7)
                    s += self.generate(stmt, indent + extra_indent, extra_indent)
                    g.debug('code so far:' + s, obj=self, level=7)

                s += indent + '}\n'
                if fake_loop: indent = old_indent
            except Exception as e:
                g.err('orio.module.loop.codegen:%s: encountered an error in C code generation for CompStmt: %s %s' % (tnode.line_no, e.__class__, e))

        elif isinstance(tnode, ast.IfStmt):
            try:
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
            except Exception as e:
                g.err('orio.module.loop.codegen:%s: encountered an error in C code generation for IfStmt: %s %s ' % (tnode.line_no, e.__class__, e))


        elif isinstance(tnode, ast.ForStmt):
            try:
                tmp = tnode.meta.get('id')
                fake_loop = False
                parent_with_id = False
                if tnode.parent:
                    if isinstance(tnode.parent, ast.CompStmt) or isinstance(tnode.parent, ast.ForStmt):
                        if tnode.parent.meta.get('id'):
                            parent_with_id = True
                if not parent_with_id and tmp and g.Globals().marker_loops: # and tmp not in self.ids:
                    #self.ids.append(tmp)
                    fake_loop = True
                    #s += tmp + ': \n'
                    fake_scope_loop = 'for (int %s=0; %s < 1; %s++)'% (tmp,tmp,tmp)
                    s += indent +  fake_scope_loop + ' {\n'
                    old_indent = indent
                    indent += extra_indent
                local_decl = True

                # In some cases, we wish loop index variables to be accessible after the
                # corresponding loop. For example, the remainder loop generated by register tiling reuses the
                # index variable from the preceding loop, hence, it is declared before the actual loop,
                # so that it can be accessed later.
                if tnode.init and tnode.meta.get('declare_vars_outside'):
                    s += indent + 'int %s;\n' % ', '.join(tnode.meta['declare_vars_outside'])
                    local_decl = False
                s += indent + 'for ('
                if tnode.init:
                    if isinstance(tnode.init, ast.BinOpExp) and local_decl:
                        #if tnode.init.lhs.name.startswith('_orio_'):  # Orio-generated variable
                        s += 'int '
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
                    self.alldecls = set([])
                else:
                    s += '\n'
                    s += self.generate(tnode.stmt, indent + extra_indent, extra_indent)

                if fake_loop and tmp:
                    s += indent + '} // ' + fake_scope_loop + '\n'
                    indent = old_indent
            except Exception as e:
                g.err('orio.module.loop.codegen:%s: encountered an error in C code generation: %s %s' % (tnode.line_no, e.__class__, e))


        elif isinstance(tnode, ast.TransformStmt):
            g.err('orio.module.loop.codegen internal error: a transformation statement is never generated as an output')

        elif isinstance(tnode, ast.VarDecl):
            qual=''
            if tnode.qualifier.strip():
                qual = str(tnode.qualifier) + ' '
            sv = indent + qual + str(tnode.type_name) + ' '
            sv += ', '.join(tnode.var_names)
            sv += ';\n'
            if not sv in self.alldecls: 
                s += sv
                self.alldecls.add(sv)

        elif isinstance(tnode, ast.VarDeclInit):
            qual=''
            if tnode.qualifier.strip():
                qual = str(tnode.qualifier) + ' '
            s += indent + qual + str(tnode.type_name) + ' '
            s += self.generate(tnode.var_name, indent, extra_indent)
            s += '=' + self.generate(tnode.init_exp, indent, extra_indent)
            s += ';'

        elif isinstance(tnode, ast.Pragma):
            s += '#pragma ' + str(tnode.pstring) + '\n'

        elif isinstance(tnode, ast.Container):
            s += self.generate(tnode.ast, indent, extra_indent)

        elif isinstance(tnode, ast.DeclStmt):
            for d in tnode.vars():
                s += self.generate(d, indent, '')
        else:
            g.err('orio.module.loop.codegen internal error: unrecognized type of AST: %s\n%s' % (tnode.__class__.__name__, str(tnode)))
        return s

# ==============================================================================================

class CodeGen_F(CodeGen):
    '''The code generator for the AST classes'''

    def __init__(self):
        '''To instantiate a code generator'''
        self.ftypes = {'int':'integer', 
                       'register int': 'integer',  
                       'long': 'integer*4', 
                       'float': 'real(single)', 
                       'double': 'real(double)'}
        pass

    #----------------------------------------------

    def generate(self, tnode, indent = '  ', extra_indent = '  ', doloop_inc = False):
        '''To generate code that corresponds to the given AST'''

        s = ''

        if isinstance(tnode, ast.NumLitExp):
            s += str(tnode.val)

        elif isinstance(tnode, ast.StringLitExp):
            s += str(tnode.val)

        elif isinstance(tnode, ast.IdentExp):
            s += str(tnode.name)

        elif isinstance(tnode, ast.ArrayRefExp):            
            # Now get all the indices
            tmpnode = tnode
            prevtmpnode = tnode
            indices = []
            while isinstance(tmpnode, ast.ArrayRefExp):
                indices.append(tmpnode.sub_exp)
                prevtmpnode = tmpnode
                tmpnode = tmpnode.exp
            
            indices.reverse()
            s += self.generate(prevtmpnode.exp, indent, extra_indent)  # the variable name
            s += '(' + ','.join([self.generate(x, indent, extra_indent) for x in indices]) + ')'

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
                s = 'NOT(' + s + ')'
            elif tnode.op_type == tnode.PRE_INC:
                s += '\n' + indent + s + ' = ' + s + ' + 1\n'
            elif tnode.op_type == tnode.PRE_DEC:
                s += '\n' + indent + s + ' = ' + s + ' - 1\n'
            elif tnode.op_type == tnode.POST_INC:
                s += s + '\n' + indent + s + ' = ' + s + ' + 1\n'
            elif tnode.op_type == tnode.POST_DEC:
                s += s + '\n' + indent + s + ' = ' + s + ' - 1\n'
            else:
                g.err('orio.module.loop.codegen internal error: unknown unary operator type: %s' % tnode.op_type)

        elif isinstance(tnode, ast.BinOpExp):
            if tnode.op_type not in [tnode.MOD, tnode.COMMA]:
                if not doloop_inc: 
                    s += self.generate(tnode.lhs, indent, extra_indent)
                if tnode.op_type == tnode.MUL:
                    s += '*'
                elif tnode.op_type == tnode.DIV:
                    s += '/'
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
                    s += '.OR.'
                elif tnode.op_type == tnode.LAND:
                    s += '.AND.'
                elif tnode.op_type == tnode.EQ_ASGN:
                    s += '='
                else:
                    g.err('orio.module.loop.codegen internal error: unknown binary operator type: %s' % tnode.op_type)
                    
                s += self.generate(tnode.rhs, indent, extra_indent)
                
            else:
                
                if tnode.op_type == tnode.MOD:
                    s += 'MOD(' + self.generate(tnode.lhs, indent, extra_indent) + ', ' \
                        + self.generate(tnode.rhs, indent, extra_indent) + ')'
                elif tnode.op_type == tnode.COMMA:
                    # TODO: We need to implement an AST canonicalization step for Fortran before generating the code.
                    print('internal warning: Fortran code generator does not fully support the comma operator -- the generated code may not compile.')
                    s += self.generate(tnode.rhs, indent, extra_indent) 
                    s += '\n' + indent + self.generate(tnode.lhs, indent, extra_indent)
                    s +='\n! ORIO Warining: check code above and fix problems.'

        elif isinstance(tnode, ast.ParenthExp):
            s += '(' + self.generate(tnode.exp, indent, extra_indent) + ')'

        elif isinstance(tnode, ast.Comment):
            s += indent
            if tnode.text:
                s += '!' + tnode.text 
            s += '\n'

        elif isinstance(tnode, ast.ExpStmt):
            if tnode.getLabel(): s += tnode.getLabel() + ' '
            s += indent
            if tnode.exp:
                s += self.generate(tnode.exp, indent, extra_indent)
            s += '\n'
            
        elif isinstance(tnode, ast.GotoStmt):
            if tnode.getLabel(): s += tnode.getLabel() + ' '
            s += indent
            if tnode.target:
                s += 'goto ' + tnode.target + '\n'

        elif isinstance(tnode, ast.CompStmt):
            s += indent + '{\n'
            if tnode.getLabel(): s += tnode.getLabel() + ' '
            for stmt in tnode.stmts:
                s += self.generate(stmt, indent + extra_indent, extra_indent)
            s += indent + '}\n'

        elif isinstance(tnode, ast.IfStmt):
            if tnode.getLabel(): s += tnode.getLabel() + ' '
            s += indent + 'if (' + self.generate(tnode.test, indent, extra_indent) + ') then \n'
            if isinstance(tnode.true_stmt, ast.CompStmt):
                tstmt_s = self.generate(tnode.true_stmt, indent, extra_indent)
                # TODO: fix below cludge -- { is missing for some reason in some compound ifs
                if tstmt_s.count('{') > 0: s += tstmt_s[tstmt_s.index('{'):]
                else: s += tstmt_s
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
            s += indent + 'end if\n'

        elif isinstance(tnode, ast.ForStmt):
            if tnode.getLabel(): s += tnode.getLabel() + ' '
            s += indent + 'do ' 
            if tnode.init:
                s += self.generate(tnode.init, indent, extra_indent)
            if not tnode.test:
                g.err('orio.module.loop.codegen:  missing loop test expression. Fortran code generation requires a loop test expression.')
                
            if not tnode.iter:
                g.err('orio.module.loop.codegen:  missing loop increment expression. Fortran code generation requires a loop increment expression.')
            s += ', '
            if not isinstance(tnode.test, ast.BinOpExp):
                g.err('orio.module.loop.codegen internal error: cannot handle code generation for loop test expression')
                
            if tnode.test.op_type not in [tnode.test.LE, tnode.test.LT, tnode.test.GE, tnode.test.GT]: 
                g.err('orio.module.loop.codegen internal error: cannot generate Fortran loop, only <, >, <=, >= are recognized in the loop limit test')
            
            # Generate the loop bound        
            s += self.generate(tnode.test.rhs, indent, extra_indent)
            
            # Check whether we need to change the bound
            #if tnode.test.op_type == tnode.test.LE: s += ' + 1'
            #if tnode.test.op_type == tnode.test.GE: s += ' - 1'
            
            s += ', '
            # Generate the loop increment/decrement step
            
            if not isinstance(tnode.iter, (ast.BinOpExp, ast.UnaryExp)):
                g.err('orio.module.loop.codegen internal error: cannot handle code generation for loop increment expression')
 
            unary = False
            if isinstance(tnode.iter, ast.UnaryExp):
                incr_decr = [tnode.iter.POST_DEC, tnode.iter.PRE_DEC, tnode.iter.POST_INC, tnode.iter.PRE_INC]
                unary = True
                
            if not ((isinstance(tnode.iter, ast.BinOpExp) and tnode.iter.op_type == tnode.iter.EQ_ASGN)
                    or (isinstance(tnode.iter, ast.UnaryExp) and tnode.iter.op_type in incr_decr)): 
                g.err('orio.module.loop.codegen internal error: cannot handle code generation for loop increment expression')

            if tnode.test.op_type in [tnode.test.GT, tnode.test.GE] \
                and unary and tnode.iter.op_type in [tnode.iter.PRE_DEC, tnode.iter.POST_DEC]:
                s += '-'
               
            if unary and tnode.iter.op_type in incr_decr:   # ++i
                s += '1'
            else:
                s += self.generate(tnode.iter.rhs, indent, extra_indent, True)
            
            if isinstance(tnode.stmt, ast.CompStmt): 
                s += self.generate(tnode.stmt, indent, extra_indent)
                s += '\n' + indent + 'end do\n'
            else:
                s += '\n'
                s += self.generate(tnode.stmt, indent + extra_indent, extra_indent)
                s += '\n' + indent + 'end do\n'

        elif isinstance(tnode, ast.TransformStmt):
            g.err('orio.module.loop.codegen internal error: a transformation statement is never generated as an output')

        elif isinstance(tnode, ast.VarDecl):
            
            if tnode.type_name not in list(self.ftypes.keys()):
                g.err('orio.module.loop.codegen internal error: Cannot generate Fortran type for ' + tnode.type_name)
                
            s += indent + str(self.ftypes[tnode.type_name]) + ' '
            s += ', '.join(tnode.var_names)
            s += '\n'

        elif isinstance(tnode, ast.Pragma):
            s += '$pragma ' + str(tnode.pstring) + '\n'

        elif isinstance(tnode, ast.Container):
            if tnode.getLabel(): s += tnode.getLabel() + ' '
            s += self.generate(tnode.ast, indent, extra_indent)

        else:
            g.err('orio.module.loop.codegen internal error: unrecognized type of AST: %s' % tnode.__class__.__name__)

        return s




