#
# A module to build a symbol table from a C source file.
#
from parsers.pycparser import c_parser, c_ast, parse_file


class STBuilder(c_ast.NodeVisitor):
    def __init__(self, srcfile):
        self.srcfile = srcfile
        self.st = {}

    def generic_visit(self, n):
        print('generic:', type(n))
        for c in n.children():
            childst = self.visit(c)
            print('c_st:', childst)
            self.st.update(childst)

    def visit_IdentifierType(self, n):
        n.show()
        return ' '.join(n.names)

    def visit_Decl(self, n):
        print('visit_Decl: %s, %s' % (n.name, n.type))
        
        self._generate_type(n.type)
        
        return {n.name : 'mytype'}
    
    def visit_FuncDef(self, n):
        print('visit_FuncDef: %s at %s' % (n.decl.name, n.decl.coord))
        decl = self.visit(n.decl)
        old_st = self.st.copy()
        self.st.update(decl)
        body = self.visit(n.body)
        print('local st:', body)
        self.st = old_st
        return {n.decl.name: 'void'}
        
    def _generate_type(self, n, modifiers=[]):
        """ Recursive generation from a type node. n is the type node. 
            modifiers collects the PtrDecl, ArrayDecl and FuncDecl modifiers 
            encountered on the way down to a TypeDecl, to allow proper
            generation from it.
        """
        typ = type(n)
        print('_generate_type:', typ)
        #~ print(n, modifiers)
        
        if typ == c_ast.TypeDecl:
            s = ''
            if n.quals: s += ' '.join(n.quals) + ' '
            s += self.visit(n.type)
            
            nstr = n.declname if n.declname else ''
            # Resolve modifiers.
            # Wrap in parens to distinguish pointer to array and pointer to
            # function syntax.
            #
            for i, modifier in enumerate(modifiers):
                if isinstance(modifier, c_ast.ArrayDecl):
                    if (i != 0 and isinstance(modifiers[i - 1], c_ast.PtrDecl)):
                        nstr = '(' + nstr + ')'
                    nstr += '[' + self.visit(modifier.dim) + ']'
                elif isinstance(modifier, c_ast.FuncDecl):
                    if (i != 0 and isinstance(modifiers[i - 1], c_ast.PtrDecl)):
                        nstr = '(' + nstr + ')'
                    nstr += '(' + self.visit(modifier.args) + ')'
                elif isinstance(modifier, c_ast.PtrDecl):
                    nstr = '*' + nstr
            if nstr: s += ' ' + nstr
            print('_generate_type returning:', nstr, s)
            return {nstr: s}
        elif typ == c_ast.Decl:
            return self._generate_decl(n.type)
        elif typ == c_ast.Typename:
            return self._generate_type(n.type)
        elif typ == c_ast.IdentifierType:
            return ' '.join(n.names) + ' '
        elif typ in (c_ast.ArrayDecl, c_ast.PtrDecl, c_ast.FuncDecl):
            return self._generate_type(n.type, modifiers + [n])
        else:
            return self.visit(n)

    def build_st(self):
        ast = parse_file(self.srcfile, use_cpp=True)
        #ast.show()
        return self.visit(ast)

