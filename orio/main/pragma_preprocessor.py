#
# Parser to extract annotations from the source code
#

import re
import sys
from orio.main.util.globals import *
from orio.module.loop import parser, ast, codegen, astvisitors
from .ann_gen import LoopAnnotationGenerator
from .tuning_spec_template import template_string, default_params, default_perf_params, default_input_type


# ----------------------------------------

class PragmaPreprocessor:
    """The parser used for auatomated loop annotation using orio loop pragmas."""

    # regular expressions
    __any_re = r'(.|\n)'
    __label_re = r'".*"'

    __pragma_beg = r'\#pragma\s+orio\s+loop\s+begin\s+' + __label_re
    __pragma_end = r'\#pragma\s+orio\s+loop\s+end\s+'
    __pragma_re = re.compile(r'#pragma orio')
    __leader_pragma_re = re.compile(__pragma_beg)
    __trailer_pragma_re = re.compile(__pragma_end)
    __non_indent_char_re = re.compile(r'[^ \t]')

    # ----------------------------------------

    def __init__(self):
        self.codegen = codegen.CodeGen(language='C')
        self.annotation_counter = 0
        self.tspec_params = default_params
        self.indent = '    '

    # ----------------------------------------

    @staticmethod
    def leaderPragmaRE():
        return PragmaPreprocessor.__leader_pragma_re

    # ----------------------------------------

    def removePragmas(self, code):
        """Remove all annotations from the given code"""
        code = self.__leader_pragma_re.sub('', code)
        return self.__trailer_pragma_re.sub('', code)

    # ----------------------------------------

    def preprocess(self, code, line_no=1):
        """Parse the code and insert Orio annotations in loops annotated with
            #pragma orio loop
        """

        debug("PARSING PRAGMAS", self)

        new_code = {}

        for m_begin in self.leaderPragmaRE().finditer(code):
            loop_code=''
            pos = m_begin.start()
            pos_end = m_begin.end()
            pragma_line = m_begin.group()
            # Extract the loops

            m_end = self.__trailer_pragma_re.search(code, pos)
            if m_end:
                loop_code = code[pos_end:m_end.start()]
            else:
                err('Did not find expected #pragma orio loop end', doexit=True)
            debug('Found Orio pragma in position %d: %s. Loop:%s' % (pos, pragma_line, loop_code), self)
            self.annotation_counter += 1

            # parse the loop
            line_no = code[:pos_end + 1].count('\n')

            # stmts = parser.getParser(line_no).parse(loop_code,tracking=1,debug=1)
            stmts = parser.getParser(line_no).parse(loop_code)
            debug("Parsed pragma-annotated loop:\n %s" % stmts, self, level=4)

            # Extract information needed for generating the annotation
            loop_info = LoopInfoVisitor()
            loop_info.visit(stmts[0])
            debug("LOOP info: bounds={}, maxnest={}, vars={}".format(
                repr(loop_info.loop_bounds), loop_info.maxnest, repr(loop_info.vars)),self)
            """Example loop_info:
            #pragma orio loop begin "C = C * beta". Loop:
            for (i = 0; i < n1; i++) {
                for (j = 0; j < n2; j++) {
                     C[ldc * i + j] *= beta;
               }
            }
            ('LOOP inFo:', {'i': ('0', 'n1'), 'j': ('0', 'n2')}, 2, set(['n1', 'n2']))
            """

            # Process the AST to generate the annotation and the associated tuning spec
            # This also updates the tuning parameters self.tspec_params
            ann = self._generate_annotation(stmts[0],loop_info)
            new_code[(pos, pos_end)] = ann


        # Insert the annotation in the place of the #pragma orio begin loop
        annotated_code = ''
        prev = 0
        for (pos, pos_end), ann in sorted(new_code.items()):
            annotated_code += code[prev:pos] + ann
            prev = pos_end + 1
        annotated_code += code[prev:]
        annotated_code = annotated_code.replace('#pragma orio loop end', '/*@ end @*/')
        #debug('Annotated code:\n{}'.format(annotated_code),self)

        # Generate the tuning spec for this file
        tuning_spec = self._generate_tuning_spec()
        debug('Tuning spec:\n{}'.format(tuning_spec), self)
        return annotated_code

    def _generate_annotation(self, stmt, loop_info):
        """ Generate an Orio annotation from the loop node stmt """

        if not isinstance(stmt, ast.ForStmt):
            warn("Not a loop, cannot generate annotation", self)
            return ''

        loop_ann = LoopAnnotationGenerator(
            loop_counter=loop_info.maxnest,
            loop_vars=loop_info.loop_bounds.keys(),
            tiling_levels=min(loop_info.maxnest,2))
        leader_annotation = "\n/*@ begin Loop(\n %s\n" % loop_ann.getComposite() \
                            + self.codegen.generate(stmt) \
                            + "\n@*/\n"

        self._update_params(loop_ann.get_perf_params(),loop_info)
        return leader_annotation

    def _generate_tuning_spec(self) :
        self.tuning_spec = template_string

        print(self.tspec_params)

        for section, val in self.tspec_params.items():
            buf = ''
            if section in ['performance_params','input_params','input_vars','constraints']:
                print(repr(self.tspec_params[section].values()))
                buf = ''.join(self.tspec_params[section].values())
            elif section in ['build','performance_counter','search']:
                for k,v in self.tspec_params[section].items():
                    buf += self.indent + 'arg %s = %s;\n' % (k,str(v))
            else:
                warn("Unknown tuning spec section \"%s\" encountered, this should never happen!" % section )

            self.tuning_spec = self.tuning_spec.replace('@%s@'%section, buf)
        return self.tuning_spec

    def _update_params(self, perf_params, loop_info):
        """Update the self.tspec_params performance_parameter section with specific variables
        and their default values (from the tuning_spec_template.py)

        :param perf_params: dict(tiling=[],unroll=[],scalar_replacement=[],vector=[],openmp=[])
        """
        for vartype in perf_params.keys():
            for var in perf_params[vartype]:
                self.tspec_params['performance_params'][var] = \
                    self.indent + 'param %s[] = %s;\t#%s\n' % (var, repr(default_perf_params[vartype]), vartype)

        #loop_info.vars: set of input vars

class LoopInfoVisitor(astvisitors.ASTVisitor):
    def __init__(self):
        astvisitors.ASTVisitor.__init__(self)
        self.maxnest = 0
        self._nest = 0
        self.vars = set()
        self.loop_bounds = {}

    def visit(self, node, params={}):
        """Invoke accept method for specified AST node"""
        if not node: return
        if isinstance(node,list):
            for item in list:
                self.visit(item)
        try:
            if isinstance(node, ast.NumLitExp):
                pass

            elif isinstance(node, ast.StringLitExp):
                pass

            elif isinstance(node, ast.IdentExp):
                if params.get('in') == 'loop_header':
                    if not node.name in self.loop_bounds.keys():
                        self.vars.add(node.name)

            elif isinstance(node, ast.ArrayRefExp):
                self.visit(node.exp)  # array variable
                self.visit(node.sub_exp)  # array index

            elif isinstance(node, ast.FunCallExp):
                self.visit(node.args)

            elif isinstance(node, ast.UnaryExp):
                self.visit(node.exp)  # the operand

            elif isinstance(node, ast.BinOpExp):
                self.visit(node.lhs,params)
                self.visit(node.rhs,params)

            elif isinstance(node, ast.ParenthExp):
                self.visit(node.exp)

            elif isinstance(node, ast.Comment):
                pass

            elif isinstance(node, ast.ExpStmt):
                self.visit(node.exp)

            elif isinstance(node, ast.GotoStmt):
                pass

            elif isinstance(node, ast.CompStmt):
                for s in node.stmts:
                    self.visit(s)

            elif isinstance(node, ast.IfStmt):
                self.visit(node.test)
                self.visit(node.true_stmt)
                self.visit(node.false_stmt)

            elif isinstance(node, ast.ForStmt):
                self._nest += 1
                self.loop_bounds[node.init.lhs.name] = (str(node.init.rhs),str(node.test.rhs))
                self.visit(node.init)
                self.visit(node.test,params={'in':'loop_header'})
                self.visit(node.iter,params={'in':'loop_header'})
                self.visit(node.stmt,params={'in':'loop_header'})
                if self._nest > self.maxnest: self.maxnest = self._nest
                self._nest -= 1


            elif isinstance(node, ast.TransformStmt):
                pass

            elif isinstance(node, ast.VarDecl):
                pass

            elif isinstance(node, ast.VarDeclInit):
                self.visit(node.init_exp)

            elif isinstance(node, ast.Pragma):
                pass

            elif isinstance(node, ast.Container):
                self.visit(node.ast)

            elif isinstance(node, ast.DeclStmt):
                for decl in node.decls:
                    self.visit(decl)
            else:
                err('internal error: unrecognized type of AST: %s' % node.__class__.__name__, self)
        except Exception as e:
            err("Exception in node %s: %s" % (node.__class__, e), self)
