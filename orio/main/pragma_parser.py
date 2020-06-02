#
# Parser to extract annotations from the source code
#

import re
import sys
from orio.main.util.globals import *
from orio.module.loop import parser, ast, codegen


# ----------------------------------------

class PragmaParser:
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
    __loop_depth=0

    # ----------------------------------------

    def __init__(self):
        self.codegen = codegen.CodeGen(language='C')
        self.loop_depth = 0

    # ----------------------------------------

    def leaderPragmaRE():
        return PragmaParser.__leader_pragma_re

    leaderPragmaRE = staticmethod(leaderPragmaRE)

    # ----------------------------------------

    def removePragmas(self, code):
        """Remove all annotations from the given code"""
        code = self.__leader_pragma_re.sub('', code)
        return self.__trailer_pragma_re.sub('', code)

    # ----------------------------------------

    def parse(self, code, line_no=1):
        """Parse the code and insert Orio annotations in loops annotated with
            #pragma orio loop
        """

        debug("PARSING PRAGMAS", self)

        new_code = {}

        for m_begin in self.leaderPragmaRE().finditer(code):
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

            # parse the loop
            line_no = code[:pos_end + 1].count('\n')

            # stmts = parser.getParser(line_no).parse(loop_code,tracking=1,debug=1)
            stmts = parser.getParser(line_no).parse(loop_code)
            debug("Parsed pragma-annotated loop:\n %s" % stmts, self)

            PragmaParser.loop_depth = 0
            ann = self._generate_annotation(stmts)

            new_code[(pos, pos_end)] = ann

        # return the sequence of code fragments
        annotated_code = ''
        prev = 0
        for (pos, pos_end), ann in sorted(new_code.items()):
            annotated_code += code[prev:pos] + ann
            prev = pos_end + 1
        annotated_code += code[prev:]

        annotated_code = annotated_code.replace('#pragma orio loop end', '/*@ end @*/')
        print(annotated_code)
        return annotated_code

    def _generate_annotation(self, stmts):
        """ Generate an Orio annotation from the loops in stmts (AST) """
        print("_generate_annotation: ", stmts)
        for s in stmts:
            if not s: continue
            if isinstance(s, ast.CompStmt):
                self._generate_annotation(s.stmts)
            if isinstance(s, ast.ForStmt):
                PragmaParser.loop_depth += 1
                print("LOOP[%d]: " % PragmaParser.loop_depth, s.init, s.test, s.iter)
                if isinstance(s.stmt, ast.CompStmt):
                    self._generate_annotation(s.stmt.stmts)
                else:
                    self._generate_annotation([s.stmt])
            elif isinstance(s, ast.IfStmt):
                if s.true_stmt:
                    self._generate_annotation([s.true_stmt])
                if s.false_stmt:
                    self._generate_annotation([s.false_stmt])


        final_annotation = "\n/*@ HAAAAAA! %d\n" % PragmaParser.loop_depth + self.codegen.generate(s) + "\n@*/\n"
        return final_annotation

