# 
# CodeFragment
#  |
#  +-- Ann
#  |    |
#  |    +-- LeaderAnn
#  |    +-- TrailerAnn
#  |
#  +-- NonAnn
#  |
#  +-- AnnCodeRegion
#

#-----------------------------------------

class CodeFragment:

    def __init__(self):
        '''Instantiate a code fragment'''
        self.id="None"
        pass

#-----------------------------------------

class Ann(CodeFragment):

    def __init__(self, code, line_no, indent_size):
        '''Instantiate an annotation'''

        CodeFragment.__init__(self)
        self.code = code
        self.line_no = line_no
        self.indent_size = indent_size

#-----------------------------------------

class LeaderAnn(Ann):

    def __init__(self, code, line_no, indent_size, mod_name, mod_name_line_no,
                 mod_code, mod_code_line_no):

        Ann.__init__(self, code, line_no, indent_size)
        self.mod_name = mod_name
        self.mod_name_line_no = mod_name_line_no
        self.mod_code = mod_code
        self.mod_code_line_no = mod_code_line_no
        self.id = mod_name

#-----------------------------------------

class TrailerAnn(Ann):

    def __init__(self, code, line_no, indent_size):
        '''Instantiate a trailer annotation'''

        Ann.__init__(self, code, line_no, indent_size)

#-----------------------------------------

class NonAnn(CodeFragment):

    def __init__(self, code, line_no, indent_size):
        '''Instantiate a non-annotation'''

        CodeFragment.__init__(self)
        self.code = code
        self.line_no = line_no
        self.indent_size = indent_size

#-----------------------------------------

class AnnCodeRegion(CodeFragment):

    def __init__(self, leader_ann, cfrags, trailer_ann):
        '''Instantiate an annotated code region'''

        CodeFragment.__init__(self)
        self.leader_ann = leader_ann
        self.cfrags = cfrags
        self.trailer_ann = trailer_ann
        self.id = leader_ann

    
        

