#
# This Orio module does not perform any significant code transformations at all.
# The purpose of this Orio module is to simply preserve global declarations in the generated code without
# embedding them in the main function (such embedding is not supported by some compilers).
#

import orio.module.module

#-----------------------------------------

class Preserve(orio.module.module.Module):
    '''A simple rewriting module.'''

    def __init__(self, perf_params, module_body_code, annot_body_code, line_no, indent_size, language='C', tinfo=None):
        '''To instantiate a simple rewriting module.'''

        orio.module.module.Module.__init__(self, perf_params, module_body_code, annot_body_code,
                                      line_no, indent_size, language)

    #---------------------------------------------------------------------
    
    def transform(self):
        '''Simply output the annotated code without changing it.'''

        # to create a comment containing information about the class attributes
        comment = '''
        /*
         perf_params = %s
         module_body_code = "%s"
         annot_body_code = "%s"
         line_no = %s
         indent_size = %s
        */
        ''' % (self.perf_params, self.module_body_code, self.annot_body_code, self.line_no, self.indent_size)

        # to rewrite the annotated code, with the class-attribute comment being prepended
        output_code = comment + self.annot_body_code

        # return the output code
        return output_code

