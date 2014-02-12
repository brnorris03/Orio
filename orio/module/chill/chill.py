#
# This Orio 
#

import orio.module.module

#-----------------------------------------

class CHiLL(orio.module.module.Module):
    '''Orio's interface to the CHiLL source transformation infrastructure. '''

    def __init__(self, perf_params, module_body_code, annot_body_code, line_no, indent_size, language='C'):
        '''To instantiate the CHiLL rewriting module.'''

        orio.module.module.Module.__init__(self, perf_params, module_body_code, annot_body_code,
                                      line_no, indent_size, language)

    #---------------------------------------------------------------------
    
    def transform(self):
        '''To simply rewrite the annotated code'''

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


        # CHiLL annotations are of the form:
        # /*@ begin Chill ( transform Recipe(recipe filename) ) @*/
        # ...
        # The code to be transformed by CHiLL here
        # ... 
        # /*@ end Chill @*/


        # Do nothing except output the code annotated with a comment for the parameters that were specified
        # TODO: this is where we use the provided CHiLL recipe or generate a new one
        # Then invoke CHiLL to produce the output_code
        output_code = comment + self.annot_body_code


        # return the output code
        return output_code

