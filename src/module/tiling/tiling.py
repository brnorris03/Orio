#
# The main file (and class) for the tiling transformation module
#

import sys
import ann_parser, module.module

#-----------------------------------------

class Tiling(module.module.Module):
    '''The class definition for the tiling transformation module'''
    
    def __init__(self, perf_params, module_body_code, annot_body_code, cmd_line_opts,
                 line_no, indent_size):
        '''To instantiate a tiling transformation module'''
        
        module.module.Module.__init__(self, perf_params, module_body_code, annot_body_code,
                                      cmd_line_opts, line_no, indent_size)
        
    #---------------------------------------------------------------------
    
    def transform(self):
        '''To apply loop tiling on the annotated code'''

        # parse the text in the annotation module body to extract tiling information
        tile_info_list = ann_parser.AnnParser(self.perf_params).parse(self.module_body_code)

        print tile_info_list
        print '----------'
        print self.module_body_code
        print '----------'
        print self.annot_body_code
        
        sys.exit(1)
        
        # parse to obtain AST
        stmts = parser.getParser(self.line_no).parse(self.annot_body_code)
        
        print '----- force to exit -----'
        sys.exit(1)

