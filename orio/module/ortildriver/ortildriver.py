#
# The orio.main.file (and class) for OrTil's optimization driver
#

import re, sys
import ann_parser, ast, code_parser, orio.module.module, transformation
from orio.main.util.globals import *

#-----------------------------------------

class OrTilDriver(orio.module.module.Module):
    '''The class definition for OrTil's optimization driver'''
    
    def __init__(self, perf_params, module_body_code, annot_body_code,
                 line_no, indent_size, language='C'):
        '''To instantiate an OrTil's optimization driver'''
        
        orio.module.module.Module.__init__(self, perf_params, module_body_code, annot_body_code,
                                      line_no, indent_size, language)
        
    #---------------------------------------------------------------------

    def __extractCoreTiles(self, code, tile_sizes):
        '''To extract code regions that cover the full core tiles'''
        
        # used regular expressions
        header_ann_re = r'/\* start full core tiles:\s*((([A-Za-z_]\w*)\,?)*?)\s*\*/'
        trailer_ann_re = r'/\* end full core tiles:\s*((([A-Za-z_]\w*)\,?)*?)\s*\*/'

        # initialize the code regions
        code_regions = []
        
        # create a table that maps tile size variables to their corresponding tile size values
        tile_size_table = dict(tile_sizes)

        # extract all full core-tile codes
        while True:

            # get the next header directive
            m = re.search(header_ann_re, code)
            if not m:
                code_regions.append(code)
                break
            header_iters = m.group(1)

            # update the code
            code_regions.append(code[:m.start()])
            code = code[m.end():]

            # get the next trailer directive
            m = re.search(trailer_ann_re, code)
            if not m:
                err('orio.module.ortildriver.ortildriver: missing closing directive for full core tiles region')
            trailer_iters = m.group(1)
            body_code = code[:m.start()]

            # update the code
            code = code[m.end():]

            # compare both header and trailer directives
            if header_iters != trailer_iters:
                err('orio.module.ortildriver.ortildriver: different loop iterators in the opening and closing ' +
                       'directives of full core tiles region', doexit=True)

            # get the specified loop iterators
            iters = header_iters.split(',')

            # derive the corresponding tile size variables
            tile_size_vars = [('T1%s' % i) for i in iters]
            for v in tile_size_vars:
                if v not in tile_size_table:
                    err('orio.module.ortildriver.ortildriver: undefined tile size variable: "%s"' % v)

            # get the tile size values
            tile_size_vals = [tile_size_table[v] for v in tile_size_vars]

            # insert the full core-tile core region information
            code_regions.append((iters, tile_size_vals, body_code))

        # return all extracted code regions
        return code_regions

    #---------------------------------------------------------------------
    
    def transform(self):
        '''To apply loop tiling on the annotated code'''

        # parse the text in the annotation orio.module.body to extract variable value pairs
        var_val_pairs = ann_parser.AnnParser(self.perf_params).parse(self.module_body_code)

        # filter out some variables used for turning on/off the optimizations
        unroll = 1
        vectorize = 1
        scalar_replacement = 1
        constant_folding = 1
        tile_sizes = []
        for var,val in var_val_pairs:
            if var == 'unroll':
                unroll = val
            elif var == 'vectorize':
                vectorize = val
            elif var == 'scalar_replacement':
                scalar_replacement = val
            elif var == 'constant_folding':
                constant_folding = val
            else:
                tile_sizes.append((var, val))

        # remove all annotations from the annotation body text
        ann_re = r'/\*@\s*(.|\n)*?\s*@\*/'
        code = re.sub(ann_re, '', self.annot_body_code)

        # extract code regions that cover the full core tiles
        code_regions = self.__extractCoreTiles(code, tile_sizes)
        
        # parse the full core-tile code and generate a corresponding AST
        n_code_regions = []
        for cr in code_regions:
            if isinstance(cr, str):
                n_code_regions.append(cr)
                continue
            i,v,c = cr
            stmts = code_parser.getParser().parse(c)
            if len(stmts) != 1 or not isinstance(stmts[0], ast.ForStmt):
                err('orio.module.ortildriver.ortildriver: invalid full core-tile code')
            n_code_regions.append((i,v,stmts[0]))
        code_regions = n_code_regions
        
        # transform the full core-tile code 
        transformed_code = ''
        for cr in code_regions:
            if isinstance(cr, str):
                transformed_code += cr
                continue
            i,v,s = cr
            t = transformation.Transformation(unroll, vectorize, scalar_replacement, constant_folding)
            transformed_code += t.transform(i,v,s)

        # insert the declaration code for the tile sizes
        decl_code = ''
        for i, (tvar, tval) in enumerate(tile_sizes):
            decl_code += '  %s = %s;\n' % (tvar, tval)
        if transformed_code[0] != '\n':
            transformed_code = '\n' + transformed_code
        transformed_code = '\n' + decl_code + transformed_code
        
        # return the transformed code
        return transformed_code


