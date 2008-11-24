#
# The implementation of annotation parser
#

import re, sys

#----------------------------------------------------------------

class AnnParser:
    '''The class definition for the annotation parser'''

    # regular expressions
    __vname_re = r'[A-Za-z_]\w*'
    __tile_info_re = r'\s*\(\s*(' + __vname_re + ')\s*,\s*(' + __vname_re + ')\s*(,\s*(\w+)\s*)?\)\s*'

    #------------------------------------------------------------
    
    def __init__(self, perf_params):
        '''To instantiate the annotation parser'''

        self.perf_params = perf_params
    
    #------------------------------------------------------------

    def __semanticCheck(self, tile_info_list):
        '''Check the semantic correctness of the given tiling information list'''

        # iterate over each tiling information
        seen_index_names = {}
        for index_name, tile_size_name, tile_size_value in tile_info_list:

            # check the tile size value
            if tile_size_value:
                try:
                    v = eval(tile_size_value, self.perf_params)
                except Exception, e:
                    print ('error:Tiling: failed to evaluate the tile size value expression: "%s"' %
                           tile_size_value)
                    print ' --> %s: %s' % (e.__class__.__name__, e)
                    sys.exit(1)
                if not isinstance(v, int) or v <= 0:
                    print ('error:Tiling: tile size value must be a positive integer, obtained: "%s"'
                           % tile_size_value)
                    sys.exit(1)

            # check if duplicity of the index name exists
            if index_name in seen_index_names:
                print 'error:Tiling: illegal multiple uses of loop index name "%s"' % index_name
                sys.exit(1)
            seen_index_names[index_name] = None

    #------------------------------------------------------------
    
    def parse(self, code):
        '''Parse the given text to extract tiling information '''

        # scan each tuple to get the tiling information
        tile_info_list = []
        text = code
        while True:

            # scan each tuple
            m = re.match(self.__tile_info_re, text)
            if not m:
                print 'error:Tiling: syntax error in the annotation code: "%s"' % code
                sys.exit(1)

            # get all needed tiling information
            index_name = m.group(1)
            tile_size_name = m.group(2)
            tile_size_value = m.group(4)
            tile_info_list.append((index_name, tile_size_name, tile_size_value))

            # update the text
            text = text[m.end():]

            # remove any trailing comma (if exists)
            if text and text[0] == ',':
                text = text[1:]

            # no more to scan?
            if text == '' or text.isspace():
                break

        # check the semantics of the tiling information
        self.__semanticCheck(tile_info_list)

        # return all tiling information
        return tile_info_list

