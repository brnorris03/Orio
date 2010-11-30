#!/usr/bin/env python
#
# File: $id$
# @Package: orio
# @version: $keRevision$
# @lastrevision: $Date$
# @modifiedby: $LastChangedBy$
# @lastmodified: $LastChangedDate$
#
# Description: Fortran 2003 parser (supports Fortran 77 and later)
# 
# Copyright (c) 2009 UChicago, LLC
# Produced at Argonne National Laboratory
# Author: Boyana Norris (norris@mcs.anl.gov)
#
# For detailed license information refer to the LICENSE file in the top-level
# source directory.

import sys, os
import ast, orio.tool.ply.yacc
import orio.main.parsers.flexer as lexer
from orio.main.parsers.fAST import *
from orio.main.util.globals import *


# Get the token map
tokens = lexer.tokens
baseTypes = {}


# R201
def p_program(p):
    'program : program_unit_list'
    p[0] = p[1]

def p_program_unit_list_1(p):
    'program_unit_list : program_unit'
    p[0] = [p[1]]
    
def p_program_unit_list_2(p):
    'program_unit_list : program_unit_list program_unit'
    p[0] = p[1] + [p[2]]

# R202
def p_program_unit(p):
    '''program_unit : main_program
                    | external_subprogram
                    | module
                    '''
    p[0] = p[1]
    
##---- Names (also see Constants)

# R304
def p_name(p):
    '''name : ID'''
    p[0] = p[1]



# R1101
def p_main_program(p):
    'main_program : program_stmt specification_part execution_part internal_subprogram_part_opt end_program_stmt'
    if p[1]: lineno = p.linespan(1)[0]
    elif p[2]: lineno = p.linespan(2)[0]
    elif p[3]: lineno = p.linespan(3)[0]
    elif p[4]: lineno = p.linespan(4)[0]
    else: lineno = p.linespan(5)[0]
    p[0] = MainProgram(str(lineno), p[0], p[2], p[3])

# R1102
def p_program_stmt_1(p):
    'program_stmt : PROGRAM ID' 
    p[0] = p[2]

def p_program_stmt_2(p):
    'program_stmt : empty'
    p[0] = ''
    
# R1103
def p_end_program_stmt_1(p):
    'end_program_stmt : END'
    pass

def p_end_program_stmt_2(p):
    'end_program_stmt : END PROGRAM opt_id'
    p[0] = p[3]

    
###--- Specification part -------------------------------    
# R204   
def p_specification_part_1(p):
    'specification_part : use_stmt_part import_stmt_list implicit_part declaration_construct_list'
    #TODO
    p[0] = p[2] 

def p_specification_part_2(p):
    'specification_part : empty'
    p[0] = []
    
    
def p_use_stmt_part_1(p):
    'use_stmt_part : use_stmt'
    pass

def p_use_stmt_part_2(p):
    'use_stmt_part : use_stmt_part opt_semi use_stmt'
    pass

def p_use_stmt_part_3(p):
    'use_stmt_part : empty'
    pass

# R1109
def p_use_stmt_1(p):
    'use_stmt : USE comma_module_nature_dcolon module_name comma_rename_list'
    pass

def p_use_stmt_2(p):
    'use_stmt : USE comma_module_nature_dcolon module_name COMMA ONLY COLON only_list'
    pass

def p_comma_module_nature_dcolon_1(p):
    'comma_module_nature_dcolon : COMMA module_nature COLON_COLON'
    pass

def p_comma_module_nature_dcolon_2(p):
    'comma_module_nature_dcolon : empty'
    pass

# R1110
def p_module_nature(p):
    '''module_nature : INTRINSIC
                    | NON_INTRINSIC
                    '''
    p[0] = p[1]
    
def p_module_name(p):
    'module_name : ID'
    p[0] = p[1]

def p_rename_1(p):
    'rename : ID EQ_GT ID'
    pass

def p_rename_2(p):
    'rename : OPERATOR LPAREN defined-operator RPAREN EQ_GT LPAREN defined-operator RPAREN'
    pass

def p_only_list_1(p):
    'only_list : only'
    p[0] = p[1]
    
def p_only_list_2(p):
    'only_list : only_list COMMA only'
    p[0] = p[1] + [p[3]]

# R1112
def p_only(p):
    '''only : generic_spec 
            | only_use_name
            | rename
            '''
    pass

# R1113
def p_only_use_name(p):
    'only_use_name : use_name'
    p[0] = p[1]


# R1114 and R1115
def p_defined_operator(p):
    '''defined_operator : defined_unary_op
                    | defined_binary_op
                    '''

# R1116
def p_block_data(p):
    'block_data : block_data_stmt specification_part end_block_data_stmt'
    pass

# R1117
def p_block_data_stmt(p):
    'block_data_stmt : BLOCK DATA opt_id'
    p[0] = p[1]

# R1118
def p_end_block_data_stmt_1(p):
    'end_block_data_stmt : END BLOCK DATA opt_id'
    p[0] = p[4]

def p_end_block_data_stmt_2(p):
    'end_block_data_stmt : END'
    p[0] = ''

# R205
def p_implicit_part_1(p):
    'implicit_part : implicit_stmt'
    pass

def p_implicit_part_2(p):
    'implicit_part : implicit_part_stmt implicit_stmt'
    pass
 
# R206
def p_implicit_part_stmt(p):
    '''implicit_part_stmt : implicit_stmt
                        | parameter_stmt
                        | format_stmt
                        | entry_stmt
                        '''
    pass

# R207
def p_declaration_construct(p):
    '''declaration_construct : derived_type_def
                            | entry_stmt
                            | enum_def
                            | format_stmt
                            | interface_block
                            | parameter_stmt
                            | procedure_declaration_stmt
                            | specification_stmt
                            | type_declaration_stmt
                            | stmt_function_stmt
                            '''
                            
        
###--- Execution part -----------------------------------

# R208
def p_execution_part_1(p):
    'execution_part : executable_construct'
    pass

def p_execution_part_3(p):
    'execution_part : execution_part_construct_part'
    pass

def p_execution_part_construct_partt_1(p):
    'execution_part_construct_part : execution_part_construct execution_part_construct_part'
    pass

def p_execution_part_construct_part_2(p):
    'execution_part_construct_part : empty'
    pass

# R209
def p_execution_part_construct(p):
    '''execution_part_construct :  executable_construct
                            | formal_stmt
                            | entry_stmt
                            | data_stmt
                            '''
    pass

# R210
def p_internal_subprogram_part(p):
    'internal_subprogram_part : contains_stmt internal_subprogram_part'
    pass

def p_internal_subprogram_part_1(p):
    'internal_subprogram_part : internal_subprogram'
    pass

def p_internal_subprogram_part_2(p):
    'internal_subprogram_part : internal_subprogram_part internal_subprogram'
    p[0] = p[1] + [p[0]]

# R211
def p_internal_subprogram(p):
    '''internal_subprogram : function_subprogram
                        | subroutine_subprogram
                        '''
    pass

# R212
def p_specification_stmt(p):
    '''specification_stmt : access_stmt
                        | allocatable_stmt
                        | asynchronous_stmt
                        | bind_stmt
                        | common_stmt
                        | data_stmt
                        | dimension_stmt
                        | equivalence_stmt
                        | external_stmt
                        | intent_stmt
                        | intrinsic_stmt
                        | namelist_stmt
                        | optional_stmt
                        | pointer_stmt
                        | protected_stmt
                        | save_stmt
                        | target_stmt
                        | volatile_stmt
                        | value_stmt
                        '''
    pass

# R213
def p_executable_construct(p):
    '''executable_construct : action_stmt
                        | associate_construct
                        | case_construct
                        | do_construct
                        | forall_construct
                        | if_construct
                        | select_type_construct
                        | where_construct
                        '''
    pass

# R214
def p_action_stmt(p):
    '''action_stmt : allocate_stmt
                        | assignment_stmt
                        | backspace_stmt
                        | call_stmt
                        | close_stmt
                        | continue_stmt
                        | cycle_stmt
                        | deallocate_stmt
                        | endfile_stmt
                        | end_function_stmt
                        | end_program_stmt
                        | end_subroutine_stmt
                        | exit_stmt
                        | flush_stmt
                        | forall_stmt
                        | goto_stmt
                        | if_stmt
                        | inquire_stmt
                        | nullify_stmt
                        | open_stmt
                        | pointer_assignment_stmt
                        | print_stmt
                        | read_stmt
                        | return_stmt
                        | rewind_stmt
                        | stop_stmt
                        | wait_stmt
                        | where_stmt
                        | write_stmt
                        | arithmetic_if_stmt
                        | computed_goto_stmt
                        '''
    p[0] = p[1]

# R215
def p_keyword(p):
    'keyword : ID'
    p[0] = p[1]

###--- Constants --------------------------------

# R305
def p_constant(p):
    '''constant : literal_constant 
                | named_constant
                '''
    p[0] = p[1]
    
# R306
def p_literal_constant(p):
    '''literal_constant : int_literal_constant
                        | real_literal_constant
                        | complex_literal_constant
                        | logical_literal_constant
                        | char_literal_constant
                        | boz_literal_constant
                        '''
    p[0] = p[1]
    
 
# R307
def p_named_constant(p):
    'named_constant : ID'
    p[0] = p[1] 
    
# R308
def p_int_constant(p):
    'int_constant : constant'
    pass

# R309
def p_char_constant(p):
    'char_constant : constant'
    pass

# R310
def p_intrinsic_operator(p):
    '''intrinsic_operator : power_op
                        | mult_op
                        | add_op
                        | concat_op
                        | rel_op
                        | not_op
                        | and_op
                        | or_op
                        | equiv_op
                        '''
    p[0] = p[1]
    
# R311
def p_defined_operator_1(p):
    '''defined_operator : defined_unary_op
                        | defined_binary_op
                        | extended_intrinsic_op
                        '''
    p[0] = p[1]
    
# R312
def p_extended_intrinsic_op(p):
    'extended_intrinsic_op : intrinsic_operator'
    p[0] = p[1]
    
# R313
def p_label_1(p):
    'label : DIGIT_STRING'
  
###--- Types -------------------------------

# R401
def p_type_spec(p):
    '''type_spec : intrinsic_type_spec
                | derived_type_spec
                '''
    p[0] = p[1]
                
# R402
def p_type_param_value(p):
    '''type_param_value : scalar_int_expr
                        | TIMES
                        | COLON
                        '''
    p[0] = p[1]


# R403
def p_intrinsic_type_spec(p):
    '''intrinsic_type_spec : INTEGER kind_selector
                            | REAL kind_selector
                            | DOUBLE PRECISION
                            | COMPLEX kind_selector
                            | CHARACTER kind_selector
                            | LOGICAL kind_selector
                            '''
    pass

# R404
def p_kind_selector_1(p):
    'kind_selector : scalar_int_initialization_expr'
    p[0] = p[1]

def p_kind_selector_2(p):
    'kind_selector : KIND EQUALS scalar_int_initialization_expr'
    p[0] = p[3]

def p_opt_kind_selector(p):
    '''opt_kind_selector : kind_selector
                        | empty
                        '''
    p[0] = p[1]
                        
# R405
def p_signed_int_literal_constant_1(p):
    'signed_int_literal_constant : signed_digit_string'
    p[0] = p[1]

# R406-407 are in lexer

# R408
def p_signed_digit_string_1(p):
    'signed_digit_string : DIGIT_STRING'
    p[0] = p[1]
        
def p_signed_digit_string_2(p):
    'signed_digit_string : sign DIGIT_STRING'
    p[0] = p[1]
    
# R409 is in lexer

# R410
def p_sign(p):
    '''sign : PLUS
            | MINUS
            '''
    p[0] = p[1]
    
# R411
def p_boz_literal_constant(p):
    '''boz_literal_constant : binary_constant
                        | octal_constant
                        | hex_constant
                        '''
    p[0] = p[1]
    
# R412
def p_binary_constant(p):
    'binary_constant : BCONST'
    p[0] = p[1]

# R413
def p_octal_constant(p):
    '''octal_constant : OCONST_S
                    | OCONST_D
                    '''
    p[0] = p[1]

# R414
def p_hex_constant(p):
    '''hex_constant : HCONST_S
                    | HCONST_D
                    '''
    p[0] = p[1]
    
# R415 is in lexer

# R416
def p_signed_real_literal_constant_1(p):
    'signed_real_literal_constant : real_literal_constant'
    p[0] = p[1]

def p_signed_real_literal_constant_2(p):
    'signed_real_literal_constant : sign real_literal_constant'
    #p[2].setSign(p[1])
    p[0] = p[2]
    
# R417
def p_real_literal_constant(p):
    'real_literal_constant : FCONST'
    p[0] = p[1]
    
# R418 - R420 are in lexer

# R421
def p_complex_literal_constant(p):
    'complex_literal_constant : LPAREN real_part COMMA imag_part RPAREN'
    pass

# R422
def p_real_part(p):
    '''real_part : signed_int_literal_constant 
                | signed_real_literal_constant
                '''
    p[0] = p[1]

# R423    
def p_imag_part(p):
    '''imag_part : signed_int_literal_constant 
                | signed_real_literal_constant
                | named_constant
                '''
    p[0] = p[1]
    
# R424
def p_char_selector_1(p):
    'char_selector : length_selector'
    p[0] = p[1]
    
def p_char_selector_2(p):
    'char_selector : LPAREN LEN EQUALS type_param_value COMMA KIND EQUALS scalar_int_initialization_expr RPAREN'
    pass

def p_char_selector_3(p):
    'char_selector : LPAREN type_param_value COMMA KIND EQUALS scalar_int_initialization_expr RPAREN'
    pass

def p_char_selector_4(p):
    'char_selector : LPAREN type_param_value COMMA scalar_int_initialization_expr RPAREN'
    pass

def p_char_selector_5(p):
    'char_selector : LPAREN KIND EQUALS scalar_int_initialization_expr RPAREN'
    pass

def p_char_selector_6(p):
    'char_selector : LPAREN KIND EQUALS scalar_int_initialization_expr COMMA LEN EQUALS type_param_value RPAREN'
    pass

# R425
def p_length_selector_1(p):
    'length_selector : type_parameter_value'
    p[0] = p[1]
    
def p_length_selector_2(p):
    'length_selector : LEN EQUALS type_parameter_value'
    p[0] = p[3]
    
def p_length_selector_3(p):
    'length_selector : TIMES char_length opt_comma'
    p[0] = p[2]
    
# R426
def p_char_length_1(p):
    'char_length : LPAREN type_param_value RPAREN'
    p[0] = p[2]   

def p_char_length_2(p):
    'char_length : scalar_int_literal_constant'
    p[0] = p[1]

# R427
def p_char_literal_constant(p):
    '''char_literal_constant : SCONST_S
                            | SCONST_D
                            | partial_string_list
                            '''
    if isinstance(p[1],list):
        p[0] = p[1]
    else:
        p[0] = [p[1]]
    
def p_partial_string(p):
    '''partial_string : PSCONST_S
                    | PSCONST_D
                    '''
    p[0] = p[1]
    
def p_partial_string_list_1(p):
    'partial_string_list : partial_string'
    p[0] = p[1]
    
def p_partial_string_list_2(p):
    'partial_string_list : partial_string_list partial_string'
    p[0] = p[1] + [p[2]]
    
# R428
def p_logical_literal_constant(p):
    '''logical_literal_constant : TRUE 
                                | FALSE 
                                '''
    pass

# R429
def p_derived_type_def(p):
    '''derived_type_def : derived_type_stmt 
                        type_param_def_stmt_list 
                        private_or_sequence_list 
                        component_part 
                        type_bound_procedure_part
                        end_type_stmt
                        '''

# R430
def p_derived_type_stmt_1(p):
    'derived_type_stmt : TYPE type_name'
    # TODO
    pass


def p_derived_type_stmt_2(p):
    'derived_type_stmt : TYPE comma_type_attr_spec_list opt_colon_colon type_name'
    # TODO
    pass

def p_derived_type_stmt_3(p):
    'derived_type_stmt : TYPE comma_type_attr_spec_list opt_colon_colon type_name LPAREN type_param_name_list RPAREN'
    # TODO
    pass

def p_comma_type_attr_spec_list(p):
    '''comma_type_attr_spec_list : COMMA type_attr_spec_list
                                | empty
                                '''
    if len(p) < 3: p[0] = []
    else: p[0] = p[2]
    pass

# R431
def p_type_attr_spec_1(p):
    '''type_attr_spec : access_spec 
                    | ABSTRACT
                    '''
    # TODO
    pass

def p_type_attr_spec_2(p):
    '''type_attr_spec : EXTENDS LPAREN parent_type_name RPAREN
                    | BIND LPAREN ID RPAREN
                    '''
    # ID should only be the character C
    #TODO
    pass

def p_type_attr_spec_list_1(p):
    'type_attr_spec_list : type_attr_spec'
    p[0] = [p[1]]

def p_type_attr_spec_list_2(p):
    'type_attr_spec_list : type_attr_spec_list COMMA type_attr_spec'
    p[3].append(p[1])
    p[0] = p[3]

# R432
def p_private_or_sequence(p):
    '''private_or_sequence : private_components_stmt
                            | sequence_stmt
                            '''
    p[0] = p[1]
    
def p_private_or_sequence_part_1(p):
    'private_or_sequence_part : private_or_sequence private_or_sequence_part'
    p[2].append(p[1])
    p[0] = p[2]
    
def p_private_or_sequence_part_2(p):
    'private_or_sequence_part : empty'
    p[0] = []   

# R433
def p_end_type_stmt(p):
    'end_type : END TYPE opt_id'
    pass

# R434
def p_sequence_stmt(p):
    'sequence_stmt : SEQUENCE'
    pass

# R435
def p_type_param_def_stmt(p):
    'type_param_def_stmt : INTEGER opt_kind_selector COMMA type_param_attr_spec COLON_COLON type_param_decl_part'
    # TODO
    pass

def p_type_param_def_stmt_part_1(p):
    'type_param_def_stmt_part : type_param_def_stmt type_param_def_stmt_part'
    p[2].append(p[1])
    p[0] = p[2]
    
def p_type_param_def_stmt_part_2(p):
    'type_param_def_stmt_part : empty'
    p[0] = []   

# R436
def p_type_param_decl_1(p):
    'type_param_decl : type_param_name'
    # TODO
    pass

def p_type_param_decl_2(p):
    'type_param_decl : type_param_name EQUALS scalar_int_initialization_expr'
    # TODO
    pass

def p_type_param_name(p):
    'type_param_name : id'
    p[0] = p[1]
    
def p_type_param_name_list_1(p):
    'type_param_name_list : type_param_name'
    p[0] = [p[1]]
    
def p_type_param_name_list_2(p):
    'type_param_name_list : type_param_name_list COMMA type_param_name'
    p[3].append(p[1])
    p[0] = p[3]
    
    
# R437
def p_type_param_attr_spec(p):
    '''type_param_attr_spec : KIND
                            | LEN
                            '''
    pass

# R438
def p_component_part_1(p):
    'component_part : component_def_stmt component_part'
    p[2].append(p[1])
    p[0] = p[2]

def p_component_part_2(p):
    'component_part : empty'
    p[0] = []
    
# R439
def p_component_def_stmt(p):
    '''component_def_stmt : data_component_def_stmt
                        | proc_component_def_stmt
                        '''
    p[0] = p[1]

# R440
def p_data_component_def_stmt_1(p):
    'data_component_def_stmt : declaration_type_spec opt_colon_colon component_decl_list'
    # TODO 
    pass

def p_data_component_def_stmt_2(p):
    'data_component_def_stmt : declaration_type_spec COMMA component_attr_spec_list COLON_COLON component_decl_list'
    # TODO 
    pass

# R441
def p_component_attr_spec_1(p):
    'component_attr_spec : DIMENSION LPAREN component_array_spec RPAREN'
    # TODO
    pass

def p_component_attr_spec_2(p):
    '''component_attr_spec : POINTER
                            | ALLOCATABLE
                            | access_spec
                            '''
    p[0] = p[1]
    
# R442
def p_component_decl(p):
    'component_decl : component_name opt_component_array_spec opt_char_length opt_component_initialization'
    # TODO 
    pass

def p_opt_component_array_spec(p):
    '''opt_component_array_spec : LPAREN component_array_spec RPAREN
                                | empty
                                '''
    if len(p) < 4: p[0] = None
    else: p[0] = p[2]


def p_opt_component_initialization(p):
    '''opt_component_initialization : component_initialization
                                    | empty
                                    '''
    if len(p) < 2: p[0] = None
    else: p[0] = p[1]
    
# R443
def p_component_array_spec(p):
    '''component_array_spec : explicit_shape_spec_list
                            | deferred_shape_spec_list
                            '''
    p[0] = p[1]

        
# R444
def p_component_initialization(p):
    '''component_initialization : EQUALS initialization_expr
                                | EQ_GT null_init
                                '''
    p[0] = p[1]

# R445
def p_proc_component_def_stmt(p):
    '''proc_component_def_stmt : PROCEDURE LPAREN opt_proc_interface RPAREN COMMA
                                proc_component_attr_spec_list COLON_COLON proc_decl_list
                                '''
    # TODO
    pass

def p_opt_proc_interface(p):
    '''opt_proc_interface : proc_interface
                        | empty
    '''
    p[0] = p[1]

# R446
def p_proc_component_attr_spec_1(p):
    '''proc_component_attr_spec : POINTER
                                | NOPASS
                                | PASS
                                | access_spec
                                '''
    # TODO 
    pass

def p_proc_component_attr_spec_2(p):
    'proc_component_attr_spec : PASS LPAREN arg_name RPAREN'
    # TODO
    pass

def p_proc_component_attr_spec_list(p):
    '''proc_component_attr_spec_list : proc_component_attr_spec_list COMMA proc_component_attr_spec
                                    | empty
                                    '''
    if len(p) < 4: p[0] = []
    else: 
        p[3].append(p[1])
        p[0] = p[3] 

# R447
def p_private_components_stmt(p):
    'private_components_stmt : PRIVATE'
    pass

# R448
def p_proc_binding_stmt(p):
    '''proc_binding_stmt : specific_binding
                        | generic_binding
                        | final_binding
                        '''
    p[0] = p[1]
    
# R449
def p_specific_binding_1(p):
    'specific_binding : PROCEDURE opt_colon_colon binding_name'
    # TODO
    pass

def p_specific_binding_2(p):
    'specific_binding : PROCEDURE opt_interface_name opt_binding_attr_list binding_name opt_procedure_name'
    # TODO
    pass

def p_opt_interface_name(p):
    '''opt_interface_name : LPAREN interface_name RPAREN
                        | empty
                        '''
    if len(p) < 4: p[0] = None
    else: p[0] = p[2]

def p_opt_binding_attr_list(p):
    '''opt_binding_attr_list : COMMA binding_attr_list COLON_COLON
                        | empty
                        '''
    if len(p) < 4: p[0] = []
    else: p[0] = p[2]
    
def p_opt_procedure_name(p):
    '''opt_interface_name : EQ_GT ID
                        | empty
                        '''
    if len(p) < 3: p[0] = None
    else: p[0] = p[2]
    
# R452
def p_generic_binding(p):
    'generic_binding : GENERIC opt_access_spec COLON_COLON generic_spec EQ_GT binding_name_list'
    # TODO
    pass

def p_opt_access_spec(p):
    '''opt_access_spec : COMMA access_spec
                        | empty
                        '''
    if len(p) < 3: p[0] = None
    else: p[0] = p[2]
    
# R453
def p_binding_attr_1(p):
    '''bidning_attr : PASS
                    | NOPASS
                    | NON_OVERRIDABLE
                    | DEFERRED
                    | access_spec
                    '''
    #TODO
    pass

def p_binding_attr_2(p):
    'binding_attr : PASS LPAREN arg_name RPAREN'
    # TODO
    pass

# R454
def p_final_binding(p):
    'findal_binding : FINAL opt_colon_colon final_subroutine_name_list'
    # TODO
    pass

# R455
def p_derived_type_spec_1(p):
    'type_name : ID'
    # TODO
    pass

def p_derived_type_spec_2(p):
    'type_name : ID LPAREN type_param_spec_list RPAREN'
    # TODO
    pass

# R456
def p_type_param_spec_1(p):
    'type_param_spec : type_param_value'
    # TODO
    pass

def p_type_param_spec_2(p):
    'type_param_spec : keyword EQUALS type_param_value'
    # TODO
    pass

def p_type_param_spec_list(p):
    '''type_param_spec : type_param_spec_list COMMA type_param_spec
                        | empty
                        '''
    if len(p) < 4: p[0] = []
    else: 
        p[3].append(p[1])
        p[0] = p[3]
        
# R457
def p_structure_constructor_1(p):
    'structure_constructor : derived_type_spec'
    # TODO
    pass

def p_structure_constructor_2(p):
    'structure_constructor : derived_type_spec LPAREN component_spec_list RPAREN'
    #TODO
    pass


# R458
def p_component_spec_1(p):
    'component_spec : component_data_source'
    # TODO
    pass

def p_component_spec_2(p):
    'component_spec : keyword EQUALS component_data_source'
    # TODO
    pass

def p_component_spec_list(p):
    '''component_spec_list : component_spec_list COMMA component_spec
                            | component_spec
                            '''
    if len(p) < 4: p[0] = [p[1]]
    else: 
        p[3].append(p[1])
        p[0] = p[3]
          
# R459
def p_component_data_source(p):
    '''component_data_source : expr
                            | data_target
                            | proc_target
                            '''
    p[0] = p[1]

# RR460
def p_enum_def(p):
    'enum_def : enum_def_stmt_part end_enum_def_stmt'
    # TODO
    pass

def p_enum_def_stmt_part_1(p):
    'enum_def_stmt_part : enum_def_stmt'
    p[0] = [p[1]]

def p_enum_def_stmt_part_2(p):
    'enum_def_stmt_part : enum_def_stmt enum_def_stmt_part'
    p[2].append(p[1]) 
    p[0] = p[2] 
    
# R461
def p_enum_def_stmt(p):
    'enum_def_stmt : ENUM COMMA BIND LPAREN ID RPAREN'
    # ID must be the character C
    # TODO
    pass

# R462
def p_enumerator_def_stmt(p):
    'enumerator_def-stmt : ENUMERATOR opt_colon_colon enumerator_list'
    # TODO
    pass

# R463
def p_enumerator_1(p):
    'enumerator : named_constant'
    # TODO
    pass

def p_enumerator_2(p):
    'enumerator : named_constant EQUALS scalar_int_initialization_expr'
    # TODO
    pass

# R464
def p_end_enum_stmt(p):
    'end_enum_stmt : END ENUM'
    pass

# R465
def p_array_constructor_1(p):
    'array_constructor : LPAREN DIVIDE ac_spec DIVIDE RPAREN'
    # TODO
    pass

def p_array_constructor_2(p):
    'array_constructor : LBRACKET ac_spec RBRACKET'
    # TODO
    pass

# R466
def p_ac_spec_1(p):
    'ac_spec : type_spec COLON_COLON'
    # TODO 
    pass

def p_ac_spec_2(p):
    'ac_spec : type_spec COLON_COLON ac_value_list'
    # TODO 
    pass

def p_ac_spec_3(p):
    'ac_spec : ac_value_list'
    # TODO 
    pass

# R467 and R468 are in lexer

# R469
def p_ac_value(p):
    '''ac_value : expr
                | ac_implied_do
                '''
    # TODO
    pass

def p_ac_value_list(p):
    '''ac_value_list : ac_value_list COMMA ac_value
                    | ac_value
                    '''
    if len(p) < 4: p[0] = [p[1]]
    else: 
        p[3].append(p[1])
        p[0] = p[3]
        
# R470
def p_ac_implied_do(p):
    'ac_implied_do : LPAREN ac_value_list COMMA ac_implied_do_control RPAREN'
    # TODO
    pass

# R471
def p_impled_do_control(p):
    'implied_do_control : ac_do_variable EQUALS scalar_int_expr COMMA scalar_int_expr_list'
    # TODO
    pass

# R472
def p_ac_do_variable(p):
    'ac_do_variable : scalar_int_variable'
    p[0] = p[1]
    

###--- Sec 5 types --------------------------------

# R501
def p_type_declaration_stmt_1(p):
    'type_declaration_stmt : declaration_type_spec opt_colon_colon entity_decl_list'
    # TODO 
    pass


def p_type_declaration_stmt_2(p):
    'type_declaration_stmt : declaration_type_spec opt_attr_spec_part entity_decl_list'
    # TODO 
    pass

def p_opt_attr_spec_part(p):
    '''opt_attr_spec_part : COMMA attr_spec_part COLON_COLON
                        | empty
                        '''
    if len(p) < 4: p[0] = None
    else: p[0] = p[2]
    
def p_attr_spec_part(p):
    '''attr_spec_part : attr_spec attr_spec_part 
                    | empty
                    '''
    if len(p) < 3: p[0] = []
    else: 
        p[2].append(p[1])
        p[0] = p[2]
        
# R502
def p_declaration_type_spec_1(p):
    'declaration_type_spec : intrinsic_type_spec'
    p[0] = p[1]
    
def p_declaration_type_spec_2(p):
    '''declaration_type_spec : TYPE LPAREN derived_type_spec RPAREN
                            | CLASS LPAREN derived_type_spec RPAREN
                            | CLASS LPAREN TIMES RPAREN
                            '''
    # TODO
    pass

# R503
def p_attr_spec_1(p):
    '''attr_spec : access_spec 
                | ALLOCATABLE
                | ASYNCHRONOUS
                | EXTERNAL
                | INTRINSIC
                | language_binding_spec
                | OPTIONAL
                | PARAMETER
                | POINTER
                | PROTECTED
                | SAVE
                | TARGET
                | VALUE
                | VOLATILE
                '''
    # TODO
    pass

def p_attr_spec_2(p):
    'attr_spec : DIMENSION LPAREN array_spec RPAREN'
    # TODO
    pass

# R504
def p_entity_decl_1(p):
    'entity_decl : object_name opt_array_spec opt_char_length opt_initialization'
    # TODO (ID is object name)
    pass

def p_entity_decl_2(p):
    'entity_decl : function_name opt_char_length'
    # TODO
    pass

def p_opt_array_spec(p):
    '''opt_array_spec : LPAREN array_spec RPAREN
                    | empty
                    '''
    if len(p) < 4: p[0] = None
    else: p[0] = p[2]

def p_opt_char_length(p):
    '''opt_char_legth : TIMES char_length
                    | empty
                    '''
    if len(p) < 3: p[0] = None
    else: p[0] = p[2]
    
def p_opt_initialization(p):
    '''opt_initialization : initialization
                        | empty
                        '''
    if len(p) < 2: p[0] = None
    else: p[0] = p[1]
    
# R505 
def p_object_name(p):
    'object_name : ID'
    p[0] = p[1]
    
# R506 
def p_initialization(p):
    '''initialization : EQUALS initialization_expr
                    | EQ_GT null_init
                    '''
    p[0] = p[2]

# R507
def p_null_init(p):
    'null_init : function_reference'
    #  C506 The function_reference shall be a reference to the NULL intrinsic function
    # with no arguments
    p[0] = p[1]

# R508
def p_access_spec(p):
    '''access_spec : PUBLIC
                    | PRIVATE
                    '''
    p[0] = p[1]
    
# R509
def p_language_binding_spec_1(p):
    'language_binding_spec : BIND LPAREN ID RPAREN'
    # ID must be C
    #TODO
    pass

def p_language_binding_spec_2():
    'language_binding_spec : BIND LPAREN ID COMMA NAME EQUALS scalar_char_initialization_expr RPAREN'
    # ID must be C
    # TODO
    pass

# R510
def p_array_spec(p):
    '''array_spec : explicit_shape_spec_list
                | assumed_shape_spec_list
                | deferred_shape_spec_list
                | assumed_size_spec
                '''
    p[0] = p[1]
    
# R511
def p_explicit_shape_spec_1(p):
    'explicit_shape_spec : upper_bound'
    # TODO
    pass

def p_explicit_shape_spec_2(p):
    'explicit_shape_spec : lower_bound COLON upper_bound'
    # TODO
    pass

def p_explicit_shape_spec_list(p):
    '''explicit_shape_spec_list : explicit_shape_list COMMA explicit_shape_spec
                                | explicit_shape
                                '''
    if len(p) < 4: p[0] = [p[1]]
    else: 
        p[3].append(p[1])
        p[0] = p[3]

def p_lower_bound(p):
    'lower_bound : specification_expr'
    p[0] = p[1]

def p_upper_bound(p):
    'upper_bound : specification_expr'
    p[0] = p[1]

        
# R514 
def p_assumed_shape_spec_1(p):
    'assumed_shape_spec : COLON'
    pass

def p_assumed_shape_spec_2(p):
    'assumed_shape_spec : lower_bound COLON'
    pass
    
def p_assumed_shape_spec_list(p):
    '''assumed_shape_spec_list : explicit_shape_list COMMA assumed_shape_spec
                                | explicit_shape
                                '''
    if len(p) < 4: p[0] = [p[1]]
    else: 
        p[3].append(p[1])
        p[0] = p[3]


# R515
def p_deferred_shape_spec(p):
    'deferred_shape_spec : COLON'
    pass

def p_deferred_shape_spec_list(p):
    '''deferred_shape_spec_list : deferred_shape_list COMMA deferred_shape_spec
                                | deferred_shape
                                '''
    if len(p) < 4: p[0] = [p[1]]
    else: 
        p[3].append(p[1])
        p[0] = p[3]
        

# R516
def p_assumed_size_spec(p):
    'assumed_size_spec : opt_explicit_shape_spec_list opt_lower_bound TIMES'
    # TODO
    pass

def p_opt_explicit_shape_spec_list(p):
    '''opt_explicit_shape_spec_list : explicit_shape_spec_list COMMA explicit_shape
                                    | empty
                                    '''
    if len(p) < 3: p[0] = None
    else: p[0] = p[1]
    
def p_opt_lower_bound(p):
    '''opt_lower_bound : lower_bound COLON
                        | empty
                        '''
    if len(p) < 3: p[0] = None
    else: p[0] = p[1]
    
    
# R517
def p_intent_spec(p):
    '''intent_spec : IN
                    | OUT
                    | INOUT
                    '''
    p[0] = p[1]
    pass

# R518
def p_access_stmt_1(p):
    'access_stmt : access_spec'
    # TODO
    pass

def p_access_stmt_2(p):
    'access_stmt : access_spec_list opt_colon_colon access_id'
    # TODO
    pass

# R519
def p_access_id(p):
    '''access_id : use_name
                | generic_spec
                '''
    p[0] = p[1]
    
# R520
def p_allocatable_stmt(p):
    'allocatable_stmt : ALLOCATABLE opt_colon_colon object_name_with_opt_deferred_shape_list'
    # TODO            
    pass

def p_opt_deferred_shape_spec_list_in_paren(p):
    '''opt_deferred_shape_spec_list_in_paren : LPAREN deferred_shape_spec_list RPAREN
                                            | empty
                                            '''
    if len(p) < 4: p[0] = None
    else: p[0] = p[2]
    
def p_object_name_with_opt_deferred_shape_list_1(p):
    'object_name_with_opt_deferred_shape_list : object_name opt_deferred_shape_spec_list_in_paren'
    # TODO
    pass

def p_object_name_with_opt_deferred_shape_list_2(p):
    'object_name_with_opt_deferred_shape_list : object_name opt_deferred_shape_spec_list_in_paren COMMA object_name_with_opt_deferred_shape_list'
    # TODO
    pass
    
# R521
def p_asyncrhonous_stmt(p):
    'asynchronous_stmt : ASYNCHRONOUS opt_colon_colon object_name_list'
    # TODO
    pass

def p_object_name_list_1(p):
    'object_name_list : object_name'
    p[0] = [p[1]]

def p_object_name_list_2(p):
    'object_name_list : object_name_list COMMA object_name'
    p[3].append(p[1])
    p[0] = p[3]

# R522
def p_bind_stmt(p):
    'bind_stmt : language_binding_spec opt_colon_colon bind_entity_list'
    # TODO
    pass

# R523
def p_bind_entity_1(p):
    'bind_entity : entity_name'
    pass

def p_bind_entity_2(p):
    'bind_entity : DIVIDE common_block_name DIVIDE'
    pass


###--- Expressions --------------------------------

# R701
def p_primary_1(p):
    '''primary : constant
                | designator
                | array_constructor
                | structure_constructor
                | function_reference
                | type_param_inquiry
                | type_param_name
                '''
                
    p[0] = p[1]

def p_primary_2(p):
    'primary : LPAREN expr RPAREN'
    p[0] = p[2]
    
# R702
def level_one_expr_1(p):
    'level_one_expr : primary'
    p[0] = p[1]
    
def level_one_expr_2(p):
    'level_one_expr : defined_unary_op primary'
    #p[2].setop(p[1])
    p[0] = p[2]

# R703
def p_defined_unary_operator(p):
    'defined_unary_operator : DEFINED_UNARY_OP'
    p[0] = p[1]
    
# R704
def p_mult_operand_1(p):
    'mult_operand : level_one_expr'
    p[0] = p[1]
    
def p_mult_operand_2(p):
    'mult_operand : level_one_expr power_op mult_operand'
    # TODO: create new binary op node and add operands
    pass

# R705
def p_add_operand_1(p):
    'add_operand : mult_operand'
    p[0] = p[1]
    
def p_add_operand_2(p):
    'add_operand : add_operand mult_op mult_operand'
    # TODO: create new binary op node and add operands
    pass

# R706
def p_level_two_expr_1(p):
    'level_two_expr : add_operand'
    p[0] = p[1]
    
def p_level_two_expr_2(p):
    'level_two_expr : add_op add_operand'
    p[0] = p[1]
    
def p_level_two_expr_3(p):
    'level_two_expr : level_two_expr add_op add_operand'
    # TODO: create new binary op node and add operands
    pass

# R707
def p_power_op(p):
    'power_op : TIMES TIMES'
    pass

# R708
def p_mult_op(p):
    '''mult_op : TIMES
                | DIVIDE
                '''
    pass

# R709
def p_add_op(p):
    '''add_op : PLUS
            | MINUS
            '''
    pass

# R710
def p_level_three_expr_1(p):
    'level_three_expr : level_two_expr'
    p[0] = p[1]
    
def p_level_three_expr_2(p):
    'level_three_expr : level_three_expr concat_op level_three_expr'
    # TODO
    pass

# R711
def p_concat_op(p):
    'concat_op : DIVIDE DIVIDE'
    pass

# R712
def p_level_four_expr_1(p):
    'level_four_expr : level_three_expr'
    p[0] = p[1]
    
def p_level_four_expr_2(p):
    'level_four_expr : level_three_expr rel_op level_three_expr'
    # TODO 
    pass

# R713
def p_rel_op_1(p):
    '''rel_op : EQ 
            | NE
            | LT
            | LE
            | GT
            | GE
            | LESSTHAN
            | LESSTHAN_EQ
            | GREATERTHAN
            | GREATERTHAN_EQ
            | EQ_GT
            | EQ_EQ
            | SLASH_EQ
            '''
    pass

# R714
def p_and_operand_1(p):
    'and_operand : level_four_expr'
    pass

def p_and_operand_2(p):
    'and_operand : NOT level_four_expr'
    pass

# R715
def p_or_operand_1(p):
    'or_operand : and_operand'
    pass

def p_or_operand_2(p):
    'or_operand : or_operand AND and_operand'
    # TODO
    pass

# R716
def p_equiv_operand_1(p):
    'equiv_operand : or_operand'
    # TODO
    pass

def p_equiv_operand_2(p):
    'equiv_operand : equiv_operand OR or_operand'
    # TODO
    pass

# R717
def p_level_five_expr_1(p):
    'level_five_expr : equiv_operand'
    pass

def p_level_five_expr_2(p):
    'level_five_expr : level_five_expr equiv_op equiv_operand'
    # TODO
    pass

# R718 - R720 are in lexer

# R721
def p_equiv_op(p):
    '''equiv_op : EQV
                | NEQV
                '''
    p[0] = p[1]

# R722
def p_expr_1(p):
    'expr : level_five_expr'
    p[0] = p[1]
    
def p_expr_2(p):
    'expr : expr DEFINED_BINARY_OP level_five_expr'
    # TODO
    pass

# R723 is in lexer

# R724
def p_logical_expr(p):
    'logical_expr : expr'
    p[0] = p[1]
    
# R725
def p_char_expr(p):
    'char_expr : expr'
    p[0] = p[1]

# R726
def p_default_char_expr(p):
    'default_char_expr : expr'
    p[0] = p[1]
    
# R727
def p_int_expr(p):
    'int_expr : expr'
    p[0] = p[1]

# R728
def p_numeric_expr(p):
    'numeric_expr : expr'
    p[0] = p[1]

# R729
def p_specification_expr(p):
    'specification_expr : scalar_int_expr'
    p[0] = p[1]

# R730
def p_initialization_expr(p):
    'initialization_expr : expr'
    p[0] = p[1]

# R731
def p_char_initialization_expr(p):
    'char_initialization_expr : char_expr'
    p[0] = p[1]

# R732
def p_int_initialization_expr(p):
    'int_initialization_expr : int_expr'
    p[0] = p[1]

# R733
def p_logical_initialization_expr(p):
    'logical_initialization_expr : logical_expr'
    p[0] = p[1]

# R734
def p_assignment_stmt(p):
    'assignment_stmt : variable EQUALS expr'
    # TODO
    pass

# R735
def p_pointer_assignment_stmt_1(p):
    '''pointer_assignment_stmt : data_pointer_object EQ_GT data_target
                                | proc_pointer_object EQ_GT proc_target
                                '''
    # TODO
    pass

def p_pointer_assignment_stmt_2(p):
    '''pointer_assignment_stmt : data_pointer_object LPAREN bounds_spec_part RPAREN EQ_GT data_target
                    | data_pointer_object LPAREN bounds_remapping_part RPAREN EQ_GT data_target
                    '''
    # TODO
    pass

# R736
def p_data_pointer_object_1(p):
    'data_pointer_object : variable_name'
    p[0] = p[1]

def p_data_pointer_object_2(p):
    'data_pointer_object : variable MOD data_pointer_component_name'
    p[0] = p[1]

# R737
def p_bounds_spec(p):
    'bounds_spec : lower_bound_expr COLON'
    # TODO
    pass

# R738
def p_bounds_remapping(p):
    'bounds_remapping : lower_bound_expr COLON upper_bound_expr'
    # TODO
    pass

# R739
def p_data_target(p):
    '''data_target : variable
                    | expr
                    '''
    p[0] = p[1]
    
# R740
def p_proc_pointer_object(p):
    '''proc_pointer_object : proc_pointer_name
                            | proc_component_ref
                            '''
    p[0] = p[1]
    
# R741
def p_proc_component_ref(p):
    'proc_component_ref : variable MOD procedure_component_name'
    # TODO
    pass

# R742
def p_proc_target(p):
    '''proc_target : expr
                    | procedure_name
                    | proc_component_ref
                    '''
    p[0] = p[1]
    
def p_scalar_int_expression(p):
    'scalar_int_expression : expr'
    # Restricted expression, see p. 125 in standard (sec. 7.1.6)
    p[0] = p[1]
    
def p_scalar_int_expression_list(p):
    '''scalar_int_expression_list : scalar_int_expression_list COMMA scalar_int_expression
                                | scalar_int_expression
                                '''
    if len(p) < 4: p[0] = [p[1]]
    else:
        p[3].append(p[1])
        p[0] = p[3]
    
###--- Statements ------------------------


# R743
def p_where_stmt(p):
    'where_stmt : WHERE LPAREN mask_expr RPAREN where_assignment_stmt'
    # TODO
    pass

# R744
def p_where_construct_1(p):
    'where_construct : where_construct_stmt end_where_stmt'
    # TODO 
    pass

def p_where_stmt_2(p):
    'where_construct : where_construct_stmt where_body_construct_part masked_elsewhere_stmt_part elsewhere_stmt_part end_where_stmt'
    # TODO 
    pass

def p_where_body_construct_part_1(p):
    'where_body_construct_part : where_body_construct'
    p[0] = [p[1]]
    
def p_where_body_construct_part_2(p):
    'where_body_construct_part : where_body_construct where_body_construct_part'
    p[2].append(p[1])
    p[0] = p[2]
    
def p_where_body_construct_part_3(p):
    'where_body_construct_part : empty'
    p[0] = []

def p_masked_elsewhere_stmt_part(p):
    'masked_elsewhere_stmt_part : masked_elsewhere_stmt where_body_construct_part'
    # TODO
    pass

def p_elsewhere_stmt_part(p):
    'elsewhere_stmt_part : elsewhere_stmt where_body_construct_part'
    # TODO 
    pass

# R745
def p_where_construct_stmt_1(p):
    'where_construct_stmt : WHERE LPAREN mask_expr RPAREN'
    # TODO
    pass

def p_where_construct_stmt_2(p):
    'where_construct_stmt : where_construct_name COLON WHERE LPAREN mask_expr RPAREN'
    # TODO
    pass

# R746
def p_where_body_construct(p):
    '''where_body_construct : where_assignment_stmt
                            | where_stmt
                            | where_construct
                            '''
    p[0] = p[1]
    
# R747 
def p_where_assignment_stmt(p):
    'where_assignment_stmt : assignment_stmt'
    p[0] = p[1]
    
# R748
def p_mask_expr(p):
    'mask_expr : logical_expr'
    p[0] = p[1]
    
# R749
def p_masked_elsewhere_stmt_1(p):
    'masked_elsewhere_stmt : ELSEWHERE LPAREN masked_expr RPAREN'
    # TODO
    pass

def p_masked_elsewhere_stmt_2(p):
    'masked_elsewhere_stmt : ELSEWHERE LPAREN masked_expr RPAREN where_construct_name'
    # TODO
    pass

# R750
def p_elsewhere_stmt_1(p):
    'elsewhere_stmt : ELSEWHERE'
    pass

def p_elsewhere_stmt_2(p):
    'elsewhere_stmt : ELSEWHERE where_construct_name'
    pass

# R751
def p_end_where_stmt_1(p):
    'end_where_stmt : END WHERE'
    pass

def p_end_where_stmt_2(p):
    'end_where_stmt : END WHERE where_construct_name'
    pass

# R752
def p_forall_construct(p):
    'forall_construct : forall_construct_stmt forall_body_construct_part end_forall_stmt'
    # TODO
    pass

def p_forall_body_construct_part_1(p):
    'forall_body_construct_part : forall_body_construct forall_body_construct_part'
    p[2].append(p[1])
    p[0] = p[2]

def p_forall_body_construct_part_2(p):
    'forall_body_construct_part : empty'
    p[0] = []

# R753
def p_forall_construct_stmt_1(p):
    'forall_construct_stmt : FORALL forall_header'
    # TODO 
    pass

def p_forall_construct_stmt_2(p):
    'forall_construct_stmt : forall_construct_name COLON FORALL forall_header'
    # TODO 
    pass

# R754
def p_forall_header_1(p):
    'forall_header : LPAREN forall_triplet_spec_list RPAREN'
    # TODO
    pass

def p_forall_header_2(p):
    'forall_header : LPAREN forall_triplet_spec_list COMMA scalar_mask_expr RPAREN'
    # TODO
    pass

# R755
def p_forall_triplet_spec_1(p):
    'forall_triplet_spec : index_name EQUALS subscript COLON subscript'
    # TODO
    pass

def p_forall_triplet_spec_2(p):
    'forall_triplet_spec : index_name EQUALS subscript COLON subscript COLON stride'
    # TODO
    pass

def p_forall_triplet_spec_list_1(p):
    'forall_triplet_spec_list : forall_triplet_spec_list COMMA forall_triplet_spec'
    p[3].append(p[1])
    p[0] = p[3]

def p_forall_triplet_spec_list_2(p):
    'forall_triplet_spec_list : forall_triplet_spec'
    p[0] = [p[1]]

# R756
def p_forall_body_construct(p):
    '''forall_body_construct : forall_assignment_stmt
                            | where_stmt
                            | where_construct
                            | forall_construct
                            | forall_stmt
                            '''
    p[0] = p[1]
    
# R757
def p_forall_assignment_stmt(p):
    '''forall_assignment_stmt : assignment_stmt
                            | pointer_assignment_stmt
                            '''
    p[0] = p[1]

# R758
def p_end_forall_stmt_1(p):
    'end_forall_stmt : END FORALL'
    pass

def p_end_forall_stmt_2(p):
    'end_forall_stmt : END FORALL forall_construct_name'
    pass

# R759
def p_forall_stmt(p):
    'forall_stmt : FORALL forall_header forall_assignment_stmt'
    # TODO
    pass


###--- Procedures and interfaces  ------------------------------------------------

# R1201
def p_interface_block(p):
    'interface_block : interface_stmt interface_spec_list end_interface_stmt'
    p[0] = p[2]
    
def p_interface_spec_list(p):
    '''interface_spec_list : interface_spec_list interface_spec 
                            | empty
                            '''
    if len(p) > 1:
        p[1].append(p[2])
        p[0] = p[1]
    else:
        p[0] = []
        
# R2102
def p_interface_spec(p):
    'interface_spec : interface_body'
    p[0] = p[1]

# R1214
def p_proc_decl_1(p):
    'proc_decl : procedure_entity_name'
    p[0] = p[1]
    
def p_proc_decl_2(p):
    'proc_decl : procedure_entity_name EQ_GT null_init'
    p[0] = p[1]
    
def p_proc_decl_list(p):
    '''proc_decl_list : proc_decl_list COMMA proc_decl
                    | empty
                    '''
    if len(p) < 4: p[0] = []
    else: 
        p[3].append(p[1])
        p[0] = p[3]
        
# R1215
#C1212 (R1215) The name shall be the name of an abstract interface or of a procedure that has an 
#explicit interface. If name is declared by a procedure-declaration-stmt it shall be previously 
#declared. If name denotes an intrinsic procedure it shall be one that is listed in 13.6 and not 
#marked with a bullet. 
#C1213 (R1215) The name shall not be the same as a keyword that specifies an intrinsic type. 
#C1214 If a procedure entity has the INTENT attribute or SAVE attribute, it shall also have the 
#POINTER attribute. 
def p_iterface_name(p):
    'interface_name : name'
    p[0] = p[1]
    



    


###--- Miscellaneous --------------------------------------------
    
def p_opt_semi(p):
    '''opt_semi : SEMI
                | empty
                '''
    pass

def p_opt_comma(p):
    '''opt_comma : COMMA
                | empty
                '''
    pass

def p_opt_colon_colon(p):
    '''opt_colon_colon : COLON_COLON 
                        | empty
                        '''
    pass

def p_opt_id_1(p):
    'opt_id : ID'
    p[0] = p[1]
    
def p_opt_id_2(p):
    'opt_id : empty'
    p[0] = ''


def p_empty(p):
    'empty :'
    p[0] = None


    
###--- End Fortran parser rules ---------------------------------


























# statement-list
def p_statement_list_opt_1(p):
    'statement_list_opt :'
    p[0] = []
    
def p_statement_list_opt_2(p):
    'statement_list_opt : statement_list'
    p[0] = p[1]
    
def p_statement_list_1(p):
    'statement_list : statement'
    p[0] = [p[1]]
    
def p_statement_list_2(p):
    'statement_list : statement_list statement'
    p[1].append(p[2])
    p[0] = p[1]
    
# statement:
def p_statement(p):
    '''statement : expression_statement
                 | compound_statement
                 | selection_statement
                 | iteration_statement
                 | transformation_statement
                 '''
    p[0] = p[1]
    
# expression-statement:
def p_expression_statement(p):
    'expression_statement : expression_opt SEMI'
    p[0] = ast.ExpStmt(p[1], p.lineno(1) + __start_line_no - 1)

# compound-statement:
def p_compound_statement(p):
    'compound_statement : LBRACE statement_list_opt RBRACE'
    p[0] = ast.CompStmt(p[2], p.lineno(1) + __start_line_no - 1)
    
# selection-statement
# Note:
#   This results in a shift/reduce conflict. However, such conflict is not harmful
#   because PLY resolves such conflict in favor of shifting.
def p_selection_statement_1(p):
    'selection_statement : IF LPAREN expression RPAREN statement'
    p[0] = ast.IfStmt(p[3], p[5], None, p.lineno(1) + __start_line_no - 1)
    
def p_selection_statement_2(p):
    'selection_statement : IF LPAREN expression RPAREN statement ELSE statement'
    p[0] = ast.IfStmt(p[3], p[5], p[7], p.lineno(1) + __start_line_no - 1)

# iteration-statement
def p_iteration_statement(p):
    'iteration_statement : FOR LPAREN expression_opt SEMI expression_opt SEMI expression_opt RPAREN statement'
    p[0] = ast.ForStmt(p[3], p[5], p[7], p[9], p.lineno(1) + __start_line_no - 1)

# transformation-statement
def p_transformation_statement(p):
    'transformation_statement : TRANSFORM ID LPAREN transformation_argument_list_opt RPAREN statement'
    p[0] = ast.TransformStmt(p[2], p[4], p[6], p.lineno(1) + __start_line_no - 1)

# transformation-argument-list
def p_transformation_argument_list_opt_1(p):
    'transformation_argument_list_opt :'
    p[0] = []

def p_transformation_argument_list_opt_2(p):
    'transformation_argument_list_opt : transformation_argument_list'
    p[0] = p[1]

def p_transformation_argument_list_1(p):
    'transformation_argument_list : transformation_argument'
    p[0] = [p[1]]

def p_transformation_argument_list_2(p):
    'transformation_argument_list : transformation_argument_list COMMA transformation_argument'
    p[1].append(p[3])
    p[0] = p[1]

# transformation-argument
def p_transformation_argument(p):
    'transformation_argument : ID EQUALS py_expression'
    p[0] = [p[1], p[3], p.lineno(1) + __start_line_no - 1]

# expression:
def p_expression_opt_1(p):
    'expression_opt :'
    p[0] = None

def p_expression_opt_2(p):
    'expression_opt : expression'
    p[0] = p[1]

def p_expression_1(p):
    'expression : assignment_expression'
    p[0] = p[1]

def p_expression_2(p):
    'expression : expression COMMA assignment_expression'
    p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.COMMA, p.lineno(1) + __start_line_no - 1)

# assignment_expression:
def p_assignment_expression_1(p):
    'assignment_expression : logical_or_expression'
    p[0] = p[1]

def p_assignment_expression_2(p):
    'assignment_expression : unary_expression assignment_operator assignment_expression'
    if (p[2] == '='):
        p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.EQ_ASGN, p.lineno(1) + __start_line_no - 1)
    elif p[2] in ('*=', '/=', '%=', '+=', '-='):
        lhs = p[1].replicate()
        rhs = None
        if (p[2] == '*='):
            rhs = ast.BinOpExp(p[1], p[3], ast.BinOpExp.MUL, p.lineno(1) + __start_line_no - 1)
        elif (p[2] == '/='):
            rhs = ast.BinOpExp(p[1], p[3], ast.BinOpExp.DIV, p.lineno(1) + __start_line_no - 1)
        elif (p[2] == '%='):
            rhs = ast.BinOpExp(p[1], p[3], ast.BinOpExp.MOD, p.lineno(1) + __start_line_no - 1)
        elif (p[2] == '+='):
            rhs = ast.BinOpExp(p[1], p[3], ast.BinOpExp.ADD, p.lineno(1) + __start_line_no - 1)
        elif (p[2] == '-='):
            rhs = ast.BinOpExp(p[1], p[3], ast.BinOpExp.SUB, p.lineno(1) + __start_line_no - 1)
        else:
            err('orio.main.parsers.fparser internal error:  missing case for assignment operator')
        p[0] = ast.BinOpExp(lhs, rhs, ast.BinOpExp.EQ_ASGN, p.lineno(1) + __start_line_no - 1)
    else:
        err('orio.main.parsers.fparser internal error:  unknown assignment operator')

# assignment-operator:
def p_assignment_operator(p):
    '''assignment_operator : EQUALS
                           | TIMESEQUAL
                           | DIVEQUAL
                           | MODEQUAL
                           | PLUSEQUAL
                           | MINUSEQUAL
                           '''
    p[0] = p[1]

# logical-or-expression
def p_logical_or_expression_1(p):
    'logical_or_expression : logical_and_expression'
    p[0] = p[1]

def p_logical_or_expression_2(p):
    'logical_or_expression : logical_or_expression LOR logical_and_expression'
    p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.LOR, p.lineno(1) + __start_line_no - 1)

# logical-and-expression
def p_logical_and_expression_1(p):
    'logical_and_expression : equality_expression'
    p[0] = p[1]

def p_logical_and_expression_2(p):
    'logical_and_expression : logical_and_expression LAND equality_expression'
    p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.LAND, p.lineno(1) + __start_line_no - 1)

# equality-expression:
def p_equality_expression_1(p):
    'equality_expression : relational_expression'
    p[0] = p[1]

def p_equality_expression_2(p):
    'equality_expression : equality_expression equality_operator relational_expression'
    if p[2] == '==':
        p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.EQ, p.lineno(1) + __start_line_no - 1)
    elif p[2] == '!=':
        p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.NE, p.lineno(1) + __start_line_no - 1)
    else:
        err('orio.main.parsers.fparser internal error:  unknown equality operator')

# equality-operator:
def p_equality_operator(p):
    '''equality_operator : EQ
                         | NE'''
    p[0] = p[1]

# relational-expression:
def p_relational_expression_1(p):
    'relational_expression : additive_expression'
    p[0] = p[1]

def p_relational_expression_2(p):
    'relational_expression : relational_expression relational_operator additive_expression'
    if (p[2] == '<'):
        p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.LT, p.lineno(1) + __start_line_no - 1)
    elif (p[2] == '>'):
        p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.GT, p.lineno(1) + __start_line_no - 1)
    elif (p[2] == '<='):
        p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.LE, p.lineno(1) + __start_line_no - 1)
    elif (p[2] == '>='):
        p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.GE, p.lineno(1) + __start_line_no - 1)
    else:
        err('orio.main.parsers.fparser internal error:  unknown relational operator')
        
# relational-operator
def p_relational_operator(p):
    '''relational_operator : LT
                           | GT
                           | LE
                           | GE'''
    p[0] = p[1]

# additive-expression
def p_additive_expression_1(p):
    'additive_expression : multiplicative_expression'
    p[0] = p[1]

def p_additive_expression_2(p):
    'additive_expression : additive_expression additive_operator multiplicative_expression'
    if (p[2] == '+'):
        p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.ADD, p.lineno(1) + __start_line_no - 1)
    elif (p[2] == '-'):
        p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.SUB, p.lineno(1) + __start_line_no - 1)
    else:
        err('orio.main.parsers.fparser internal error:  unknown additive operator' )

# additive-operator:
def p_additive_operator(p):
    '''additive_operator : PLUS
                         | MINUS'''
    p[0] = p[1]

# multiplicative-expression
def p_multiplicative_expression_1(p):
    'multiplicative_expression : unary_expression'
    p[0] = p[1]

def p_multiplicative_expression_2(p):
    'multiplicative_expression : multiplicative_expression multiplicative_operator unary_expression'
    if (p[2] == '*'):
        p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.MUL, p.lineno(1) + __start_line_no - 1)
    elif (p[2] == '/'):
        p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.DIV, p.lineno(1) + __start_line_no - 1)
    elif (p[2] == '%'):
        p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.MOD, p.lineno(1) + __start_line_no - 1)
    else:
        err('orio.main.parsers.fparser internal error:  unknown multiplicative operator')

# multiplicative-operator
def p_multiplicative_operator(p):
    '''multiplicative_operator : TIMES
                               | DIVIDE
                               | MOD'''
    p[0] = p[1]

# unary-expression:
def p_unary_expression_1(p):
    'unary_expression : postfix_expression'
    p[0] = p[1]

def p_unary_expression_2(p):
    'unary_expression : PLUSPLUS unary_expression'
    p[0] = ast.UnaryExp(p[2], ast.UnaryExp.PRE_INC, p.lineno(1) + __start_line_no - 1)

def p_unary_expression_3(p):
    'unary_expression : MINUSMINUS unary_expression'
    p[0] = ast.UnaryExp(p[2], ast.UnaryExp.PRE_DEC, p.lineno(1) + __start_line_no - 1)

def p_unary_expression_4(p):
    'unary_expression : unary_operator unary_expression'
    if p[1] == '+':
        p[0] = ast.UnaryExp(p[2], ast.UnaryExp.PLUS, p.lineno(1) + __start_line_no - 1)
    elif p[1] == '-':
        p[0] = ast.UnaryExp(p[2], ast.UnaryExp.MINUS, p.lineno(1) + __start_line_no - 1)
    elif p[1] == '!':
        p[0] = ast.UnaryExp(p[2], ast.UnaryExp.LNOT, p.lineno(1) + __start_line_no - 1)
    else:
        err('orio.main.parsers.fparser internal error:  unknown unary operator')

# unary-operator
def p_unary_operator(p):
    '''unary_operator : PLUS
                      | MINUS
                      | LNOT '''
    p[0] = p[1]

# postfix-expression
def p_postfix_expression_1(p):
    'postfix_expression : primary_expression'
    p[0] = p[1]

def p_postfix_expression_2(p):
    'postfix_expression : postfix_expression LBRACKET expression RBRACKET'
    p[0] = ast.ArrayRefExp(p[1], p[3], p.lineno(1) + __start_line_no - 1)

def p_postfix_expression_3(p):
    'postfix_expression : postfix_expression LPAREN argument_expression_list_opt RPAREN'
    p[0] = ast.FunCallExp(p[1], p[3], p.lineno(1) + __start_line_no - 1)

def p_postfix_expression_4(p):
    'postfix_expression : postfix_expression PLUSPLUS'
    p[0] = ast.UnaryExp(p[1], ast.UnaryExp.POST_INC, p.lineno(1) + __start_line_no - 1)

def p_postfix_expression_5(p):
    'postfix_expression : postfix_expression MINUSMINUS'
    p[0] = ast.UnaryExp(p[1], ast.UnaryExp.POST_DEC, p.lineno(1) + __start_line_no - 1)

# primary-expression
def p_primary_expression_1(p):
    'primary_expression : ID'
    p[0] = ast.IdentExp(p[1], p.lineno(1) + __start_line_no - 1)

def p_primary_expression_2(p):
    'primary_expression : ICONST'
    val = int(p[1])
    p[0] = ast.NumLitExp(val, ast.NumLitExp.INT, p.lineno(1) + __start_line_no - 1)

def p_primary_expression_3(p):
    'primary_expression : FCONST'
    val = float(p[1])
    p[0] = ast.NumLitExp(val, ast.NumLitExp.FLOAT, p.lineno(1) + __start_line_no - 1)

def p_primary_expression_4(p):
    'primary_expression : SCONST_D'
    p[0] = ast.StringLitExp(p[1], p.lineno(1) + __start_line_no - 1)

def p_primary_expression_5(p):
    '''primary_expression : LPAREN expression RPAREN'''
    p[0] = ast.ParenthExp(p[2], p.lineno(1) + __start_line_no - 1)

# argument-expression-list:
def p_argument_expression_list_opt_1(p):
    'argument_expression_list_opt :'
    p[0] = []
     
def p_argument_expression_list_opt_2(p):
    'argument_expression_list_opt : argument_expression_list'
    p[0] = p[1]

def p_argument_expression_list_1(p):
    'argument_expression_list : assignment_expression' 
    p[0] = [p[1]]

def p_argument_expression_list_2(p):
    'argument_expression_list : argument_expression_list COMMA assignment_expression' 
    p[1].append(p[3])
    p[0] = p[1]

# grammatical error
def p_error(p):
    err('orio.main.parsers.fparser:%s: grammatical error: "%s"' % ((p.lineno + __start_line_no - 1), p.value))
    sys.exit(1)

#------------------------------------------------

# Below is a grammar subset of Python expression
# py-expression
def p_py_expression_1(p):
    'py_expression : py_m_expression'
    p[0] = p[1]

def p_py_expression_2(p):
    'py_expression : py_conditional_expression'
    p[0] = p[1]

# py-expression-list
def p_py_expression_list_opt_1(p):
    'py_expression_list_opt : '
    p[0] = ''

def p_py_expression_list_opt_2(p):
    'py_expression_list_opt : py_expression_list'
    p[0] = p[1]

def p_py_expression_list_1(p):
    'py_expression_list : py_expression'
    p[0] = p[1]

def p_py_expression_list_2(p):
    'py_expression_list : py_expression_list COMMA py_expression'
    p[0] = p[1] + p[2] + p[3]

# py-conditional-expression
def p_py_conditional_expression(p):
    'py_conditional_expression : py_m_expression IF py_m_expression ELSE py_m_expression'
    p[0] = p[1] + ' ' + p[2] + ' ' + p[3] + ' ' + p[4] + ' ' + p[5]

# py-m-expression
def p_py_m_expression_1(p):
    'py_m_expression : py_u_expression'
    p[0] = p[1]

def p_py_m_expression_2(p):
    'py_m_expression : py_m_expression py_binary_operator py_u_expression'
    p[0] = p[1] + ' ' +  p[2] + ' ' + p[3]

# py-binary-operator
def p_py_binary_operator(p):
    '''py_binary_operator : PLUS
                          | MINUS
                          | TIMES
                          | DIVIDE
                          | MOD
                          | LT
                          | GT
                          | LE
                          | GE
                          | EQ
                          | NE
                          | AND
                          | OR'''
    p[0] = p[1]

# py-u-expression
def p_py_u_expression_1(p):
    'py_u_expression : py_primary'
    p[0] = p[1]

def p_py_u_expression_2(p):
    '''py_u_expression : PLUS py_u_expression
                       | MINUS py_u_expression
                       | NOT py_u_expression'''
    p[0] = p[1] + ' ' + p[2]

# py-primary
def p_py_primary(p):
    '''py_primary : py_atom
                  | py_subscription
                  | py_attribute_ref
                  | py_call
                  | py_list_display
                  | py_dict_display'''
    p[0] = p[1]

# py-subscription
def p_py_subscription(p):
    'py_subscription : py_primary LBRACKET py_expression_list RBRACKET'
    p[0] = p[1] + p[2] + p[3] + p[4]

# py-attribute-ref
def p_py_attribute_ref(p):
    'py_attribute_ref : py_primary PERIOD ID'
    p[0] = p[1] + p[2] + p[3]

# py-call
def p_py_call(p):
    'py_call : py_primary LPAREN py_expression_list_opt RPAREN'
    p[0] = p[1] + p[2] + p[3] + p[4]

# py-list-display
def p_py_list_display(p):
    'py_list_display : LBRACKET py_expression_list_opt RBRACKET'
    p[0] = p[1] + p[2] + p[3]

# py-dict-display
def p_py_dict_display(p):
    'py_dict_display : LBRACE py_key_datum_list_opt RBRACE'
    p[0] = p[1] + p[2] + p[3]

# py-key-datum-list
def p_py_key_datum_list_opt_1(p):
    'py_key_datum_list_opt : '
    p[0] = ''

def p_py_key_datum_list_opt_2(p):
    'py_key_datum_list_opt : py_key_datum_list'
    p[0] = p[1]

def p_py_key_datum_list_1(p):
    'py_key_datum_list : py_key_datum'
    p[0] = p[1]

def p_py_key_datum_list_2(p):
    'py_key_datum_list : py_key_datum_list COMMA py_key_datum'
    p[0] = p[1] + p[2] + p[3]

# py-key-datum
def p_py_key_datum(p):
    'py_key_datum : py_expression COLON py_expression'
    p[0] = p[1] + p[2] + p[3]

# py-atom
def p_py_atom_1(p):
    '''py_atom : ID
               | ICONST
               | FCONST
               | SCONST_D
               | SCONST_S'''
    p[0] = p[1]

def p_py_atom_2(p):
    'py_atom : LPAREN py_expression_list_opt RPAREN'
    p[0] = p[1] + p[2] + p[3]


#------------------------------------------------

def getParser(start_line_no):
    '''Create the parser for the annotations language'''

    # set the starting line number
    global __start_line_no
    __start_line_no = start_line_no

    # create the lexer and parser
    lexer = tool.ply.lex.lex()
    parser = tool.ply.yacc.yacc(method='LALR', debug=0)

    # return the parser
    return parser

# Compute column. 
#     input is the input text string
#     token is a token instance
def find_column(input,token):
    i = token.lexpos
    startline = input[:i].rfind('\n')
    endline = startline + input[startline+1:].find('\n') 
    line = input[startline+1:endline+1]
    while i > 0:
        if input[i] == '\n': break
        i -= 1
    column = (token.lexpos - i)
    return line, column

# Driver (regenerates parse table)
def setup_regen(debug = 1, outputdir='.'):
    global parser
    
    # Remove the old parse table
    parsetabfile = os.path.join(os.path.abspath(outputdir),'parsetab.py')
    try: os.remove(parsetabfile)
    except: pass

    parser = parse.ply.yacc.yacc(debug=debug, optimize=1, tabmodule='parsetab', write_tables=1, outputdir=os.path.abspath(outputdir))

    return parser


if __name__ == '__orio.main._':
    '''To regenerate the parse tables, invoke iparse.py with --regen as the last command-line
        option, for example:
            iparse.py somefile.sidl --regen
    '''
    #import visitor.printer
    
    DEBUGSTREAM = Devnull()

    if True or sys.argv[-1] == '--regen':
        del sys.argv[-1]
        DEBUGSTREAM = sys.stderr
        setup_regen(debug=0, outputdir=os.path.dirname(sys.argv[0]))
    else:
        setup()

    lex = lexer.FLexer()
    lex.build(optimize=1)                     # Build the lexer

    for i in range(1, len(sys.argv)):
        fname = sys.argv[i]
        debug("orio.main.parsers.fparser: About to parse %s" % fname, level=1)
        f = open(fname,"r")
        s = f.read()
        f.close()
        # debug("orio.main.parsers.fparser: Contents of %s: %s" % (fname, s))
        if s == '' or s.isspace(): sys.exit(0)
        if not s.endswith('\n'): 
            warn('orio.main.parser.fparser: file does not end with newline.')
            s += '\n'
        
        lex.reset(fname)
        ast = parser.parse(s, lexer=lex.lexer, debug=0)
        debug('orio.main.parsers.fparser: Successfully parsed %s' % fname, level=1)

        
        #printer = visitor.printer.Printer()
        #ast.accept(printer)


