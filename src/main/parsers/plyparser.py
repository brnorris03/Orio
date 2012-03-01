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
import orio.tool.ply.yacc as yacc
#import orio.main.parsers.flexer as lexer
#import orio.module.inline.sublexer as lexer
from orio.main.util.globals import *
import orio.module.inline.subroutine as subroutine
   
# Get the token map
tokens = lexer.SubroutineLexer.tokens
baseTypes = {}
filename = ''

# R201
def p_program(p):
    'program : program_unit_list'
    p[0] = p[1]

def p_program_unit_list(p):
    '''program_unit_list : program_unit
                        | program_unit_list program_unit
                        | empty
                        '''
    p[0] = processList(p,2)


# R202
def p_program_unit(p):
    '''program_unit : main_program
                    | module_definition
                    | subprogram_definition_list
                    '''
    p[0] = p[1]
    
def p_main_program(p):
    'main_program : PROGRAM optional_specification_part stuff_we_dont_care_about ENDPROGRAM ID'
    p[0] = p[4]   # TODO
    
def p_stuff_we_dont_care_about(p):
    '''stuff_we_dont_care_about : idlist
    '''
    p[0] = None
    
def p_module_definition(p):
    'module_definition : MODULE ID module_body ENDMODULE optional_name'
    p[0] = p[2]     # TODO
    
def p_module_body(p):
    '''module_body : optional_specification_part optional_module_subprogram_part
                    | stuff_we_dont_care_about
                    '''
    p[0] = p[1]
    
def p_optional_module_subprogram_part(p):
    '''optional_module_subprogram_part : module_subprogram_part 
                                    | empty
                                    '''
    p[0] = p[1]
    
def p_module_subprogram_part(p):
    'module_subprogram_part : CONTAINS module_subprogram module_subprogram_list'
    p[0] = [p[2]] + p[3]
    
def p_module_subprogram_list(p):
    '''module_subprogram_list : module_subprogram
                            | module_subprogram_list module_subprogram
                            | empty
                            '''
    p[0] = processList(p,2)

def p_module_subprogram(p):
    'module_subprogram : subprogram_definition'
    p[0] = p[1]
    
def p_optional_name(p):
    '''optional_name : ID
                    | empty
                    '''
    p[0] = p[1]

def p_optional_specification_part(p):
    '''optional_specification_part : specification_part 
                                | empty
                                '''
    p[0] = p[1]

def p_specification_part(p):
    'specification_part : use_stmt_part import_stmt_list implicit_part declaration_construct_list' 
    p[0] = (p[1],p[2],p[3],p[4])
    

def p_use_stmt_part(p):
    '''use_stmt_part : use_stmt
                    | use_stmt_part opt_semi use_stmt
                    | empty
                    '''
    p[0] = processList(p,3)
    
# TODO (placeholder for now)
def p_use_stmt(p):
    '''use_stmt : USE opt_module_nature optional_double_colon module_name rename_list 
                | USE opt_module_nature optional_double_colon module_name COMMA ONLY COLON optional_only_list
                '''
    p[0] = p[4]
    
def p_opt_module_nature(p):
    '''opt_module_nature : COMMA INTRINSIC
                        | COMMA NON_INTRINSIC
                        | empty'''
    if len(p) > 2: p[0] = p[2]
    else: p[0] = None
    
def p_module_name(p):
    'module_name : ID'
    p[0] = p[1]
    
def p_optional_only_list(p):
    '''optional_only_list : name_list
                          | empty'''
    # TODO: incomplete
    p[0] = p[1]
    
def p_rename_list(p):
    '''rename_list : rename
                            | rename_list rename
                            | empty
                            '''
    p[0] = processList(p,2)

def p_rename(p): 
    '''rename : ID EQ_GT ID'''
    p[0]= (p[1],p[3])
    
def p_import_stmt_list(p):
    '''import_stmt_list : import_stmt 
                        | import_stmt_list import_stmt 
                        | empty
                        '''
    p[0] = processList(p)      
    
def p_import_stmt(p):
    'import_stmt : IMPORT optional_import_list opt_semi'
    # TODO: ignore for now
    p[0] = None
    
def p_optional_import_list(p):
    '''optional_import_list : optional_double_colon name_list
                            | empty
                            '''
    if len(p) > 2: p[0] = p[2]
    else: p[0] = p[1]
    
def p_implicit_part(p):
    '''implicit_part : implicit_stmt 
                    | implicit_part implicit_stmt 
                    | empty
                    '''
    p[0] = processList(p,2)
    
def p_implicit_stmt(p):
    '''implicit_stmt : IMPLICITNONE opt_semi
                    | IMPLICIT implicit_spec_list opt_semi
                    '''
    if len(p) > 2:
        p[0] = p[2]
    else:
        p[0] = []
                    
def p_implicit_spec_list(p):
    'implicit_spec_list : stuff_we_dont_care_about'
    p[0] = p[1]
    
    
def p_declaration_construct_list(p):
    '''declaration_construct_list : declaration_construct
                                | declaration_construct_list declaration_construct
                                | stuff_we_dont_care_about
                                | empty
                                '''
    p[0] = processList(p,2)
    
def p_interface_block(p):
    'interface_block : interface_stmt interface_specification_list ENDINTERFACE'
    p[0] = p[2]
    
def p_interface_stmt(p):
    '''interface_stmt : INTERFACE
                       | INTERFACE generic_spec
                       '''
    if len(p) > 2: p[0] = p[2]
    else: p[0] = None
    
def p_generic_spec(p):
    '''generic_spec : generic_name'''
    # TODO
    p[0] = p[1]

def p_generic_name(p):
    'generic_name : ID'
    p[0] = p[1]
    
def p_interface_specification_list(p):
    '''interface_specification_list : interface_specification 
                                    | interface_specification_list interface_specification
                                    | empty
                                    '''
    p[0] = processList(p,2)

def p_interface_specification(p):
    '''interface_specification : interface_body
                                | procedure_stmt
                                '''
    p[0] = p[1]
    
def p_procedure_stmt(p):
    'procedure_stmt : optional_module PROCEDURE name_list'
    p[0] = p[3]
    
def p_interface_body(p):
    'interface_body : subprogram_declaration'
    p[0] = p[1]
    
    
# ==================================================================
# Subprograms 

def p_subprogram_declaration(p):
    '''subprogram_declaration : subroutine_declaration
                            | function_declaration
                            '''
    p[0] = p[1]
    
def p_subroutine_declaration(p):
    'subroutine_declaration : subroutine_header optional_specification_part varref_list ENDSUBROUTINE optional_name'
    p[0] = subroutine.SubroutineDeclaration(p[1], p[3], p[4])
    #print p[0]

def p_function_declaration(p):
    'function_declaration : function_header optional_specification_part varref_list ENDFUNCTION optional_name'
    p[0] = subroutine.SubroutineDeclaration(p[1], p[3], p[4],function=True)
    #print p[0]
    
def p_subprogram_definition_list(p):
    '''subprogram_definition_list : subprogram_definition
                                | subprogram_definition_list subprogram_definition
                                | empty
                                '''
    p[0] = processList(p,2)
    
def p_subprogram_definition(p):
    '''subprogram_definition : subroutine_definition
                            | function_definition
                            '''
    p[0] = p[1]
    
def p_subroutine_definition(p):
    '''subroutine_definition : subroutine_header optional_specification_part varref_list ENDSUBROUTINE optional_name'''
    p[0] = subroutine.SubroutineDefinition(p[1], p[3], p[4])

    #print p[0]
    
def p_subroutine_header(p):
    '''subroutine_header : subroutine_stmt LPAREN argument_list RPAREN
                        | subroutine_stmt 
                        '''
    if len(p) > 4:
        p[0] = (p[1][0],p[4])
    else:
        p[0] = (p[1][0],[])

def p_subroutine_stmt(p):
    '''subroutine_stmt : SUBROUTINE subroutine_name
                        | prefix_spec_list SUBROUTINE subroutine_name
                        '''
    # TODO: incomplete
    if len(p) > 3:
        p[0] = (p[3],p[0])
    else:
        p[0] = (p[2],None)
    
def p_prefix_spec_list(p):
    '''prefix_spec_list : prefix_spec
                        | prefix_spec_list prefix_spec
                        '''
    p[0] = processList(p,2)
    
def p_prefix_spec(p):
    '''prefix_spec : RECURSIVE
                    | PURE
                    | ELEMENTAL
                    | declaration_type_spec
                    '''
    p[0] = p[1]

def p_declaration_type_spec(p):
    '''declaration_type_spec : intrinsic_type_spec
                            | TYPE LPAREN derived_type_spec RPAREN
                            | CLASS LPAREN derived_type_spec RPAREN
                            | CLASS LPAREN ASTERISK RPAREN
                            '''
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = p[2]

def p_intrinsic_type_spec(p):
    '''intrinsic_type_spec : INTEGER opt_kind_selector 
                            | REAL opt_kind_selector
                            | DOUBLE PRECISION
                            | COMPLEX opt_kind_selector
                            | CHARACTER opt_char_selector
                            | LOGICAL opt_kind_selector
                            '''
    p[0] = (p[1],p[2])

def p_opt_kind_selector(p):
    '''opt_kind_selector : kind_selector 
                        | empty
                        '''
    p[0] = p[1]

def p_opt_char_selector(p):
    '''opt_char_selector : empty'''
    # TODO    
    p[0] = p[1]
    
def p_kind_selector(p):
    'kind_selector : empty'
    # TODO
    'kind_selector : optional_kind scalar_int_initialization_expr'
    p[0] = p[1]
    
def p_derived_type_spec(p):
    'derived_type_spec : ID'
    # TODO
    p[0] = p[1]
    
def p_function_definition(p):
    'function_definition : function_header optional_specification_part varref_list ENDFUNCTION optional_name'
    p[0] = subroutine.SubroutineDefinition(p[1], p[3], p[4],function=True)
    #print p[0]

def p_function_header(p):
    '''function_header : function_stmt LPAREN argument_list RPAREN
                    | function_stmt 
                    '''
    if len(p) > 4:
        p[0] = (p[1][0],p[4])
    else:
        p[0] = (p[1][0],[])


def p_function_stmt(p):
    '''function_stmt : FUNCTION subroutine_name
                    | prefix_spec_list FUNCTION subroutine_name
                    '''
    if len(p) > 3:
        p[0] = (p[3],p[0])
    else:
        p[0] = (p[2],None)
        
def p_subroutine_name(p):
    'subroutine_name : ID'
    p[0] = p[1]
    
def p_argument_list(p):
    '''argument_list : ID
                    | argument_list COMMA ID
                    | empty
                    '''
    p[0] = processList(p,3)

def p_varref_list(p):
    '''varref_list : ID
                    | varref_list ID
                    | empty
                    '''
    if len(p) > 2:
        p[1].append((p[2],p.lexspan(2)))
        p[0] = p[1]
    elif p[1]:
        p[0] = [(p[1],p.lexspan(1))]
    else:
        p[0] = []
      

# ==============================================
# Declarations

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
    p[0] = p[1]
    
# ---------------------------------------------------- ------

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
    p[0] = p[1]
    
# ----------------------------------------------------- ----
    
# R429
def p_derived_type_def(p):
    'derived_type_def : derived_type_stmt opt_type_param_def_stmt_seq \
                                    opt_private_or_sequence_seq \
                                    opt_component_part \
                                    opt_type_bound_procedure_part \
                        end_type_stmt'
    p[0] = None
    
# ------------------ end derived_type_def -------------------
    
# ===========================================================
# =========== Specification statements section ==============
# R518
def p_access_stmt(p):
    'access_stmt : access_spec optional_access_id_list opt_semi'
    p[0] = ('access_stmt', p[2])
    
def p_optional_access_id_list(p):
    '''optional_access_id_list : optional_double_colon access_id_list
                                | empty
                                '''
    if len(p) > 2: p[0] = p[2]
    else: p[0] = None

def p_access_id_list(p):
    'access_id_list : nonempty_name_list'
    p[0] = p[1]
    
# R519
def p_access_id(p):
    '''access_id : use_name 
                | generic_spec
                '''
    p[0] = p[1]
    
# C549
def p_use_name(p):
    'use_name : ID'
    #C549 (R519) Each use-name shall be the name of a named variable, procedure, derived type, 
    # named constant, or namelist group.
    p[0] = p[1]
    
# ----------------------------- end access_stmt ---------------------------------------

# R520 
def p_allocatable_stmt(p):
    'allocatable_stmt : ALLOCATABLE optional_double_colon object_name optional_deferred_shape_spec_list optional_obj_def_shape_list opt_semi'
    p[0] = ('allocatable_stmt', [(p[3],p[4])] + p[5])
    
def p_optional_obj_def_shape_list(p):
    '''optional_obj_def_shape_list : COMMA object_name optional_deferred_shape_spec_list
                                    | optional_obj_def_shape_list COMMA object_name optional_deferred_shape_spec_list
                                    | empty
                                    '''
    if len(p) > 4:
        p[0] = p[1] + [(p[3],p[4])]
    elif len(p) == 4:
        p[0] = [(p[2],p[3])]        
    else: p[0] = []
        
def p_optional_deferred_shape_spec_list(p):
    '''optional_deferred_shape_spec_list : LPAREN deferred_shape_spec_list RPAREN
                                        | empty
                                        '''
    if len(p) > 2: 
        p[0] = p[2]
    else: p[0] = []
    
def p_deferred_shape_spec_list(p):
    '''deferred_shape_spec_list : deferred_shape
                                | deferred_shape_list COMMA deferred_shape
                                | empty
                                '''
    p[0] = processList(p,3)
    
def p_deferred_shape(p):
    'deferred_shape : COLON'
    p[0] = p[1]
    
def p_object_name(p):
    'object_name : ID'
    p[0] = p[1]
    
# ------------------------ end allocatable_stmt ----------------------------
        
# R521 
def p_asynchronous_stmt(p):
    'asynchronous_stmt : ASYNCHRONOUS optional_double_colon object_name_list opt_semi'
    p[0] = p[3]

def p_object_name_list(p):
    '''object_name_list : object
                        | object_name_list COMMA object
                        '''
    p[0] = processList(p,3)
    
# ---------------------- end asynchronous_stmt -----------------------------

# R522
def p_bind_stmt(p):
    'bind_stmt : language_binding_spec optional_double_colon bind_entity_list opt_semi'
    p[0] = ('bind_stmt', p[1],p[3])

# R509
def p_language_binding_spec(p):
    '''language_binding_spec : BIND LPAREN ID RPAREN
                            | BIND LPAREN ID COMMA NAME EQUALS scalar_char_initialization_expr
                            '''
    if p[3].upper() != 'C': 
        # TODO: error
        pass
    
    if len(p) > 5: p[0] = p[7]
    else: p[0] = None
    
                            
def p_bind_entity_list(p):
    '''bind_entity_list : bind_entity
                        | bind_entity_list COMMA bind_entity
                        '''
    p[0] = processList(p,3)

# R523
def p_bind_entity(p):
    '''bind_entity : entity_name
                    | DIVIDE common_block_name DIVIDE
                    '''
    if len(p) > 2: p[0] = p[2]
    else: p[0] = p[1]
    
    
    
# --------------------- end bind_stmt -------------------------------

# R557
def p_common_stmt(p):
    'common_stmt : COMMON optional_common_block_name common_block_object_list opt_common_block_list opt_semi'
    p[0] = ('common_stmt', [(p[2],p[3])] + p[4])
    
def p_opt_common_block_list(p):
    '''opt_common_block_list : COMMA optional_common_block_name common_block_object_list
                            | opt_common_block_list COMMA optional_common_block_name common_block_object_list
                            | empty
                            '''
    if len(p) > 4:
        p[0] = p[1] + [(p[3],p[4])]
    elif len(p) == 4:
        p[0] = [(p[2],p[3])]        
    else: p[0] = []
                            
def p_optional_common_block_name(p):
    '''optional_common_block_name : ID
                                | empty
                                '''
    p[0] = p[1]
    
def p_common_block_object_list(p):
    '''common_block_object_list : common_block_object
                                | common_block_object_list COMMA common_block_object
                                '''
    p[0] = processList(p,3)
    
def p_common_block_object(p):
    '''common_block_object : variable_name optional_explicit_shape_spec_list
                            | proc_pointer_name
                            '''
    if len(p) > 2: p[0] = (p[1],p[2])
    else: p[0] = (p[1],None)
    
def p_variable_name(p):
    'variable_name : name'
    p[0] = p[1]

def p_optional_explicit_shape_spec_list(p):
    '''optional_explicit_shape_spec_list : LPAREN explicit_shape_spec_list RPAREN
                                        | empty
                                        '''
    if len(p) > 2: p[0] = p[2]
    else: p[0] = []
    
def p_explicit_shape_spec_list(p):
    '''explicit_shape_spec_list : explicit_shape_spec
                                | explicit_shape_spec_list COMMA explicit_shape_spec
                                '''
    p[0] = processList(p,3)
    
# R511
def p_explicit_shape_spec(p):
    'explicit_shape_spec : optional_lower_bound upper_bound'
    p[0] = (p[1],p[2])
    
def p_optional_lower_bound(p):
    '''optional_lower_bound : specification_expr COLON
                            | empty
                            '''
    p[0] = p[1]
    
# R513
def p_upper_bound(p):
    'upper_bound : specification_expr'
    p[0] = p[1]

# R512
def p_lower_bound(p):
    'lower_bound : specification_expr'
    p[0] = p[1]
    
def p_specification_expr(p):
    'specification_expr : scalar_int_expr'
    p[0] = p[1]
    
def p_proc_pointer_name(p):
    'proc_pointer_name : ID'
    p[0] = p[1]
    
# --------------------- end common_block_stmt -------------------------------------

# R524
def p_data_stmt(p):
    'data_stmt : DATA data_stmt_set optional_data_stmt_set_list opt_semi'
    p[0] = ('data_stmt',[p[2]] + p[3])
    
def p_optional_data_stmt_set_list(p):
    '''optional_data_stmt_set_list : optional_comma data_stmt_set_list
                                    | empty
                                    '''
    if p[1] : p[0] = p[1]
    else: p[0] = []
    
def p_data_stmt_set_list(p):
    '''data_stmt_set_list : data_stmt_set
                        | data_stmt_set_list optional_comma data_stmt_set
                        '''
    p[0] = processList(p,3)
    

# R525
def p_data_stmt_set(p):
    'data_stmt_set : data_stmt_object_list DIVIDE data_stmt_value_list DIVIDE'
    p[0] = (p[1],p[3])
    
def p_data_stmt_object_list(p):
    '''data_stmt_object_list : data_stmt_object
                            | data_stmt_object_list COMMA dat_stmt_object
                            '''
    p[0] = processList(p,3)

# R526
def p_data_stmt_object(p):
    '''data_stmt_object : variable 
                        | data_implied_do
                        '''
    p[0] = p[1]
    
def d_ata_stmt_value_list(p):
    '''data_stmt_value_list : data_stmt_value
                            | data_stmt_value_list COMMA data_stmt_value
                            '''
    p[0] = processList(p,3)

# R530
def p_data_stmt_value(p):
    'data_stmt_value : optional_data_stmt_repeat data_stmt_constant'
    # TODO
    p[0] = (p[1],p[2])
   
# R531
def p_data_stmt_repeat(p):
    '''data_stmt_repeat : scalar_int_constant
                        | scalar_int_constant_subobject
                        '''
    p[0] = p[1]

# R527  
def p_data_implied_do(p):
    'data_implied_do : LPAREN data_i_do_object_list COMMA data_i_do_variable EQUALS scalar_int_expr COMMA scalar_int_expr optional_scalar_int_expr'
    # TODO
    p[0] = (p[2], p[4], p[6], p[8], p[9])

def p_optional_scalar_int_expr(p):
    'optional_scalar_int_expr : COMMA scalar_int_expr'
    p[0] = p[1]
    
# R528
def p_data_i_do_object(p):
    '''data_i_do_object : array_element
                        | scalar_structure_component
                        | data_implied_do
                        '''
    p[0] = p[1]

# R529
def p_data_i_do_variable(p):
    'data_i_do_variable : scalar_int_variable'
    p[0] = p[1]


def p_scalar_structure_component(p):
    'scalar_structure_component : variable'
    p[0] = p[1]
    

# ----------------------------- end data_stmt ---------------------------------------

# R535
def p_dimension_stmt(p):
    'dimension_stmt : DIMENSION optional_double_colon array_spec_list opt_semi'
    p[0] = ('dimension_stmt',p[3])

def p_array_spec_list(p):
    '''array_spec_list : array_name LPAREN array_spec RPAREN
                        | array_spec_list COMMA array_name LPAREN array_spec RPAREN
                        '''
    if len(p) > 5:
        p[0] = p[1] + [(p[3],p[5])]
    else:
        p[0] = [(p[1],p[3])]

# R510
def p_array_spec(p):
    '''array_spec : explicit_shape_spec_list
                | assumed_shape_spec_list
                | deferred_shape_spec_list
                | assumed_size_spec
                '''
    p[0] = p[1]
    
# R514
def p_assumed_shape_spec(p):
    '''assumed_shape_spec : COLON
                        | lower_bound COLON
                        '''
    if len(p) > 2: p[0] = p[1]
    else: p[0] = None
    
# R515
def p_deferred_shape_spec(p):
    'deferred_shape_spec : COLON'
    p[0] = ('deferred')
    
# R516
def p_assumed_size_spec(p):
    'assumed_size_spec : opional_explicit_shape_spec_list_comma optional_lower_bound MULT'
    p[0] = ('assumed',p[1],p[2])
    
def p_optional_explicit_spec_list_comma(p):
    '''optional_explicit_shape_spec_list_comma : explicit_shape_spec_list COMMA
                                                | empty
                                                '''
    if len(p) > 2: p[0] = p[1]
    else: p[0] = []
    
# ---------------------------- end dimension_stmt ------------------------------------
    
# R554
def p_equivalence_stmt(p):
    'equivalence_stmt : EQUIVALENCE equivalence_set_list opt_semi'
    p[0] = ('equivalence_stmt',p[2])

def p_equivalence_set_list(p):
    '''equivalence_set_list : equivalence_set 
                            | equivalence_set_list COMMA equivalence_set
                            '''
    p[0] = processList(p,3)

# R555
def p_equivalence_set(p):
    'equivalence_set : LPAREN equivalence_object COMMA equivalence_object_list RPAREN'
    p[0] = (p[2],p[4])

def p_equivalance_object_list(p):
    '''equivalence_object_list : equivalence_object 
                                | equivalence_object_list COMMA equivalence_object
                                '''
    p[0] = processList(p,3)
    
# R556
def p_equivalence_object(p):
    '''equivalence_object : variable_name
                        | array_element
                        | substring
                        '''
    p[0] = p[1]

# -------------------------- end equivalence_stmt ----------------------------------

# R1210
def p_external_stmt(p):
    'external_stmt : EXTERNAL optional_double_colon external_name_list opt_semi'
    p[0] = ('external_stmt', p[3])

def p_external_name_list(p):
    '''external_name_list : external_name 
                        | external_name_list COMMA external_name
                        '''
    p[0] = processList(p,3)
    
def p_external_name(p):
    'external_name : name'
    # Each external name shall be the name of an external procedure, a dummy argument, 
    # a procedure pointer or a block data program unit.
    p[0] = p[1]
    
# ------------------------- end external_stmt ----------------------------------------

# R536
def p_intent_stmt(p):
    'intent_stmt : INTENT LPAREN intent_spec RPAREN optional_double_colon dummy_arg_name_list opt_semi'
    p[0] = ('intent_stmt',p[3],p[6])
    
def p_intent_spec(p):
    '''intent_spec : IN
                    | OUT
                    | INOUT
                    '''
    p[0] = p[1]
    
def p_dummy_arg_name_list(p):
    '''dummy_arg_name_list : dummy_arg_name 
                            | dummy_arg_name_list COMMA dummy_arg_name
                            '''
    p[0] = processList(p,3)
    
def p_dummy_arg_name(p):
    'dummy_arg_name : name'
    p[0] = p[1]
    
# ------------------------ end intent_stmt  ----------------------------------------------

# R1216
def p_intrinsic_stmt(p):
    'intrinsic_stmt : INTRINSIC optional_double_colon intrinsic_procedure_name_list opt_semi'
    p[0] = ('intrinsic_stmt',p[3])
    
def p_intrinsic_procedure_name_list(p):
    '''intrinsic_procedure_name_list : intrinsic_procedure_name
                                    | intrinsic_procedure_name_list COMMA intrinsic_procedure
                                    '''
    p[0] = processList(p,3)

def p_intrinsic_procedure_name(p):
    'intrinsic_procedure_name : name'
    p[0] = p[1]
    
# ------------------------- end intrinsic_stmt ---------------------------------------------

# R552
def p_namelist_stmt(p):
    'namelist_stmt : NAMELIST namelist_groups opt_semi'
    p[0] = ('namelist_stmt', p[2])
    
def p_namelist_groups(p): 
    '''namelist_groups : DIVIDE namelist_group_name DIVIDE namelist_group_object_list
                        | namelist_groups optional_comma DIVIDE namelist_group_name DIVIDE namelist_group_object_list
                        '''
    if len(p) > 5:
        p[0] = p[1] + [(p[3],p[5])]
    else:
        p[0] = [(p[2],p[4])]
        
def p_namelist_group_object(p):
    'namelist_group_object : variable_name'
    p[0] = p[1]
    
def p_namelist_group_name(p):
    'namelist_group_name : name'
    # C573 (R552) The namelist-group-name shall not be a name made accessible by use association.
    p[0] = p[1]

# ---------------------- end namelist_stmt -----------------------------------------------

# R537 
def p_optional_stmt(p):
    'optional_stmt : OPTIONAL optional_double_colon dummy_arg_name_list opt_semi'
    p[0] = ('optional_stmt', p[3])
    
# -------------------- end optional_stmt ------------------------------------------------

# R540 
def p_pointer_stmt(p):
    'pointer_stmt : POINTER optional_double_colon pointer_decl_list opt_semi'
    p[0] = ('pointer_stmt', p[3])
    
def p_pointer_decl_list(p):
    '''pointer_decl_list : pointer_decl
                        | pointer_decl_list COMMA pointer_decl
                        '''
    p[0] = processList(p,3)

# R541
def p_pointer_decl(p):
    '''pointer_decl : object_name optional_deferred_shape_spec_list
                    | proc_entity_name
                    '''
    if len(p) > 2:
        p[0] = (p[1],p[2])
    else:
        p[0] = (p[1],None)
        
def p_proc_entity_name(p):
    'proc_entity_name : name'
    p[0] = p[1]
      
# -------------------- end pointer_stmt --------------------------------------------------

# R542
def p_protected_stmt(p):
    'protected_stmt : PROTECTED optional_double_colon entity_name_list opt_semi'
    p[0] = ('protected_stmt', p[3])
    
def p_entity_name_list(p):
    '''entity_name_list : entity_name
                        | entity_name_list COMMA entity_name
                        '''
    p[0] = processList(p,3)
    
def p_entity_name(p):
    'entity_name : name'
    p[0] = p[1]
    
# ------------------- end protected_stmt ---------------------------------------------

# R543
def p_save_stmt(p):
    'save_stmt : SAVE optional_double_colon saved_entity_list opt_semi'
    p[0] = ('save_stmt', p[3])
    
def p_saved_entity_list(p):
    '''saved_entity_list : saved_entity
                        | saved_entity_list COMMA saved_entity
                        '''
    p[0] = processList(p,3)
    
# R544
def p_saved_entity(p):
    '''saved_entity : object_name
                    | proc_pointer_name
                    | DIVIDE common_block_name DIVIDE
                    '''
    if len(p) < 4:
        p[0] = p[1]
    else:
        p[0] = p[2]
        
# ---------------------- end save_stmt -----------------------------------------

# R546
def p_target_stmt(p):
    'target_stmt : TARGET optional_double_colon array_spec_list opt_semi'
    p[0] = ('target_stmt', p[3])
    
# --------------------- end target_stmt ----------------------------------------

# R547
def p_value_stmt(p):
    'value_stmt : VALUE optional_double_colon dummy_arg_name_list opt_semi'
    p[0] = ('value_stmt', p[3])
    
# -------------------- end value_stmt -----------------------------------------

# R548
def p_volatile_stmt(p):
    'volatile_stmt : VOLATILE optional_double_colon object_name_list opt_semi'
    p[0] = ('volatile_stmt', p[3])
    
# ------------------- end volatile_stmt --------------------------------------
        
# ==============================================


# Variables and constants
# R 601
def p_variable(p):
    'variable : designator'
    p[0] = p[1]

def p_name(p):
    'name : ID'
    p[0] = p[1]
    
def p_scalar_variable_name(p):
    'scalar_variable_name : ID'
    p[0] = p[1]
    
    
def p_scalar_int_variable(p):
    'scalar_int_variable : ID'
    # restricted to integer
    p[0] = p[1]
    
def p_scalar_int_constant(p):
    'scalar_int_constant : ID'
    # TODO: fix to restrict to int
    p[0] = p[1]

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
                        | 
                        '''
        

# ================================================
# Arrays, strings, data references

# R609 
def p_substring(p):
    'substring : parent_string LPAREN substring_range RPAREN'
    p[0] = (p[1], p[3])
    
# R610
def p_parent_string(p):
    '''parent_string : scalar_variable_name
                    | array_element
                    | scalar_structure_component
                    | scalar_constant
                    '''
    p[0] = p[1]
    
# R611
def p_substring_range(p):
    'substring_range : scalar_int_expr COLON scalar_int_expr'
    p[0] = (p[1],p[3])

# R616
def p_array_element(p):
    'array_element : data_ref'
    p[0] = p[1]
    
# R617
def p_array_section(p):
    'array_section : data_ref optional_substring_range_in_paren'
    p[0] = (p[1],p[2])
    

    
# R612
def p_data_ref(p):
    '''data_ref : part_ref
                | data_ref MOD part_ref
                '''
    # part-ref [ % part-ref ] ...
    p[0] = processList(p,3)    
     
# R613
def p_part_ref(p):
    'part_ref : part_name optional_section_subscript_list_in_paren'
    p[0] = (p[1],p[2])
    
def p_optional_section_subscript_list_in_paren(p):
    '''optional_section_subscript_list_in_paren : LPAREN section_subscript_list RPAREN
                                                | empty
                                                '''
    if len(p) > 2: p[0] = p[2]
    else: p[0] = []
    
def section_subscript_list(p):
    '''section_subscript_list : section_subscript
                            | section_subscript_list COMMA section_subscript
                            '''
    p[0] = processList(p,3)
    
# R618
    


# ==============================================
# Expressions

# R729
def p_scalar_int_expr(p):
    'scalar_int_expr : expr'
    # Restricted expression: each operation is intrinsic and each primary is a constant or subobject of a constant (C710)
    p[0] = p[1]

# R730
def p_initialization_expr(p):
    'initialization_expr : expr'
    p[0] = p[1]

# R731 
def p_char_initialization_expr(p):
    'char_initialization_expr : char_expr'
    p[0] = p[1]

# R725
def p_char_expr(p):
    'char_expr : expr'
    # of type character
    p[0] = p[1]
    
# R726
def p_default_char_expr(p):
    'default_char_expr : expr'
    p[0] = p[1]

# R732 
def p_int_initialization_expr(p):
    'int_initialization_expr : int_expr'
    p[0] = p[1]

# R727
def p_int_expr(p):
    'int_expr : expr'
    # of type integer
    p[0] = p[1]
    
# R733 
def p_logical_initialization_expr(p):
    'logical_initialization_expr : logical_expr'
    p[0] = p[1]

# R724
def p_logical_expr(p):
    'logical_expr : expr'
    # of type logical
    p[0] = p[1]

# R728
def p_numeric_expr(p):
    'numeric_expr : expr'
    # numeric_expr shall be of type integer, real or complex
    p[0] = p[1]

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
    p[0] = p[1]

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



# ==============================================
# Miscellaneous
# Lists of identifiers outside subroutines  
def p_idlist(p):
    '''idlist : ID
            | idlist ID
            | empty
            '''
    p[0] = []

def p_nonempty_name_list(p):
    '''name_list : ID
                | name_list COMMA ID
                '''
    p[0] = processList(p,3)

def p_name_list(p):
    '''name_list : nonempty_name_list
                | empty
                '''
    p[0] = p[1]
    
def p_opt_semi(p):
    '''opt_semi : SEMICOLON
                | empty
                '''
    p[0] = p[1]
    
def p_optional_comma(p):
    '''optional_comma : COMMA
                    | empty
                    '''
    p[0] = p[1]

def p_optional_double_colon(p):
    '''optional_double_colon : COLON_COLON
                            | empty
                            '''
    p[0] = p[1]
    
def p_optional_module(p):
    '''optional_module : MODULE 
                        | empty
                        '''
    p[0] = p[1]
    
def p_empty(p):
    'empty : '
    p[0] = None
    
def p_error(t):
    global filename
    if t:
        line,col = find_column(t.lexer.lexdata,t)
        pos = (col-1)*' '
        err("[orio.module.inline.fparser] %s: unexpected symbol '%s' at line %s, column %s:\n\t%s\n\t%s^" \
            % (filename, t.value, t.lexer.lineno, col, line, pos))
    else:
        err("[orio.module.inline.fparser] internal error, please email source code to norris@mcs.anl.gov")
    
# Generic token list processing
def processList(p,tokenindex=2):
    if len(p) > 2:
        p[0] = p[1] + [p[tokenindex]]
    elif len(p) > tokenindex and p[tokenindex]:
        p[0] = [p[tokenindex]]
    else:
        p[0] = []
    return p[0]

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

    parser = yacc.yacc(debug=debug, optimize=1, tabmodule='parsetab', write_tables=1, outputdir=os.path.abspath(outputdir))

    return parser

# Driver (does not regenerate parse table)
def setup(debug = 0, outputdir='.'):
    global parser

    parser = yacc.yacc(debug = debug, optimize=1, write_tables=0)
    return parser

def getParser(start_line_no):
    '''Create the parser for the annotations language'''

    # set the starting line number
    global __start_line_no
    __start_line_no = start_line_no

    # create the lexer and parser
    parser = yacc.yacc(method='LALR', debug=0)

    # return the parser
    return parser

def getFileName(lexer):
    return lexer.getFileName()

if __name__ == '__main__':
    '''To regenerate the parse tables, invoke iparse.py with --regen as the last command-line
        option, for example:
            iparse.py somefile.sidl --regen
    '''
    #import visitor.printer
    #import visitor.commentsmerger
    import sys
    
    #import profile
    # Build the grammar
    #profile.run("yacc.yacc()")
    
    if sys.argv[-1] == '--regen':
        del sys.argv[-1]
        setup_regen(debug=0, outputdir=os.path.dirname(sys.argv[0]))
    else:
        setup()

    lex = lexer.SubroutineLexer()
    lex.build(optimize=1)                     # Build the lexer

    
    for i in range(1, len(sys.argv)):
        debug("[inliner parse] About to parse %s" % sys.argv[i])
        f = os.popen('gfortran -E %s' % sys.argv[i])
        #f = open(sys.argv[i],"r")
        s = f.read()
        f.close()
        # print "Contents of %s: %s" % (sys.argv[i], s)
        if s == '' or s.isspace(): sys.exit(0)
        
        #print 'Comments: \n', comments
        
        filename = sys.argv[i]
        lex.reset()
        sub = parser.parse(s, lexer=lex.lexer, debug=1)
        debug('[inliner parse] Successfully parsed %s' % sys.argv[i])
        parser.restart()
        
        #printer = visitor.printer.Printer()
        #ast.accept(printer)
