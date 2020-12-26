# 
# The evaluator for the TSpec (Tuning Specifier) language
#

import io, sys, tokenize
import builtins, itertools, string
from orio.main.util.globals import *

#--------------------------------------------------------------------------------

class TSpecEvaluator:
    '''The evaluator for the TSpec language'''

    def __init__(self):
        '''To instantiate a TSpec evaluator'''
        pass
    
    #----------------------------------------------------------------------------

    def __extractVars(self, code):
        '''Return all variables that are present in the given code'''

        # tokenize the given expression code
        gtoks = tokenize.generate_tokens(io.StringIO(code).readline)

        # iterate over each token and replace any matching token with its corresponding value
        vnames = []
        for toknum, tokval, _, _, _ in gtoks:
            if toknum == tokenize.NAME:
                vnames.append(tokval)

        # return all the found variable names
        return vnames

    #----------------------------------------------------------------------------

    def __substituteVars(self, code, env):
        '''
        Expand any variables that exist in the given environment to their corresponding values
        '''

        # tokenize the given expression code
        gtoks = tokenize.generate_tokens(io.StringIO(code).readline)

        # iterate over each token and replace any matching token with its corresponding value
        tokens = []
        for toknum, tokval, _, _, _ in gtoks:
            if toknum == tokenize.NAME and tokval in env:
                ntoks = tokenize.generate_tokens(io.StringIO(str(env[tokval])).readline)
                tokens.extend(ntoks)
            else:
                tokens.append((toknum, tokval))

        # convert the tokens back to a string
        code = tokenize.untokenize(tokens)

        # remove all the leading and trailing spaces
        code = code.strip()

        # return the modified string
        return code

    #----------------------------------------------------------------------------

    def __evalArg(self, stmt, env, name_space):
        '''To evaluate the given "let" statement'''

        # unpack the statement
        keyw, line_no, (id_name, id_line_no), (rhs, rhs_line_no) = stmt

        # check for illegal variable references
        for vname in self.__extractVars(rhs):
            try:
                eval(vname, env)
            except:
                err('orio.main.tspec.eval: %s: invalid reference: "%s"' % (rhs_line_no, vname), doexit=True)

        # evaluate the RHS expression
        try:
            rhs_val = eval(rhs, env)
        except Exception as e:
            err('orio.main.tspec.eval: %s: failed to evaluate the RHS expression\n --> %s: %s' % (rhs_line_no, e.__class__.__name__, e))

        # return the evaluated statement
        return (keyw, line_no, (id_name, id_line_no), (rhs_val, rhs_line_no))

    #----------------------------------------------------------------------------

    def __evalConstraint(self, stmt, env, name_space):
        '''To evaluate the given "constraint" statement'''

        # unpack the statement
        keyw, line_no, (id_name, id_line_no), (rhs, rhs_line_no) = stmt
        
        # substitute all environment variables with their corresponding values
        rhs = self.__substituteVars(rhs, env)

        # return the evaluated statement
        return (keyw, line_no, (id_name, id_line_no), (rhs, rhs_line_no))

    #----------------------------------------------------------------------------

    def __evalDecl(self, stmt, env, name_space):
        '''To evaluate the given "decl" statement'''

        # unpack the statement
        keyw, line_no, (id_name, id_line_no), type_seq, dim_exp_seq, (rhs, rhs_line_no) = stmt
        
        # check for types
        type_names = []
        for t, l in type_seq:
            if t in type_names:
                err('orio.main.tspec.eval: %s: repeated type name: "%s"' % (l, t))
            type_names.append(t)

        # substitute all environment variables in each dimension expression
        n_dim_exp_seq = []
        for e, l in dim_exp_seq:
            e = self.__substituteVars(e, env)
            n_dim_exp_seq.append((e, l))
        dim_exp_seq = n_dim_exp_seq

        # substitute all environment variables in the RHS expression
        if rhs and rhs != 'random':
            rhs = self.__substituteVars(rhs, env)

        # return the evaluated statement
        return (keyw, line_no, (id_name, id_line_no), type_seq, dim_exp_seq, (rhs, rhs_line_no))

    #----------------------------------------------------------------------------

    def __evalDef(self, stmt, env, name_space):
        '''To evaluate the given "def" statement'''

        # copy the environment and name space
        env = env.copy()
        name_space = name_space.copy()

        # unpack the statement
        keyw, line_no, (id_name, id_line_no), stmt_seq = stmt
        
        # evaluate each statement in the definition statement body
        stmt_seq = self.__evaluate(stmt_seq, env, name_space)
        
        # return the evaluated statement
        return (keyw, line_no, (id_name, id_line_no), stmt_seq)

    #----------------------------------------------------------------------------

    def __evalLet(self, stmt, env, name_space):
        '''To evaluate the given "let" statement'''

        # unpack the statement
        keyw, line_no, (id_name, id_line_no), (rhs, rhs_line_no) = stmt

        # check for illegal variable references
        for vname in self.__extractVars(rhs):
            try:
                eval(vname, env)
            except:
                err('orio.main.tspec.eval: %s: invalid reference: "%s"' % (rhs_line_no, vname))

        # evaluate the RHS expression
        try:
            rhs_val = eval(rhs, env)
        except Exception as e:
            err('orio.main.tspec.eval: %s: failed to evaluate the RHS expression\n --> %s: %s' % (rhs_line_no, e.__class__.__name__, e))

        # update the environment
        env[id_name] = rhs_val

        # return the evaluated statement
        return (keyw, line_no, (id_name, id_line_no), (rhs_val, rhs_line_no))

    #----------------------------------------------------------------------------

    def __evalParam(self, stmt, env, name_space):
        '''To evaluate the given "param" statement'''

        # unpack the statement
        keyw, line_no, (id_name, id_line_no), is_range, (rhs, rhs_line_no) = stmt

        # check for illegal variable references
        for vname in self.__extractVars(rhs):
            try:
                eval(vname, env)
            except:
                err('orio.main.tspec.eval: %s: invalid reference: "%s"' % (rhs_line_no, vname))

        # evaluate the RHS expression
        try:
            rhs_val = eval(rhs, env)
        except Exception as e:
            err('orio.main.tspec.eval: %s: failed to evaluate the RHS expression\n --> %s: %s' % (rhs_line_no, e.__class__.__name__, e))

        if isinstance(rhs_val, range): rhs_val = list(rhs_val)

        # check the RHS value
        if is_range:
            if not isinstance(rhs_val, list) and not isinstance(rhs_val, tuple):
                err('orio.main.tspec.eval: %s: RHS must be a list/tuple' % rhs_line_no)
            if len(rhs_val) == 0:
                err('orio.main.tspec.eval: %s: RHS must not be an empty list' % rhs_line_no)
            etype = type(rhs_val[0])
            for e in rhs_val:
                if not isinstance(e, etype):
                    err('orio.main.tspec.eval: %s: RHS must be a list of equal-typed elements' % rhs_line_no)
        
        # return the evaluated statement
        return (keyw, line_no, (id_name, id_line_no), is_range, (rhs_val, rhs_line_no))
        
    def __evalOption(self, stmt, env, name_space):
        '''Evaluate the given "option" statement'''
        # Example:  
        # def cmdline_params {
        #    option '-a' = [0,1,2];
        # }

        # unpack the statement
        keyw, line_no, (option_string, id_line_no), is_range, (rhs, rhs_line_no) = stmt

        # check for illegal variable references in the RHS
        for vname in self.__extractVars(rhs):
            try:
                eval(vname, env)
            except:
                err('orio.main.tspec.eval: %s: invalid reference: "%s"' % (rhs_line_no, vname))

        # evaluate the RHS expression
        try:
            rhs_val = eval(rhs, env)
        except Exception as e:
            err('orio.main.tspec.eval: %s: failed to evaluate the RHS expression\n --> %s: %s' % (rhs_line_no, e.__class__.__name__, e))

        # check the RHS value
        if is_range:
            if not isinstance(rhs_val, list) and not isinstance(rhs_val, tuple):
                err('orio.main.tspec.eval: %s: RHS must be a list/tuple' % rhs_line_no)
            if len(rhs_val) == 0:
                err('orio.main.tspec.eval: %s: RHS must not be an empty list' % rhs_line_no)
            etype = type(rhs_val[0])
            for e in rhs_val:
                if not isinstance(e, etype):
                    err('orio.main.tspec.eval: %s: RHS must be a list of equal-typed elements' % rhs_line_no)
        
        # return the evaluated statement
        return (keyw, line_no, (option_string, id_line_no), is_range, (rhs_val, rhs_line_no))

    #----------------------------------------------------------------------------

    def __evalSpec(self, stmt, env, name_space):
        '''To evaluate the given "spec" statement'''

        # copy the environment and name space
        env = env.copy()
        name_space = name_space.copy()

        # unpack the statement
        keyw, line_no, (id_name, id_line_no), stmt_seq = stmt

        # evaluate each statement in the specification statement body
        stmt_seq = self.__evaluate(stmt_seq, env, name_space)

        # return the evaluated statement
        return (keyw, line_no, (id_name, id_line_no), stmt_seq)

    #----------------------------------------------------------------------------

    def __evaluate(self, stmt, env, name_space):
        '''
        To evaluate the given statement. Note that the given statement could be a statement sequence.
        '''

        # in the case of a single statement
        if isinstance(stmt, tuple):

            # get keyword, line number, and identifier name
            keyw = stmt[0]
            line_no = stmt[1]
            (id_name, id_line_no) = stmt[2]
                

            # check for any predefined name
            if id_name in name_space:
                err('orio.main.tspec.eval: %s: name "%s" already defined' % (id_line_no, id_name))

            # first update the name space before evaluation (if necessary)
            if keyw in ('def', 'spec'):
                name_space[id_name] = keyw
    
            # evaluate each statement
            if keyw == 'arg':
                e = self.__evalArg(stmt, env, name_space)
            elif keyw == 'constraint':
                e = self.__evalConstraint(stmt, env, name_space)
            elif keyw == 'decl':
                e = self.__evalDecl(stmt, env, name_space)
            elif keyw == 'def':
                e = self.__evalDef(stmt, env, name_space)
            elif keyw == 'let':
                e = self.__evalLet(stmt, env, name_space)
            elif keyw == 'param':
                e = self.__evalParam(stmt, env, name_space)
            elif keyw == 'option':
                e = self.__evalOption(stmt, env, name_space)
            elif keyw == 'spec':
                e = self.__evalSpec(stmt, env, name_space)
            else:
                err('orio.main.tspec.eval internal error: %s: unrecognized TSpec statement' % line_no)

            # update the name_space
            name_space[id_name] = keyw

            # return the evaluated statement
            return e
            
        # in the case of a sequence of statements
        elif isinstance(stmt, list):

            # evaluate each statement
            e = [self.__evaluate(s, env, name_space) for s in stmt]

            # return the evaluated statement sequence
            return e

        # unexpected input
        else:
            err('orio.main.tspec.eval internal error:  unexpected type of TSpec statement')
            
    #----------------------------------------------------------------------------

    def evaluate(self, stmt_seq):
        '''To evaluate the given statement sequence'''
        return self.__evaluate(stmt_seq, dict(list(builtins.__dict__.items()) + list(itertools.__dict__.items()) + list(string.__dict__.items())), {})



