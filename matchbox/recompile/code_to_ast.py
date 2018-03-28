# This code is MODIFIED from the version in Astor (github.com/berkerpeksag/astor)
# found at https://github.com/berkerpeksag/astor/blob/master/astor/file_util.py
#
# Part of the astor library for Python AST manipulation.
# License: 3-clause BSD
# Copyright (c) 2012-2015 Patrick Maupin
# Copyright (c) 2013-2015 Berker Peksag

import ast
import sys
import os

import gast

class CodeToAst(object):
    """Given a module, or a function that was compiled as part
    of a module, re-compile the module into an AST and extract
    the sub-AST for the function.  Allow caching to reduce
    number of compiles.
    Also contains static helper utility functions to
    look for python files, to parse python files, and to extract
    the file/line information from a code object.
    """

    @staticmethod
    def parse_file(fname):
        """Parse a python file into an AST.
        This is a very thin wrapper around ast.parse
            TODO: Handle encodings other than the default (issue #26)
        """
        try:
            with open(fname, 'r') as f:
                fstr = f.read()
        except IOError:
            if fname != 'stdin':
                raise
            sys.stdout.write('\nReading from stdin:\n\n')
            fstr = sys.stdin.read()
        fstr = fstr.replace('\r\n', '\n').replace('\r', '\n')
        if not fstr.endswith('\n'):
            fstr += '\n'
        return ast.parse(fstr, filename=fname)

    @staticmethod
    def get_file_info(codeobj):
        """Returns the file and line number of a code object.
            If the code object has a __file__ attribute (e.g. if
            it is a module), then the returned line number will
            be 0
        """
        fname = getattr(codeobj, '__file__', None)
        linenum = 0
        if fname is None:
            func_code = codeobj.__code__
            fname = func_code.co_filename
            linenum = func_code.co_firstlineno
        fname = fname.replace('.pyc', '.py')
        return fname, linenum

    def __init__(self, cache=None):
        self.cache = cache or {}

    def __call__(self, codeobj):
        cache = self.cache
        key = self.get_file_info(codeobj)
        result = cache.get(key)
        if result is not None:
            return result
        fname = key[0]
        cache[(fname, 0)] = mod_ast = gast.ast_to_gast(self.parse_file(fname))
        for obj in gast.walk(mod_ast):
            if isinstance(obj, gast.FunctionDef):
                cache[(fname, obj.lineno)] = obj
        return cache[key]

code_to_ast = CodeToAst()
