# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#      Unless required by applicable law or agreed to in writing, software
#      distributed under the License is distributed on an "AS IS" BASIS,
#      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#      See the License for the specific language governing permissions and
#      limitations under the License.
"""Going from AST or source code to executable code."""
from __future__ import absolute_import
import os
import tempfile
from uuid import uuid4

import astor
import gast
import six
if six.PY3:
    from importlib import util
else:
    import imp

def compile_file(source, globals_=None):
    """Compile by saving to file and importing that.
    Compiling the AST/source code this way ensures that the source code is
    readable by e.g. `pdb` or `inspect`.
    Args:
    source: The code to compile, either as a string or as an AST.
    globals_: A dictionary of variables that should be available as globals in
        the compiled module. They will be monkey patched after importing the
        module.
    Returns:
    A module object containing the compiled source code.
    """
    if isinstance(source, gast.AST):
        source = astor.to_source(gast.gast_to_ast(source))

    # Write source to temporary file
    tempdir = tempfile.mkdtemp()
    uuid = str(uuid4().hex[:4])
    tmpname = os.path.join(tempdir, 'matchbox_%s.py' % uuid)
    with open(tmpname, 'w') as f:
        f.write(source)

    # Load the temporary file as a module
    module_name = 'matchbox_%s' % uuid
    if six.PY3:
        spec = util.spec_from_file_location(module_name, tmpname)
        m = util.module_from_spec(spec)
        spec.loader.exec_module(m)
    else:
        m = imp.load_source(module_name, tmpname)

    # Update the modules namespace
    if globals_:
        m.__dict__.update(globals_)
    return m


def compile_function(node, globals_=None):
    """Convert an AST into a function with inspectable source.
    This function uses `compile_file` internally, but instead of returning the
    entire module it will return the function only.
    Args:
    node: A `FunctionDef` node or a `Module` node which contains at least one
        `FunctionDef` node. If a module contains multiple functions, a handle
        to the first one will be returned.
    globals_: See `compile_file`
    Returns:
    A handle to the compiled function.
    Raises:
    TypeError: If the input is not a string or AST.
    ValueError: If no function can be found.
    """
    module = compile_file(node, globals_)
    return getattr(module, node.name)

# """
# Part of the astor library for Python AST manipulation.
# License: 3-clause BSD
# Copyright (c) 2012-2015 Patrick Maupin
# Copyright (c) 2013-2015 Berker Peksag
# Functions that interact with the filesystem go here.
# """

import ast
import sys
import os


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
