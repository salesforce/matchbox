# Copyright (c) 2018, salesforce.com, inc.
# All rights reserved.
# Licensed under the BSD 3-Clause license.
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

from collections import defaultdict

import astor
import gast

from .recompile import compile_function, code_to_ast

class FuseAttributes(gast.NodeTransformer):
    '''Transform foo.bar to foo_DOT_bar'''
    def visit_Attribute(self, node):
        self.generic_visit(node)
        if not isinstance(node.value, gast.Name):
            return node
        attrname = node.attr if isinstance(node.attr, str) else node.attr.id
        return gast.Name(node.value.id + '_DOT_' + attrname,
                         node.value.ctx, None)

class SplitAttributes(gast.NodeTransformer):
    '''Transform foo_DOT_bar to foo.bar'''
    def visit_Name(self, node):
        if '_DOT_' not in node.id:
            return node
        value, attr = node.id.split('_DOT_')
        return gast.Attribute(gast.Name(value, node.ctx, None),
                              attr, node.ctx)

class ExecutionMasking(gast.NodeTransformer):
    def __init__(self):
        super().__init__()
        self.execution_mask_stack = []

    @property
    def execution_mask(self):
        def compose(masks):
            if len(masks) == 0:
                # return `None`
                return gast.NameConstant(value=None)
            if len(masks) == 1:
                return masks[0]
            # return masks[-1] * compose(masks[:-1])
            return gast.BinOp(masks[-1], gast.Mult, compose(masks[:-1]))
        return compose(self.execution_mask_stack)

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        def is_batch_decorator(d):
            if isinstance(d, gast.Name):
                return d.id == 'batch'
            elif isinstance(d, gast.Attribute):
                return is_batch_decorator(d.attr)
            return d == 'batch'
        node.decorator_list = [d for d in node.decorator_list
                               if not is_batch_decorator(d)]
        return node

    def visit_For(self, node):
        self.generic_visit(node)
        return self.synchronize_lcds(node)

    def visit_If(self, node):
        self.execution_mask_stack.append(node.test)
        for child in node.body:
            self.generic_visit(child)
        self.execution_mask_stack.pop()
        if len(node.orelse) > 0:
            test_inverse = gast.BinOp(1, gast.Sub, node.test)
            self.execution_mask_stack.append(test_inverse)
            for child in node.orelse:
                self.generic_visit(child)
            self.execution_mask_stack.pop()
        node.test = gast.Call(gast.Attribute( # TODO any over dim 0
            node.test, gast.Name('any', gast.Load(), None), None), [], [])
        #TODO move orelse into its own If so it can have test_inverse.any()

    def visit_While(self, node):
        if len(node.orelse) > 0:
            raise NotImplementedError("cannot process while-else")
        self.execution_mask_stack.append(node.test)
        self.generic_visit(node)
        self.execution_mask_stack.pop()
        node.test = gast.Call(gast.Attribute( # TODO any over dim 0
            node.test, gast.Name('any', gast.Load(), None), None), [], [])
        return self.synchronize_lcds(node)

    def visit_Assign(self, node):
        if len(node.targets) > 1:
            raise NotImplementedError("cannot process multiple assignment")
        node.value = gast.Call(
            gast.Attribute(
                gast.Name(node.targets[0].id, gast.Load(), None),
                gast.Name('_update', gast.Load(), None),
                None),
            [node.value, self.execution_mask], [])
        return node

    def synchronize_lcds(self, node):
        node = FuseAttributes().visit(node)
        loads, lcds = defaultdict(list), set()
        for child in node.body:
            for n in gast.walk(child):
                if isinstance(n, gast.Name) and isinstance(n.ctx, gast.Load):
                    loads[n.id].append(n)
            if isinstance(child, gast.Assign):
                name = child.targets[0].id
                if name in loads:
                    if name in lcds:
                        raise NotImplementedError("cannot process LCD "
                                                  "stored to twice")
                    lcds.add(name)
        node = SplitAttributes().visit(node)
        synchronizes = []
        for name in lcds:
            synchronize = gast.Assign(
                [gast.Name(name, gast.Store(), None)],
                gast.Call(
                    gast.Attribute(
                        gast.Name(name, gast.Load(), None),
                        gast.Name('_synchronize', gast.Load(), None),
                        None),
                    [], []))
            synchronizes.append(synchronize)
        node.body.extend(synchronizes)
        return node

def batch(fn):
    node = code_to_ast(fn)
    node = ExecutionMasking().visit(node)
    return compile_function(node, fn.__globals__)
