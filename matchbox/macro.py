# Copyright (c) 2018, salesforce.com, inc.
# All rights reserved.
# Licensed under the BSD 3-Clause license.
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

from collections import defaultdict

import astor
import gast

from .recompile import compile_function, code_to_ast

def pushmask(mask_expr):
    return gast.Expr(gast.Call(
        gast.Attribute(gast.Name('matchbox', gast.Load(), None),
                       gast.Name('push_execution_mask', gast.Load(), None),
                       gast.Load()),
        [mask_expr], []))

popmask = gast.Expr(gast.Call(
    gast.Attribute(gast.Name('matchbox', gast.Load(), None),
                   gast.Name('pop_execution_mask', gast.Load(), None),
                   gast.Load()),
    [], []))

def any_active(mask_expr):
    return gast.Call(gast.Attribute( # TODO any over dim 0
        mask_expr, gast.Name('any', gast.Load(), None), gast.Load()), [], [])

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

    def visit_FunctionDef(self, node):
        node = self.generic_visit(node)
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
        node = self.generic_visit(node)
        return self.synchronize_lcds(node)

    def visit_If(self, node):
        node = self.generic_visit(node)
        node = self.add_mask(node, node.test)
        nodes = [node]
        if len(node.orelse) > 0:
            test_inverse = gast.Call(
                gast.Attribute(
                    node.test, gast.Name('eq', gast.Load(), None), gast.Load()),
                [gast.Num(0)], [])
            else_node = gast.If(any_active(test_inverse), node.orelse, [])
            node.orelse = []
            self.add_mask(else_node, test_inverse)
            nodes.append(else_node)
        node.test = any_active(node.test)
        return nodes

    def visit_While(self, node):
        if len(node.orelse) > 0:
            raise NotImplementedError("cannot process while-else")
        node = self.generic_visit(node)
        node = self.add_mask(node, node.test)
        node.test = any_active(node.test)
        node = self.synchronize_lcds(node)
        return node

    def visit_Assign(self, node):
        if len(node.targets) > 1:
            raise NotImplementedError("cannot process multiple assignment")
        if not isinstance(node.targets[0], gast.Name):
            raise NotImplementedError("cannot process indexed assignment")
        # $lhs = $lhs.update_($rhs, matchbox.EXECUTION_MASK) if (lhs in vars()
        # or lhs in globals()) and isinstance($lhs, (matchbox.MaskedBatch,
        # matchbox.TENSOR_TYPE)) else $rhs
        node.value = gast.IfExp(
            gast.BoolOp(gast.And(),
                [gast.BoolOp(gast.Or(),
                    [gast.Compare(gast.Str(node.targets[0].id), [gast.In()],
                        [gast.Call(gast.Name('vars', gast.Load, None),
                                   [], [])]),
                     gast.Compare(gast.Str(node.targets[0].id), [gast.In()],
                        [gast.Call(gast.Name('globals', gast.Load, None),
                                   [], [])])]),
                 # gast.Compare(
                 #    gast.Attribute(
                 #      gast.Name('matchbox', gast.Load(), None),
                 #      gast.Name('EXECUTION_MASK', gast.Load(), None),
                 #      gast.Load()),
                 #    [gast.IsNot()],
                 #    [gast.NameConstant(None)]),
                 gast.Call(gast.Name('isinstance', gast.Load(), None),
                           [node.targets[0], gast.Tuple(
                            [gast.Attribute(
                                gast.Name('matchbox', gast.Load(), None),
                                gast.Name('MaskedBatch', gast.Load(), None),
                                gast.Load()),
                             gast.Attribute(
                                gast.Name('matchbox', gast.Load(), None),
                                gast.Name('TENSOR_TYPE', gast.Load(), None),
                                gast.Load())], gast.Load())], [])]),
            gast.Call(
                gast.Attribute(
                    gast.Name(node.targets[0].id, gast.Load(), None),
                    gast.Name('_update', gast.Load(), None),
                    gast.Load()),
                [node.value, gast.Attribute(
                  gast.Name('matchbox', gast.Load(), None),
                  gast.Name('EXECUTION_MASK', gast.Load(), None),
                  gast.Load())], []),
            node.value)
        return node

    def add_mask(self, node, mask):
        node.body = [pushmask(mask)] + node.body + [popmask]
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
