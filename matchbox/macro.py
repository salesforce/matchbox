from collections import defaultdict

import astor
import gast

from .recompile import compile_function, code_to_ast

class FuseAttributes(gast.NodeTransformer):
    def visit_Attribute(self, node):
        self.generic_visit(node)
        if not isinstance(node.value, gast.Name):
            return node
        return gast.Name(node.value.id + '_DOT_' + node.attr,
                         node.value.ctx, None)

class SplitAttributes(gast.NodeTransformer):
    def visit_Name(self, node):
        if '_DOT_' not in node.id:
            return node
        value, attr = node.id.split('_DOT_')
        return gast.Attribute(gast.Name(value, node.ctx, None),
                              attr, node.ctx)

class LoopAccumulation(gast.NodeTransformer):
    def generic_visit(self, node):
        super().generic_visit(node)
        #print('generic:', astor.dump_tree(node))
        return node
    def visit_For(self, node):
        self.generic_visit(node)
        return self.visit_loop(node)
    def visit_While(self, node):
        self.generic_visit(node)
        if len(node.orelse) > 0:
            raise NotImplementedError("cannot process while-else")
        test = node.test
        node.test = gast.Call(gast.Attribute(
            test, gast.Name('any', gast.Load(), None), None), [], [])
        return self.visit_loop(node, test)
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        node.decorator_list = [d for d in node.decorator_list
                               if d.id != 'batch']
        return node
    def visit_loop(self, node, update_mask=gast.NameConstant(value=None)):
        node = FuseAttributes().visit(node)
        loads, stores = defaultdict(list), set()
        for child in node.body:
            for n in gast.walk(child):
                if isinstance(n, gast.Name) and isinstance(n.ctx, gast.Load):
                    loads[n.id].append(n)
            if isinstance(child, gast.Assign):
                if len(child.targets) > 1:
                    raise NotImplementedError("cannot process LCD that is "
                                              "part of multiple assignment")
                name = child.targets[0].id
                if name in loads:
                    if name in stores:
                        raise NotImplementedError("cannot process LCD "
                                                  "stored to twice")
                    # $var = $expr -> $var = $var._update($expr)
                    child.value = gast.Call(
                        gast.Attribute(
                            gast.Name(name, gast.Load(), None),
                            gast.Name('_update', gast.Load(), None),
                            None),
                        [child.value, update_mask], [])
                    stores.add(name)
        node = SplitAttributes().visit(node)
        synchronizes = []
        for name in stores:
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
    node = LoopAccumulation().visit(node)
    return compile_function(node, fn.__globals__)
