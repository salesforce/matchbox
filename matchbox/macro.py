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
        return self.visit_loop(node)
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        node.decorator_list = [d for d in node.decorator_list if d.id != 'wrap']
        return node
    def visit_loop(self, node):
        node = FuseAttributes().visit(node)
        loads, stores = set(), set()
        for child in node.body:
            for n in gast.walk(child):
                if isinstance(n, gast.Name) and isinstance(n.ctx, gast.Load):
                    loads.add(n.id)
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
                        [child.value], [])
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
        return [node] + synchronizes

def wrap(fn):
    node = code_to_ast(fn)
    node = LoopAccumulation().visit(node)
    return compile_function(node)
