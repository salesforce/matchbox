from tangent.transformers import TreeTransformer
import astor
import gast

def simple_if(x):
    if x[0, 0] > 0:
        return x
    elif x[0, 0] < 0:
        pass
    else:
        pass
    return -x

def rnn(x, h0, cell):
    h = h0
    for xt in x.unbind(1):
        h = cell(xt, h)
    return h

# si_ast = astor.code_to_ast(simple_if)
# print(astor.dump_tree(si_ast))
# print(astor.to_source(si_ast))
rnn_ast = gast.ast_to_gast(astor.code_to_ast(rnn))
print(astor.dump_tree(rnn_ast))
print(astor.to_source(gast.gast_to_ast(rnn_ast)))

import random, string
def gensym():
    return '_' + ''.join(random.choice(string.ascii_lowercase) for _ in range(10))

class LambdaLift(TreeTransformer):
    def generic_visit(self, node):
        super().generic_visit(node)
        #print('generic:', astor.dump_tree(node))
        return node
    def visit_For(self, node):
        super().generic_visit(node)
        #print('for:', astor.dump_tree(node))
        fn_name = gensym()
        self.prepend(gast.Assign([
            gast.Name('_', gast.Store(), None)],
            gast.Call(gast.Name('_for', gast.Load(), None),
                [gast.Name(fn_name, gast.Load(), None), node.iter], [])))
        self.prepend(gast.FunctionDef(
            fn_name, gast.arguments([node.target], None, [], [], None, []),
            node.body, [], None))
        return None

# class LiftReturns(ast.NodeTransformer):
#     def __init__(self):
#         super().__init__()
#         self.functions = []
#     def generic_visit(self, node):
#         super().generic_visit(node)
#         print('generic:', astor.dump_tree(node))
#         return node
#     def visit_FunctionDef(self, node):
#         self.functions.append(node)
#         self.generic_visit(node)
#         self.functions.pop()
#         return node
#     def visit_If(self, node):
#         print('if:', astor.dump_tree(node))
#
#         # if any(isinstance(stmt, ast.Return) for stmt in node.body):
#         #      or
#         #     any(isinstance(stmt, ast.Return) for stmt in node.orelse):
#
#         for line in node.body:
#             if isinstance(line, ast.Return):
#                 function = self.functions[-1]
#                 #function.body[]
#                 #
#                 #break
#                 print(' return:', astor.dump_tree(line))
#         return node

class MyWalker(astor.TreeWalk):
    pass

rnn_ast = LambdaLift().visit(rnn_ast)
print(astor.dump_tree(rnn_ast))
print(astor.to_source(gast.gast_to_ast(rnn_ast)))


# things we need to do:
# - lift `return` out of body
#   - this sounds difficult; check onelinerizer
#   - possible equivalent? return:
#     a. deletes later code in body,
#     b. moves all code after the if to the end of the else, and
#     c. verifies that all branches end in a return, then lifts to a variable and moves the return to the end of the function
# - lift body to function
#   - this means identifying variables updated during a loop
#   - not sure if we also have to lift the closure parameters
# - lift control flow to higher-order function
