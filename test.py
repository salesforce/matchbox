import torch
from torch.autograd import Variable
from matchbox import MaskedBatch
from matchbox import functional as F
from torch import nn

data = Variable(torch.rand(3, 3))
mask = Variable(torch.Tensor([[1, 1, 1], [1, 0, 0], [1, 1, 0]]))
x = MaskedBatch(data * mask, mask, (True,))

print(x)
x = F.softmax(x, dim=-1)
print(x)

import random

examples = [Variable(torch.rand(1, random.randint(1, 5))) for i in range(3)]
print(examples)
batch = MaskedBatch.fromlist(examples, (True,))
print(batch)

examples = [Variable(torch.rand(1, random.randint(1, 5), random.randint(1, 5)))
            for i in range(3)]
print(examples)
batch = MaskedBatch.fromlist(examples, (True, True))
print(batch)

batch = F.softmax(batch, dim=-1)
print(batch)

srcs = [Variable(torch.rand(1, random.randint(1, 5), 4))
            for i in range(3)]
trgs = [Variable(torch.rand(1, random.randint(1, 5), 4))
            for i in range(3)]
src = MaskedBatch.fromlist(srcs, (True, False))
trg = MaskedBatch.fromlist(trgs, (True, False))
print(src)
print(trg)

res = src @ trg.transpose(1, 2)
print(res)

res = res[:, 0]
print(res)

def rnn(x, h0, cell):
    h = h0
    for xt in F.unbind(x, 1):
        h = cell(xt, h)
    return h

# TODO make this work
# def rnn(x, h0, cell):
#     h = [h0]
#     for xt in F.unbind(x, 1):
#         h.append(cell(xt, h[-1]))
#     return h[1:]

b, t, ci, co = 4, 3, 2, 2
cell = nn.RNNCell(ci, co)
h0 = MaskedBatch(Variable(torch.rand(b, co)),
                 Variable(torch.FloatTensor([[1]])), (False,))
x = MaskedBatch.fromlist([Variable(torch.rand(1, random.randint(1, t), ci))
                          for i in range(b)], (True, False))
print(h0)
print(x)

h = rnn(x, h0, cell)
print(h)
