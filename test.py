import torch
from torch.autograd import Variable
from matchbox import MaskedBatch
from matchbox import functional as F

data = Variable(torch.rand(3, 3))
mask = Variable(torch.Tensor([[1, 1, 1], [1, 0, 0], [1, 1, 0]]))
x = MaskedBatch(data * mask, mask, (True,))

print(x)
x = F.softmax(x, dim=-1)
print(x)
