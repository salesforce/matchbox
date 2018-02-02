import torch
from torch import nn
from torch.autograd import Variable
from torchtext import data, datasets
import matchbox
from matchbox import functional as F
from matchbox import MaskedBatch

import random

from transformer import LayerNorm

def test_LayerNorm():
    b, t, c = 4, 3, 2
    layernorm = LayerNorm(c)
    xs = [Variable(torch.rand(1, random.randint(1, t), c)) for i in range(b)]
    xb = MaskedBatch.fromlist(xs, (True, False))
    ys = [layernorm(x) for x in xs]
    assert ys == list(layernorm(xb).examples())
