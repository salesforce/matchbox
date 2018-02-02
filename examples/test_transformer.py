import torch
from torch import nn
from torch.autograd import Variable
from torchtext import data, datasets
import matchbox
from matchbox import functional as F
from matchbox import MaskedBatch

import random

from transformer import LayerNorm

def _mbtest(f, *dimsarr):
    xsarr, xbarr = [], []
    for dims in dimsarr:
        sizes = (1, *(random.randint(1, size) if b else size
                      for b, size in dims[1:]))
        xs = [Variable(torch.rand(*sizes)) for i in range(dims[0])]
        xsarr.append(xs)
        xbarr.append(MaskedBatch.fromlist(xs, tuple(b for b, d in dims[1:])))
    ys = [f(*(xs[j] for xs in xsarr)) for j in range(len(xs[0]))]
    ybs = f(*xbarr).examples()
    assert all(y.eq(yb).all() for y, yb in zip(ys, ybs))

def test_LayerNorm():
    _mbtest(LayerNorm(2), (4, (True, 3), (False, 2)))
