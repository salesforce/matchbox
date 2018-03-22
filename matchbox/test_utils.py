import torch
from torch.autograd import Variable
import matchbox
from matchbox import functional as F
from matchbox import MaskedBatch

import random
import numpy as np

def mb_rand(*dims):
    dims = [dim for dim in dims if dim != ()]
    xs = [Variable(torch.rand(1, *(random.randint(1, size) if b else size
                  for b, size in dims[1:]))) for i in range(dims[0])]
    xb = MaskedBatch.fromlist(xs, tuple(b for b, d in dims[1:]))
    return xs, xb

def mb_assert_allclose(xs, ybs):
    if isinstance(ybs, Variable):
        np.testing.assert_allclose(xs.data.numpy(), ybs.data.numpy(), rtol=1e-3)
    elif isinstance(ybs, MaskedBatch):
        mb_assert_allclose(xs, ybs.examples())
    else:
        if isinstance(ybs, (list, tuple)):
            for j, yb in enumerate(ybs):
                for i, y in enumerate(yb.examples()):
                    mb_assert_allclose(xs[i][j], y)
        else:
            for x, yb in zip(xs, ybs):
                mb_assert_allclose(x, yb)

def mb_assert(f, xsarr, xbarr, bs):
    ys = [f(*(xs[j] if isinstance(xs, list) else xs for xs in xsarr))
          for j in range(bs)]
    ybs = f(*xbarr)
    mb_assert_allclose(ys, ybs)

def mb_test(f, *dimsarr):
    xsarr, xbarr = [], []
    bs = None
    for dims in dimsarr:
        if not isinstance(dims, tuple):
            xsarr.append(xsarr[dims])
            xbarr.append(xbarr[dims])
        elif isinstance(dims[-1], tuple):
            bs = dims[0]
            xs, xb = mb_rand(*dims)
            xsarr.append(xs)
            xbarr.append(xb)
        else:
            x = Variable(torch.rand(*dims))
            xsarr.append(x)
            xbarr.append(x)
    mb_assert(f, xsarr, xbarr, bs)
