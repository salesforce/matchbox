import torch
from torch.autograd import Variable
import matchbox
from matchbox import functional as F
from matchbox import MaskedBatch

import random
import numpy as np

def mb_test(f, *dimsarr):
    xsarr, xbarr = [], []
    bs = None
    for dims in dimsarr:
        if not isinstance(dims, tuple):
            xsarr.append(xsarr[dims])
            xbarr.append(xbarr[dims])
        elif isinstance(dims[-1], tuple):
            bs = dims[0]
            sizes = (1, *(random.randint(1, size) if b else size
                          for b, size in dims[1:]))
            xs = [Variable(torch.rand(*sizes)) for i in range(dims[0])]
            xsarr.append(xs)
            xbarr.append(MaskedBatch.fromlist(xs, tuple(b for b, d in dims[1:])))
        else:
            x = Variable(torch.rand(*dims))
            xsarr.append(x)
            xbarr.append(x)
    mb_assert(f, xsarr, xbarr, bs)

def mb_assert(f, xsarr, xbarr, bs):
    ys = [f(*(xs[j] if isinstance(xs, list) else xs for xs in xsarr))
          for j in range(bs)]
    ybs = f(*xbarr).examples()
    for y, yb in zip(ys, ybs):
        np.testing.assert_allclose(y.data.numpy(), yb.data.numpy(), rtol=1e-6)
    #assert all(y.eq(yb).all() for y, yb in zip(ys, ybs))
