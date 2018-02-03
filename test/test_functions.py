import torch
from torch.autograd import Variable
import matchbox
from matchbox import functional as F
from matchbox import MaskedBatch
from matchbox.test_utils import mb_test, mb_assert

import random

def test_embedding():
    xs = [Variable(torch.rand(1, random.randint(1, 3)).mul_(5).long())
          for i in range(4)]
    W = Variable(torch.rand(5, 2))
    xb = MaskedBatch.fromlist(xs, (True,))
    mb_assert(F.embedding, (xs, W), (xb, W), 4)

def test_mean():
    mb_test(lambda x: x.mean(2), (4, (True, 3), (False, 2)))

def test_std():
    mb_test(lambda x: x.std(2), (4, (True, 3), (False, 2)))

def test_matmul():
    mb_test(lambda a, b: a @ b, (4, (True, 3), (False, 2)), (4, (False, 2), (True, 3)))
