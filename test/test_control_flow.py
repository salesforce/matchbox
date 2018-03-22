import torch
from torch.autograd import Variable
from torch import nn
import matchbox
from matchbox import functional as F
from matchbox import MaskedBatch, batch
from matchbox.test_utils import mb_test, mb_assert

import random

@batch
def while_loop(x):
    while x > 0:
        x = x - 1
    return x

def test_while():
    mb_test(while_loop, (4, ()))
