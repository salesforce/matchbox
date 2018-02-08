import torch
from torch.autograd import Variable
from torch import nn
import matchbox
from matchbox import functional as F
from matchbox import MaskedBatch, wrap
from matchbox.test_utils import mb_test, mb_assert

import random

def test_rnn_cell():
    mb_test(nn.RNNCell(2, 2), (4, (False, 2)), (4, (False, 2)))

@wrap
def simple_rnn(x, h0, cell):
    h = h0
    for xt in x.unbind(1):
        h = cell(xt, h)
    return h

def test_rnn():
    def SimpleRNN(cell):
        def inner(x, h0):
            return simple_rnn(x, h0, cell)
        return inner
    mb_test(SimpleRNN(nn.RNNCell(2, 2)),
            (4, (True, 3), (False, 2)), (4, (False, 2)))

class RNNClass(nn.Module):
    def __init__(self, cell):
        super().__init__()
        self.cell = cell
    @wrap
    def forward(self, x, h0=None):
        h = x.new(1, x.size(-1)).zero_() if h0 is None else h0
        for xt in x.unbind(1):
            h = self.cell(xt, h)
        return h

def test_rnn_class():
    mb_test(RNNClass(nn.RNNCell(2, 2)),
            (4, (True, 3), (False, 2)))
